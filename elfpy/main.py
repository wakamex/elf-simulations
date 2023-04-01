"""Main body of the elfpy simulation"""
from __future__ import annotations
from dataclasses import dataclass, field  # types will be strings by default in 3.11
from enum import Enum
from abc import ABC
from copy import deepcopy
from decimal import Decimal

import logging
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Type, Any, Dict, Optional

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from elfpy.agent import Agent

# Setup barebones logging without a handler for users to adapt to their needs.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# This is the minimum allowed value to be passed into calculations to avoid
# problems with sign flips that occur when the floating point range is exceeded.
WEI = 1e-18  # smallest denomination of ether

# The maximum allowed difference between the base reserves and bond reserves.
# This value was calculated using trial and error and is close to the maximum
# difference between the reserves that will not result in a sign flip when a
# small trade is put on.
MAX_RESERVES_DIFFERENCE = 2e10

# The maximum allowed precision error.
# This value was selected based on one test not passing without it.
# apply_delta() below checks if reserves are negative within the threshold,
# and sets them to 0 if so.
# TODO: we shouldn't have to adjsut this -- we need to reesolve rounding errors
PRECISION_THRESHOLD = 1e-8


def freezable(frozen: bool = False, no_new_attribs: bool = False) -> Type:
    r"""A wrapper that allows classes to be frozen, such that existing member attributes cannot be changed"""

    def decorator(cls: Type) -> Type:
        @wraps(wrapped=cls, updated=())
        class FrozenClass(cls):
            """Subclass cls to enable freezing of attributes

            .. todo:: resolve why pyright cannot access member "freeze" when instantiated_class.freeze() is called
            """

            def __init__(self, *args, frozen=frozen, no_new_attribs=no_new_attribs, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                super().__setattr__("frozen", frozen)
                super().__setattr__("no_new_attribs", no_new_attribs)

            def __setattr__(self, attrib: str, value: Any) -> None:
                if hasattr(self, attrib) and hasattr(self, "frozen") and getattr(self, "frozen"):
                    raise AttributeError(f"{self.__class__.__name__} is frozen, cannot change attribute '{attrib}'.")
                if not hasattr(self, attrib) and hasattr(self, "no_new_attribs") and getattr(self, "no_new_attribs"):
                    raise AttributeError(
                        f"{self.__class__.__name__} has no_new_attribs set, cannot add attribute '{attrib}'."
                    )
                super().__setattr__(attrib, value)

            def freeze(self) -> None:
                """disallows changing existing members"""
                super().__setattr__("frozen", True)

            def disable_new_attribs(self) -> None:
                """disallows adding new members"""
                super().__setattr__("no_new_attribs", True)

        return FrozenClass

    return decorator


@freezable(frozen=False, no_new_attribs=True)
@dataclass
class Config:
    """Data object for storing user simulation config parameters

    .. todo:: Rename the {trade/redemption}_fee_percent variables so that they doesn't use "percent"
    """

    # lots of configs!
    # pylint: disable=too-many-instance-attributes

    # Market
    target_liquidity: float = field(
        default=1e6, metadata={"description": "total size of the market pool (bonds + shares"}
    )
    target_volume: float = field(default=0.01, metadata={"description": "fraction of pool liquidity"})
    init_vault_age: float = field(default=0, metadata={"description": "fraction of a year since the vault was opened"})
    base_asset_price: float = field(default=2e3, metadata={"description": "market price"})
    variable_rate: list[float] = field(
        init=False, metadata={"description": "the underlying (variable) variable APR at each time step"}
    )
    init_share_price: float = field(
        init=False, metadata={"description": "initial market share price for the vault asset"}
    )

    # AMM
    pricing_model_name: str = field(
        default="Hyperdrive", metadata={"description": 'Must be "Hyperdrive", or "YieldSpace"'}
    )
    trade_fee_percent: float = field(
        default=0.05, metadata={"description": "LP fee factor (decimal) to charge for trades"}
    )
    redemption_fee_percent: float = field(
        default=0.05, metadata={"description": "LP fee factor (decimal) to charge for redemption"}
    )
    target_fixed_rate: float = field(default=0.1, metadata={"description": "desired fixed apr for as a decimal"})
    floor_fee: float = field(default=0, metadata={"description": "minimum fee percentage (bps)"})

    # Simulation
    # durations
    title: str = field(default="elfpy simulation", metadata={"description": "Text description of the simulation"})
    num_trading_days: int = field(default=3, metadata={"description": "in days; should be <= pool_duration"})
    num_blocks_per_day: int = field(default=3, metadata={"description": "int; agents execute trades each block"})
    num_position_days: int = field(
        default=90, metadata={"description": "time lapse between token mint and expiry as days"}
    )

    # users
    shuffle_users: bool = field(
        default=True, metadata={"description": "Shuffle order of action (as if random gas paid)"}
    )
    agent_policies: list = field(default_factory=list, metadata={"description": "List of strings naming user policies"})
    init_lp: bool = field(default=True, metadata={"description": "If True, use an initial LP agent to seed pool"})

    # vault
    compound_vault_rate: bool = field(
        default=True,
        metadata={"description": "Whether or not to use compounding revenue for the underlying yield source"},
    )
    init_vault_age: float = field(default=0, metadata={"description": "initial vault age"})

    # logging
    log_level: int = field(
        default=logging.INFO, metadata={"description": "Logging level, as defined by stdlib logging"}
    )
    log_filename: str = field(default="simulation.log", metadata={"description": "filename for output logs"})

    # numerical
    precision: int = field(default=64, metadata={"description": "precision of calculations; max is 64"})

    # random
    random_seed: int = field(default=1, metadata={"description": "int to be used for the random seed"})
    rng: Generator = field(
        init=False, compare=False, metadata={"description": "random number generator used in the simulation"}
    )

    def __post_init__(self) -> None:
        r"""init_share_price & rng are a function of other random variables"""
        self.rng = np.random.default_rng(self.random_seed)
        self.variable_rate = [0.05] * self.num_trading_days
        self.init_share_price = (1 + self.variable_rate[0]) ** self.init_vault_age
        self.disable_new_attribs()  # disallow new attributes # pylint: disable=no-member # type: ignore

    def __getitem__(self, key) -> None:
        return getattr(self, key)

    def __setattr__(self, attrib, value) -> None:
        if attrib == "variable_rate":
            if hasattr(self, "variable_rate"):
                self.check_variable_rate()
            super().__setattr__("variable_rate", value)
        elif attrib == "init_share_price":
            super().__setattr__("init_share_price", value)
        else:
            super().__setattr__(attrib, value)

    def check_variable_rate(self) -> None:
        r"""Verify that the variable_rate is the right length"""
        if not isinstance(self.variable_rate, list):
            raise TypeError(
                f"ERROR: variable_rate must be of type list, not {type(self.variable_rate)}."
                f"\nhint: it must be set after Config is initialized."
            )
        if not hasattr(self, "num_trading_days") and len(self.variable_rate) != self.num_trading_days:
            raise ValueError(
                "ERROR: variable_rate must have len equal to num_trading_days = "
                + f"{self.num_trading_days},"
                + f" not {len(self.variable_rate)}"
            )


@freezable(frozen=True, no_new_attribs=True)
@dataclass
class TradeBreakdown:
    r"""A granular breakdown of a trade.

    This includes information relating to fees and slippage.
    """

    without_fee_or_slippage: float
    with_fee: float
    without_fee: float
    fee: float


@freezable(frozen=True, no_new_attribs=True)
@dataclass
class TradeResult:
    r"""The result of performing a trade.

    This includes granular information about the trade details,
    including the amount of fees collected and the total delta.
    Additionally, breakdowns for the updates that should be applied
    to the user and the market are computed.
    """

    user_result: Wallet
    market_result: Wallet
    breakdown: TradeBreakdown


@dataclass
class Quantity:
    r"""An amount with a unit"""

    amount: float
    unit: TokenType


class TokenType(Enum):
    r"""A type of token"""

    BASE = "base"
    PT = "pt"


class MarketActionType(Enum):
    r"""
    The descriptor of an action in a market

    .. todo:: Add INITIALIZE_MARKET = "initialize_market"
    """

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"

    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"

    ADD_LIQUIDITY = "add_liquidity"
    REMOVE_LIQUIDITY = "remove_liquidity"


@dataclass
class Position:
    r"""A Long or Short position

    Parameters
    ----------
    balance : float
        The amount of bonds that the position is short.
    open_share_price: float
        The share price at the time the short was opened.
    """

    balance: float = 0
    open_share_price: float = 0


@dataclass(frozen=False)
class Wallet:
    r"""Stores what is in the agent's wallet

    Parameters
    ----------
    address : int
        The trader's address.
    base : float
        The base assets that held by the trader.
    lp_tokens : float
        The LP tokens held by the trader.
    longs : Dict[float, Position]
        The long positions held by the trader.
    shorts : Dict[float, Position]
        The short positions held by the trader.
    fees_paid : float
        The fees paid by the wallet.
    """

    # pylint: disable=too-many-instance-attributes
    # dataclasses can have many attributes

    # agent identifier
    address: int = field(default=0)

    # reserves
    base: float = 0
    bonds: float = 0
    base_buffer: float = 0
    bond_buffer: float = 0
    lp: float = 0

    # non-fungible (identified by key=mint_time, stored as dict)
    longs: Dict[float, Position] = field(default_factory=Dict)
    shorts: Dict[float, Position] = field(default_factory=Dict)

    share_price: float = 0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


def get_state(wallet_, simulation_state_: SimulationState) -> dict:
    r"""The wallet's current state of public variables
    .. todo:: TODO: return a dataclass instead of dict to avoid having to check keys & the get_state_keys func
    """
    lp_token_value = 0
    if (
        wallet_.lp_tokens > 0 and simulation_state_.market_state.lp_reserves > 0
    ):  # check if LP, and avoid divide by zero
        share_of_pool = wallet_.lp_tokens / simulation_state_.market_state.lp_reserves
        pool_value = (
            simulation_state_.market_state.bond_reserves * simulation_state_.market_state.spot_price  # in base
            + simulation_state_.market_state.share_reserves * simulation_state_.market_state.share_price  # in base
        )
        lp_token_value = pool_value * share_of_pool  # in base
    share_reserves = simulation_state_.market_state.share_reserves
    # compute long values in units of base
    longs_value = 0
    longs_value_no_mock = 0
    for mint_time, long in wallet_.longs.items():
        base = (
            close_long(
                mint_time=mint_time,
                wallet_address=wallet_.address,
                simulation_state_=simulation_state_,
                trade_amount=long.balance,
            )[1].base
            if long.balance > 0 and share_reserves
            else 0.0
        )
        longs_value += base
        base_no_mock = long.balance * simulation_state_.market_state.spot_price
        longs_value_no_mock += base_no_mock
    # compute short values in units of base
    shorts_value = 0
    shorts_value_no_mock = 0
    for mint_time, short in wallet_.shorts.items():
        base = (
            close_short(
                simulation_state_=simulation_state_,
                wallet_address=wallet_.address,
                open_share_price=short.open_share_price,
                trade_amount=short.balance,
                mint_time=mint_time,
            )[1].base
            if short.balance > 0 and share_reserves
            else 0.0
        )
        shorts_value += base
        base_no_mock = short.balance * (1 - simulation_state_.market_state.spot_price)
        shorts_value_no_mock += base_no_mock
    return {
        f"agent_{wallet_.address}_base": wallet_.base,
        f"agent_{wallet_.address}_lp_tokens": lp_token_value,
        f"agent_{wallet_.address}_num_longs": len(wallet_.longs),
        f"agent_{wallet_.address}_num_shorts": len(wallet_.shorts),
        f"agent_{wallet_.address}_total_longs": longs_value,
        f"agent_{wallet_.address}_total_shorts": shorts_value,
        f"agent_{wallet_.address}_total_longs_no_mock": longs_value_no_mock,
        f"agent_{wallet_.address}_total_shorts_no_mock": shorts_value_no_mock,
    }


@freezable(frozen=False, no_new_attribs=True)
@dataclass
class MarketAction:
    r"""Market action specification"""
    action_type: MarketActionType = field(metadata={"description": "type of action to execute"})
    trade_amount: float = field(metadata={"description": "amount to supply for the action"})
    min_amount_out: float = field(default=0, metadata={"properties": "slippage on output, always sets a minimum"})
    wallet: Wallet = field(default_factory=Wallet, metadata={"description": "the wallet to execute the action on"})
    mint_time: Optional[float] = field(default=None, metadata={"description": "the mint time of the position to close"})


def get_pricing_model(model: str):
    """Get the pricing model from the string"""  # sourcery skip: assign-if-exp, reintroduce-else
    model = model.lower()
    if model == "yieldspace":
        return YieldSpacePricingModel
    if model == "hyperdrive":
        return HyperdrivePricingModel
    return PricingModel


@freezable(frozen=False, no_new_attribs=False)
@dataclass
class MarketState:
    r"""The state of an AMM

    Implements a class for all that that an AMM smart contract would hold or would have access to
    For example, reserve numbers are local state variables of the AMM.  The variable_rate will most
    likely be accessible through the AMM as well.

    Attributes
    ----------
    share_reserves: float
        Quantity of shares stored in the market
    bond_reserves: float
        Quantity of bonds stored in the market
    base_buffer: float
        Base amount set aside to account for open longs
    bond_buffer: float
        Bond amount set aside to account for open shorts
    lp: float
        Amount of lp tokens
    variable_rate: float
        .. todo: fill this in
    share_price: float
        .. todo: fill this in
    init_share_price: float
        .. todo: fill this in
    trade_fee_percent : float
        The percentage of the difference between the amount paid without
        slippage and the amount received that will be added to the input
        as a fee.
    redemption_fee_percent : float
        A flat fee applied to the output.  Not used in this equation for Yieldspace.
    """

    # dataclasses can have many attributes
    # pylint: disable=too-many-instance-attributes

    pricing_model: PricingModel = field(default_factory=get_pricing_model("hyperdrive"))
    time: float = 0.0
    term_length_in_days: float = 90
    time_stretch: float = 1
    share_reserves: float = 0.0
    bond_reserves: float = 0.0
    base_buffer: float = 0.0
    bond_buffer: float = 0.0
    lp_reserves: float = 0.0
    variable_rate: float = 0.0
    share_price: float = 1.0
    init_share_price: float = 1.0
    trade_fee_percent: float = 0.0
    redemption_fee_percent: float = 0.0

    @property
    def term_length_in_years(self) -> float:
        """Returns the term length in years"""
        return self.term_length_in_days / 365

    @property
    def apr(self) -> float:
        """Returns the current market apr (returns nan if shares are zero)"""
        return (
            np.nan
            if self.share_reserves <= 0
            else calc_apr_from_spot_price(price=self.spot_price, time_remaining_in_years=self.term_length_in_years)
        )

    @property
    def spot_price(self) -> float:
        """Returns the current market price of the share reserves (returns nan if shares are zero)"""
        return (
            np.nan
            if self.share_reserves == 0
            else self.pricing_model.calc_spot_price_from_reserves(
                market_state=self,
                time_remaining_in_years=self.term_length_in_years,
                time_stretch=self.time_stretch,
            )
        )

    def copy(self) -> "MarketState":
        """Returns a copy of the market state"""
        return deepcopy(self)


def apply_delta(simulation_state_, delta: Wallet) -> None:
    r"""Applies a delta to the market state."""
    simulation_state_.share_reserves += delta.base / simulation_state_.share_price
    simulation_state_.bond_reserves += delta.bonds
    simulation_state_.base_buffer += delta.base_buffer
    simulation_state_.bond_buffer += delta.bond_buffer
    simulation_state_.lp += delta.lp
    simulation_state_.share_price += delta.share_price

    # TODO: issue #146
    # this is an imperfect solution to rounding errors, but it works for now
    # ideally we'd find a more thorough solution than just catching errors
    # when they are.
    for key, value in simulation_state_.__dict__.items():
        if 0 > value > -PRECISION_THRESHOLD:
            logging.debug(
                ("%s=%s is negative within PRECISION_THRESHOLD=%f, setting it to 0"),
                key,
                value,
                PRECISION_THRESHOLD,
            )
            setattr(simulation_state_, key, 0)
        else:
            assert (
                value > -PRECISION_THRESHOLD
            ), f"MarketState values must be > {-PRECISION_THRESHOLD}. Error on {key} = {value}"


def trade_and_update(simulation_state_, action_details: tuple[int, MarketAction]) -> tuple[int, Wallet, Wallet]:
    r"""Execute a trade in the simulated market

    check which of 6 action types are being executed, and handles each case:

    open_long
    .. todo:: fill this in

    close_long
    .. todo:: fill this in

    open_short
    .. todo:: fill this in

    close_short
    .. todo:: fill this in

    add_liquidity
        pricing model computes new market deltas
        market updates its "liquidity pool" wallet, which stores each trade's mint time and user address
        LP tokens are also stored in user wallet as fungible amounts, for ease of use

    remove_liquidity
        market figures out how much the user has contributed (calcualtes their fee weighting)
        market resolves fees, adds this to the agent_action (optional function, to check AMM logic)
        pricing model computes new market deltas
        market updates its "liquidity pool" wallet, which stores each trade's mint time and user address
        LP tokens are also stored in user wallet as fungible amounts, for ease of use
    """
    agent_id, agent_action = action_details
    # TODO: add use of the Quantity type to enforce units while making it clear what units are being used
    # issue 216
    # for each position, specify how to forumulate trade and then execute
    if agent_action.action_type == MarketActionType.OPEN_LONG:  # buy to open long
        market_deltas, agent_deltas = simulation_state_.open_long(
            wallet_address=agent_action.wallet.address,
            trade_amount=agent_action.trade_amount,  # in base: that's the thing in your wallet you want to sell
        )
    elif agent_action.action_type == MarketActionType.CLOSE_LONG:  # sell to close long
        # TODO: python 3.10 includes TypeGuard which properly avoids issues when using Optional type
        mint_time = float(agent_action.mint_time or 0)
        market_deltas, agent_deltas = simulation_state_.close_long(
            wallet_address=agent_action.wallet.address,
            trade_amount=agent_action.trade_amount,  # in bonds: that's the thing in your wallet you want to sell
            mint_time=mint_time,
        )
    elif agent_action.action_type == MarketActionType.OPEN_SHORT:  # sell PT to open short
        market_deltas, agent_deltas = simulation_state_.open_short(
            wallet_address=agent_action.wallet.address,
            trade_amount=agent_action.trade_amount,  # in bonds: that's the thing you want to short
        )
    elif agent_action.action_type == MarketActionType.CLOSE_SHORT:  # buy PT to close short
        # TODO: python 3.10 includes TypeGuard which properly avoids issues when using Optional type
        mint_time = float(agent_action.mint_time or 0)
        open_share_price = agent_action.wallet.shorts[mint_time].open_share_price
        market_deltas, agent_deltas = simulation_state_.close_short(
            wallet_address=agent_action.wallet.address,
            trade_amount=agent_action.trade_amount,  # in bonds: that's the thing you owe, and need to buy back
            mint_time=mint_time,
            open_share_price=open_share_price,
        )
    elif agent_action.action_type == MarketActionType.ADD_LIQUIDITY:
        market_deltas, agent_deltas = simulation_state_.add_liquidity(
            wallet_address=agent_action.wallet.address,
            trade_amount=agent_action.trade_amount,
        )
    elif agent_action.action_type == MarketActionType.REMOVE_LIQUIDITY:
        market_deltas, agent_deltas = simulation_state_.remove_liquidity(
            wallet_address=agent_action.wallet.address,
            trade_amount=agent_action.trade_amount,
        )
    else:
        raise ValueError(f'ERROR: Unknown trade type "{agent_action.action_type}".')
    logging.debug(
        "%s\n%s\nagent_deltas = %s\npre_trade_market = %s",
        agent_action,
        market_deltas,
        agent_deltas,
        simulation_state_.market_state,
    )
    return agent_id, agent_deltas, market_deltas


### Spot Price and APR ###
def calc_apr_from_spot_price(price: float, time_remaining_in_years: float):
    """Returns the APR (decimal) given the current (positive) base asset price and the remaining pool duration"""
    assert price > 0, (
        "utils.price.calc_apr_from_spot_price: ERROR: "
        f"Price argument should be greater or equal to zero, not {price}"
    )
    assert time_remaining_in_years > 0, (
        "utils.price.calc_apr_from_spot_price: ERROR: "
        f"time_remaining_in_years should be greater than zero, not {time_remaining_in_years}"
    )
    return (1 - price) / (price * time_remaining_in_years)  # r = ((1/p)-1)/t = (1-p)/(pt)


def calc_spot_price_from_apr(apr_: float, time_remaining_in_years):
    """Returns the current spot price based on the current APR (decimal) and the remaining pool duration"""
    return 1 / (1 + apr_ * time_remaining_in_years)  # price = 1 / (1 + r * t)


def open_short(
    simulation_state_,
    wallet_address: int,
    trade_amount: float,
) -> tuple[Wallet, Wallet]:
    """
    shorts need their margin account to cover the worst case scenario (p=1)
    margin comes from 2 sources:
    - the proceeds from your short sale (p)
    - the max value you cover with base deposted from your wallet (1-p)
    these two components are both priced in base, yet happily add up to 1.0 units of bonds
    so we have the following identity:
    total margin (base, from proceeds + deposited) = face value of bonds shorted (# of bonds)

    this guarantees that bonds in the system are always fully backed by an equal amount of base
    """
    # Perform the trade.
    trade_quantity = Quantity(amount=trade_amount, unit=TokenType.PT)
    simulation_state_.pricing_model.check_input_assertions(
        quantity=trade_quantity,
        market_state=simulation_state_.market_state,
        time_remaining=simulation_state_.position_duration,
    )
    trade_result = simulation_state_.pricing_model.calc_out_given_in(
        in_=trade_quantity,
        market_state=simulation_state_.market_state,
        time_remaining=simulation_state_.position_duration,
    )
    simulation_state_.pricing_model.check_output_assertions(trade_result=trade_result)
    # Return the market and wallet deltas.
    market_deltas = Wallet(
        base=trade_result.market_result.d_base,
        bonds=trade_result.market_result.d_bonds,
        bond_buffer=trade_amount,
    )
    # amount to cover the worst case scenario where p=1. this amount is 1-p. see logic above.
    max_loss = trade_amount - trade_result.user_result.d_base
    agent_deltas = Wallet(
        address=wallet_address,
        base=-max_loss,
        shorts={
            simulation_state_.time: Position(
                balance=trade_amount, open_share_price=simulation_state_.market_state.share_price
            )
        },
    )
    return market_deltas, agent_deltas


def close_short(
    simulation_state_,
    wallet_address: int,
    open_share_price: float,
    trade_amount: float,
    mint_time: float,
) -> tuple[Wallet, Wallet]:
    """
    when closing a short, the number of bonds being closed out, at face value, give us the total margin returned
    the worst case scenario of the short is reduced by that amount, so they no longer need margin for it
    at the same time, margin in their account is drained to pay for the bonds being bought back
    so the amount returned to their wallet is trade_amount minus the cost of buying back the bonds
    that is, d_base = trade_amount (# of bonds) + trade_result.user_result.d_base (a negative amount, in base))
    for more on short accounting, see the open short method
    """

    # Clamp the trade amount to the bond reserves.
    if trade_amount > simulation_state_.market_state.bond_reserves:
        logging.warning(
            (
                "markets._close_short: WARNING: trade amount = %g"
                "is greater than bond reserves = %g. "
                "Adjusting to allowable amount."
            ),
            trade_amount,
            simulation_state_.market_state.bond_reserves,
        )
        trade_amount = simulation_state_.market_state.bond_reserves

    time_remaining_in_years = simulation_state_.term_length_in_years - (simulation_state.time - mint_time)

    # Perform the trade.
    trade_quantity = Quantity(amount=trade_amount, unit=TokenType.PT)
    simulation_state_.pricing_model.check_input_assertions(
        quantity=trade_quantity,
        market_state=simulation_state_.market_state,
        time_remaining_in_years=time_remaining_in_years,
    )
    trade_result = simulation_state_.pricing_model.calc_in_given_out(
        out=trade_quantity,
        market_state=simulation_state_.market_state,
        time_remaining_in_years=time_remaining_in_years,
    )
    simulation_state_.pricing_model.check_output_assertions(trade_result=trade_result)
    # Return the market and wallet deltas.
    market_deltas = Wallet(
        base=trade_result.market_result.d_base,
        bonds=trade_result.market_result.d_bonds,
        bond_buffer=-trade_amount,
    )
    agent_deltas = Wallet(
        address=wallet_address,
        base=(simulation_state_.market_state.share_price / open_share_price) * trade_amount
        + trade_result.user_result.d_base,  # see CLOSING SHORT LOGIC above
        shorts={
            mint_time: Position(
                balance=-trade_amount,
                open_share_price=0,
            )
        },
    )
    return market_deltas, agent_deltas


def open_long(
    simulation_state_,
    wallet_address: int,
    trade_amount: float,  # in base
) -> tuple[Wallet, Wallet]:
    """
    take trade spec & turn it into trade details
    compute wallet update spec with specific details
    will be conditional on the pricing model
    """
    # TODO: Why are we clamping elsewhere but we don't apply the trade at all here?
    # issue #146
    if trade_amount <= simulation_state_.market_state.bond_reserves:
        # Perform the trade.
        trade_quantity = Quantity(amount=trade_amount, unit=TokenType.BASE)
        simulation_state_.pricing_model.check_input_assertions(
            quantity=trade_quantity,
            market_state=simulation_state_.market_state,
            time_remaining=simulation_state_.position_duration,
        )
        trade_result = simulation_state_.pricing_model.calc_out_given_in(
            in_=trade_quantity,
            market_state=simulation_state_.market_state,
            time_remaining=simulation_state_.position_duration,
        )
        simulation_state_.pricing_model.check_output_assertions(trade_result=trade_result)
        # Get the market and wallet deltas to return.
        market_deltas = Wallet(
            base=trade_result.market_result.d_base,
            bonds=trade_result.market_result.d_bonds,
            base_buffer=trade_result.user_result.d_bonds,
        )
        agent_deltas = Wallet(
            address=wallet_address,
            base=trade_result.user_result.d_base,
            longs={simulation_state_.time: Position(trade_result.user_result.d_bonds)},
        )
    else:
        market_deltas = Wallet()
        agent_deltas = Wallet(address=wallet_address, base=0)
    return market_deltas, agent_deltas


def close_long(
    simulation_state_,
    wallet_address: int,
    trade_amount: float,  # in bonds
    mint_time: float,
) -> tuple[Wallet, Wallet]:
    """
    take trade spec & turn it into trade details
    compute wallet update spec with specific details
    will be conditional on the pricing model
    """
    time_remaining_in_years = simulation_state_.term_length_in_years - (simulation_state.time - mint_time)

    # Perform the trade.
    trade_quantity = Quantity(amount=trade_amount, unit=TokenType.PT)
    simulation_state_.pricing_model.check_input_assertions(
        quantity=trade_quantity,
        market_state=simulation_state_.market_state,
        time_remaining_in_years=time_remaining_in_years,
    )
    trade_result = simulation_state_.pricing_model.calc_out_given_in(
        in_=trade_quantity,
        market_state=simulation_state_.market_state,
        time_remaining_in_years=time_remaining_in_years,
    )
    simulation_state_.pricing_model.check_output_assertions(trade_result=trade_result)
    # Return the market and wallet deltas.
    market_deltas = Wallet(
        base=trade_result.market_result.d_base,
        bonds=trade_result.market_result.d_bonds,
        base_buffer=-trade_amount,
    )
    agent_deltas = Wallet(
        address=wallet_address,
        base=trade_result.user_result.d_base,
        longs={mint_time: Position(trade_result.user_result.d_bonds)},
    )
    return market_deltas, agent_deltas


def initialize_market(
    simulation_state_,
    wallet_address: int,
    contribution: float,
    target_apr: float,
) -> tuple[Wallet, Wallet]:
    """Allows an LP to initialize the market"""
    share_reserves = contribution / simulation_state_.market_state.share_price
    bond_reserves = simulation_state_.pricing_model.calc_bond_reserves(
        target_apr=target_apr,
        time_remaining=simulation_state_.position_duration,
        market_state=MarketState(
            share_reserves=share_reserves,
            init_share_price=simulation_state_.market_state.init_share_price,
            share_price=simulation_state_.market_state.share_price,
        ),
    )
    market_deltas = Wallet(
        base=contribution,
        bonds=bond_reserves,
    )
    agent_deltas = Wallet(
        address=wallet_address,
        base=-contribution,
        lp=2 * bond_reserves + contribution,  # 2y + cz
    )
    return (market_deltas, agent_deltas)


def add_liquidity(
    simulation_state_,
    wallet_address: int,
    trade_amount: float,
) -> tuple[Wallet, Wallet]:
    """Computes new deltas for bond & share reserves after liquidity is added"""
    # get_rate assumes that there is some amount of reserves, and will throw an error if share_reserves is zero
    if (
        simulation_state_.market_state.share_reserves == 0 and simulation_state_.market_state.bond_reserves == 0
    ):  # pool has not been initialized
        rate = 0
    else:
        rate = simulation_state_.apr
    # sanity check inputs
    simulation_state_.pricing_model.check_input_assertions(
        quantity=Quantity(amount=trade_amount, unit=TokenType.PT),  # temporary Quantity object just for this check
        market_state=simulation_state_.market_state,
        time_remaining=simulation_state_.position_duration,
    )
    # perform the trade
    lp_out, d_base_reserves, d_token_reserves = simulation_state_.pricing_model.calc_lp_out_given_tokens_in(
        d_base=trade_amount,
        rate=rate,
        market_state=simulation_state_.market_state,
        time_remaining=simulation_state_.position_duration,
    )
    market_deltas = Wallet(
        base=d_base_reserves,
        bonds=d_token_reserves,
        lp=lp_out,
    )
    agent_deltas = Wallet(
        address=wallet_address,
        base=-d_base_reserves,
        lp=lp_out,
    )
    return market_deltas, agent_deltas


def remove_liquidity(
    simulation_state_,
    wallet_address: int,
    trade_amount: float,
) -> tuple[Wallet, Wallet]:
    """Computes new deltas for bond & share reserves after liquidity is removed"""
    # sanity check inputs
    simulation_state_.pricing_model.check_input_assertions(
        quantity=Quantity(amount=trade_amount, unit=TokenType.PT),  # temporary Quantity object just for this check
        market_state=simulation_state_.market_state,
        time_remaining=simulation_state_.position_duration,
    )
    # perform the trade
    lp_in, d_base_reserves, d_token_reserves = simulation_state_.pricing_model.calc_tokens_out_given_lp_in(
        lp_in=trade_amount,
        rate=simulation_state_.apr,
        market_state=simulation_state_.market_state,
        time_remaining=simulation_state_.position_duration,
    )
    market_deltas = Wallet(
        base=-d_base_reserves,
        bonds=-d_token_reserves,
        lp=-lp_in,
    )
    agent_deltas = Wallet(
        address=wallet_address,
        base=d_base_reserves,
        lp=-lp_in,
    )
    return market_deltas, agent_deltas


def log_market_step_string(simulation_state_) -> None:
    """Logs the current market step"""
    # TODO: This is a HACK to prevent test_sim from failing on market shutdown
    # when the market closes, the share_reserves are 0 (or negative & close to 0) and several logging steps break
    if simulation_state_.market_state.share_reserves <= 0:
        spot_price = str(np.nan)
        rate = str(np.nan)
    else:
        spot_price = simulation_state_.spot_price
        rate = simulation_state_.apr
    logging.debug(
        ("t = %g" "\nx = %g" "\ny = %g" "\nlp = %g" "\nz = %g" "\nx_b = %g" "\ny_b = %g" "\np = %s" "\npool apr = %s"),
        simulation_state_.time,
        simulation_state_.market_state.share_reserves * simulation_state_.market_state.share_price,
        simulation_state_.market_state.bond_reserves,
        simulation_state_.market_state.lp,
        simulation_state_.market_state.share_reserves,
        simulation_state_.market_state.base_buffer,
        simulation_state_.market_state.bond_buffer,
        str(spot_price),
        str(rate),
    )


def init_market_state(
    simulation_state_: SimulationState, config: Config, pricing_model: PricingModel, init_target_liquidity: float = 1
):
    """Calculate reserves required to hit init targets and assign them to market_state"""
    term_length_in_years = config.num_position_days / 365
    time_stretch = pricing_model.calc_time_stretch(config.target_fixed_rate)
    adjusted_target_apr = config.target_fixed_rate * config.num_position_days / 365
    share_reserves_direct, bond_reserves_direct = pricing_model.calc_liquidity(
        simulation_state_=simulation_state_,
        target_liquidity=init_target_liquidity,
        target_apr=adjusted_target_apr,
        term_length_in_years=term_length_in_years,
        time_stretch=time_stretch,
    )
    simulation_state_.market_state.time_stretch = time_stretch
    simulation_state_.market_state.share_reserves = share_reserves_direct
    simulation_state_.market_state.bond_reserves = bond_reserves_direct


@dataclass
class SimulationState:
    """stores Simulator State"""

    config: Config = field(default_factory=lambda: Config())  # pylint: disable=unnecessary-lambda
    logging.info("%s", config)
    market_state: MarketState = field(default_factory=lambda: MarketState())  # pylint: disable=unnecessary-lambda
    agents: dict[int, Agent] = field(default_factory=dict)

    # Simulation variables
    run_number = 0
    block_number = 0
    seconds_in_a_day = 86400
    run_trade_number = 0

    @property
    def time(self) -> float:
        """Returns the current time in the simulation (in seconds)"""
        return self.block_number * self.time_between_blocks

    @property
    def time_in_days(self) -> float:
        """Returns the current time in the simulation (in days)"""
        return self.time / self.seconds_in_a_day

    @property
    def time_in_years(self) -> float:
        """Returns the current time in the simulation (in years)"""
        return self.time / self.seconds_in_a_day / 365

    def __post_init__(self):
        self.time_between_blocks = self.seconds_in_a_day / self.config.num_blocks_per_day
        self.config.check_variable_rate()
        self.config.freeze()  # pylint: disable=no-member # type: ignore
        self.rng = self.config.rng
        logging.info("%s %s %s", "#" * 20, self.config.pricing_model_name, "#" * 20)
        if self.config.pricing_model_name.lower() == "hyperdrive":
            self.market_state.pricing_model = HyperdrivePricingModel()
        elif self.config.pricing_model_name.lower() == "yieldspace":
            self.market_state.pricing_model = YieldSpacePricingModel()
        else:
            raise ValueError(
                f'pricing_config.pricing_model_name must be "Hyperdrive", or "YieldSpace", not {self.config.pricing_model_name}'
            )
        init_market_state(self, config=self.config, pricing_model=self.market_state.pricing_model)
        if self.config.init_lp is True:  # Instantiate and add the initial LP agent, if desired
            current_market_liquidity = self.market_state.pricing_model.calc_total_liquidity_from_reserves_and_price(
                market_state=self.market_state, share_price=self.market_state.share_price
            )
            lp_amount = self.config.target_liquidity - current_market_liquidity
            init_lp_agent = import_module("elfpy.policies.init_lp").Policy(wallet_address=0, budget=lp_amount)
            self.agents.update({0: init_lp_agent})
        collect_and_execute_trades(simulation_state_=self)  # Initialize the simulator using only the initial LP
        self.agents.update(setup_agents(self.config))


def validate_custom_parameters(policy_instruction):
    """separate the policy name from the policy arguments and validate the arguments"""
    policy_name, policy_args = policy_instruction.split(":")
    try:
        policy_args = policy_args.split(",")
    except AttributeError as exception:
        logging.info("ERROR: No policy arguments provided")
        raise exception
    try:
        policy_args = [arg.split("=") for arg in policy_args]
    except AttributeError as exception:
        logging.info("ERROR: Policy arguments must be provided as key=value pairs")
        raise exception
    try:
        kwargs = {key: float(value) for key, value in policy_args}
    except ValueError as exception:
        logging.info("ERROR: Policy arguments must be provided as key=value pairs")
        raise exception
    return policy_name, kwargs


def setup_agents(config, agent_policies=None) -> dict[int, Agent]:
    """setup agents"""
    agent_policies = config.agent_policies if agent_policies is None else agent_policies
    agents = {}
    for agent_id, policy_instruction in enumerate(agent_policies):
        if ":" in policy_instruction:  # we have custom parameters
            policy_name, not_kwargs = validate_custom_parameters(policy_instruction)
        else:  # we don't have custom parameters
            policy_name = policy_instruction
            not_kwargs = {}
        wallet_address = agent_id + 1
        policy = import_module("elfpy.policies.{policy_name}").Policy
        agent = policy(wallet_address=wallet_address, budget=1000)  # first policy goes to init_lp_agent
        for key, value in not_kwargs.items():
            if hasattr(agent, key):  # check if parameter exists
                setattr(agent, key, value)
            else:
                raise AttributeError(f"Policy {policy_name} does not have parameter {key}")
        agent.log_status_report()
        agents[wallet_address] = agent
    return agents


def collect_trades(simulation_state_, agent_ids: list[int], liquidate: bool = False) -> list[tuple[int, MarketAction]]:
    r"""Collect trades from a set of provided agent IDs.

    Parameters
    ----------
    agent_ids: list[int]
        A list of agent IDs. These IDs must correspond to agents that are
        registered in the simulator.

    liquidate: bool
        If true, have agents collect their liquidation trades. Otherwise, agents collect their normal trades.


    Returns
    -------
    list[tuple[int, MarketAction]]
        A list of trades associated with specific agents.
    """
    agents_and_trades = []
    for agent_id in agent_ids:
        agent = simulation_state_.agents[agent_id]
        if liquidate:
            logging.debug("Collecting liquiditation trades for market closure")
            trades = agent.get_liquidation_trades(simulation_state_.market)
        else:
            trades = agent.get_trades(simulation_state_.market)
        agents_and_trades.extend((agent_id, trade) for trade in trades)
    return agents_and_trades


def collect_and_execute_trades(simulation_state_, last_block_in_sim: bool = False) -> None:
    r"""Get trades from the agent list, execute them, and update states

    Parameters
    ----------
    last_block_in_sim : bool
        If True, indicates if the current set of trades are occuring on the final block in the simulation
    """
    if simulation_state_.config.shuffle_users:
        if last_block_in_sim:
            agent_ids: list[int] = simulation_state_.rng.permutation(  # shuffle wallets except init_lp
                [key for key in simulation_state_.agents if key > 0]  # exclude init_lp before shuffling
            ).tolist()
            if simulation_state_.config.init_lp:
                agent_ids.append(0)  # add init_lp so that they're always last
        else:
            agent_ids = simulation_state_.rng.permutation(
                list(simulation_state_.agents)
            ).tolist()  # random permutation of keys (agent wallet addresses)
    else:  # we are in a deterministic mode
        agent_ids = (
            list(simulation_state_.agents)[
                ::-1
            ]  # close their trades in reverse order to allow withdrawing of LP tokens
            if last_block_in_sim
            else list(simulation_state_.agents)  # execute in increasing order
        )
    agent_trades = collect_trades(simulation_state_, agent_ids, liquidate=last_block_in_sim)
    for trade in agent_trades:
        agent_id, agent_deltas, market_deltas = simulation_state_.market.trade_and_update(trade)
        simulation_state_.market_state.apply_delta(market_deltas)
        agent = simulation_state_.agents[agent_id]
        logging.debug("agent #%g wallet deltas:\n%s", agent.wallet.address, agent_deltas)
        agent.update_wallet(agent_deltas, simulation_state_.market)
        agent.log_status_report()
        # TODO: need to log deaggregated trade informaiton, i.e. trade_deltas
        # issue #215
        update_simulation_state(simulation_state_)
        simulation_state_.run_trade_number += 1


def run_simulation(simulation_state_, liquidate_on_end: bool = True) -> None:
    r"""Run the trade simulation and update the output state dictionary

    This is the primary function of the Simulator class.
    The PricingModel and Market objects will be constructed.
    A loop will execute a group of trades with random volumes and directions for each day,
    up to `self.config.num_trading_days` days.

    Parameters
    ----------
    liquidate_on_end : bool
        if True, liquidate trades when the simulation is complete

    Returns
    -------
    There are no returns, but the function does update the simulation_state member variable
    """
    last_block_in_sim = False
    for day in range(simulation_state_.config.num_trading_days):
        simulation_state_.day = day
        simulation_state_.market.market_state.vault_apr = simulation_state_.config.vault_apr[simulation_state_.day]
        # Vault return can vary per day, which sets the current price per share
        if simulation_state_.day > 0:  # Update only after first day (first day set to init_share_price)
            if simulation_state_.config.compound_vault_apr:  # Apply return to latest price (full compounding)
                price_multiplier = simulation_state_.market.market_state.share_price
            else:  # Apply return to starting price (no compounding)
                price_multiplier = simulation_state_.market.market_state.init_share_price
            delta = Wallet(
                share_price=(
                    simulation_state_.market.market_state.vault_apr  # current day's apy
                    / 365  # convert annual yield to daily
                    * price_multiplier
                )
            )
            simulation_state_.market.update_market(delta)
        for daily_block_number in range(simulation_state_.config.num_blocks_per_day):
            simulation_state_.daily_block_number = daily_block_number
            last_block_in_sim = (simulation_state_.day == simulation_state_.config.num_trading_days - 1) and (
                simulation_state_.daily_block_number == simulation_state_.config.num_blocks_per_day - 1
            )
            liquidate = last_block_in_sim and liquidate_on_end
            simulation_state_.collect_and_execute_trades(liquidate)
            logging.debug(
                "day = %d, daily_block_number = %d\n", simulation_state_.day, simulation_state_.daily_block_number
            )
            simulation_state_.market.log_market_step_string()
            if not last_block_in_sim:
                simulation_state_.market.time += simulation_state_.market_step_size()
                simulation_state_.block_number += 1
    # simulation has ended
    for agent in simulation_state_.agents.values():
        agent.log_final_report(simulation_state_.market)


def update_simulation_state(simulation_state_) -> None:
    r"""Increment the list for each key in the simulation_state output variable

    .. todo:: This gets duplicated in notebooks when we make the pandas dataframe.
        Instead, the simulation_state should be a dataframe.
        issue #215
    """
    # pylint: disable=too-many-statements
    parameter_list = [
        "model_name",
        "run_number",
        "day",
        "block_number",
        "market_time",
        "run_trade_number",
        "market_step_size",
        "position_duration",
        "fixed_apr",
        "variable_rate",
    ]
    for parameter in parameter_list:
        if not hasattr(simulation_state_, parameter):
            setattr(simulation_state_, parameter, [])
    simulation_state_.model_name.append(simulation_state_.market.pricing_model.model_name())
    simulation_state_.run_number.append(simulation_state_.run_number)
    simulation_state_.day.append(simulation_state_.day)
    simulation_state_.block_number.append(simulation_state_.block_number)
    simulation_state_.market_time.append(simulation_state_.market.time)
    simulation_state_.run_trade_number.append(simulation_state_.run_trade_number)
    simulation_state_.market_step_size.append(simulation_state_.market_step_size)
    simulation_state_.position_duration.append(simulation_state_.market.position_duration)
    simulation_state_.fixed_apr.append(simulation_state_.market.apr)
    simulation_state_.variable_rate.append(simulation_state_.config.variable_rate[simulation_state_.day])
    simulation_state_.add_dict_entries({f"config.{key}": val for key, val in simulation_state_.config.__dict__.items()})
    simulation_state_.add_dict_entries(simulation_state_.market.market_state.__dict__)
    for agent in simulation_state_.agents.values():
        simulation_state_.add_dict_entries(agent.wallet.get_state(simulation_state_))
    # TODO: This is a HACK to prevent test_sim from failing on market shutdown
    # when the market closes, the share_reserves are 0 (or negative & close to 0) and several logging steps break
    if simulation_state_.market.market_state.share_reserves > 0:  # there is money in the market
        simulation_state_.spot_price.append(simulation_state_.market.spot_price)
    else:
        simulation_state_.spot_price.append(np.nan)


def add_agents(self, agent_list: list[Agent]) -> None:
    r"""Append the agents and simulation_state member variables

    If trades have already happened (as indicated by self.run_trade_number), then empty wallet states are
    prepended to the simulation_state for each new agent so that the state can still easily be converted into
    a pandas dataframe.

    Parameters
    ----------
    agent_list : list[Agent]
        A list of instantiated Agent objects
    """
    for agent in agent_list:
        self.agents.update({agent.wallet.address: agent})
        for key in agent.wallet.__dict__.keys():
            setattr(self.simulation_state, key, [None] * self.run_trade_number)


def execute_trades(simulation_state_, agent_trades: list[tuple[int, MarketAction]]) -> None:
    r"""Execute a list of trades associated with agents in the simulator.

    Parameters
    ----------
    trades : list[tuple[int, list[MarketAction]]]
        A list of agent trades. These will be executed in order.
    """
    for trade in agent_trades:
        agent_id, agent_deltas, market_deltas = simulation_state_.market.trade_and_update(trade)
        simulation_state_.market.update_market(market_deltas)
        agent = simulation_state_.agents[agent_id]
        logging.debug(
            "agent #%g wallet deltas:\n%s",
            agent.wallet.address,
            agent_deltas,
        )
        agent.update_wallet(agent_deltas, simulation_state_.market)
        # TODO: Get simulator, market, pricing model, agent state strings and log
        agent.log_status_report()
        # TODO: need to log deaggregated trade informaiton, i.e. trade_deltas
        # issue #215
        simulation_state_.update_simulation_state()
        simulation_state_.run_trade_number += 1


class PricingModel(ABC):
    """Contains functions for calculating AMM variables

    Base class should not be instantiated on its own; it is assumed that a user will instantiate a child class
    """

    def calc_in_given_out(
        self,
        out: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> TradeResult:
        """Calculate fees and asset quantity adjustments"""
        raise NotImplementedError

    def calc_out_given_in(
        self,
        in_: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> TradeResult:
        """Calculate fees and asset quantity adjustments"""
        raise NotImplementedError

    def calc_lp_out_given_tokens_in(
        self,
        d_base: float,
        rate: float,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float, float]:
        """Computes the amount of LP tokens to be minted for a given amount of base asset"""
        raise NotImplementedError

    def calc_lp_in_given_tokens_out(
        self,
        d_base: float,
        rate: float,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float, float]:
        """Computes the amount of LP tokens to be minted for a given amount of base asset"""
        raise NotImplementedError

    def calc_tokens_out_given_lp_in(
        self,
        lp_in: float,
        rate: float,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float, float]:
        """Calculate how many tokens should be returned for a given lp addition"""
        raise NotImplementedError

    def model_name(self) -> str:
        """Unique name given to the model, can be based on member variable states"""
        raise NotImplementedError

    def model_type(self) -> str:
        """Unique identifier given to the model, should be lower snake_cased name"""
        raise NotImplementedError

    def _calc_k_const(
        self,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> Decimal:
        """Returns the 'k' constant variable for trade mathematics"""
        raise NotImplementedError

    def calc_bond_reserves(
        self,
        target_apr: float,
        time_remaining_in_years: float,
        time_stretch: float,
        simulation_state_: SimulationState,
    ) -> float:
        """Returns the assumed bond (i.e. token asset) reserve amounts given
        the share (i.e. base asset) reserves and APR

        Parameters
        ----------
        target_apr : float
            Target fixed APR in decimal units (for example, 5% APR would be 0.05)
        time_remaining : StretchedTime
            Amount of time left until bond maturity
        simulation_state_: SimulationState
            Simulation state

        Returns
        -------
        float
            The expected amount of bonds (token asset) in the pool, given the inputs

        .. todo:: Write a test for this function
        """
        # Only want to renormalize time for APR ("annual", so hard coded to 365)
        # Don't want to renormalize stretched time
        annualized_time = simulation_state_.time_in_years
        market_state = simulation_state_.market_state
        bond_reserves = (market_state.share_reserves / 2) * (
            market_state.init_share_price
            * (1 + target_apr * annualized_time) ** (1 / (time_remaining_in_years / time_stretch))
            - market_state.share_price
        )  # y = z/2 * (mu * (1 + rt)**(1/tau) - c)
        return bond_reserves

    def calc_share_reserves(
        self,
        target_apr: float,
        bond_reserves: float,
        time_remaining_in_years: float,
        time_stretch: float,
        init_share_price: float = 1,
    ):
        """Returns the assumed share (i.e. base asset) reserve amounts given
        the bond (i.e. token asset) reserves and APR

        Parameters
        ----------
        target_apr : float
            Target fixed APR in decimal units (for example, 5% APR would be 0.05)
        bond_reserves : float
            Token asset (pt) reserves in the pool
        days_remaining : float
            Amount of days left until bond maturity
        time_stretch : float
            Time stretch parameter, in years
        init_share_price : float
            Original share price when the pool started
        share_price : float
            Current share price

        Returns
        -------
        float
            The expected amount of base asset in the pool, calculated from the provided parameters

        .. todo:: Write a test for this function
        """

        # y = (z / 2) * (mu * (1 + rt)**(1/tau) - c)
        # z = (2 * y) / (mu * (1 + rt)**(1/tau) - c)
        # Only want to renormalize time for APR ("annual", so hard coded to 365)
        # Don't want to renormalize stretched time
        share_reserves = (
            2
            * bond_reserves
            / (
                init_share_price
                * (1 - target_apr * time_remaining_in_years) ** (1 / (time_remaining_in_years / time_stretch))
                - init_share_price
            )
        )

        return share_reserves

    def calc_base_for_target_apr(
        self,
        target_apr: float,
        bond: float,
        term_length_in_years: float,
        share_price: float = 1.0,
    ) -> float:
        """Returns the base for a given target APR."""
        base = share_price * bond * (1 - target_apr * term_length_in_years)

        assert base >= 0, "base value negative"
        return base

    def calc_bond_for_target_apr(
        self,
        target_apr: float,
        base: float,
        term_length_in_years,
        share_price: float = 1.0,
    ) -> float:
        """Calculates the bond for a given target APR."""
        bond = (base / share_price) / (1 - target_apr * term_length_in_years)

        assert bond >= 0, "bond value negative"
        return bond

    def calc_liquidity(
        self,
        simulation_state_: SimulationState,
        target_liquidity: float,
        target_apr: float,
        # TODO: Fields like position_duration and fee_percent could arguably be
        # wrapped up into a "MarketContext" value that includes the state as
        # one of its fields.
        term_length_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float]:
        """Returns the reserve volumes and total supply

        The scaling factor ensures bond_reserves and share_reserves add
        up to target_liquidity, while keeping their ratio constant (preserves apr).

        total_liquidity = in base terms, used to target liquidity as passed in
        total_reserves  = in arbitrary units (AU), used for yieldspace math

        Parameters
        ----------
        simulation_state_ : SimulationState
            The simulation state
        target_liquidity_usd : float
            Amount of liquidity that the simulation is trying to achieve in a given market
        target_apr : float
            Desired APR for the seeded market
        position_duration : StretchedTime
            The duration of bond positions in this market

        Returns
        -------
        (float, float)
            Tuple that contains (share_reserves, bond_reserves)
            calculated from the provided parameters
        """
        market_state = simulation_state_.market_state
        share_reserves = target_liquidity / market_state.share_price
        # guarantees only that it hits target_apr
        bond_reserves = self.calc_bond_reserves(
            target_apr=target_apr,
            time_remaining_in_years=term_length_in_years,
            time_stretch=time_stretch,
            simulation_state_=simulation_state_,
        )
        total_liquidity = self.calc_total_liquidity_from_reserves_and_price(
            MarketState(
                share_reserves=share_reserves,
                bond_reserves=bond_reserves,
                base_buffer=market_state.base_buffer,
                bond_buffer=market_state.bond_buffer,
                lp_reserves=market_state.lp_reserves,
                share_price=market_state.share_price,
                init_share_price=market_state.init_share_price,
            ),
            market_state.share_price,
        )
        # compute scaling factor to adjust reserves so that they match the target liquidity
        scaling_factor = target_liquidity / total_liquidity  # both in token units
        # update variables by rescaling the original estimates
        bond_reserves = bond_reserves * scaling_factor
        share_reserves = share_reserves * scaling_factor
        return share_reserves, bond_reserves

    def calc_total_liquidity_from_reserves_and_price(self, market_state: MarketState, share_price: float) -> float:
        """Returns the total liquidity in the pool in terms of base

        Parameters
        ----------
        MarketState : MarketState
            The following member variables are used:
                share_reserves : float
                    Base asset reserves in the pool
                bond_reserves : float
                    Token asset (pt) reserves in the pool
        share_price : float
            Variable (underlying) yield source price

        Returns
        -------
        float
            Total liquidity in the pool in terms of base, calculated from the provided parameters

        .. todo:: Write a test for this function
        """
        return market_state.share_reserves * share_price

    def calc_spot_price_from_reserves(
        self,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> float:
        r"""
        Calculates the spot price of base in terms of bonds.

        The spot price is defined as:

        .. math::
            \begin{align}
            p = (\frac{2y + cz}{\mu z})^{-\tau}
            \end{align}

        Parameters
        ----------
        market_state: MarketState
            The reserves and share prices of the pool.
        time_remaining : StretchedTime
            The time remaining for the asset (uses time stretch).

        Returns
        -------
        float
            The spot price of principal tokens.
        """
        return float(
            self._calc_spot_price_from_reserves_high_precision(
                market_state=market_state, time_remaining_in_years=time_remaining_in_years, time_stretch=time_stretch
            )
        )

    def _calc_spot_price_from_reserves_high_precision(
        self,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> Decimal:
        r"""
        Calculates the current market spot price of base in terms of bonds.
        This variant returns the result in a high precision format.

        The spot price is defined as:

        .. math::
            \begin{align}
            p = (\frac{2y + cz}{\mu z})^{-\tau}
            \end{align}

        Parameters
        ----------
        market_state: MarketState
            The reserves and share prices of the pool.
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch).

        Returns
        -------
        Decimal
            The spot price of principal tokens.
        """
        # TODO: in general s != y + c*z, we'll want to update this to have s = lp_reserves
        # issue #94
        # s = y + c*z
        total_reserves = Decimal(market_state.bond_reserves) + Decimal(market_state.share_price) * Decimal(
            market_state.share_reserves
        )
        # p = ((y + s)/(mu*z))^(-tau) = ((2y + cz)/(mu*z))^(-tau)
        spot_price = (
            (Decimal(market_state.bond_reserves) + total_reserves)
            / (Decimal(market_state.init_share_price) * Decimal(market_state.share_reserves))
        ) ** Decimal(-time_remaining_in_years / time_stretch)
        return spot_price

    def get_max_long(
        self,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float]:
        r"""
        Calculates the maximum long the market can support

        .. math::
            \begin{align}
            \Delta z' = \mu^{-1} \cdot (\frac{\mu}{c} \cdot (k-(y+c \cdot z)^{1-\tau(d)}))^{\frac{1}{1-\tau(d)}}
            -c \cdot z
            \end{align}

        Parameters
        ----------
        market_state : MarketState
            The reserves and share prices of the pool
        fee_percent : float
            The fee percent charged by the market
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch)

        Returns
        -------
        float
            The maximum amount of base that can be used to purchase bonds.
        float
            The maximum amount of bonds that can be purchased.
        """
        base = self.calc_in_given_out(
            out=Quantity(market_state.bond_reserves - market_state.bond_buffer, unit=TokenType.PT),
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        ).breakdown.with_fee
        bonds = self.calc_out_given_in(
            in_=Quantity(amount=base, unit=TokenType.BASE),
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        ).breakdown.with_fee
        return (base, bonds)

    def get_max_short(
        self,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float]:
        r"""
        Calculates the maximum short the market can support using the bisection
        method.

        \begin{align}
        \Delta y' = \mu^{-1} \cdot (\frac{\mu}{c} \cdot k)^{\frac{1}{1-\tau(d)}}-2y-c \cdot z
        \end{align}

        Parameters
        ----------
        market_state : MarketState
            The reserves and share prices of the pool.
        fee_percent : float
            The fee percent charged by the market.
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch).

        Returns
        -------
        float
            The maximum amount of base that can be used to short bonds.
        float
            The maximum amount of bonds that can be shorted.
        """
        bonds = self.calc_in_given_out(
            out=Quantity(
                market_state.share_reserves - market_state.base_buffer / market_state.share_price,
                unit=TokenType.PT,
            ),
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        ).breakdown.with_fee
        base = self.calc_out_given_in(
            in_=Quantity(amount=bonds, unit=TokenType.PT),
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        ).breakdown.with_fee
        return (base, bonds)

    def calc_time_stretch(self, apr) -> float:
        """Returns fixed time-stretch value based on current apr (as a decimal)"""
        apr_percent = apr * 100  # bounded between 0 and 100
        return 3.09396 / (0.02789 * apr_percent)  # bounded between ~1.109 (apr=1) and inf (apr=0)

    def check_input_assertions(
        self,
        quantity: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ):
        """Applies a set of assertions to the input of a trading function."""

        assert quantity.amount >= WEI, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected quantity.amount >= {WEI}, not {quantity.amount}!"
        )
        assert market_state.share_reserves >= 0, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected share_reserves >= {WEI}, not {market_state.share_reserves}!"
        )
        assert market_state.bond_reserves >= 0, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected bond_reserves >= {WEI} or bond_reserves == 0, not {market_state.bond_reserves}!"
        )
        assert market_state.share_price >= market_state.init_share_price, (
            f"pricing_models.check_input_assertions: ERROR: "
            f"expected share_price >= {market_state.init_share_price}, not share_price={market_state.share_price}"
        )
        assert market_state.init_share_price >= 1, (
            f"pricing_models.check_input_assertions: ERROR: "
            f"expected init_share_price >= 1, not share_price={market_state.init_share_price}"
        )
        reserves_difference = abs(market_state.share_reserves * market_state.share_price - market_state.bond_reserves)
        assert reserves_difference < MAX_RESERVES_DIFFERENCE, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected reserves_difference < {MAX_RESERVES_DIFFERENCE}, not {reserves_difference}!"
        )
        assert 1 >= market_state.trade_fee_percent >= 0, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected 1 >= trade_fee_percent >= 0, not {market_state.trade_fee_percent}!"
        )
        assert 1 >= market_state.redemption_fee_percent >= 0, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected 1 >= redemption_fee_percent >= 0, not {market_state.redemption_fee_percent}!"
        )
        assert 1 >= time_remaining_in_years / time_stretch >= 0, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected 1 > time_remaining_in_years / time_stretch >= 0, not {time_remaining_in_years / time_stretch}!"
        )
        assert 1 >= time_remaining_in_years >= 0, (
            "pricing_models.check_input_assertions: ERROR: "
            f"expected 1 > time_remaining_in_years >= 0, not {time_remaining_in_years}!"
        )

    # TODO: Add checks for TradeResult's other outputs.
    # issue #57
    def check_output_assertions(
        self,
        trade_result: TradeResult,
    ):
        """Applies a set of assertions to a trade result."""

        assert isinstance(trade_result.breakdown.fee, float), (
            "pricing_models.check_output_assertions: ERROR: "
            f"fee should be a float, not {type(trade_result.breakdown.fee)}!"
        )
        assert trade_result.breakdown.fee >= 0, (
            "pricing_models.check_output_assertions: ERROR: "
            f"Fee should not be negative, but is {trade_result.breakdown.fee}!"
        )
        assert isinstance(trade_result.breakdown.without_fee, float), (
            "pricing_models.check_output_assertions: ERROR: "
            f"without_fee should be a float, not {type(trade_result.breakdown.without_fee)}!"
        )
        assert trade_result.breakdown.without_fee >= 0, (
            "pricing_models.check_output_assertions: ERROR: "
            f"without_fee should be non-negative, not {trade_result.breakdown.without_fee}!"
        )


class YieldSpacePricingModel(PricingModel):
    """
    YieldSpace Pricing Model

    This pricing model uses the YieldSpace invariant with modifications to
    enable the base reserves to be deposited into yield bearing vaults
    """

    # TODO: The too many locals disable can be removed after refactoring the LP
    #       functions.
    #
    # pylint: disable=too-many-locals
    # pylint: disable=duplicate-code

    def model_name(self) -> str:
        return "YieldSpace"

    def model_type(self) -> str:
        return "yieldspace"

    def calc_lp_out_given_tokens_in(
        self,
        d_base: float,
        rate: float,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float, float]:
        r"""
        Computes the amount of LP tokens to be minted for a given amount of base asset

        .. math::
            y = \frac{(z + \Delta z)(\mu \cdot (\frac{1}{1 + r \cdot t(d)})^{\frac{1}{\tau(d_b)}} - c)}{2}

        """
        d_shares = d_base / market_state.share_price
        if market_state.share_reserves > 0:  # normal case where we have some share reserves
            # TODO: We need to update these LP calculations to address the LP
            #       exploit scenario.
            lp_out = (d_shares * market_state.lp_reserves) / (market_state.share_reserves - market_state.base_buffer)
        else:  # initial case where we have 0 share reserves or final case where it has been removed
            lp_out = d_shares
        # TODO: Move this calculation to a helper function.
        d_bonds = (market_state.share_reserves + d_shares) / 2 * (
            market_state.init_share_price
            * (1 + rate * time_remaining_in_years) ** (1 / (time_remaining_in_years / time_stretch))
            - market_state.share_price
        ) - market_state.bond_reserves
        logging.debug(
            (
                "inputs: d_base=%g, share_reserves=%d, "
                "bond_reserves=%d, base_buffer=%g, "
                "init_share_price=%g, share_price=%g, "
                "lp_reserves=%g, rate=%g, "
                "time_remaining_in_years=%g, time_stretch=%g"
                "\nd_shares=%g (d_base / share_price = %g / %g)"
                "\nlp_out=%g\n"
                "(d_share_reserves * lp_reserves / (share_reserves - base_buffer / share_price) = "
                "%g * %g / (%g - %g / %g))"
                "\nd_bonds=%g\n"
                "((share_reserves + d_share_reserves) / 2 * (init_share_price * (1 + rate * time_remaining) ** "
                "(1 / stretched_time_remaining) - share_price) - bond_reserves = "
                "(%g + %g) / 2 * (%g * (1 + %g * %g) ** "
                "(1 / %g) - %g) - %g)"
            ),
            d_base,
            market_state.share_reserves,
            market_state.bond_reserves,
            market_state.base_buffer,
            market_state.init_share_price,
            market_state.share_price,
            market_state.lp_reserves,
            rate,
            time_remaining_in_years,
            time_stretch,
            d_shares,
            d_base,
            market_state.share_price,
            lp_out,
            d_shares,
            market_state.lp_reserves,
            market_state.share_reserves,
            market_state.base_buffer,
            market_state.share_price,
            d_bonds,
            market_state.share_reserves,
            d_shares,
            market_state.init_share_price,
            rate,
            time_remaining_in_years,
            time_remaining_in_years / time_stretch,
            market_state.share_price,
            market_state.bond_reserves,
        )
        return lp_out, d_base, d_bonds

    # TODO: Delete this function from here & base? It is not used or tested.
    def calc_lp_in_given_tokens_out(
        self,
        d_base: float,
        rate: float,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float, float]:
        r"""
        Computes the amount of LP tokens to be minted for a given amount of base asset

        .. math::
            y = \frac{(z - \Delta z)(\mu \cdot (\frac{1}{1 + r \cdot t(d)})^{\frac{1}{\tau(d_b)}} - c)}{2}

        """
        d_shares = d_base / market_state.share_price
        lp_in = (d_shares * market_state.lp_reserves) / (
            market_state.share_reserves - market_state.base_buffer / market_state.share_price
        )
        # TODO: Move this calculation to a helper function.
        d_bonds = (market_state.share_reserves - d_shares) / 2 * (
            market_state.init_share_price
            * (1 + rate * time_remaining_in_years) ** (1 / (time_remaining_in_years / time_stretch))
            - market_state.share_price
        ) - market_state.bond_reserves
        return lp_in, d_base, d_bonds

    def calc_tokens_out_given_lp_in(
        self,
        lp_in: float,
        rate: float,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> tuple[float, float, float]:
        """Calculate how many tokens should be returned for a given lp addition

        .. todo:: add test for this function; improve function documentation w/ parameters, returns, and equations used
        """
        d_base = (
            market_state.share_price
            * (market_state.share_reserves - market_state.base_buffer)
            * lp_in
            / market_state.lp_reserves
        )
        d_shares = d_base / market_state.share_price
        # TODO: Move this calculation to a helper function.
        # rate is an APR, which is annual, so we normalize time by 365 to correct for units
        d_bonds = (market_state.share_reserves - d_shares) / 2 * (
            market_state.init_share_price
            * (1 + rate * time_remaining_in_years) ** (1 / (time_remaining_in_years / time_stretch))
            - market_state.share_price
        ) - market_state.bond_reserves
        logging.debug(
            (
                "inputs:\n\tlp_in=%g,\n\tshare_reserves=%d, "
                "bond_reserves=%d,\n\tbase_buffer=%g, "
                "init_share_price=%g,\n\tshare_price=%g,\n\tlp_reserves=%g,\n\t"
                "rate=%g,\n\ttime_remaining=%g,\n\tstretched_time_remaining=%g\n\t"
                "\n\td_shares=%g\n\t(d_base / share_price = %g / %g)"
                "\n\td_bonds=%g"
                "\n\t((share_reserves - d_shares) / 2 * (init_share_price * (1 + apr * annualized_time) "
                "** (1 / stretched_time_remaining) - share_price) - bond_reserves = "
                "\n\t((%g - %g) / 2 * (%g * (1 + %g * %g) "
                "** (1 / %g) - %g) - %g =\n\t%g"
            ),
            lp_in,
            market_state.share_reserves,
            market_state.bond_reserves,
            market_state.base_buffer,
            market_state.init_share_price,
            market_state.share_price,
            market_state.lp_reserves,
            rate,
            time_remaining_in_years,
            (time_remaining_in_years / time_stretch),
            d_shares,
            d_base,
            market_state.share_price,
            d_bonds,
            market_state.share_reserves,
            d_shares,
            market_state.init_share_price,
            rate,
            time_remaining_in_years,
            (time_remaining_in_years / time_stretch),
            market_state.share_price,
            market_state.bond_reserves,
            d_bonds,
        )
        return lp_in, d_base, d_bonds

    def calc_in_given_out(
        self,
        out: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> TradeResult:
        r"""
        Calculates the amount of an asset that must be provided to receive a
        specified amount of the other asset given the current AMM reserves.

        The input is calculated as:

        .. math::
            \begin{align*}
            & p \;\;\;\; = \;\;\;\; \Bigg(\dfrac{2y + cz}{\mu z}\Bigg)^{-\tau}
            \\\\
            & in' \;\;\:  = \;\;\:
            \begin{cases}
            \\
            \text{ if $token\_in$ = "base", }\\
            \quad\quad\quad c \big(\mu^{-1} \big(\mu \cdot c^{-1} \big(k -
            \big(2y + cz - \Delta y\big)
            ^{1-\tau}\big)\big)
            ^ {\tfrac{1}{1-\tau}} - z\big)
            \\\\
            \text{ if $token\_in$ = "pt", }\\
            \quad\quad\quad (k -
            \big(c \cdot \mu^{-1} \cdot
            \big(\mu \cdot\big(z - \Delta z \big)\big)
            ^{1 - \tau} \big)^{\tfrac{1}{1 - \tau}}) - \big(2y + cz\big)
            \\\\
            \end{cases}
            \\\\
            & f \;\;\;\; = \;\;\;\;
            \begin{cases}
            \\
            \text{ if $token\_in$ = "base", }\\\\
            \quad\quad\quad (1 - p) \phi\;\; \Delta y
            \\\\
            \text{ if $token\_in$ = "pt", }\\\\
            \quad\quad\quad (p^{-1} - 1) \enspace \phi \enspace (c \cdot \Delta z)
            \\\\
            \end{cases}
            \\\\\\
            & in = in' + f
            \\
            \end{align*}

        Parameters
        ----------
        out : Quantity
            The quantity of tokens that the user wants to receive (the amount
            and the unit of the tokens).
        market_state : MarketState
            The state of the AMM's reserves and share prices.
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch).

        Returns
        -------
        float
            The amount the user pays without fees or slippage. The units
            are always in terms of bonds or base.
        float
            The amount the user pays with fees and slippage. The units are
            always in terms of bonds or base.
        float
            The amount the user pays with slippage and no fees. The units are
            always in terms of bonds or base.
        float
            The fee the user pays. The units are always in terms of bonds or
            base.
        """
        # Calculate some common values up front
        time_elapsed = 1 - Decimal((time_remaining_in_years / time_stretch))
        init_share_price = Decimal(market_state.init_share_price)
        share_price = Decimal(market_state.share_price)
        scale = share_price / init_share_price
        share_reserves = Decimal(market_state.share_reserves)
        bond_reserves = Decimal(market_state.bond_reserves)
        total_reserves = share_price * share_reserves + bond_reserves
        spot_price = self._calc_spot_price_from_reserves_high_precision(
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        )
        out_amount = Decimal(out.amount)
        trade_fee_percent = Decimal(market_state.trade_fee_percent)
        # We precompute the YieldSpace constant k using the current reserves and
        # share price:
        #
        # k = (c / mu) * (mu * z)**(1 - tau) + (2y + cz)**(1 - tau)
        k = self._calc_k_const(market_state, time_remaining_in_years, time_stretch)
        if out.unit == TokenType.BASE:
            in_reserves = bond_reserves + total_reserves
            out_reserves = share_reserves
            d_shares = out_amount / share_price
            # The amount the user pays without fees or slippage is simply the
            # amount of base the user would receive times the inverse of the
            # spot price of base in terms of bonds. The amount of base the user
            # receives is given by c * d_z where d_z is the number of shares the
            # pool will need to unwrap to give the user their base. If we let p
            # be the conventional spot price, then we can write this as:
            #
            # without_fee_or_slippage = (1 / p) * c * d_z
            without_fee_or_slippage = (1 / spot_price) * share_price * d_shares
            # We solve the YieldSpace invariant for the bonds paid to receive
            # the requested amount of base. We set up the invariant where the
            # user pays d_y' bonds and receives d_z shares:
            #
            # (c / mu) * (mu * (z - d_z))**(1 - tau) + (2y + cz + d_y')**(1 - tau)) = k
            #
            # Solving for d_y' gives us the amount of bonds the user must pay
            # without including fees:
            #
            # d_y' = (k - (c / mu) * (mu * (z - d_z))**(1 - tau))**(1 / (1 - tau)) - (2y + cz)
            #
            # without_fee = d_y'
            without_fee = (k - scale * (init_share_price * (out_reserves - d_shares)) ** time_elapsed) ** (
                1 / time_elapsed
            ) - in_reserves
            # The fees are calculated as the difference between the bonds paid
            # without slippage and the base received times the fee percentage.
            # This can also be expressed as:
            #
            # fee = ((1 / p) - 1) * phi * c * d_z
            fee = ((1 / spot_price) - 1) * trade_fee_percent * share_price * d_shares
            logging.debug(
                (
                    "fee = ((1 / spot_price) - 1) * _fee_percent * share_price * d_shares = "
                    "((1 / %g) - 1) * %g * %g * %g = %g"
                ),
                spot_price,
                trade_fee_percent,
                share_price,
                d_shares,
                fee,
            )
            # To get the amount paid with fees, add the fee to the calculation that
            # excluded fees. Adding the fees results in more tokens paid, which
            # indicates that the fees are working correctly.
            with_fee = without_fee + fee
            # Create the user and market trade results.
            user_result = Wallet(
                base=out.amount,
                bonds=float(-with_fee),
            )
            market_result = Wallet(
                base=-out.amount,
                bonds=float(with_fee),
            )
        elif out.unit == TokenType.PT:
            in_reserves = share_reserves
            out_reserves = bond_reserves + total_reserves
            d_bonds = out_amount
            # The amount the user pays without fees or slippage is simply
            # the amount of bonds the user would receive times the spot price of
            # base in terms of bonds. If we let p be the conventional spot price,
            # then we can write this as:
            #
            # without_fee_or_slippage = p * d_y
            without_fee_or_slippage = spot_price * d_bonds
            # We solve the YieldSpace invariant for the base paid for the
            # requested amount of bonds. We set up the invariant where the user
            # pays d_z' shares and receives d_y bonds:
            #
            # (c / mu) * (mu * (z + d_z'))**(1 - tau) + (2y + cz - d_y)**(1 - tau) = k
            #
            # Solving for d_z' gives us the amount of shares the user pays
            # without including fees:
            #
            # d_z' = (1 / mu) * ((k - (2y + cz - d_y)**(1 - tau)) / (c / mu))**(1 / (1 - tau)) - z
            #
            # We really want to know the value of d_x', the amount of base the
            # user pays. This is given by d_x' = c * d_z'.
            #
            # without_fee = d_x'
            without_fee = (
                (1 / init_share_price) * ((k - (out_reserves - d_bonds) ** time_elapsed) / scale) ** (1 / time_elapsed)
                - in_reserves
            ) * share_price
            # The fees are calculated as the difference between the bonds
            # received and the base paid without slippage times the fee
            # percentage. This can also be expressed as:
            #
            # fee = (1 - p) * phi * d_y
            fee = (1 - spot_price) * trade_fee_percent * d_bonds
            logging.debug(
                ("fee = (1 - spot_price) * _fee_percent * d_bonds = (1 - %g) * %g * %g = %g"),
                spot_price,
                trade_fee_percent,
                d_bonds,
                fee,
            )
            # To get the amount paid with fees, add the fee to the calculation that
            # excluded fees. Adding the fees results in more tokens paid, which
            # indicates that the fees are working correctly.
            with_fee = without_fee + fee
            # Create the user and market trade results.
            user_result = Wallet(
                base=float(-with_fee),
                bonds=out.amount,
            )
            market_result = Wallet(
                base=float(with_fee),
                bonds=-out.amount,
            )
        else:
            raise AssertionError(
                # pylint: disable-next=line-too-long
                f"pricing_models.calc_in_given_out: ERROR: expected out.unit to be {TokenType.BASE} or {TokenType.PT}, not {out.unit}!"
            )
        return TradeResult(
            user_result=user_result,
            market_result=market_result,
            breakdown=TradeBreakdown(
                without_fee_or_slippage=float(without_fee_or_slippage),
                with_fee=float(with_fee),
                without_fee=float(without_fee),
                fee=float(fee),
            ),
        )

    # TODO: The high slippage tests in tests/test_pricing_model.py should
    # arguably have much higher slippage. This is something we should
    # consider more when thinking about the use of a time stretch parameter.
    def calc_out_given_in(
        self,
        in_: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> TradeResult:
        r"""
        Calculates the amount of an asset that must be provided to receive a
        specified amount of the other asset given the current AMM reserves.

        The output is calculated as:

        .. math::
            \begin{align*}
            & p \;\;\;\; = \;\;\;\; \Bigg(\dfrac{2y + cz}{\mu z}\Bigg)^{-\tau}
            \\\\
            & out'\;\; = \;\;
            \begin{cases}
            \\
            \text{ if $token\_out$ = "base", }\\
            \quad\quad\quad c \big(z - \mu^{-1}
            \big(c \cdot \mu^{-1} \big(k - \big(2y + cz + \Delta y\big)
            ^{1 - \tau}\big)\big)
            ^{\tfrac{1}{1 - \tau}}\big)
            \\\\
            \text{ if $token\_out$ = "pt", }\\
            \quad\quad\quad 2y + cz - (k - c \cdot
            \mu^{-1} \cdot (\mu (z + \Delta z))^{1 - \tau})
            ^{\tfrac{1}{(1 - \tau)}}
            \\\\
            \end{cases}
            \\\\
            & f \;\;\;\; = \;\;\;\;
            \begin{cases}
            \\
            \text{ if $token\_out$ = "base", }\\\\
            \quad\quad\quad (1 - p) \phi\;\; \Delta y
            \\\\
            \text{ if $token\_out$ = "pt", }\\\\
            \quad\quad\quad (p^{-1} - 1) \enspace \phi \enspace (c \cdot \Delta z)
            \\\\
            \end{cases}
            \\\\\\
            & out = out' + f
            \\
            \end{align*}

        Parameters
        ----------
        in_ : Quantity
            The quantity of tokens that the user wants to pay (the amount
            and the unit of the tokens).
        market_state : MarketState
            The state of the AMM's reserves and share prices.
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch).

        Returns
        -------
        float
            The amount the user receives without fees or slippage. The units
            are always in terms of bonds or base.
        float
            The amount the user receives with fees and slippage. The units are
            always in terms of bonds or base.
        float
            The amount the user receives with slippage and no fees. The units are
            always in terms of bonds or base.
        float
            The fee the user pays. The units are always in terms of bonds or
            base.
        """
        # Calculate some common values up front
        time_elapsed = 1 - Decimal((time_remaining_in_years / time_stretch))
        init_share_price = Decimal(market_state.init_share_price)
        share_price = Decimal(market_state.share_price)
        scale = share_price / init_share_price
        share_reserves = Decimal(market_state.share_reserves)
        bond_reserves = Decimal(market_state.bond_reserves)
        total_reserves = share_price * share_reserves + bond_reserves
        spot_price = self._calc_spot_price_from_reserves_high_precision(
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        )
        in_amount = Decimal(in_.amount)
        trade_fee_percent = Decimal(market_state.trade_fee_percent)
        # We precompute the YieldSpace constant k using the current reserves and
        # share price:
        #
        # k = (c / mu) * (mu * z)**(1 - tau) + (2y + cz)**(1 - tau)
        k = self._calc_k_const(market_state, time_remaining_in_years, time_stretch)
        if in_.unit == TokenType.BASE:
            d_shares = in_amount / share_price  # convert from base_asset to z (x=cz)
            in_reserves = share_reserves
            out_reserves = bond_reserves + total_reserves
            # The amount the user would receive without fees or slippage is
            # the amount of base the user pays times inverse of the spot price
            # of base in terms of bonds. If we let p be the conventional spot
            # price, then we can write this as:
            #
            # (1 / p) * c * d_z
            without_fee_or_slippage = (1 / spot_price) * share_price * d_shares
            # We solve the YieldSpace invariant for the bonds received from
            # paying the specified amount of base. We set up the invariant where
            # the user pays d_z shares and receives d_y' bonds:
            #
            # (c / mu) * (mu * (z + d_z))**(1 - tau) + (2y + cz - d_y')**(1 - tau) = k
            #
            # Solving for d_y' gives us the amount of bonds the user receives
            # without including fees:
            #
            # d_y' = 2y + cz - (k - (c / mu) * (mu * (z + d_z))**(1 - tau))**(1 / (1 - tau))
            without_fee = out_reserves - (
                k - scale * (init_share_price * (in_reserves + d_shares)) ** time_elapsed
            ) ** (1 / time_elapsed)
            # The fees are calculated as the difference between the bonds
            # received without slippage and the base paid times the fee
            # percentage. This can also be expressed as:
            #
            # ((1 / p) - 1) * phi * c * d_z
            fee = ((1 / spot_price) - 1) * trade_fee_percent * share_price * d_shares
            # To get the amount paid with fees, subtract the fee from the
            # calculation that excluded fees. Subtracting the fees results in less
            # tokens received, which indicates that the fees are working correctly.
            with_fee = without_fee - fee
            # Create the user and market trade results.
            user_result = Wallet(
                base=-in_.amount,
                bonds=float(with_fee),
            )
            market_result = Wallet(
                base=in_.amount,
                bonds=float(-with_fee),
            )
        elif in_.unit == TokenType.PT:
            d_bonds = in_amount
            in_reserves = bond_reserves + total_reserves
            out_reserves = share_reserves
            # The amount the user would receive without fees or slippage is the
            # amount of bonds the user pays times the spot price of base in
            # terms of bonds. If we let p be the conventional spot price, then
            # we can write this as:
            #
            # p * d_y
            without_fee_or_slippage = spot_price * d_bonds
            # We solve the YieldSpace invariant for the base received from
            # selling the specified amount of bonds. We set up the invariant
            # where the user pays d_y bonds and receives d_z' shares:
            #
            # (c / mu) * (mu * (z - d_z'))**(1 - tau) + (2y + cz + d_y)**(1 - tau) = k
            #
            # Solving for d_z' gives us the amount of shares the user receives
            # without fees:
            #
            # d_z' = z - (1 / mu) * ((k - (2y + cz + d_y)**(1 - tau)) / (c / mu))**(1 / (1 - tau))
            #
            # We really want to know the value of d_x', the amount of base the
            # user receives without fees. This is given by d_x' = c * d_z'.
            #
            # without_fee = d_x'
            without_fee = (
                share_reserves
                - (1 / init_share_price) * ((k - (in_reserves + d_bonds) ** time_elapsed) / scale) ** (1 / time_elapsed)
            ) * share_price
            # The fees are calculated as the difference between the bonds paid
            # and the base received without slippage times the fee percentage.
            # This can also be expressed as:
            #
            # fee = (1 - p) * phi * d_y
            fee = (1 - spot_price) * trade_fee_percent * d_bonds
            # To get the amount paid with fees, subtract the fee from the
            # calculation that excluded fees. Subtracting the fees results in less
            # tokens received, which indicates that the fees are working correctly.
            with_fee = without_fee - fee
            # Create the user and market trade results.
            user_result = Wallet(
                base=float(with_fee),
                bonds=-in_.amount,
            )
            market_result = Wallet(
                base=float(-with_fee),
                bonds=in_.amount,
            )
        else:
            raise AssertionError(
                f"pricing_models.calc_out_given_in: ERROR: expected in_.unit"
                f" to be {TokenType.BASE} or {TokenType.PT}, not {in_.unit}!"
            )
        return TradeResult(
            user_result=user_result,
            market_result=market_result,
            breakdown=TradeBreakdown(
                without_fee_or_slippage=float(without_fee_or_slippage),
                with_fee=float(with_fee),
                without_fee=float(without_fee),
                fee=float(fee),
            ),
        )

    def _calc_k_const(self, market_state: MarketState, time_remaining_in_years: float, time_stretch: float) -> Decimal:
        """
        Returns the 'k' constant variable for trade mathematics

        .. math::
            k = \frac{c / mu} (mu z)^{1 - \tau} + (2y + c z)^(1 - \tau)

        Parameters
        ----------
        market_state : MarketState
            The state of the AMM
        time_remaining : StretchedTime
            Time until expiry for the token

        Returns
        -------
        Decimal
            'k' constant used for trade mathematics, calculated from the provided parameters
        """
        scale = Decimal(market_state.share_price) / Decimal(market_state.init_share_price)
        total_reserves = Decimal(market_state.bond_reserves) + Decimal(market_state.share_price) * Decimal(
            market_state.share_reserves
        )
        time_elapsed = Decimal(1) - Decimal((time_remaining_in_years / time_stretch))
        return (
            scale * (Decimal(market_state.init_share_price) * Decimal(market_state.share_reserves)) ** time_elapsed
            + (Decimal(market_state.bond_reserves) + Decimal(total_reserves)) ** time_elapsed
        )


class HyperdrivePricingModel(YieldSpacePricingModel):  # type: ignore
    """
    Hyperdrive Pricing Model

    This pricing model uses a combination of the Constant Sum and Yield Space
    invariants with modifications to the Yield Space invariant that enable the
    base reserves to be deposited into yield bearing vaults
    """

    def model_name(self) -> str:
        return "Hyperdrive"

    def model_type(self) -> str:
        return "hyperdrive"

    def calc_in_given_out(
        self,
        out: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> TradeResult:
        r"""
        Calculates the amount of an asset that must be provided to receive a
        specified amount of the other asset given the current AMM reserves.

        The input is calculated as:

        .. math::
            \begin{align*}
            & p \;\;\;\; = \;\;\;\; \Bigg(\dfrac{2y + cz}{\mu z}\Bigg)^{-\tau}
            \\\\
            & in' \;\;\:  = \;\;\:
            \begin{cases}
            \\
            \text{ if $token\_in$ = "base", }\\
            \quad\quad\quad c \big(\mu^{-1} \big(\mu \cdot c^{-1}
            \big(k - \big(2y + cz - \Delta y \cdot t\big)
            ^{1-\tau}\big)\big)
            ^ {\tfrac{1}{1-\tau}} - z\big) + \Delta y \cdot\big(1 - \tau\big)
            \\\\
            \text{ if $token\_in$ = "pt", }\\
            \quad\quad\quad (k - \big(
            c \cdot \mu^{-1} \cdot\big(\mu \cdot
            \big(z - \Delta z \cdot t\big)\big)^{1 - \tau} \big))
            ^{\tfrac{1}{1 - \tau}} - \big(2y + cz\big)
            + c \cdot \Delta z \cdot\big(1 - \tau\big)
            \\\\
            \end{cases}
            \\\\
            & f \;\;\;\; = \;\;\;\;
            \begin{cases}
            \\
            \text{ if $token\_in$ = "base", }\\\\
            \quad\quad\quad (1 - p) \phi\;\; \Delta y
            \\\\
            \text{ if $token\_in$ = "pt", }\\\\
            \quad\quad\quad (p^{-1} - 1) \enspace \phi \enspace (c \cdot \Delta z)
            \\\\
            \end{cases}
            \\\\\\
            & in = in' + f
            \\
            \end{align*}

        Parameters
        ----------
        out : Quantity
            The quantity of tokens that the user wants to receive (the amount
            and the unit of the tokens).
        market_state : MarketState
            The state of the AMM's reserves and share prices.
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch).

        Returns
        -------
        float
            The amount the user pays without fees or slippage. The units
            are always in terms of bonds or base.
        float
            The amount the user pays with fees and slippage. The units are
            always in terms of bonds or base.
        float
            The amount the user pays with slippage and no fees. The units are
            always in terms of bonds or base.
        float
            The fee the user pays. The units are always in terms of bonds or
            base.
        """

        # Calculate some common values up front
        out_amount = Decimal(out.amount)
        normalized_time = Decimal(time_remaining_in_years)
        share_price = Decimal(market_state.share_price)
        d_bonds = out_amount * (1 - normalized_time)
        d_shares = d_bonds / share_price

        market_state = market_state.copy()

        # TODO: This is somewhat strange since these updates never actually hit the reserves.
        # Redeem the matured bonds 1:1 and simulate these updates hitting the reserves.
        if out.unit == TokenType.BASE:
            market_state.share_reserves -= float(d_shares)
            market_state.bond_reserves += float(d_bonds)
        elif out.unit == TokenType.PT:
            market_state.share_reserves += float(d_shares)
            market_state.bond_reserves -= float(d_bonds)
        else:
            raise AssertionError(
                "pricing_models.calc_in_given_out: ERROR: "
                f"Expected out.unit to be {TokenType.BASE} or {TokenType.PT}, not {out.unit}!"
            )
        # Trade the bonds that haven't matured on the YieldSpace curve.
        curve = super().calc_in_given_out(
            out=Quantity(amount=float(out_amount * normalized_time), unit=out.unit),
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        )

        # Compute flat part with fee
        flat_without_fee = out_amount * (1 - normalized_time)
        redemption_fee = flat_without_fee * Decimal(market_state.redemption_fee_percent)
        flat_with_fee = flat_without_fee + redemption_fee

        # Compute the user's trade result including both the flat and the curve parts of the trade.
        if out.unit == TokenType.BASE:
            user_result = Wallet(
                base=out.amount,
                bonds=float(-flat_with_fee + Decimal(curve.user_result.bonds)),
            )
            market_result = Wallet(
                base=-out.amount,
                bonds=curve.market_result.bonds,
            )
        elif out.unit == TokenType.PT:
            user_result = Wallet(
                base=float(-flat_with_fee + Decimal(curve.user_result.base)),
                bonds=out.amount,
            )
            market_result = Wallet(
                base=float(flat_with_fee + Decimal(curve.market_result.base)),
                bonds=curve.market_result.bonds,
            )
        else:
            raise AssertionError(
                "pricing_models.calc_in_given_out: ERROR: "
                f"Expected out.unit to be {TokenType.BASE} or {TokenType.PT}, not {out.unit}!"
            )

        return TradeResult(
            user_result=user_result,
            market_result=market_result,
            breakdown=TradeBreakdown(
                without_fee_or_slippage=float(flat_without_fee + Decimal(curve.breakdown.without_fee_or_slippage)),
                without_fee=float(flat_without_fee + Decimal(curve.breakdown.without_fee)),
                fee=float(redemption_fee + Decimal(curve.breakdown.fee)),
                with_fee=float(flat_with_fee + Decimal(curve.breakdown.with_fee)),
            ),
        )

    # TODO: The high slippage tests in tests/test_pricing_model.py should
    # arguably have much higher slippage. This is something we should
    # consider more when thinking about the use of a time stretch parameter.
    def calc_out_given_in(
        self,
        in_: Quantity,
        market_state: MarketState,
        time_remaining_in_years: float,
        time_stretch: float,
    ) -> TradeResult:
        r"""
        Calculates the amount of an asset that must be provided to receive a specified amount of the
        other asset given the current AMM reserves.

        The output is calculated as:

        .. math::
            \begin{align*}
            & p \;\;\;\; = \;\;\;\; \Bigg(\dfrac{2y + cz}{\mu z}\Bigg)^{-\tau}
            \\\\
            & out'\;\; = \;\;
            \begin{cases}
            \\
            \text{ if $token\_out$ = "base", }\\
            \quad\quad\quad c \big(z - \mu^{-1}
            \big(c \cdot \mu^{-1} \big(k - \big(2y + cz + \Delta y \cdot t\big)
            ^{1 - \tau}\big)\big)
            ^{\tfrac{1}{1 - \tau}}\big) + \Delta y \cdot (1 - \tau)
            \\\\
            \text{ if $token\_out$ = "pt", }\\
            \quad\quad\quad 2y + cz - (k - c \cdot \mu^{-1} \cdot
            (\mu (z + \Delta z \cdot t))^{1 - \tau})
            ^{\tfrac{1}{1 - \tau}} + c \cdot \Delta z \cdot (1 - \tau)
            \\\\
            \end{cases}
            \\\\
            & f \;\;\;\; = \;\;\;\;
            \begin{cases}
            \\
            \text{ if $token\_out$ = "base", }\\\\
            \quad\quad\quad (1 - p) \phi\;\; \Delta y
            \\\\
            \text{ if $token\_out$ = "pt", }\\\\
            \quad\quad\quad (p^{-1} - 1) \enspace \phi \enspace (c \cdot \Delta z)
            \\\\
            \end{cases}
            \\\\\\
            & out = out' + f
            \\
            \end{align*}

        Parameters
        ----------
        in_ : Quantity
            The quantity of tokens that the user wants to pay (the amount and the unit of the
            tokens).
        market_state : MarketState
            The state of the AMM's reserves and share prices.
        time_remaining : StretchedTime
            The time remaining for the asset (incorporates time stretch).

        Returns
        -------
        TradeResult
            The result of performing the trade.
        """

        # Calculate some common values up front
        in_amount = Decimal(in_.amount)
        normalized_time = Decimal(time_remaining_in_years)
        share_price = Decimal(market_state.share_price)
        d_bonds = in_amount * (1 - normalized_time)
        d_shares = d_bonds / share_price

        market_state = market_state.copy()

        # TODO: This is somewhat strange since these updates never actually hit the reserves.
        # Redeem the matured bonds 1:1 and simulate these updates hitting the reserves.
        if in_.unit == TokenType.BASE:
            market_state.share_reserves += float(d_shares)
            market_state.bond_reserves -= float(d_bonds)
        elif in_.unit == TokenType.PT:
            market_state.share_reserves -= float(d_shares)
            market_state.bond_reserves += float(d_bonds)
        else:
            raise AssertionError(
                "pricing_models.calc_out_given_in: ERROR: "
                f"Expected in_.unit to be {TokenType.BASE} or {TokenType.PT}, not {in_.unit}!"
            )

        # Trade the bonds that haven't matured on the YieldSpace curve.
        curve = super().calc_out_given_in(
            in_=Quantity(amount=float(in_amount * normalized_time), unit=in_.unit),
            market_state=market_state,
            time_remaining_in_years=time_remaining_in_years,
            time_stretch=time_stretch,
        )

        # Compute flat part with fee
        flat_without_fee = in_amount * (1 - normalized_time)
        redemption_fee = flat_without_fee * Decimal(market_state.redemption_fee_percent)
        flat_with_fee = flat_without_fee - redemption_fee

        # Compute the user's trade result including both the flat and the curve parts of the trade.
        if in_.unit == TokenType.BASE:
            user_result = Wallet(
                base=-in_.amount,
                bonds=float(flat_with_fee + Decimal(curve.user_result.bonds)),
            )
            market_result = Wallet(
                base=in_.amount,
                bonds=curve.market_result.bonds,
            )
        elif in_.unit == TokenType.PT:
            user_result = Wallet(
                base=float(flat_with_fee + Decimal(curve.user_result.base)),
                bonds=-in_.amount,
            )
            market_result = Wallet(
                base=float(-flat_with_fee + Decimal(curve.market_result.base)),
                bonds=curve.market_result.bonds,
            )
        else:
            raise AssertionError(
                "pricing_models.calc_out_given_in: ERROR: "
                f"Expected in_.unit to be {TokenType.BASE} or {TokenType.PT}, not {in_.unit}!"
            )

        return TradeResult(
            user_result=user_result,
            market_result=market_result,
            breakdown=TradeBreakdown(
                without_fee_or_slippage=float(flat_without_fee + Decimal(curve.breakdown.without_fee_or_slippage)),
                without_fee=float(flat_without_fee + Decimal(curve.breakdown.without_fee)),
                fee=float(Decimal(curve.breakdown.fee) + redemption_fee),
                with_fee=float(flat_with_fee + Decimal(curve.breakdown.with_fee)),
            ),
        )


if __name__ == "__main__":
    simulation_state = SimulationState()
    print(f"{simulation_state=}")
    run_simulation(simulation_state_=simulation_state)
