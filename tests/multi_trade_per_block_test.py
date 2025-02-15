"""System test for end to end testing of elf-simulations"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

import pandas as pd
from agent0 import build_account_key_config_from_agent_config
from agent0.base.config import AgentConfig, EnvironmentConfig
from agent0.hyperdrive.exec import run_agents
from agent0.hyperdrive.policies import HyperdrivePolicy
from agent0.hyperdrive.state import HyperdriveActionType, HyperdriveMarketAction, HyperdriveWallet
from chainsync.db.hyperdrive.interface import get_ticker, get_transactions, get_wallet_deltas
from chainsync.exec import acquire_data, data_analysis
from elfpy.types import MarketType, Trade
from eth_typing import URI
from ethpy import EthConfig
from ethpy.base.errors import ContractCallException
from fixedpointmath import FixedPoint
from numpy.random._generator import Generator as NumpyGenerator
from sqlalchemy.orm import Session
from web3 import HTTPProvider

if TYPE_CHECKING:
    from ethpy.hyperdrive import HyperdriveAddresses
    from ethpy.hyperdrive.api import HyperdriveInterface
    from ethpy.test_fixtures.local_chain import DeployedHyperdrivePool


class MultiTradePolicy(HyperdrivePolicy):
    """A agent that submits multiple trades per block"""

    # Using default parameters
    def __init__(
        self,
        budget: FixedPoint,
        rng: NumpyGenerator | None = None,
        slippage_tolerance: FixedPoint | None = None,
        # When this policy doesn't have a config and doesn't define a custom config object
        # we still need it in the constructor since the object factory still calls with this arg
        policy_config: HyperdrivePolicy.Config | None = None,  # pylint: disable=unused-argument
    ):
        # We want to do a sequence of trades one at a time, so we keep an internal counter based on
        # how many times `action` has been called.
        self.counter = 0
        self.made_trade = False
        super().__init__(budget, rng, slippage_tolerance)

    def action(
        self, interface: HyperdriveInterface, wallet: HyperdriveWallet
    ) -> tuple[list[Trade[HyperdriveMarketAction]], bool]:
        """This agent simply opens all trades for a fixed amount and closes them after, one at a time"""
        # pylint: disable=unused-argument
        action_list = []

        if self.made_trade:
            # We want this bot to exit and crash after it's done the trades it needs to do
            # In this case, if this exception gets thrown, this means an invalid trade went through
            raise AssertionError("This policy's action shouldn't get called again after failure")

        # Adding in 4 trades at the same time:

        # Add liquidity
        action_list.append(
            Trade(
                market_type=MarketType.HYPERDRIVE,
                market_action=HyperdriveMarketAction(
                    action_type=HyperdriveActionType.ADD_LIQUIDITY,
                    trade_amount=FixedPoint(11111),
                    wallet=wallet,
                ),
            )
        )

        # Open Long
        action_list.append(
            Trade(
                market_type=MarketType.HYPERDRIVE,
                market_action=HyperdriveMarketAction(
                    action_type=HyperdriveActionType.OPEN_LONG,
                    trade_amount=FixedPoint(22222),
                    wallet=wallet,
                ),
            )
        )

        # Open Short
        action_list.append(
            Trade(
                market_type=MarketType.HYPERDRIVE,
                market_action=HyperdriveMarketAction(
                    action_type=HyperdriveActionType.OPEN_SHORT,
                    trade_amount=FixedPoint(33333),
                    wallet=wallet,
                ),
            )
        )

        # Illegal trade
        # TODO this trade is currently returning an uninformative assertion error
        action_list.append(
            Trade(
                market_type=MarketType.HYPERDRIVE,
                market_action=HyperdriveMarketAction(
                    action_type=HyperdriveActionType.REDEEM_WITHDRAW_SHARE,
                    trade_amount=FixedPoint(99999999999),
                    wallet=wallet,
                ),
            )
        )

        self.made_trade = True

        return action_list, True


class TestMultiTradePerBlock:
    """Tests pipeline from bots making trades to viewing the trades in the db"""

    # TODO split this up into different functions that work with tests
    # pylint: disable=too-many-locals, too-many-statements
    def test_multi_trade_per_block(
        self,
        local_hyperdrive_pool: DeployedHyperdrivePool,
        db_session: Session,
        db_api: str,
    ):
        """Runs the entire pipeline and checks the database at the end.
        All arguments are fixtures.
        """
        # TODO local_hyperdrive_pool is currently being run with automining. Hence, multiple trades
        # per block can't be tested until we can parameterize anvil running without automining.
        # For now, this is simply testing that the introduction of async trades doesn't break
        # when automining.

        # Run this test with develop mode on
        os.environ["DEVELOP"] = "true"

        # Get hyperdrive chain info
        uri: URI | None = cast(HTTPProvider, local_hyperdrive_pool.web3.provider).endpoint_uri
        rpc_uri = uri if uri else URI("http://localhost:8545")
        hyperdrive_contract_addresses: HyperdriveAddresses = local_hyperdrive_pool.hyperdrive_contract_addresses

        # Build environment config
        env_config = EnvironmentConfig(
            delete_previous_logs=True,
            halt_on_errors=True,
            # We don't want tests to write lots of files
            crash_report_to_file=False,
            log_filename=".logging/multi_trade_per_block_test.log",
            log_level=logging.INFO,
            log_stdout=True,
            random_seed=1234,
            username="test",
        )

        # Build agent config
        agent_config: list[AgentConfig] = [
            AgentConfig(
                policy=MultiTradePolicy,
                number_of_agents=1,
                slippage_tolerance=None,
                base_budget_wei=FixedPoint("1_000_000").scaled_value,  # 1 million base
                eth_budget_wei=FixedPoint("100").scaled_value,  # 100 base
                policy_config=MultiTradePolicy.Config(),
            ),
        ]

        # No need for random seed, this bot is deterministic
        account_key_config = build_account_key_config_from_agent_config(agent_config)

        # Build custom eth config pointing to local test chain
        eth_config = EthConfig(
            # Artifacts_uri isn't used here, as we explicitly set addresses and passed to run_bots
            artifacts_uri="not_used",
            rpc_uri=rpc_uri,
            database_api_uri=db_api,
            # Using default abi dir
        )

        try:
            run_agents(
                env_config,
                agent_config,
                account_key_config,
                eth_config=eth_config,
                contract_addresses=hyperdrive_contract_addresses,
            )
            # If this reaches this point, the agent was successful, which means this test should fail
            assert False, "Agent was successful with known invalid trade"
        except ContractCallException as exc:
            # Expected error due to illegal trade
            # We do add an argument for invalid balance to the args, so check for that here
            assert "Invalid balance:" in exc.args[0]

        # Run acquire data to get data from chain to db
        acquire_data(
            start_block=8,  # First 7 blocks are deploying hyperdrive, ignore
            eth_config=eth_config,
            db_session=db_session,
            contract_addresses=hyperdrive_contract_addresses,
            # Exit the script after catching up to the chain
            exit_on_catch_up=True,
        )

        # Run data analysis to calculate various analysis values
        data_analysis(
            start_block=8,  # First 7 blocks are deploying hyperdrive, ignore
            eth_config=eth_config,
            db_session=db_session,
            contract_addresses=hyperdrive_contract_addresses,
            # Exit the script after catching up to the chain
            exit_on_catch_up=True,
        )

        # Ensure other 3 trades went through
        # 1. addLiquidity of 11111 base
        # 2. openLong of 22222 base
        # 3. openShort of 33333 bonds

        db_transaction_info: pd.DataFrame = get_transactions(db_session, coerce_float=False)
        # TODO transactions is logging the failed trade? Is this desired?
        assert len(db_transaction_info == 4)
        # Checking without order
        trxs = db_transaction_info["input_method"].to_list()
        assert "addLiquidity" in trxs
        assert "openLong" in trxs
        assert "openShort" in trxs
        assert "redeemWithdrawalShares" in trxs

        db_ticker: pd.DataFrame = get_ticker(db_session, coerce_float=False)
        assert len(db_ticker == 3)
        ticker_ops = db_ticker["trade_type"].to_list()
        assert "addLiquidity" in ticker_ops
        assert "openLong" in ticker_ops
        assert "openShort" in ticker_ops

        wallet_deltas: pd.DataFrame = get_wallet_deltas(db_session, coerce_float=False)
        # Ensure deltas only exist for valid trades
        # 2 for each trade
        assert len(wallet_deltas) == 6
