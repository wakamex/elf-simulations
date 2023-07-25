"""Example custom agent strategy."""
from __future__ import annotations

from fixedpointmath import FixedPoint
from numpy.random._generator import Generator as NumpyGenerator

from elfpy.agents.policies import BasePolicy
from elfpy.markets.hyperdrive import HyperdriveMarketAction, MarketActionType
from elfpy.markets.hyperdrive.hyperdrive_market import HyperdriveMarket
from elfpy.types import MarketType, Trade
from elfpy.wallet.wallet import Wallet

# pylint: disable=too-few-public-methods, missing-function-docstring, invalid-name, too-many-arguments, line-too-long, missing-class-docstring
LP, CLOSE_LP, LONG, SHORT, CLOSE_LONG, CLOSE_SHORT = "add_liquidity", "remove_liquidity", "open_long", "open_short", "close_long", "close_short"

def new_trade(action_type: str, amount: FixedPoint | int, wallet: Wallet, mint_time=None) -> Trade:
    amount = amount if isinstance(amount, FixedPoint) else FixedPoint(amount)
    action = HyperdriveMarketAction(MarketActionType(action_type), wallet, amount, FixedPoint(0), mint_time)
    return Trade(market=MarketType.HYPERDRIVE, trade=action)

class DBot(BasePolicy):
    def __init__(self, budget: FixedPoint, rng: NumpyGenerator | None = None, amount: FixedPoint | None = None):
        self.amount = FixedPoint(100) if amount is None else amount
        self.trade_count = 0
        self.trade_list = [(LP, 100), (LONG, 100), (SHORT, 100), (CLOSE_SHORT, 100)]
        super().__init__(budget, rng)

    def action(self, market: HyperdriveMarket, wallet: Wallet) -> list[Trade]:
        action, amount = self.trade_list[self.trade_count]
        mint_time = list(wallet.longs)[0] if action == CLOSE_LONG else list(wallet.shorts)[0] if action == CLOSE_SHORT else None
        p = market.spot_price
        print(f"market is:\n fixed_apr={market.fixed_apr}\n spot_price={p}")
        print(f"for 100 shorts I expect to pay slightly more than {100*(1-p)} ({float(100*(1-p))=:.4f}) ({float(1-p)=:.6f})")
        # for d in dir(market):
        #     if not d.startswith("_"):
        #         print(d)
        self.trade_count += 1
        return [new_trade(action_type=action, amount=amount, wallet=wallet, mint_time=mint_time)]
