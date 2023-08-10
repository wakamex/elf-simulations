"""Example custom agent strategy."""
# pylint: disable=line-too-long, missing-class-docstring
from __future__ import annotations
from agent0 import FixedPoint, Rng, BasePolicy, HyperdriveMarketAction, HyperdriveActionType, HyperdriveWallet, HyperdriveMarketState, MarketType, Trade

class DBot(BasePolicy[HyperdriveMarketState, HyperdriveWallet]):
    def __init__(self, budget: FixedPoint, rng: Rng | None = None, trade_list: list[str] | None = None):
        self.trade_list = trade_list or [("add_liquidity", 100), ("open_long", 100), ("open_short", 100)]
        super().__init__(budget, rng)

    def action(self, market: HyperdriveMarketState, wallet: HyperdriveWallet) -> list[Trade]:
        action_type, amount = self.trade_list.pop(0)
        mint_time = next(iter({"close_long": wallet.longs, "close_short": wallet.shorts}.get(action_type, [])), None)
        action = HyperdriveMarketAction(HyperdriveActionType(action_type), wallet, FixedPoint(amount), None, mint_time)
        return [Trade(market_type=MarketType.HYPERDRIVE, market_action=action)]
