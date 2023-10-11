"""Deterministically trade things."""
# pylint: disable=line-too-long, missing-class-docstring
from __future__ import annotations
from dataclasses import dataclass
from agent0 import FixedPoint, Rng, HyperdrivePolicy, HyperdriveMarketAction, HyperdriveActionType, HyperdriveWallet, HyperdriveInterface, MarketType, Trade

class DBot(HyperdrivePolicy):
    @dataclass
    class Config(HyperdrivePolicy.Config):
        trade_list: list[tuple[str, int]]

    def __init__(self, budget: FixedPoint, rng: Rng | None = None, policy_config: Config | None = None):
        self.trade_list = policy_config.trade_list if policy_config else [("add_liquidity", 100), ("open_long", 100), ("open_short", 100)]
        self.starting_length = len(self.trade_list)
        super().__init__(budget, rng)

    def action(self, interface: HyperdriveInterface, wallet: HyperdriveWallet) -> tuple[list[Trade[HyperdriveMarketAction]], bool]:
        print(f"ACTION LOG {len(self.trade_list)}/{self.starting_length}")
        if not self.trade_list:
            return [], True  # done trading
        action_type, amount = self.trade_list.pop(0)
        mint_time = next(iter({"close_long": wallet.longs, "close_short": wallet.shorts}.get(action_type, [])), None)
        action = HyperdriveMarketAction(HyperdriveActionType(action_type), wallet, FixedPoint(amount), None, mint_time)
        return [Trade(market_type=MarketType.HYPERDRIVE, market_action=action)], False