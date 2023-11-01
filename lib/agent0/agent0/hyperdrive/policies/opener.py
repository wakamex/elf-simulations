"""Deterministically trade things."""
# pylint: disable=line-too-long, missing-class-docstring
from __future__ import annotations
from dataclasses import dataclass
from agent0 import FixedPoint, Rng, HyperdrivePolicy, HyperdriveMarketAction, HyperdriveActionType, HyperdriveWallet, HyperdriveInterface, MarketType, Trade

class Opener(HyperdrivePolicy):
    @dataclass
    class Config(HyperdrivePolicy.Config):
        trade_amount: FixedPoint = FixedPoint(10_000)

    def __init__(self, budget: FixedPoint, rng: Rng | None = None, policy_config: Config | None = None):
        self.policy_config = policy_config or self.Config()
        self.long = True
        self.exposure = FixedPoint(0)
        super().__init__(budget, rng)

    def action(self, interface: HyperdriveInterface, wallet: HyperdriveWallet) -> tuple[list[Trade[HyperdriveMarketAction]], bool]:
        total_exposure = sum(long.balance for long in wallet.longs.values()) + sum(short.balance for short in wallet.shorts.values())
        print(f"{self.exposure=}")
        print(f"{total_exposure=}")
        if 0 < total_exposure <= self.exposure:
            self.long = not self.long  # reverse, we didn't open more exposure last go-around!
        self.exposure = total_exposure
        action_type = "open_long" if self.long else "open_short"
        trade_amount = None
        if self.long:
            trade_amount = interface.get_max_long(wallet.balance.amount)  # in base
        else:
            max_base = interface.get_max_short(wallet.balance.amount)  # in base
            trade_amount = max_base * interface.spot_price  # in bonds
        action = HyperdriveMarketAction(HyperdriveActionType(action_type), wallet, trade_amount)
        return [Trade(market_type=MarketType.HYPERDRIVE, market_action=action)], False
