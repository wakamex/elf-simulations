"""Example custom agent strategy."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fixedpointmath import FixedPoint

from elfpy import WEI
from elfpy.agents.policies import BasePolicy
from elfpy.markets.hyperdrive import HyperdriveMarketAction, MarketActionType
from elfpy.types import MarketType, Trade

if TYPE_CHECKING:
    from numpy.random._generator import Generator as NumpyGenerator

    from elfpy.markets.hyperdrive import HyperdriveMarket
    from elfpy.wallet.wallet import Wallet

# pylint: disable=too-few-public-methods, missing-function-docstring, invalid-name


def long_action(amount: int, wallet):
    return HyperdriveMarketAction(MarketActionType.OPEN_LONG, trade_amount=FixedPoint(amount), wallet=wallet)


def short_action(amount: int, wallet):
    return HyperdriveMarketAction(MarketActionType.OPEN_SHORT, trade_amount=FixedPoint(amount), wallet=wallet)


def LP_action(amount: int, wallet):
    return HyperdriveMarketAction(MarketActionType.ADD_LIQUIDITY, trade_amount=FixedPoint(amount), wallet=wallet)


def long(amount: int, wallet) -> Trade:
    return Trade(market=MarketType.HYPERDRIVE, trade=long_action(amount, wallet))


def short(amount: int, wallet) -> Trade:
    return Trade(market=MarketType.HYPERDRIVE, trade=short_action(amount, wallet))


def LP(amount: int, wallet):
    return Trade(market=MarketType.HYPERDRIVE, trade=LP_action(amount, wallet))


class DBot(BasePolicy):
    """Deterministic bot."""

    def __init__(self, budget: FixedPoint, rng: NumpyGenerator | None = None, trade_amount: FixedPoint | None = None):
        if trade_amount is None:
            self.trade_amount = FixedPoint(100)
            logging.warning("Policy trade_amount not set, using 100.")
        else:
            self.trade_amount: FixedPoint = trade_amount
        self.trade_count = 0
        self.trade_list = [(LP, 100), (long, 100), (short, 100)]
        super().__init__(budget, rng)

    def action(self, market: HyperdriveMarket, wallet: Wallet) -> list[Trade]:
        """Pick a trade from the pre-determined list based on trade_count.

        Returns
        -------
        list[MarketAction]
            list of actions
        """
        if self.trade_count >= len(self.trade_list):
            return []
        trade, amount = self.trade_list[self.trade_count]
        self.trade_count += 1
        return [trade(amount, wallet)]
