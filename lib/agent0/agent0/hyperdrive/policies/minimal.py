# pylint: disable=line-too-long, missing-class-docstring, missing-module-docstring, wildcard-import, unused-import, missing-final-newline
from . import *
class MBot(BasePolicy):
    def __init__(self, budget, rng):
        self.trade_list = ["add_liquidity", "open_short", "open_long", "close_short"]
        super().__init__(budget, rng)
    def action(self, _, wallet):
        mint_time = next(iter({"close_long": wallet.longs, "close_short": wallet.shorts}.get(self.trade_list[0], [])), None)
        return [Trade(MarketType.HYPERDRIVE, Action(Type(self.trade_list.pop(0)), wallet, fp100, fp0, mint_time))]