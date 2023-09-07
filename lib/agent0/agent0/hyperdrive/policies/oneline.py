# pylint: disable=line-too-long, missing-class-docstring, missing-module-docstring, wildcard-import, unused-import, missing-final-newline
from . import *
class OBot(BasePolicy):
    def __init__(self, budget, rng):
        self.trade_list = ["add_liquidity", "open_long", "open_short"]
        super().__init__(budget, rng)
    def action(self, _, wallet):
        return [Trade(HYPERDRIVE, Action(Type(self.trade_list.pop(0)), wallet, fp100, fp0))]