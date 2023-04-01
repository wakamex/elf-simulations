"""Initializer agent

Special reserved user strategy that is used to initialize a market with a desired amount of share & bond reserves
"""
from elfpy.agent import Agent
from elfpy.main import MarketActionType

# pylint: disable=duplicate-code


class Policy(Agent):
    """Adds a large LP"""

    def action(self):
        """
        User strategy adds liquidity and then takes no additional actions
        """
        if self.wallet.lp > 0:  # has already opened the lp
            return []
        return [self.create_agent_action(action_type=MarketActionType.ADD_LIQUIDITY, trade_amount=self.budget)]
