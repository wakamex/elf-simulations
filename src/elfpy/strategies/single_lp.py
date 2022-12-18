"""
User strategy that adds base liquidity and doesn't remove until liquidation
"""
# TODO: the init calls are replicated across each strategy, which looks like duplicate code
#     this should be resolved once we fix user inheritance
# pylint: disable=duplicate-code

import logging

from elfpy.strategies.basic import BasicPolicy


class Policy(BasicPolicy):
    """
    single LP
    only has one LP open at a time
    """

    def __init__(self, market, rng, wallet_address, budget=1000, **kwargs):
        """call basic policy init then add custom stuff"""
        # these are default values only, they get overwritten by custom values in kwargs
        self.base_to_lp = 100
        super().__init__(
            market=market,
            rng=rng,
            wallet_address=wallet_address,
            budget=budget,
            **kwargs,
        )

    def action(self):
        """
        implement agent strategy
        LP if you can, but only do it once
        """
        action_list = []
        has_lp = self.wallet.lp_in_wallet > 0
        can_lp = self.wallet.base_in_wallet >= self.base_to_lp
        logging.debug(
            ("evaluating LP, base_in_wallet: %g, can_lp: %g, has_lp: %g"),
            self.wallet.base_in_wallet,
            can_lp,
            has_lp,
        )
        if can_lp and not has_lp:
            action_list.append(self.create_agent_action(action_type="add_liquidity", trade_amount=self.base_to_lp))
        return action_list
