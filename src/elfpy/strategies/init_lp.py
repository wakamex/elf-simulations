"""
Special reserved user strategy that is used to initialize a market with a desired amount of share & bond reserves
"""
# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments

import logging

from elfpy.strategies.basic import BasicPolicy
from elfpy.pricing_models import ElementPricingModel


class Policy(BasicPolicy):
    """
    simple LP
    only has one LP open at a time
    """

    def __init__(
        self,
        market,
        rng,
        wallet_address,
        budget=1000,
        **kwargs,
    ):
        """call basic policy init then add custom stuff"""
        # these are default values only, they get overwritten by custom values in kwargs
        self.base_to_lp = 100
        self.pt_to_short = 100
        super().__init__(
            market=market,
            rng=rng,
            wallet_address=wallet_address,
            budget=budget,
            **kwargs,
        )
        logging.debug(
            "initializing init_lp strategy with base_to_lp: %g, pt_to_short: %g, kwargs: %s",
            self.base_to_lp,
            self.pt_to_short,
            kwargs,
        )

    def action(self):
        """
        implement agent strategy
        LP if you can, but only do it once
        short if you can, but only do it once
        """
        has_lp = self.wallet.lp_in_wallet > 0
        if has_lp:
            action_list = []
        else:
            if self.market.pricing_model.model_name == ElementPricingModel().model_name():
                # TODO: This doesn't work correctly -- need to add PT
                action_list = [
                    self.create_agent_action(action_type="add_liquidity", trade_amount=self.base_to_lp),
                ]
            else:
                action_list = [
                    self.create_agent_action(action_type="add_liquidity", trade_amount=self.base_to_lp),
                    self.create_agent_action(action_type="open_short", trade_amount=self.pt_to_short),
                ]
        return action_list
