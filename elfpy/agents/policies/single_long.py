"""User strategy that opens a long position and then closes it after a certain amount of time has passed."""
from elfpy.agents import agent
from elfpy.markets.hyperdrive import hyperdrive_actions
from elfpy.markets.hyperdrive import hyperdrive_market
from elfpy import types

# pylint: disable=too-many-arguments
# pylint: disable=duplicate-code


class Policy(agent.Agent):
    """simple long
    only has one long open at a time.
    """

    def action(self, market: hyperdrive_market.Market) -> "list[types.Trade]":
        """Specify action."""
        longs = list(self.wallet.longs.values())
        has_opened_long = len(longs) > 0
        time_to_wait_until_closing = 0.01
        action_list = []
        if has_opened_long:
            mint_time = list(self.wallet.longs)[-1]
            enough_time_has_passed = market.block_time.time - mint_time > time_to_wait_until_closing
            if enough_time_has_passed:
                action_list.append(
                    types.Trade(
                        market=types.MarketType.HYPERDRIVE,
                        trade=hyperdrive_actions.MarketAction(
                            action_type=hyperdrive_actions.MarketActionType.CLOSE_LONG,
                            trade_amount=longs[-1].balance,
                            wallet=self.wallet,
                            mint_time=mint_time,
                        ),
                    )
                )
        else:
            trade_amount = self.get_max_long(market) / 2
            action_list.append(
                types.Trade(
                    market=types.MarketType.HYPERDRIVE,
                    trade=hyperdrive_actions.MarketAction(
                        action_type=hyperdrive_actions.MarketActionType.OPEN_LONG,
                        trade_amount=trade_amount,
                        wallet=self.wallet,
                    ),
                )
            )
        return action_list
