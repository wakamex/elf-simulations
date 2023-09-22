"""Trade around a target fixed rate area."""
# pylint: disable=line-too-long, missing-class-docstring, too-many-locals
from __future__ import annotations
from agent0 import FixedPoint, Rng, BasePolicy, HyperdriveMarketAction, HyperdriveActionType, HyperdriveWallet, HyperdriveMarketState, MarketType, Trade, WEI

class ShengTsung(BasePolicy[HyperdriveMarketState, HyperdriveWallet]):
    def __init__(
        self,
        budget: FixedPoint,
        rng: Rng | None = None,
        slippage_tolerance: FixedPoint | None = None,
        trade_amount: FixedPoint | None = None,
        high_fixed_rate_thresh: FixedPoint = FixedPoint(0.51),
        low_fixed_rate_thresh: FixedPoint = FixedPoint(0.049),
    ):
        self.high_fixed_rate_thresh = high_fixed_rate_thresh
        self.low_fixed_rate_thresh = low_fixed_rate_thresh
        if trade_amount is None:
            self.trade_amount = FixedPoint(100)
            print("Policy trade_amount not set, using 100.")
        else:
            self.trade_amount: FixedPoint = trade_amount

        if slippage_tolerance is None:
            super().__init__(budget, rng)
        else:
            super().__init__(budget, rng, slippage_tolerance)

    def action(self, market: HyperdriveMarketState, wallet: HyperdriveWallet) -> list[Trade]:
        """Specify actions.

        Arguments
        ---------
        market : HyperdriveMarketState
            the trading market
        wallet : HyperdriveWallet
            agent's wallet

        Returns
        -------
        list[MarketAction]
            list of actions
        """
        if wallet.balance.amount <= WEI:
            return []

        # Calculate fixed rate
        init_share_price = market.market_state.init_share_price
        share_reserves = market.market_state.share_reserves
        bond_reserves = market.market_state.bond_reserves
        time_stretch = FixedPoint(1) / market.time_stretch_constant
        annualized_time = market.position_duration.days / (365)
        spot_price = ((init_share_price * share_reserves) / bond_reserves) ** time_stretch
        fixed_rate = (1 - spot_price) / (spot_price * annualized_time)

        # Make trade amount for longs based on how much higher the fixed rate is vs the threshold
        open_long_perc = fixed_rate / self.high_fixed_rate_thresh

        action_list = []
        # Close longs if it's matured
        longs = list(wallet.longs.values())
        has_opened_long = len(longs) > 0
        if has_opened_long:
            mint_time = list(wallet.longs)[0]  # get the mint time of the open long
            if market.block_time.time - mint_time >= market.position_duration.years:
                action = HyperdriveMarketAction(action_type=HyperdriveActionType.CLOSE_LONG,trade_amount=longs[0].balance,wallet=wallet,mint_time=mint_time)
                action_list.append(Trade(market_type=MarketType.HYPERDRIVE,market_action=action))

        # High fixed rate, close any open shorts and open longs
        if fixed_rate >= self.high_fixed_rate_thresh:
            if len(wallet.shorts) > 0:
                for mint_time, short in wallet.shorts.items():
                    action = HyperdriveMarketAction(action_type=HyperdriveActionType.CLOSE_SHORT,trade_amount=short.balance,wallet=wallet,mint_time=mint_time)
                    action_list.append(Trade(market_type=MarketType.HYPERDRIVE,market_action=action))
            action = HyperdriveMarketAction(action_type=HyperdriveActionType.OPEN_LONG,trade_amount=self.trade_amount * open_long_perc,wallet=wallet)
            action_list.append(Trade(market_type=MarketType.HYPERDRIVE,market_action=action))

        # Low fixed rate, close all longs and open short
        if fixed_rate <= self.low_fixed_rate_thresh:
            if len(wallet.longs) > 0:
                for mint_time, long in wallet.longs.items():
                    action = HyperdriveMarketAction(action_type=HyperdriveActionType.CLOSE_LONG,trade_amount=long.balance,wallet=wallet,mint_time=mint_time)
                    action_list.append(Trade(market_type=MarketType.HYPERDRIVE,market_action=action))
            action = HyperdriveMarketAction(action_type=HyperdriveActionType.OPEN_SHORT,trade_amount=self.trade_amount,wallet=wallet)
            action_list.append(Trade(market_type=MarketType.HYPERDRIVE,market_action=action))

        return action_list
