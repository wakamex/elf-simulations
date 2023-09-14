"""Calculate the fixed interest rate."""
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext

import numpy as np
import pandas as pd

from .calc_spot_price import calc_spot_price

def calc_fixed_rate(spot_price: pd.Series, position_duration: Decimal):
    """Calculate fixed rate from spot price."""
    # sourcery skip: inline-immediately-returned-variable
    # Position duration (in seconds) in terms of fraction of year
    # This div should round up
    # This replicates div up in fixed point
    with localcontext() as ctx:
        ctx.prec = 18
        ctx.rounding = ROUND_UP
        annualized_time = position_duration / Decimal(60 * 60 * 24 * 365)

    # Pandas is smart enough to be able to broadcast with internal Decimal types at runtime
    # We keep things in 18 precision here
    with localcontext() as ctx:
        ctx.prec = 18
        ctx.rounding = ROUND_DOWN
        fixed_rate = (1 - spot_price) / (spot_price * annualized_time)  # type: ignore
    return fixed_rate

def calc_fixed_rate_df(trade_data, config_data):
    """Calculate fixed rate from trade and config data."""
    trade_data["rate"] = np.nan
    annualized_time = config_data["positionDuration"] / Decimal(60 * 60 * 24 * 365)
    spot_price = calc_spot_price(
        trade_data["share_reserves"],
        trade_data["bond_reserves"],
        config_data["initialSharePrice"],
        config_data["invTimeStretch"],
    )
    fixed_rate = (Decimal(1) - spot_price) / (spot_price * annualized_time)
    x_data = trade_data["timestamp"]
    y_data = fixed_rate
    return x_data, y_data
