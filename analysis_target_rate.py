"""Script to query bot experiment data."""

# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import, bare-except, wrong-import-position, redefined-outer-name, pointless-statement, missing-final-newline, missing-function-docstring, line-too-long
# %%
# bot script setup
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from script_functions import check_docker
check_docker(restart=True)
from script_setup import *
print("\n ==== Pool Config ===")
for k,v in config_data.items():
    print(f"{k:20} | {v}")

# %%
# constants
MAX_ITER = 10
fp0 = FixedPoint(0)
fp1 = FixedPoint(1)
fp2 = FixedPoint(2)
fp12 = FixedPoint(12)
fp_seconds_in_year = FixedPoint(365 * 24 * 60 * 60)
# target_apr = FixedPoint(0.01)  # ONE percent
target_apr = FixedPoint(0.10)  # TEN percent

# calculate amount to long
fixed_rate = hyperdrive.fixed_rate
# variable_rate = hyperdrive.variable_rate
variable_rate = FixedPoint("0.05")
print(f"start {float(fixed_rate):.0%}, target {float(target_apr):.0%}, ", end="")

def calc_bond_reserves(share_reserves, share_price, apr, position_duration, time_stretch):
    return share_price * share_reserves * ((fp1 + apr * position_duration / fp_seconds_in_year) ** time_stretch)

def calc_k(share_price, initial_share_price, share_reserves, bond_reserves, time_stretch):
    # (c / mu) * (mu * z) ** (1 - t) + y ** (1 - t)
    return (share_price / initial_share_price) * (initial_share_price * share_reserves) ** (
        fp1 - time_stretch
    ) + bond_reserves ** (fp1 - time_stretch)

def get_shares_in_for_bonds_out(
    bond_reserves, share_price, initial_share_price, share_reserves, bonds_out, time_stretch, curve_fee, gov_fee, one_block_return
):
    # y_term = (y - out) ** (1 - t)
    # z_val = (k_t - y_term) / (c / mu)
    # z_val = z_val ** (1 / (1 - t))
    # z_val /= mu
    # return z_val - z
    # pylint: disable=too-many-arguments
    k_t = calc_k(
        share_price,
        initial_share_price,
        share_reserves,
        bond_reserves,
        time_stretch,
    )
    y_term = (bond_reserves - bonds_out) ** (fp1 - time_stretch)
    z_val = (k_t - y_term) / (share_price / initial_share_price)
    z_val = z_val ** (fp1 / (fp1 - time_stretch))
    z_val /= initial_share_price
    # z_val *= one_block_return
    spot_price = calc_spot_price_local(initial_share_price, share_reserves, fp0, bond_reserves, time_stretch)
    amount_in_shares = z_val - share_reserves
    price_discount = (fp1 - spot_price)
    # price_discount = (fp1/spot_price - fp1)
    curve_fee_rate = price_discount * curve_fee
    curve_fee_amount_in_shares = amount_in_shares * curve_fee_rate
    gov_fee_amount_in_shares = curve_fee_amount_in_shares * gov_fee
    # applying fees means you pay MORE shares in for the same amount of bonds OUT
    amount_from_user_in_shares = amount_in_shares + curve_fee_amount_in_shares
    return amount_from_user_in_shares, curve_fee_amount_in_shares, gov_fee_amount_in_shares

def get_shares_out_for_bonds_in(bond_reserves, share_price, initial_share_price, share_reserves, bonds_in, time_stretch, curve_fee, gov_fee, one_block_return):
    # y_term = (y + in_) ** (1 - t)
    # z_val = (k_t - y_term) / (c / mu)
    # z_val = z_val ** (1 / (1 - t))
    # z_val /= mu
    # return z - z_val if z > z_val else 0.0
    # pylint: disable=too-many-arguments
    k_t = calc_k(
        share_price,
        initial_share_price,
        share_reserves,
        bond_reserves,
        time_stretch,
    )
    y_term = (bond_reserves + bonds_in) ** (fp1 - time_stretch)
    z_val = (k_t - y_term) / (share_price / initial_share_price)
    z_val = z_val ** (fp1 / (fp1 - time_stretch))
    z_val /= initial_share_price
    # z_val *= one_block_return
    spot_price = calc_spot_price_local(initial_share_price, share_reserves, fp0, bond_reserves, time_stretch)
    # price_discount = (fp1/spot_price - fp1)
    price_discount = (fp1 - spot_price)
    amount_in_shares = max(fp0, share_reserves - z_val)
    curve_fee_rate = price_discount * curve_fee
    curve_fee_amount_in_shares = amount_in_shares * curve_fee_rate
    gov_fee_amount_in_shares = curve_fee_amount_in_shares * gov_fee
    # applying fee means you get LESS shares out for the same amount of bonds IN
    amount_to_user_in_shares = amount_in_shares - curve_fee_amount_in_shares
    return  amount_to_user_in_shares, curve_fee_amount_in_shares, gov_fee_amount_in_shares

def calc_spot_price_local(initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch):
    effective_share_reserves = share_reserves - share_adjustment
    return (initial_share_price * effective_share_reserves / bond_reserves) ** time_stretch


def calc_apr(share_reserves, share_adjustment, bond_reserves, initial_share_price, position_duration_seconds, time_stretch):
    annualized_time = position_duration_seconds / fp_seconds_in_year
    spot_price = calc_spot_price_local(initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch)
    return (fp1 - spot_price) / (spot_price * annualized_time)

def bonds_given_shares_and_rate(share_reserves, share_adjustment, bond_reserves, initial_share_price, time_stretch, target_rate):
    spot_price = calc_spot_price_local(initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch)
    return initial_share_price * share_reserves * spot_price ** ((spot_price * target_rate) / (spot_price - fp1))

# Calculate bonds needed to hit target APR
shares_needed = None
predicted_rate = fp0
tolerance = FixedPoint(scaled_value=1)
pool_config = hyperdrive.pool_config.copy()
pool_info = hyperdrive.pool_info.copy()

start_time = time.time()
# convert to Decimal
USE_DECIMAL = True
PRECISION = 24
if USE_DECIMAL:
    print(f"using Decimal with precision {PRECISION}...")
else:
    print("using FixedPoint")
if USE_DECIMAL:
    pool_info["shareReserves"] = Decimal(str(pool_info["shareReserves"]))
    pool_config["initialSharePrice"] = Decimal(str(pool_config["initialSharePrice"]))
    pool_config["positionDuration"] = Decimal(str(pool_config["positionDuration"]))
    pool_config["timeStretch"] = Decimal(str(pool_config["timeStretch"]))
    pool_config["invTimeStretch"] = Decimal(str(pool_config["invTimeStretch"]))
    pool_info["bondReserves"] = Decimal(str(pool_info["bondReserves"]))
    pool_info["sharePrice"] = Decimal(str(pool_info["sharePrice"]))
    pool_config["curveFee"] = Decimal(str(pool_config["curveFee"]))
    pool_config["governanceFee"] = Decimal(str(pool_config["governanceFee"]))
    fp0 = Decimal(str(fp0))
    fp1 = Decimal(str(fp1))
    fp2 = Decimal(str(fp2))
    fp12 = Decimal(str(fp12))
    fp_seconds_in_year = Decimal(str(fp_seconds_in_year))
    predicted_rate = Decimal(str(predicted_rate))
    target_apr = Decimal(str(target_apr))
    tolerance = Decimal(str(tolerance))
    fixed_rate = Decimal(str(fixed_rate))
    variable_rate = Decimal(str(variable_rate))
    getcontext().prec = PRECISION
one_block_return = (fp1 + variable_rate) ** ( fp12 / fp_seconds_in_year)

iteration = 0
while abs(predicted_rate - target_apr) > tolerance:  # max tolerance 1e-16
    iteration += 1
    target_bonds = calc_bond_reserves(
        pool_info["shareReserves"],
        pool_config["initialSharePrice"],
        target_apr,
        pool_config["positionDuration"],
        pool_config["invTimeStretch"],
    )
    # target_bonds = bonds_given_shares_and_rate(pool_info["shareReserves"], 0, pool_info["bondReserves"], pool_config["initialSharePrice"], pool_config["invTimeStretch"], target_apr)
    # print(f"{pool_config['timeStretch']=}")
    bonds_needed = (target_bonds - pool_info["bondReserves"]) / fp2
    print(f"{bonds_needed=}")
    # assert bonds_needed < 0, "To lower the fixed rate, we should require a decrease in bonds"
    if bonds_needed > 0:  # short
        shares_out, curve_fee, gov_fee = get_shares_out_for_bonds_in(
            pool_info["bondReserves"],
            pool_info["sharePrice"],
            pool_config["initialSharePrice"],
            pool_info["shareReserves"],
            bonds_needed,
            pool_config["timeStretch"],
            pool_config["curveFee"],
            pool_config["governanceFee"],
            one_block_return
        )
        # shares_out is what the user takes OUT: curve_fee less due to fees.
        # gov_fee of that doesn't stay in the pool, going OUT to governance (same direction as user flow).
        pool_info["shareReserves"] += (-shares_out - gov_fee)  * 1
    else:  # long
        shares_in, curve_fee, gov_fee = get_shares_in_for_bonds_out(
            pool_info["bondReserves"],
            pool_info["sharePrice"],
            pool_config["initialSharePrice"],
            pool_info["shareReserves"],
            -bonds_needed,
            pool_config["timeStretch"],
            pool_config["curveFee"],
            pool_config["governanceFee"],
            one_block_return
        )
        print(f"{shares_in=}")
        print(f"{curve_fee=}")
        print(f"{gov_fee=}")
        # shares_in is what the user pays IN: curve_fee more due to fees.
        # gov_fee of that doesn't go to the pool, going OUT to governance (opposite direction of user flow).
        pool_info["shareReserves"] += (shares_in - gov_fee) * 1
    pool_info["bondReserves"] += bonds_needed
    if USE_DECIMAL:
        total_shares_needed = pool_info["shareReserves"] - Decimal(str(hyperdrive.pool_info["shareReserves"]))
        total_bonds_needed = pool_info["bondReserves"] - Decimal(str(hyperdrive.pool_info["bondReserves"]))
    else:
        total_shares_needed = pool_info["shareReserves"] - hyperdrive.pool_info["shareReserves"]
        total_bonds_needed = pool_info["bondReserves"] - hyperdrive.pool_info["bondReserves"]
    predicted_rate = calc_apr(pool_info["shareReserves"], fp0, pool_info["bondReserves"], pool_config["initialSharePrice"], pool_config["positionDuration"], pool_config["timeStretch"])
    print(f"iteration {iteration:3}: {float(predicted_rate):22.18%} d_bonds={float(total_bonds_needed):27,.18f} d_shares={float(total_shares_needed):27,.18f}")
    if iteration >= MAX_ITER:
        break
print(f"predicted precision: {float(abs(predicted_rate-target_apr))}, time taken: {time.time() - start_time}s")

# %%
pool_info = hyperdrive.pool_info.copy()
bond_reserves_before = pool_info["bondReserves"]
share_reserves_before = pool_info["shareReserves"]
if USE_DECIMAL:
    bond_reserves_before = Decimal(str(bond_reserves_before))
    share_reserves_before = Decimal(str(share_reserves_before))
if total_shares_needed > fp0:  # long
    print(f"{total_shares_needed=}")
    current_share_price = hyperdrive.pool_info["sharePrice"]
    if USE_DECIMAL:
        current_share_price = Decimal(str(current_share_price))
    base_needed = total_shares_needed * current_share_price
    print(f"{base_needed=}")
    amount = float(base_needed)
    print(f"{amount=}")
    logging.log(10, f" open long of {(amount)} to hit {int(target_apr)/1e18:.0%}")
    run_trades(trade_list=[("open_long", amount)])
else:  # short
    amount = float(total_bonds_needed)
    # if USE_DECIMAL:
    #     total_shares_needed,_,_ = get_shares_out_for_bonds_in(
    #         Decimal(str(pool_info["bondReserves"])),
    #         Decimal(str(pool_info["sharePrice"])),
    #         Decimal(str(pool_config["initialSharePrice"])),
    #         Decimal(str(pool_info["shareReserves"])),
    #         total_bonds_needed,
    #         Decimal(str(pool_config["timeStretch"])),
    #         Decimal(str(pool_config["curveFee"])),
    #         Decimal(str(pool_config["governanceFee"])),
    #         one_block_return
    #     )
    # else:
    #     total_shares_needed,_,_ = get_shares_out_for_bonds_in(
    #         pool_info["bondReserves"],
    #         pool_info["sharePrice"],
    #         pool_config["initialSharePrice"],
    #         pool_info["shareReserves"],
    #         total_bonds_needed,
    #         pool_config["timeStretch"],
    #         pool_config["curveFee"],
    #         pool_config["governanceFee"],
    #         one_block_return
    #     )
    # total_shares_needed = -total_shares_needed * 1
    total_shares_needed *= 1
    logging.log(10, f" open short of {(amount)} to hit {int(target_apr)/1e18:.2%}")
    run_trades(trade_list=[("open_short", amount)])
time.sleep(2)
bond_reserves_after = hyperdrive.pool_info["bondReserves"]
share_reserves_after = hyperdrive.pool_info["shareReserves"]
if USE_DECIMAL:
    bond_reserves_after = Decimal(str(bond_reserves_after))
    share_reserves_after = Decimal(str(share_reserves_after))
d_bonds = bond_reserves_after - bond_reserves_before
d_shares = share_reserves_after - share_reserves_before
print(f"change in pool:")
print(f"  d_bonds  = {float(d_bonds):+27,.18f}", end="")
print(f" expected: {float(total_bonds_needed):+27,.18f}")
d_bonds_diff = d_bonds - total_bonds_needed
print(f"   diff: {float(d_bonds_diff):+27,.18f}", end="")
print(f" diff (%): {float(d_bonds_diff/total_bonds_needed):.1e}")
print(f"  d_shares = {float(d_shares):+27,.18f}", end="")
print(f" expected: {float(total_shares_needed):+27,.18f}")
d_shares_diff = d_shares - total_shares_needed
print(f"   diff: {float(d_shares_diff):+27,.18f}", end="")
print(f" diff (%): {float(d_shares_diff/total_shares_needed):.1e}")

# %%
# get data
time.sleep(2)
data, pool_info = get_data(session, config_data)

# %%
pool_config = hyperdrive.pool_config.copy()
pool_info = hyperdrive.pool_info.copy()
if USE_DECIMAL:
    pool_info["shareReserves"] = Decimal(str(pool_info["shareReserves"]))
    pool_config["initialSharePrice"] = Decimal(str(pool_config["initialSharePrice"]))
    pool_config["positionDuration"] = Decimal(str(pool_config["positionDuration"]))
    pool_config["timeStretch"] = Decimal(str(pool_config["timeStretch"]))
    pool_config["invTimeStretch"] = Decimal(str(pool_config["invTimeStretch"]))
    pool_info["bondReserves"] = Decimal(str(pool_info["bondReserves"]))
    pool_info["sharePrice"] = Decimal(str(pool_info["sharePrice"]))
    pool_config["curveFee"] = Decimal(str(pool_config["curveFee"]))
    pool_config["governanceFee"] = Decimal(str(pool_config["governanceFee"]))

current_rate = calc_apr(pool_info["shareReserves"], fp0, pool_info["bondReserves"], pool_config["initialSharePrice"], pool_config["positionDuration"], pool_config["timeStretch"])
print(f"target: {float(target_apr):0%} ", end="")
print(f"actual: {float(current_rate):22.18%} ")
d_apr = current_rate - target_apr
print(f" diff: {float(abs(d_apr)):22.18%}", end="")
print(f" diff (%): {float(d_apr/target_apr):.2e}")

# %%
pool_info["sharePrice"]

# %%
