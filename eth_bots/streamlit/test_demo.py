# %%
from __future__ import annotations

import time

import matplotlib.pyplot as plt
from calc_pnl import calc_closeout_pnl, calc_total_returns
from dotenv import load_dotenv

from eth_bots.data import postgres
from eth_bots.streamlit.calc_pnl import calc_total_returns
from eth_bots.streamlit.extract_data_logs import get_combined_data

load_dotenv()
session = postgres.initialize_session()
config_data = postgres.get_pool_config(session, coerce_float=False)
config_data["invTimeStretch"] = config_data["invTimeStretch"] / 10**18
config_data = config_data.iloc[0]
max_live_blocks = 14400

# Place data and plots
start_time = time.time()
txn_data = postgres.get_transactions(session, -max_live_blocks)
print(f"get_transactions in {time.time() - start_time}")
start_time = time.time()
pool_info_data = postgres.get_pool_info(session, -max_live_blocks, coerce_float=False)
print(f"get_pool_info in {time.time() - start_time}")
start_time = time.time()
combined_data = get_combined_data(txn_data, pool_info_data)
print(f"get_combined_data in {time.time() - start_time}")
start_time = time.time()
wallet_deltas = postgres.get_wallet_deltas(session, coerce_float=False)
print(f"get_wallet_deltas in {time.time() - start_time}")

# %%
# PNL IMPORTS
from decimal import Decimal

import numpy as np
import pandas as pd
from eth_typing import ChecksumAddress, HexAddress, HexStr
from fixedpointmath import FixedPoint
from web3 import Web3

from elfpy import eth
from elfpy.eth.transactions import smart_contract_preview_transaction, smart_contract_read
from eth_bots import hyperdrive_interface
from eth_bots.hyperdrive_interface.hyperdrive_assets import AssetIdPrefix, encode_asset_id


# %%
def add_unrealized_pnl_closeout_test(current_wallet: pd.DataFrame, pool_info: pd.DataFrame, use_unique: bool):
    """Calculate closeout value of agent positions."""
    web3: Web3 = eth.initialize_web3_with_http_provider("http://localhost:8546", request_kwargs={"timeout": 60})

    # send a request to the local server to fetch the deployed contract addresses and
    # all Hyperdrive contract addresses from the server response
    addresses = hyperdrive_interface.fetch_hyperdrive_address_from_url("http://localhost:8080")
    abis = eth.abi.load_all_abis("../../packages/hyperdrive/src/")
    contract = hyperdrive_interface.get_hyperdrive_contract(web3, abis, addresses)

    # Define a function to handle the calculation for each group
    def calculate_unrealized_pnl(position: pd.DataFrame, min_output: int, as_underlying: bool, use_unique: bool):
        # Extract the relevant values (you can adjust this part to match your logic)
        if position["baseTokenType"] == "BASE" or position["delta"] == 0:
            position["unrealized_pnl"] = position["pnl"]
            return position
        assert len(position.shape) == 1, "Only one position at a time for add_unrealized_pnl_closeout"
        amount = FixedPoint(str(position["delta"])).scaled_value
        if use_unique:
            address = str(position["walletAddress"])
        else:
            address = position.name[0]
        tokentype = position["baseTokenType"]
        sender = ChecksumAddress(HexAddress(HexStr(address)))
        preview_result = None
        maturity = 0
        if tokentype in ["LONG", "SHORT"]:
            maturity = position["maturityTime"]
            assert isinstance(maturity, Decimal)
            maturity = int(maturity)
            assert isinstance(maturity, int)
        assert isinstance(tokentype, str)
        # token_id = encode_asset_id(AssetIdPrefix[tokentype], maturity)
        # balance = smart_contract_read(contract, "balanceOf", *(token_id, address))["value"]
        # if balance != amount:
        #     print(f"{balance=}")
        #     print(f"{amount =}")
        #     print(f"{(balance - amount)=}")
        #     if tokentype != "LP" and balance != amount:
        #         print(f"setting amount to {balance=}")
        #         amount = balance
        if tokentype == "LONG":
            fn_args = (maturity, amount, min_output, address, as_underlying)
            preview_result = smart_contract_preview_transaction(contract, sender, "closeLong", *fn_args)
            position["unrealized_pnl"] = Decimal(preview_result["value"])/Decimal(1e18)
        elif tokentype == "SHORT":
            fn_args = (maturity, amount, min_output, address, as_underlying)
            preview_result = smart_contract_preview_transaction(contract, sender, "closeShort", *fn_args)
            position["unrealized_pnl"] = preview_result["value"]/Decimal(1e18)
        elif tokentype == "LP":
            fn_args = (amount, min_output, address, as_underlying)
            preview_result = smart_contract_preview_transaction(contract, sender, "removeLiquidity", *fn_args)
            # print(f"i tried to remove {amount} liquidity\n => and all I got was: {preview_result}")
            # actual_withdrawn_base = preview_result["baseProceeds"]
            # implied_withdrawn_lp = (
            #     actual_withdrawn_base / pool_info["sharePrice"].values[-1] / pool_info["lpSharePrice"].values[-1]
            # )
            # implied_lp_share_price = actual_withdrawn_base / pool_info["sharePrice"].values[-1] / implied_withdrawn_lp
            # total_output = implied_withdrawn_lp + preview_result["withdrawalShares"]
            # print(f" => {total_output=} vs. {amount=}\n => diff={total_output - amount=}")
            # observed_lp_share_price = pool_info["lpSharePrice"].values[-1]
            # print(f"{implied_lp_share_price=} vs. {observed_lp_share_price=}")
            # print(f" => diff={implied_lp_share_price - observed_lp_share_price=}")
            position["unrealized_pnl"] = Decimal(
                preview_result["baseProceeds"]
                + preview_result["withdrawalShares"]
                * pool_info["sharePrice"].values[-1]
                * pool_info["lpSharePrice"].values[-1]
            )/Decimal(1e18)
        elif tokentype == "WITHDRAWAL_SHARE":
            fn_args = (amount, min_output, address, as_underlying)
            preview_result = smart_contract_preview_transaction(contract, sender, "redeemWithdrawalShares", *fn_args)
            position["unrealized_pnl"] = preview_result["proceeds"]/Decimal(1e18)
        return position

    if use_unique is True:
        # get unique positions by (baseTokenType, maturityTime, delta)
        unique_positions = current_wallet.reset_index().drop_duplicates(subset=["baseTokenType", "maturityTime", "delta"])
        unique_positions = unique_positions[["walletAddress", "baseTokenType", "maturityTime", "delta"]]
        unique_positions = unique_positions.loc[unique_positions["baseTokenType"] != "BASE", :]
        unique_positions = unique_positions.loc[unique_positions["delta"] != 0, :]
        unique_positions["unrealized_pnl"] = np.nan
        print(f"{len(unique_positions)} unique positions")

        # calculate unrealized pnl for each unique position
        unique_positions = unique_positions.apply(calculate_unrealized_pnl, min_output=0, as_underlying=True, use_unique=use_unique, axis=1)  # type: ignore
        current_wallet = current_wallet.merge(unique_positions, how="left", on=["baseTokenType", "maturityTime", "delta"])
    else:
        current_wallet["unrealized_pnl"] = np.nan
        current_wallet = current_wallet.apply(calculate_unrealized_pnl, min_output=0, as_underlying=True, use_unique=use_unique, axis=1)  # type: ignore
    return current_wallet

# %%
start_time = time.time()
current_returns, current_wallet = calc_total_returns(config_data, pool_info_data, wallet_deltas)
print(f"finished current_returns in {time.time() - start_time}")

# %%
current_wallet_unique = current_wallet.copy()
current_wallet_notunique = current_wallet.copy()

# %%
# %%timeit
# start_time = time.time()
current_wallet_unique = current_wallet.copy()
out_unique = add_unrealized_pnl_closeout_test(current_wallet, pool_info_data, use_unique=True)  # add unrealized_pnl column using closeout pnl valuation method
pnls_unique = out_unique["unrealized_pnl"].values
print(f"{len(pnls_unique)} positions {len(set(list(current_wallet.index.get_level_values('walletAddress').values)))} bots")
print(",".join([f"{v}" for v in pnls_unique]))
print(f"finished add_unrealized_pnl_closeout_test in {time.time() - start_time}")

# %%
# %%timeit
start_time = time.time()
current_wallet_notunique = current_wallet.copy()
out_notunique = add_unrealized_pnl_closeout_test(current_wallet_notunique, pool_info_data, use_unique=False)  # add unrealized_pnl column using closeout pnl valuation method
pnls_notunique = out_notunique["unrealized_pnl"].values
print(f"{len(pnls_notunique)} positions {len(set(list(current_wallet.index.get_level_values('walletAddress').values)))} bots")
print(",".join([f"{v}" for v in pnls_notunique]))
print(f"finished add_unrealized_pnl_closeout_test in {time.time() - start_time}")

# %%
for r in range(len(out_unique)):
    if out_unique["unrealized_pnl"][r] != out_notunique["unrealized_pnl"][r]:
        # if not np.isnan(out_unique["unrealized_pnl"][r]) or not np.isnan(out_notunique["unrealized_pnl"][r]):
        print(f"{r=:3.0f} {out_unique['unrealized_pnl'][r]=} vs. {out_notunique['unrealized_pnl'][r]=}")
# %%
print(out_notunique)

# %%
for r in range(len(out_unique)):
    print(f"spot: {out_notunique['pnl'][r]:5.5f} vs. {out_notunique['unrealized_pnl'][r]:5.5f}")

# %%
out_notunique["slippage"] = out_notunique["unrealized_pnl"]-out_notunique["pnl"]
out_notunique["slippage_pct"] = np.nan

notnan = out_notunique["pnl"].notna()
notzero = out_notunique["pnl"] != 0
mask = notnan & notzero
out_notunique.loc[mask, "slippage_pct"] = (out_notunique.loc[mask, "unrealized_pnl"]-out_notunique.loc[mask, "pnl"]) / out_notunique.loc[mask, "pnl"]

# %%
out_notunique.sort_values("slippage_pct", ascending=True)
# %%
