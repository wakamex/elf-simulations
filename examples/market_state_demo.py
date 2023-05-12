# %% imports
import json
from pathlib import Path
from time import time
import os

import ape
import darkmode_orange  # type: ignore # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plt
import requests
from ape import Contract
from ape.api import ProviderAPI, ReceiptAPI
from ape.contracts import ContractInstance

import elfpy.pricing_models.hyperdrive as hyperdrive_pm
import elfpy.utils.apeworx_integrations as ape_utils
from elfpy import time as elfpy_time
from elfpy import types
from elfpy.agents.get_wallet_state import get_wallet_state
from elfpy.agents.wallet import Long, Short, Wallet
from elfpy.markets.hyperdrive import hyperdrive_assets, hyperdrive_market

USE_ALCHEMY = False
SECONDS_IN_YEAR = 365 * 24 * 60 * 60  # 31_536_000

examples_dir = Path.cwd() if Path.cwd().name == "examples" else Path.cwd() / "examples"

if USE_ALCHEMY:
    from dotenv import load_dotenv

    # load env from up one level
    load_dotenv(dotenv_path=examples_dir.parent / ".env")


# def is_interactive_function():  # pylint: disable=missing-function-docstring
#     import __main__ as main  # pylint: disable=import-outside-toplevel

#     return not hasattr(main, "__file__")


# IS_INTERACTIVE = is_interactive_function()  # calculate once


# def display(*args, **kwargs):  # pylint: disable=missing-function-docstring
#     if IS_INTERACTIVE:
#         display(*args, **kwargs)
#     else:
#         print(*args, **kwargs)


# %% read it in
url = "https://gist.githubusercontent.com/wakamex/30a8e92327526a5b9cb66c091377af59/raw/9402e87e13bb639b20b888fb77c143e708116409/hyperDcdTL.json"
# download into variable ephemerally
j = json.loads(requests.get(url).text)
# j = json.load(fp=open(f"{examples_dir}/hyperDcdTL.json", "r", encoding="utf-8"))
j = [i for i in j if "decoded_input" in i]
j2 = [i for i in j if "decoded_event_logs" in i and len(i["decoded_event_logs"]) > 0]

# pool_info = getPoolInfo
# market_state = MarketState(pool_info)
# agent.get_wallet_sate(market_state)

# %% STRUCTURE FOR ELFPY
# ALL-IN-ONE VERSION
# def spoof_run_simulation(trade_tape):
#     # initialize market
#     # initialize agents
#     for trade in trade_tape:
#         update_market & agent wallets from receipt
#         compute pnl (get_wallet_sate)
#         write to pnl.json
#         wait(random_value(10,30))
# show it in streamlit ==> streamlit.py

# FUTURE VERSION (separate process: one runs the bots, the other processes data and plots things)
# streamlit.py:
# listen to pnl.json
# if new_row:
#     update_plot()

# %% JSON from records, craziest comprehension
# 374 µs ± 3.06 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
start_time = time()
trades = [
    {"block_number": int(i["blockNumber"], 16)}
    | {k: v / 1e18 if isinstance(v, int) and k != "id" else v for k, v in i["decoded_event_logs"][0].items()}
    | {"function_name": i["decoded_input"]["function_name"]}
    | {"hash": i["hash"]}
    | {k: v / 1e18 if isinstance(v, int) else v for k, v in i["decoded_input"]["args"].items() if k != "_asUnderlying"}
    for i in j2
]
initialize_trade = [i for i in trades if i["function_name"] == "initialize"][0]
trades = [i for i in trades if i["function_name"] != "initialize"]
display(trades[-1])
print(f"parsed in {(time() - start_time)*1e3:0.1f}ms")
print(f"count={sum(len(row) for row in trades)}")
print(f"length={len(trades)}")

addresses = []
for trade in trades:
    if trade["operator"] not in addresses:
        addresses.append(trade["operator"])
print(f"{len(addresses)=}")
for address in addresses:
    print(f"address {address} has index {addresses.index(address)}", end="")
    print(f" and {sum(trade['operator'] == address for trade in trades)} trades")

# %% Set up ape
PROVIDER_STRING = "alchemy" if USE_ALCHEMY else "http://localhost:8547"
provider: ProviderAPI = ape.networks.parse_network_choice(f"ethereum:goerli:{PROVIDER_STRING}").push_provider()
project = ape_utils.HyperdriveProject(Path.cwd())
hyperdrive: ContractInstance = project.get_hyperdrive_contract()
dai: ContractInstance = Contract("0x11fe4b6ae13d2a6055c8d9cf65c55bac32b5d844")  # sDai
for k, v in provider.__dict__.items():
    if not k.startswith("_"):
        print(k, v)
print(f"Block number: {ape.chain.blocks[-1].number or 0}, Block time: {ape.chain.blocks[-1].timestamp}")

# %% Inspect trades
for trade in trades:
    info = {k: v for k, v in trade.items() if k != "function_name"}
    print(f"{trade['function_name']}: {info}")

# %% Set up hyperdrive
# querying by block_number is faster than querying by block (2.3x: 35ms vs. 82ms)
start_time = ape.chain.blocks[int(initialize_trade["block_number"])].timestamp
hyper_config = hyperdrive.getPoolConfig(block_identifier=initialize_trade["block_number"]).__dict__
hyper_config["timeStretch"] = 1 / (hyper_config["timeStretch"] / 1e18)
hyper_config["term_length"] = 365  # days
position_duration = elfpy_time.StretchedTime(
    hyper_config["term_length"], hyper_config["timeStretch"], hyper_config["term_length"]
)
params = {"pricing_model": hyperdrive_pm.HyperdrivePricingModel()} | {"position_duration": position_duration}

# %% Set up wallets
agent_wallets = {
    address: Wallet(
        address=addresses.index(address),
        # balance=dai.balanceOf(address, block_number=trades[0]["block_number"] - 1) / 1e18,
        balance=types.Quantity(
            amount=dai.balanceOf(address, block_identifier=trades[0]["block_number"] - 1) / 1e18,
            unit=types.TokenType.BASE,
        ),
    )
    for address in addresses
}
for address, wallet in agent_wallets.items():
    print(f"{address}: has {wallet=}")

pnl_history = []
for trade_num, trade in enumerate(trades):
    # create market
    block_number = int(trade["block_number"])
    block_time = (ape.chain.blocks[block_number].timestamp - start_time) / SECONDS_IN_YEAR
    params |= {"market_state": ape_utils.get_market_state_from_contract(hyperdrive, block_number=block_number)}
    params |= {"block_time": elfpy_time.BlockTime(time=block_time)}
    # for k, v in params.items():
    #     print(f"{k}: {v}")
    market: hyperdrive_market.Market = hyperdrive_market.Market(**params)  # type: ignore

    # get trade info
    trade_type = trade["function_name"]
    agent = trade["operator"]
    txn_receipt: ReceiptAPI = provider.get_receipt(txn_hash=str(trade["hash"]))
    txn_events = [e.dict() for e in txn_receipt.events if agent in [e.get("from"), e.get("to")]]
    dai_events = [e.dict() for e in txn_receipt.events if agent in [e.get("src"), e.get("dst")]]
    dai_in = sum(int(e["event_arguments"]["wad"]) for e in dai_events if e["event_arguments"]["src"] == agent) / 1e18
    dai_out = sum(int(e["event_arguments"]["wad"]) for e in dai_events if e["event_arguments"]["dst"] == agent) / 1e18
    prefix, maturity_timestamp = hyperdrive_assets.decode_asset_id(int(trade["id"]))
    mint_time = ((maturity_timestamp - SECONDS_IN_YEAR * hyper_config["term_length"]) - start_time) / SECONDS_IN_YEAR
    token_type = hyperdrive_assets.AssetIdPrefix(prefix)  # look up prefix in AssetIdPrefix
    address_index = addresses.index(agent)
    if trade_type == "addLiquidity":  # sourcery skip: switch
        # agent_deltas = wallet.Wallet(
        #     address=wallet_address,
        #     balance=-types.Quantity(amount=d_base_reserves, unit=types.TokenType.BASE),
        #     lp_tokens=lp_out,
        # )
        agent_deltas = Wallet(
            address=address_index,
            balance=-types.Quantity(amount=trade["_contribution"], unit=types.TokenType.BASE),
            lp_tokens=trade["value"],  # trade output
        )
    elif trade_type == "removeLiquidity":
        # agent_deltas = wallet.Wallet(
        #     address=wallet_address,
        #     balance=types.Quantity(amount=delta_base, unit=types.TokenType.BASE),
        #     lp_tokens=-lp_shares,
        #     withdraw_shares=withdraw_shares,
        # )
        agent_deltas = Wallet(
            address=address_index,
            balance=types.Quantity(amount=trade["value"], unit=types.TokenType.BASE),  # trade output
            lp_tokens=-trade["_shares"],  # negative, decreasing
            withdraw_shares=trade["_shares"],  # positive, increasing
        )
    elif trade_type == "openLong":
        # agent_deltas = wallet.Wallet(
        #     address=wallet_address,
        #     balance=types.Quantity(amount=trade_result.user_result.d_base, unit=types.TokenType.BASE),
        #     longs={market.latest_checkpoint_time: wallet.Long(trade_result.user_result.d_bonds)},
        #     fees_paid=trade_result.breakdown.fee,
        # )
        agent_deltas = Wallet(
            address=address_index,
            balance=types.Quantity(amount=-trade["_baseAmount"], unit=types.TokenType.BASE),  # negative, decreasing
            longs={block_time: Long(trade["value"])},  # trade output, increasing
        )
    elif trade_type == "closeLong":
        # agent_deltas = wallet.Wallet(
        #     address=wallet_address,
        #     balance=types.Quantity(amount=base_proceeds, unit=types.TokenType.BASE),
        #     longs={mint_time: wallet.Long(-bond_amount)},
        #     fees_paid=fee,
        # )
        agent_deltas = Wallet(
            address=address_index,
            balance=types.Quantity(amount=trade["value"], unit=types.TokenType.BASE),  # trade output
            longs={mint_time: Long(-trade["_bondAmount"])},  # negative, decreasing
        )
    elif trade_type == "openShort":
        # agent_deltas = wallet.Wallet(
        #     address=wallet_address,
        #     balance=-types.Quantity(amount=trader_deposit, unit=types.TokenType.BASE),
        #     shorts={
        #         market.latest_checkpoint_time: wallet.Short(
        #             balance=bond_amount, open_share_price=market.market_state.share_price
        #         )
        #     },
        #     fees_paid=trade_result.breakdown.fee,
        # )
        agent_deltas = Wallet(
            address=address_index,
            balance=types.Quantity(amount=-dai_in, unit=types.TokenType.BASE),  # negative, decreasing
            shorts={
                block_time: Short(
                    balance=trade["value"],  # trade output
                    open_share_price=params["market_state"].share_price,
                )
            },
        )
    else:
        assert trade_type == "closeShort", f"Unknown trade type: {trade_type}"
        # agent_deltas = wallet.Wallet(
        #     address=wallet_address,
        #     balance=types.Quantity(
        #         amount=(market.market_state.share_price / open_share_price) * bond_amount + trade_result.user_result.d_base,
        #         unit=types.TokenType.BASE,
        #     ),  # see CLOSING SHORT LOGIC above
        #     shorts={
        #         mint_time: wallet.Short(
        #             balance=-bond_amount,
        #             open_share_price=0,
        #         )
        #     },
        #     fees_paid=trade_result.breakdown.fee,
        # )
        agent_deltas = Wallet(
            address=address_index,
            balance=types.Quantity(amount=trade["value"], unit=types.TokenType.BASE),
            shorts={
                mint_time: Short(
                    balance=-trade["_bondAmount"],  # negative, decreasing
                    open_share_price=0,
                )
            },
        )
    agent_wallet = agent_wallets[agent]
    print(f"starting wallet = {agent_wallet}")
    print(f"deltas wallet = {agent_deltas}")
    agent_wallet.update(agent_deltas)
    print(f"resulting wallet = {agent_wallet}")
    print(f"{trade_num=} {trade_type=} {agent=}")
    pnl_after_this_trade = []
    for address in addresses:
        address_index = addresses.index(address)
        wallet_values_in_base = {
            f"agent_{address_index}_base",
            f"agent_{address_index}_lp_tokens",
            f"agent_{address_index}_total_longs",
            f"agent_{address_index}_total_shorts",
        }
        wallet_values_in_base_no_mock = {
            f"agent_{address_index}_base",
            f"agent_{address_index}_lp_tokens",
            f"agent_{address_index}_total_longs_no_mock",
            f"agent_{address_index}_total_shorts_no_mock",
        }
        agent_wallet = agent_wallets[address]
        wallet_state: dict[str, float] = get_wallet_state(agent_wallet, market)
        pnl_after_this_trade.insert(
            address_index, sum(v for k, v in wallet_state.items() if k in wallet_values_in_base_no_mock)
        )
    pnl_history.append(pnl_after_this_trade)

# %% plot pnl
figure = plt.figure(figsize=(12, 8))
plt.step(range(len(pnl_history)), pnl_history, where="post")
plt.gca().set_xlim(0, len(trades))
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# %% make a gif
start_time = time()
figure = plt.figure(figsize=(12, 8))

lines = plt.step(range(len(pnl_history)), np.full((len(pnl_history), len(pnl_history[0])), np.nan), where="post")


def init():  # pylint: disable=missing-function-docstring
    return lines


def update(i):  # pylint: disable=missing-function-docstring
    for idx, line in enumerate(lines):
        data = line.get_ydata()
        data[i] = pnl_history[i][idx]
        line.set_ydata(data)
    # update x and y limits based on previous plot
    plt.gca().set_ylim(ylim)
    plt.gca().set_xlim(xlim)
    return lines


# manually update each frame
for i in range(len(pnl_history)):
    update(i)
    fname = examples_dir / "pics" / f"pnl_{i:03d}.svg"
    plt.savefig(fname)
    print(f"saved {fname}")

os.system(f"cd {examples_dir / 'pics'} && ffmpeg -y -i pnl_%03d.svg -c:v h264_nvenc output.mp4")

print(f"gif took {time() - start_time} seconds to make")
