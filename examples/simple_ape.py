# %% Imports
from __future__ import annotations  # types will be strings by default in 3.11

# stdlib
from pathlib import Path
from typing import Optional, List
from time import time

# external lib
import ape
from ape import Contract
from ape.api import BlockAPI, ProviderAPI
from ape.contracts import ContractInstance
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from elfpy.utils.format_number import format_number as fmt
import darkmode_orange

load_dotenv()

ours = [
    "0x2C47e2A9948d10aD873109eB85c6F2e45186277b",
    "0x841958527DFe4499fA234A1Acc247b29C90d1C21",
    "0xFed2c446A218d26477b78f652B252a400F929436",
    "0x4612E8A93E2e089d074c25F41F3b11D853718f68",
]


# %% Helper functions
def get_gas_fees(block: BlockAPI | float | int) -> tuple[List[float], List[float]]:
    """Get the max and priority fees from a block"""
    if isinstance(block, (float, int)):
        block = ape.chain.blocks[int(block)]
    if type2 := [txn for txn in block.transactions if txn.type == 2]:
        type2 = [txn for txn in type2 if txn.max_fee is not None and txn.max_priority_fee is not None]
        _max_fees, _priority_fees = zip(*((txn.max_fee, txn.max_priority_fee) for txn in type2))
        _max_fees = [f / 1e9 for f in max_fees if f is not None]
        _priority_fees = [f / 1e9 for f in _priority_fees if f is not None]
        return _max_fees, _priority_fees
    return [], []


def get_gas_fee_stats(
    block: BlockAPI | float | int,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Get the max and avg max and priority fees from a block"""
    if isinstance(block, (float, int)):
        block = ape.chain.blocks[int(block)]
    if [txn for txn in block.transactions if txn.type == 2]:
        _max_fees, _priority_fees = get_gas_fees(block)
        _max_max_fee, _avg_max_fee = max(_max_fees), sum(max_fees) / len(max_fees)
        _max_priority_fee, _avg_priority_fee = max(_priority_fees), sum(_priority_fees) / len(_priority_fees)
        return _max_max_fee, _avg_max_fee, _max_priority_fee, _avg_priority_fee
    return None, None, None, None


# %% Set up ape
provider: ProviderAPI = ape.networks.parse_network_choice("ethereum:goerli:alchemy").__enter__()
provider.network.config.goerli.required_confirmations = 1
project_root = Path.cwd()
# if current folder is examples, move up one
if project_root.name == "examples":
    project_root = project_root.parent
project = ape.Project(path=project_root)

# %% Hyperdrive specific stuff
Dai: ContractInstance = Contract("0x11fe4b6ae13d2a6055c8d9cf65c55bac32b5d844")  # sDai
hyperdrive: ContractInstance = project.Hyperdrive.at("0xB311B825171AF5A60d69aAD590B857B1E5ed23a2")  # type: ignore
hyper_config = hyperdrive.getPoolConfig().__dict__
hyper_config["timeStretch"] = 1 / (hyper_config["timeStretch"] / 1e18)
print(f"Hyperdrive config deployed at {hyperdrive.address}:")
for k, v in hyper_config.items():
    divisor = 1e18 if k not in ["positionDuration", "checkpointDuration"] else 1
    print(f" {k}: {fmt(v/divisor)}")
hyper_config["term_length"] = 365  # days

# %% Print all ABIs in our project
start_time = time()
Path("abi").mkdir(parents=True, exist_ok=True)
for name, contract in project.contracts.items():
    # print(f"{name}: {contract=}")
    with open(f"abi/{name}.json", "w", encoding="utf-8") as f:
        for abi in contract.abi:
            f.write(f"{abi.json()}\n")
print(f"Wrote all ABIs in {time() - start_time:.2f}s")

# %% Do stuff
latest_block = ape.chain.blocks[-1]
block_number = latest_block.number or 0
block_time = latest_block.timestamp
_max_max_fee, _avg_max_fee, _max_priority_fee, _avg_priority_fee = get_gas_fee_stats(latest_block)

# %% Get gas for a specific block
BLOCK_NUMBER = 8860229
max_fees, priority_fees = get_gas_fees(BLOCK_NUMBER)

# %% Plot gas
fig, axs = plt.subplots(figsize=(12, 8), nrows=2, squeeze=True)

ax = axs[0]
ax.set_title(f"Goerli block {BLOCK_NUMBER} gas fees", fontsize=16)
ax.set_ylabel("Frequency")
ax.hist(
    x=max_fees,
    bins=100,
    label=["Total gas (gwei)"],
)

ax = axs[1]
ax.set_ylabel("Frequency")
ax.hist(
    x=priority_fees,
    bins=100,
    label=["Priority fee (gwei)"],
)

our_txns = [txn for txn in ape.chain.blocks[BLOCK_NUMBER].transactions if txn.sender in ours]
for txn in our_txns:  # plot em!
    print(f"found our txn from {txn.sender} with hash")
    for k, v in txn.__dict__.items():
        # don't print if k is data or sender
        if k in ["data", "sender", "signature"]:
            continue
        if k == "gas_price":
            print(f" => {k}: {fmt(v/1e9)}")
        elif k == "gas_limit":
            print(f" => {k}: {fmt(v)}")
        else:
            print(f" => {k}: {v}")
    if txn.max_fee:
        axs[0].axvline(txn.max_fee / 1e9, color="yellow", linestyle="solid", label="ours")
    if txn.max_priority_fee:
        axs[1].axvline(txn.max_priority_fee / 1e9, color="yellow", linestyle="solid", label="ours")
axs[0].legend(prop={"size": 14}, loc="upper center")
axs[1].legend(prop={"size": 14}, loc="upper center")
