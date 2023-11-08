"""Script to query bot experiment data."""

# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import, bare-except, wrong-import-position, redefined-outer-name, pointless-statement, missing-final-newline, missing-function-docstring, line-too-long
# %%
# bot script setup
from __future__ import annotations

import asyncio
import logging
import subprocess
import os
import time
import warnings
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
import docker
from agent0 import initialize_accounts
from agent0.accounts_config import AccountKeyConfig
from agent0.base.config import AgentConfig, EnvironmentConfig
from agent0.hyperdrive.exec import async_fund_agents, create_and_fund_user_account, run_agents, setup_experiment
from agent0.hyperdrive.policies.hyperdrive_policy import HyperdrivePolicy
from agent0.hyperdrive.state.hyperdrive_actions import HyperdriveActionType, HyperdriveMarketAction
from agent0.hyperdrive.state.hyperdrive_wallet import HyperdriveWallet
from chainsync.analysis import calc_spot_price
from chainsync.analysis.data_to_analysis import get_transactions
from chainsync.db.base import initialize_session
from chainsync.db.hyperdrive import get_pool_info
from chainsync.db.hyperdrive.interface import get_pool_config
from eth_typing import URI
from ethpy.eth_config import EthConfig, build_eth_config
from ethpy.hyperdrive import AssetIdPrefix
from ethpy.hyperdrive.addresses import fetch_hyperdrive_address_from_uri
from ethpy.hyperdrive.api import HyperdriveInterface
from fixedpointmath import FixedPoint
from numpy.random._generator import Generator
from docker.errors import DockerException

from elfpy.types import MarketType, Trade

import nest_asyncio

nest_asyncio.apply()

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="web3.contract.base_contract")


def check_docker(infra_folder: Path, restart: bool = False) -> None:
    """Check whether docker is running to your liking.

    Arguments
    ---------
    infra_folder : Path
        Path to infra repo folder.
    restart : bool
        Restart docker even if it is running.
    """
    try:
        home_dir = os.path.expanduser("~")
        socket_path = Path(f"{home_dir}/.docker/desktop/docker.sock")
        if socket_path.exists():
            logging.debug("The socket exists at %s.. using it to connect to docker", socket_path)
            _ = docker.DockerClient(base_url=f"unix://{socket_path}")
        else:
            logging.debug("No socket found at %s.. using default socket", socket_path)
            _ = docker.from_env()
    except DockerException as exc:
        raise DockerException("Failed to connect to docker.") from exc
    dockerps = _get_docker_ps_and_log()
    number_of_running_services = dockerps.count("\n") - 1
    if number_of_running_services > 0:
        preamble_str = f"Found {number_of_running_services} running services"
        if restart:
            _start_docker(f"{preamble_str}, restarting docker...", infra_folder)
        else:
            logging.info("%s, using them.", preamble_str)
    else:
        _start_docker("Starting docker.", infra_folder)
    dockerps = os.popen("docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'").read()
    logging.info(dockerps)


def _start_docker(startup_str: str, infra_folder: Path):
    logging.info(startup_str)
    _run_cmd(infra_folder, " && docker-compose down -v", "Shut down docker in ")
    cmd = "docker images | awk 'NR>1 && $2 !~ /none/ && $1 ~ /^ghcr\\.io\\// {print $1 \":\" $2}'"
    output = subprocess.getoutput(cmd)
    if output is not None:
        docker_pull_cmd = f"echo '{output}' | xargs -L1 docker pull"
        _run_cmd(infra_folder, f" && {docker_pull_cmd}", "Updated docker in ")
    else:
        logging.info("No matching images found.")
    _run_cmd(infra_folder, " && docker-compose up -d", "Started docker in ")


def _run_cmd(infra_folder: Path, cmd: str, timing_str: str):
    result = time.time()
    os.system(f"cd {infra_folder}{cmd}")
    formatted_str = f"{timing_str}{time.time() - result:.2f}s"
    logging.info(formatted_str)
    return result


def _get_docker_ps_and_log() -> str:
    dockerps = os.popen("docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'").read()
    logging.info(dockerps)
    return dockerps


class DBot(HyperdrivePolicy):
    @dataclass
    class Config(HyperdrivePolicy.Config):
        trade_list: list[tuple[str, int]]

    def __init__(
        self,
        budget: FixedPoint,
        rng: Generator | None = None,
        slippage_tolerance: FixedPoint | None = None,
        policy_config: Config | None = None,
    ):
        self.trade_list = (
            policy_config.trade_list
            if policy_config
            else [("add_liquidity", 100), ("open_long", 100), ("open_short", 100)]
        )
        self.starting_length = len(self.trade_list)
        super().__init__(budget, rng, slippage_tolerance)

    def action(
        self, interface: HyperdriveInterface, wallet: HyperdriveWallet
    ) -> tuple[list[Trade[HyperdriveMarketAction]], bool]:
        print(f"ACTION LOG {len(self.trade_list)}/{self.starting_length}")
        if not self.trade_list:
            return [], True  # done trading
        action_type, amount = self.trade_list.pop(0)
        mint_time = next(iter({"close_long": wallet.longs, "close_short": wallet.shorts}.get(action_type, [])), None)
        action = HyperdriveMarketAction(HyperdriveActionType(action_type), wallet, FixedPoint(amount), None, mint_time)
        return [Trade(market_type=MarketType.HYPERDRIVE, market_action=action)], False


def run_trades(
    env_config: EnvironmentConfig | None = None,
    agent_config: AgentConfig | None = None,
    account_key_config: AccountKeyConfig | None = None,
    eth_config: EthConfig | None = None,
    trade_list=None,
):
    """Allow running in interactive session."""
    agent_config_list = None
    if env_config is None:
        env_config = globals().get("env_config")
        assert env_config is not None, "env_config must be set"
    if agent_config is None:
        agent_config = globals().get("agent_config")
        print(f"{agent_config=}")
        assert agent_config is not None, "agent_config must be set"
    if not isinstance(agent_config, list):
        agent_config_list = [agent_config]
    else:
        agent_config_list = agent_config
    assert agent_config_list is not None, "agent_config_list failed to be set"
    if account_key_config is None:
        account_key_config = globals().get("account_key_config")
        assert account_key_config is not None, "account_key_config must be set"
    if eth_config is None:
        eth_config = globals().get("eth_config")
        assert eth_config is not None, "eth_config must be set"
    if hasattr(agent_config_list[0].policy_config, "trade_list"):
        agent_config_list[0].policy_config.trade_list = trade_list
    assert isinstance(agent_config_list, list)
    run_agents(
        environment_config=env_config,
        agent_config=agent_config_list,
        account_key_config=account_key_config,
        eth_config=eth_config,
    )


def calc_fixed_rate_df(trade_data, config_data):
    """Calculate fixed rate from trade and config data."""
    trade_data["rate"] = np.nan
    annualized_time = config_data.position_duration / Decimal(60 * 60 * 24 * 365)
    spot_price = calc_spot_price(
        trade_data["share_reserves"],
        trade_data["share_adjustment"],
        trade_data["bond_reserves"],
        config_data["initial_share_price"],
        config_data["time_stretch"],
    )
    fixed_rate = (Decimal(1) - spot_price) / (spot_price * annualized_time)
    x_data = trade_data["timestamp"]
    y_data = fixed_rate
    return x_data, y_data


def get_combined_data(txn_data, pool_info_data):
    """Combine multiple datasets into one containing transaction data, and pool info."""
    pool_info_data.index = pool_info_data.index.astype(int)
    # txn_data.index = txn_data["blockNumber"]
    # Combine pool info data and trans data by block number
    data = txn_data.merge(pool_info_data, on="block_number")

    rename_dict = {
        "event_operator": "operator",
        "event_from": "from",
        "event_to": "to",
        "event_id": "id",
        "event_prefix": "prefix",
        "event_maturity_time": "maturity_time",
        "event_value": "value",
        "input_method": "trade_type",
        "timestamp": "block_timestamp",
    }
    other_cols = {
        "bond_reserves",
        "long_exposure",
        "longs_outstanding",
        "long_average_maturity_time",
        "lp_total_supply",
        "share_price",
        "share_reserves",
        "share_adjustment",
        "short_average_maturity_time",
        "shorts_outstanding",
        "transaction_hash",
        "transaction_index",
    }

    # Filter data based on columns
    trade_data = data[list(rename_dict) + list(other_cols)]
    # Rename columns
    trade_data = trade_data.rename(columns=rename_dict)

    # Calculate trade type and timetsamp from args.id
    def decode_prefix(row):
        # Check for nans
        if row is None:
            return None
        if np.isnan(row):
            out = np.nan
        else:
            out = AssetIdPrefix(row).name
        return out

    trade_data["trade_enum"] = trade_data["prefix"].apply(decode_prefix)
    trade_data["timestamp"] = trade_data["block_timestamp"]
    trade_data["block_timestamp"] = trade_data["block_timestamp"].astype(int)

    trade_data = trade_data.sort_values("block_timestamp")

    return trade_data


check_docker(infra_folder=Path("/code/infra"),restart=False)
DEVELOP = True
ENV_FILE = "script.env"
env_config = EnvironmentConfig(
    delete_previous_logs=True,
    halt_on_errors=True,
    log_formatter="%(message)s",
    log_filename="agent0-bots",
    log_level=logging.DEBUG,
    log_stdout=True,
    random_seed=1234,
    username="Botty McBotFace",
)
agent_config: list[AgentConfig] = [
    AgentConfig(
        policy=DBot,
        number_of_agents=1,
        base_budget_wei=FixedPoint(1e9).scaled_value,  # 1 billion base
        eth_budget_wei=FixedPoint(1).scaled_value,  # 1 Eth
        policy_config=DBot.Config(trade_list=[("open_long", 100)] * 1),
    ),
]
session = initialize_session()  # initialize the postgres session
os.environ["DEVELOP"] = "true"
account_key_config = initialize_accounts(agent_config, ENV_FILE, random_seed=env_config.random_seed)
eth_config = build_eth_config()
eth_config.rpc_uri = URI("http://localhost:8546")
contract_addresses = fetch_hyperdrive_address_from_uri(os.path.join(eth_config.artifacts_uri, "addresses.json"))
user_account = create_and_fund_user_account(eth_config, account_key_config, contract_addresses)
asyncio.run(async_fund_agents(user_account, eth_config, account_key_config, contract_addresses))
hyperdrive, agent_accounts = setup_experiment(
    eth_config, env_config, agent_config, account_key_config, contract_addresses
)
config_data = get_pool_config(session, coerce_float=False)
while config_data.empty:
    print("waiting for config data")
    time.sleep(1)
    config_data = get_pool_config(session, coerce_float=False)
config_data = config_data.iloc[0]
print("\n ==== Pool Config ===")
for k, v in config_data.items():
    print(f"{k:20} | {v}")

# %%
# constants
MAX_ITER = 20
fp0 = FixedPoint(0)
fp1 = FixedPoint(1)
fp2 = FixedPoint(2)
fp12 = FixedPoint(12)
fp_seconds_in_year = FixedPoint(365 * 24 * 60 * 60)
target_apr = FixedPoint(0.01)  # ONE percent
# target_apr = FixedPoint(0.10)  # TEN percent

# calculate amount to long
fixed_rate = hyperdrive.calc_fixed_rate()
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
    bond_reserves,
    share_price,
    initial_share_price,
    share_reserves,
    bonds_out,
    time_stretch,
    curve_fee,
    gov_fee,
    one_block_return,
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
    price_discount = fp1 - spot_price
    # price_discount = (fp1/spot_price - fp1)
    curve_fee_rate = price_discount * curve_fee
    curve_fee_amount_in_shares = amount_in_shares * curve_fee_rate
    gov_fee_amount_in_shares = curve_fee_amount_in_shares * gov_fee
    # applying fees means you pay MORE shares in for the same amount of bonds OUT
    amount_from_user_in_shares = amount_in_shares + curve_fee_amount_in_shares
    return amount_from_user_in_shares, curve_fee_amount_in_shares, gov_fee_amount_in_shares


def get_shares_out_for_bonds_in(
    bond_reserves,
    share_price,
    initial_share_price,
    share_reserves,
    bonds_in,
    time_stretch,
    curve_fee,
    gov_fee,
    one_block_return,
):
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
    price_discount = fp1 - spot_price
    amount_in_shares = max(fp0, share_reserves - z_val)
    curve_fee_rate = price_discount * curve_fee
    curve_fee_amount_in_shares = amount_in_shares * curve_fee_rate
    gov_fee_amount_in_shares = curve_fee_amount_in_shares * gov_fee
    # applying fee means you get LESS shares out for the same amount of bonds IN
    amount_to_user_in_shares = amount_in_shares - curve_fee_amount_in_shares
    return amount_to_user_in_shares, curve_fee_amount_in_shares, gov_fee_amount_in_shares


def calc_spot_price_local(initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch):
    effective_share_reserves = share_reserves - share_adjustment
    return (initial_share_price * effective_share_reserves / bond_reserves) ** time_stretch


def calc_apr(
    share_reserves, share_adjustment, bond_reserves, initial_share_price, position_duration_seconds, time_stretch
):
    annualized_time = position_duration_seconds / fp_seconds_in_year
    spot_price = calc_spot_price_local(
        initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch
    )
    return (fp1 - spot_price) / (spot_price * annualized_time)


def bonds_given_shares_and_rate(
    share_reserves, share_adjustment, bond_reserves, initial_share_price, time_stretch, target_rate
):
    spot_price = calc_spot_price_local(
        initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch
    )
    return initial_share_price * share_reserves * spot_price ** ((spot_price * target_rate) / (spot_price - fp1))


# Calculate bonds needed to hit target APR
shares_needed = None
predicted_rate = fp0
tolerance = FixedPoint(scaled_value=1)
pool_config = (hyperdrive.current_pool_state.pool_config)
pool_info = (hyperdrive.current_pool_state.pool_info)

# convert to Decimal
USE_DECIMAL = True
PRECISION = 24
if USE_DECIMAL:
    print(f"using Decimal with precision {PRECISION}...")
else:
    print("using FixedPoint")
if USE_DECIMAL:
    pool_info.share_reserves = Decimal(str(pool_info.share_reserves))
    pool_config.initial_share_price = Decimal(str(pool_config.initial_share_price))
    pool_config.position_duration = Decimal(str(pool_config.position_duration))
    pool_config.time_stretch = Decimal(str(pool_config.time_stretch))
    pool_config.inv_time_stretch = Decimal(str(1/pool_config.time_stretch))
    pool_info.bond_reserves = Decimal(str(pool_info.bond_reserves))
    pool_info.share_price = Decimal(str(pool_info.share_price))
    pool_config.fees.curve = Decimal(str(pool_config.fees.curve))
    pool_config.fees.governance = Decimal(str(pool_config.fees.governance))
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
one_block_return = (fp1 + variable_rate) ** (fp12 / fp_seconds_in_year)

iteration = 0
start_time = time.time()
while abs(predicted_rate - target_apr) > tolerance:  # max tolerance 1e-16
    iteration += 1
    target_bonds = calc_bond_reserves(
        pool_info.share_reserves,
        pool_config.initial_share_price,
        target_apr,
        pool_config.position_duration,
        pool_config.inv_time_stretch,
    )
    # target_bonds = bonds_given_shares_and_rate(pool_info.share_reserves, 0, pool_info.bond_reserves, pool_config.initial_share_price, pool_config["inv_time_stretch"], target_apr)
    # print(f"{pool_config['time_stretch']=}")
    bonds_needed = (target_bonds - pool_info.bond_reserves) / fp2
    print(f"{bonds_needed=}")
    # assert bonds_needed < 0, "To lower the fixed rate, we should require a decrease in bonds"
    if bonds_needed > 0:  # short
        shares_out, curve_fee, gov_fee = get_shares_out_for_bonds_in(
            pool_info.bond_reserves,
            pool_info.share_price,
            pool_config.initial_share_price,
            pool_info.share_reserves,
            bonds_needed,
            pool_config.time_stretch,
            pool_config.fees.curve,
            pool_config.fees.governance,
            one_block_return,
        )
        # shares_out is what the user takes OUT: curve_fee less due to fees.
        # gov_fee of that doesn't stay in the pool, going OUT to governance (same direction as user flow).
        pool_info.share_reserves += (-shares_out - gov_fee) * 1
    else:  # long
        shares_in, curve_fee, gov_fee = get_shares_in_for_bonds_out(
            pool_info.bond_reserves,
            pool_info.share_price,
            pool_config.initial_share_price,
            pool_info.share_reserves,
            -bonds_needed,
            pool_config.time_stretch,
            pool_config.fees.curve,
            pool_config.fees.governance,
            one_block_return,
        )
        print(f"{shares_in=}")
        print(f"{curve_fee=}")
        print(f"{gov_fee=}")
        # shares_in is what the user pays IN: curve_fee more due to fees.
        # gov_fee of that doesn't go to the pool, going OUT to governance (opposite direction of user flow).
        pool_info.share_reserves += (shares_in - gov_fee) * 1
    pool_info.bond_reserves += bonds_needed
    if USE_DECIMAL:
        total_shares_needed = pool_info.share_reserves - Decimal(str(hyperdrive.current_pool_state.pool_info.share_reserves))
        total_bonds_needed = pool_info.bond_reserves - Decimal(str(hyperdrive.current_pool_state.pool_info.bond_reserves))
    else:
        total_shares_needed = pool_info.share_reserves - hyperdrive.current_pool_state.pool_info.share_reserves
        total_bonds_needed = pool_info.bond_reserves - hyperdrive.current_pool_state.pool_info.bond_reserves
    predicted_rate = calc_apr(
        pool_info.share_reserves,
        fp0,
        pool_info.bond_reserves,
        pool_config.initial_share_price,
        pool_config.position_duration,
        pool_config.time_stretch,
    )
    print(
        f"iteration {iteration:3}: {float(predicted_rate):22.18%} d_bonds={float(total_bonds_needed):27,.18f} d_shares={float(total_shares_needed):27,.18f}"
    )
    if iteration >= MAX_ITER:
        break
print(f"predicted precision: {float(abs(predicted_rate-target_apr))}, time taken: {time.time() - start_time}s")

# %%
pool_info = copy(hyperdrive.current_pool_state.pool_info)
bond_reserves_before = pool_info.bond_reserves
share_reserves_before = pool_info.share_reserves
if USE_DECIMAL:
    bond_reserves_before = Decimal(str(bond_reserves_before))
    share_reserves_before = Decimal(str(share_reserves_before))
if total_shares_needed > fp0:  # long
    print(f"{total_shares_needed=}")
    current_share_price = hyperdrive.pool_info.share_price
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
    #         Decimal(str(pool_info.bond_reserves)),
    #         Decimal(str(pool_info.share_price)),
    #         Decimal(str(pool_config.initial_share_price)),
    #         Decimal(str(pool_info.share_reserves)),
    #         total_bonds_needed,
    #         Decimal(str(pool_config.time_stretch)),
    #         Decimal(str(pool_config.fees.curve)),
    #         Decimal(str(pool_config.fees.governance)),
    #         one_block_return
    #     )
    # else:
    #     total_shares_needed,_,_ = get_shares_out_for_bonds_in(
    #         pool_info.bond_reserves,
    #         pool_info.share_price,
    #         pool_config.initial_share_price,
    #         pool_info.share_reserves,
    #         total_bonds_needed,
    #         pool_config.time_stretch,
    #         pool_config.fees.curve,
    #         pool_config.fees.governance,
    #         one_block_return
    #     )
    # total_shares_needed = -total_shares_needed * 1
    total_shares_needed *= 1
    if amount > 0:
        logging.log(10, f" open short of {(amount)} to hit {int(target_apr)/1e18:.2%}")
        run_trades(trade_list=[("open_short", amount)])
    logging.log(10, "no trade required.")
time.sleep(2)
bond_reserves_after = hyperdrive.current_pool_state.pool_info.bond_reserves
share_reserves_after = hyperdrive.current_pool_state.pool_info.share_reserves
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
if total_bonds_needed > 0:
    print(f" diff (%): {float(d_bonds_diff/total_bonds_needed):.1e}")
print(f"  d_shares = {float(d_shares):+27,.18f}", end="")
print(f" expected: {float(total_shares_needed):+27,.18f}")
d_shares_diff = d_shares - total_shares_needed
print(f"   diff: {float(d_shares_diff):+27,.18f}", end="")
if total_shares_needed > 0:
    print(f" diff (%): {float(d_shares_diff/total_shares_needed):.1e}")


# %%
# get data
def get_data(session, config_data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get data."""
    txn_data = get_transactions(session)
    pool_info = get_pool_info(session, coerce_float=False)
    combined_data = get_combined_data(txn_data, pool_info)
    _, fixed_rate_y = calc_fixed_rate_df(combined_data, config_data)
    combined_data["fixed_rate"] = fixed_rate_y
    combined_data["spot_price"] = calc_spot_price(
        combined_data["share_reserves"],
        combined_data["share_adjustment"],
        combined_data["bond_reserves"],
        config_data["initial_share_price"],
        config_data["time_stretch"],
    )
    combined_data["base_buffer"] = (
        combined_data["longs_outstanding"] / combined_data["share_price"] + config_data["minimum_share_reserves"]
    )
    return combined_data, pool_info


time.sleep(2)
data, pool_info = get_data(session, config_data)

# %%
pool_config = copy(hyperdrive.current_pool_state.pool_config)
pool_info = copy(hyperdrive.current_pool_state.pool_info)
if USE_DECIMAL:
    pool_info.share_reserves = Decimal(str(pool_info.share_reserves))
    pool_config.initial_share_price = Decimal(str(pool_config.initial_share_price))
    pool_config.position_duration = Decimal(str(pool_config.position_duration))
    pool_config.time_stretch = Decimal(str(pool_config.time_stretch))
    pool_config.inv_time_stretch = Decimal(str(1/pool_config.time_stretch))
    pool_info.bond_reserves = Decimal(str(pool_info.bond_reserves))
    pool_info.share_price = Decimal(str(pool_info.share_price))
    pool_config.fees.curve = Decimal(str(pool_config.fees.curve))
    pool_config.fees.governance = Decimal(str(pool_config.fees.governance))

current_rate = calc_apr(
    pool_info.share_reserves,
    fp0,
    pool_info.bond_reserves,
    pool_config.initial_share_price,
    pool_config.position_duration,
    pool_config.time_stretch,
)
print(f"target: {float(target_apr):0%} ", end="")
print(f"actual: {float(current_rate):22.18%} ")
d_apr = current_rate - target_apr
print(f" diff: {float(abs(d_apr)):22.18%}", end="")
print(f" diff (%): {float(d_apr/target_apr):.2e}")

# %%
