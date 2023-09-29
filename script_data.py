"""Script to query bot experiment data."""
# pylint: disable=invalid-name,wildcard-import,unused-wildcard-import

# %%
# setup
import nest_asyncio

nest_asyncio.apply()

# %%
# settings
from script_imports import *

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="web3.contract.base_contract")

START_BLOCK = 0
LOOKBACK_BLOCK_LIMIT = 100_000  # Look back limit for backfilling
SLEEP_AMOUNT = 1
MAX_LIVE_BLOCKS = 14400

load_dotenv()  # Get postgres env variables if exists

# %%
# initialization
session = initialize_session()  # initialize the postgres session
DEVELOP = True
ENV_FILE = "script_data.env"
env_config = EnvironmentConfig(
    delete_previous_logs=False,
    halt_on_errors=True,
    log_filename="agent0-logs",
    log_level=logging.INFO,
    log_stdout=True,
    random_seed=1234,
    username="open100long",
)
agent_config: list[AgentConfig] = [
    AgentConfig(
        policy=Policies.deterministic,
        number_of_agents=1,
        base_budget_wei=FixedPoint(1e9).scaled_value,  # 1 billion base
        eth_budget_wei=FixedPoint(1).scaled_value,  # 1 Eth
        init_kwargs={"trade_list": [("open_long", 100)] * 1},  # Open 4 longs
    ),
]
os.environ["DEVELOP"] = "true"
account_key_config = initialize_accounts(agent_config, ENV_FILE, random_seed=env_config.random_seed)
eth_config = build_eth_config()
eth_config.rpc_uri = URI("http://localhost:8546")
addresses = fetch_hyperdrive_address_from_uri(os.path.join(eth_config.rpc_uri, "addresses.json"))
web3, base_token_contract, hyperdrive_contract, agent_accounts = setup_experiment(
    eth_config, env_config, agent_config, account_key_config, addresses
)
run_agents(env_config, agent_config, account_key_config, develop=DEVELOP)

# %%
# config
txn_data = get_transactions(session, -MAX_LIVE_BLOCKS)
pool_info = get_pool_info(session, -MAX_LIVE_BLOCKS, coerce_float=False)
config_data = get_pool_config(session, coerce_float=False)
config_data = config_data.iloc[0]
print("".join(["="] * 23) + "=== Pool Config ===" + "".join(["="] * 23))
for k, v in config_data.items():
    print(f"{k:20} | {v}")

# %%
combined_data = get_combined_data(txn_data, pool_info)
wallet_deltas = get_wallet_deltas(session, coerce_float=False)

fixed_rate_x, fixed_rate_y = calc_fixed_rate_df(combined_data, config_data)
pool_analysis = get_pool_analysis(session)
ohlcv = build_ohlcv(pool_analysis, freq="5T")

start_time = time.time()
current_wallet = get_current_wallet(session, -MAX_LIVE_BLOCKS)
current_returns = current_wallet.reset_index().groupby("walletAddress")["pnl"].sum()

current_wallet.delta = current_wallet.delta.astype(float)
current_wallet.pnl = current_wallet.pnl.astype(float)
current_wallet["spot_pnl"] = calc_spot_pnl(pool_config, pool_info, current_wallet)
current_wallet["closeout_pnl"] = calc_closeout_pnl(current_wallet, pool_info, hyperdrive_contract)

print(f"calculated PNL in {time.time() - start_time=}")

# %%
plt.step(combined_data.reset_index().blockNumber, combined_data[["longs_outstanding", "shorts_outstanding"]])
plt.legend(["longs", "shorts"])

# %%
# ohlcv
# fig = plt.figure(figsize=(8, 4))
# ax_ohlcv = fig.add_subplot(1, 2, 1)
# ax_vol = fig.add_subplot(1, 2, 2)
# plot_ohlcv(ohlcv, ax_ohlcv, ax_vol)

# %%
