from agent0.accounts_config import AccountKeyConfig
from script_functions import *

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="web3.contract.base_contract")

START_BLOCK = 0
LOOKBACK_BLOCK_LIMIT = 100_000  # Look back limit for backfilling
SLEEP_AMOUNT = 1
MAX_LIVE_BLOCKS = 14400

load_dotenv()  # Get postgres env variables if exists

# %%
# setup
DEVELOP = True
ENV_FILE = "script.env"
env_config = EnvironmentConfig(
    delete_previous_logs = True,
    halt_on_errors = True,
    log_formatter = "%(message)s",
    log_filename = "agent0-bots",
    log_level = logging.DEBUG,
    log_stdout = True,
    random_seed = 1234,
    username="Botty McBotFace",
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
session = initialize_session()  # initialize the postgres session
account_key_config = initialize_accounts(agent_config, ENV_FILE, random_seed=env_config.random_seed, develop=DEVELOP)
eth_config = build_eth_config()
eth_config.RPC_URL = URI("http://localhost:8546")
contract_addresses = fetch_hyperdrive_address_from_url(os.path.join(eth_config.ARTIFACTS_URL, "addresses.json"))
user_account = create_and_fund_user_account(eth_config, account_key_config, contract_addresses)
fund_agents(user_account, eth_config, account_key_config, contract_addresses)
web3, base_token_contract, hyperdrive_contract, agent_accounts = setup_experiment(
    eth_config, env_config, agent_config, account_key_config, contract_addresses
)
config_data = []
while len(config_data) == 0:
    config_data = get_pool_config(session, coerce_float=False)
config_data = config_data.iloc[0]

# %%
# only main wrapper function goes here, because it depends on things that come before it

def run_trades(env_config:EnvironmentConfig|None=None, agent_config:AgentConfig | None =None, account_key_config:AccountKeyConfig|None=None, develop=None, trade_list=None):
    """Wrapper to run trades in interactive session, pulling in global variables for configuration."""
    if env_config is None:
        env_config = globals().get("env_config")
    if agent_config is None:
        agent_config = globals().get("agent_config")
    if account_key_config is None:
        account_key_config = globals().get("account_key_config")
    if develop is None:
        develop = True
    # trade_list= [
    #     ("add_liquidity", 100),
    #     ("open_long", 100),
    #     ("open_short", 100),
    #     ("close_short", 100),
    # ]
    # trade_list= [("open_long", 10_000)]*100
    agent_config[0].init_kwargs = {
        "trade_list": trade_list or [("open_long", 100_000)]*100
    }
    try:
        run_agents(env_config, agent_config, account_key_config, develop=develop)
    except Exception as exc:
        print(f"Failed to run agents. Exception: {exc}")
        pass