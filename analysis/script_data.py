"""Script to query bot experiment data."""
# pylint: disable=invalid-name
# %%
from chainsync.base import initialize_session
from ethpy.base import initialize_web3_with_http_provider
from dotenv import load_dotenv

START_BLOCK = 0
LOOKBACK_BLOCK_LIMIT = 100_000  # Look back limit for backfilling
SLEEP_AMOUNT = 1

load_dotenv()  # Get postgres env variables if exists

# %%
# experiment parameters
contracts_url = "http://localhost:8080/addresses.json"
ethereum_node = "http://localhost:8546"
abi_dir = "./packages/hyperdrive/src/"

# %%
# initialization
session = initialize_session()  # initialize the postgres session
web3 = initialize_web3_with_http_provider(
    ethereum_node=ethereum_node,
    request_kwargs={"timeout": 60},
)
