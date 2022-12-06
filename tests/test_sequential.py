# pylint: disable=duplicate-code
"""
basic simulation
consists of only two agents: single_LP and single_short
each only does their action once, then liquidates at the end
this tests pool bootstrap behavior since it doesn't use init_LP to seed the pool
"""

import time
from pathlib import Path

from test_trade import BaseTradeTest

from elfpy.utils.outputs import float_to_string

config_file = Path(__file__).parent.parent / "config" / "example_config.toml"

base_test = BaseTradeTest()

# run a test
override_dict = {
    # "num_blocks_per_day": int(24 * 60 * 60 / 12),  # 12 second block time
    # "verbose": True,
    "pricing_model_name": "Hyperdrive",
    "shuffle_users": False,
    "init_lp": False,
    "simulator.verbose": True,
    "amm.verbose": False,
    "vault_apy": 0.00,
}

start = time.time()
base_test.run_base_lp_test(
    user_policies=["single_lp", "single_short:amount_to_short=200"],
    config_file=config_file,
    additional_overrides=override_dict,
)
dur = time.time() - start
if dur < 1:
    output_string = f"test took {float_to_string(dur*1000,precision=2)} milliseconds"
else:
    output_string = f"test took {float_to_string(dur,precision=2)} seconds"
print(output_string)
