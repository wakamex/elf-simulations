# pylint: disable=duplicate-code
"""
basic simulation
consists of only two agents: single_LP and single_short
each only does their action once, then liquidates at the end
this tests pool bootstrap behavior since it doesn't use init_LP to seed the pool
"""

import time
import logging
from pathlib import Path

from test_trade import BaseTradeTest

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
    "share_price": 5.0,
    "init_share_price": 5.0,
    # "time_stretch_constant": 1,
}

# try:
start = time.time()
base_test.run_base_lp_test(
    user_policies=["single_lp:base_to_lp=200", "single_short:pt_to_short=500"],
    config_file=config_file,
    delete_log_file=False,
    additional_overrides=override_dict,
)
dur = time.time() - start
if dur < 1:
    output_string = f"test took {dur*1000} milliseconds"
else:
    output_string = f"test took {dur} seconds"
print(output_string)

# except:
#     # print("=== failed to run, printing partial log file ===")
#     pass

# CUSTOMIZE OUTPUTS HERE
loglevels = ["INFO"]
modules = ["simulators", "agent", "single_short"]
# loglevels = ["DEBUG"]
# modules = ["pricing_models"]

# parse log file and print to terminal
print_next_line = False
handler = logging.getLogger().handlers[0]
print(f"check if file exists: {handler.baseFilename} exists={Path(handler.baseFilename).exists()}")
file = open(handler.baseFilename, "r", encoding="utf-8")
with file as fh:
    lines = fh.readlines()
    for line in lines:
        try:
            # separate timestamp out from rest of line based on the first two spaces
            try:
                date, time, restofline = line.split(" ", 2)
                # remove trailing : on time
                time = time[:-1]
            except ValueError:
                raise ValueError("doesn't have 2 spaces")
            # check if date is a date
            if not all([x.isnumeric() for x in date.split("-")]):
                raise ValueError(f"date is not a date, date={date}")
            # check if time is a time
            if not all([x.isnumeric() for x in time.split(":")]):
                raise ValueError(f"time is not a time, time={time}")
            # separate loglevel out from the rest of the line based on the first ": "
            try:
                loglevel, restofline = restofline.split(":", 1)
            except ValueError:
                raise ValueError("loglevel is not a loglevel")
            # remove leading space
            try:
                restofline = restofline[1:]
            except IndexError:
                raise ValueError("restofline is empty")
            # separate module out from the rest of the line based on the first "."
            try:
                module, restofline = restofline.split(".", 1)
            except ValueError:
                raise ValueError("module is not a module")
            # separate function out from the rest of the line based on the first ":\n"
            try:
                function, restofline = restofline.split(":\n", 1)
            except ValueError:
                raise ValueError("function is not a function")
            if module in modules and loglevel in loglevels:
                print_next_line = True
            else:
                print_next_line = False
        except ValueError as e:
            # print(f"ValueError: {e}")
            # print(f"ValueError: {line} printing next line: {print_next_line} len(line)={len(line)} line[0:5]={line[0:5]}")
            if print_next_line and len(line) > 1 and line[0:5] != "day =":
                print(line, end="")
                # print(f"{date} {time} {module}.{function}: {line}", end="")
file.close()