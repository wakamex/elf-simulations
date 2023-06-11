"""Get configuration for a bot run."""
import logging
import os
import sys
from pathlib import Path

from ape.logging import logger as ape_logger
from dotenv import load_dotenv

from elfpy.agents.policies import LongLouie, RandomAgent, ShortSally
from elfpy.bots.bot_info import BotInfo
from elfpy.bots.get_env_args import EnvironmentArguments
from elfpy.simulators.config import Config


# TODO: this would be a really good place to create something with attr.s so that we can do run time
# checking to make sure that nothing incorrect is passed in, so that we can avoid typo errors.
def get_config(args: EnvironmentArguments) -> Config:
    """Instantiate a config object with elf-simulation parameters.
    Parameters
    ----------
    args : dict
        The arguments from environmental variables.
    Returns
    -------
    config : simulators.Config
        The config object.
    """
    # init
    ape_logger.set_level(logging.ERROR)
    config = Config()

    # general settings
    config.title = "evm bots"
    config.scratch["project_dir"] = Path.cwd().parent if Path.cwd().name == "examples" else Path.cwd()
    load_dotenv(dotenv_path=f"{config.scratch['project_dir']}/.env")
    config.log_level = args.log_level
    random_seed_file = f"{config.scratch['project_dir']}/.logging/random_seed{'_devnet' if args.devnet else ''}.txt"

    # wipe solidity cache to allow ape to build clean version
    cache_dirs = [
        f"{config.scratch['project_dir']}/hyperdrive_solidity/contracts/.cache",
        f"{config.scratch['project_dir']}/hyperdrive_solidity/forge-cache",
    ]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            os.system(f"rm -rf {cache_dir}")
            print(f"found and removed {cache_dir}")

    # get random seed if it's been saved
    if os.path.exists(random_seed_file):
        with open(random_seed_file, "r", encoding="utf-8") as file:
            config.random_seed = int(file.read()) + 1
    else:  # make parent directory if it doesn't exist
        os.makedirs(os.path.dirname(random_seed_file), exist_ok=True)
    logging.info("Random seed=%s", config.random_seed)
    with open(random_seed_file, "w", encoding="utf-8") as file:
        file.write(str(config.random_seed))

    # save all args into the config
    for key, value in args.__dict__.items():
        if hasattr(config, key):
            config[key] = value
        else:
            config.scratch[key] = value
    config.log_filename += "_devnet" if args.devnet else ""

    # experiment specific settings
    config.scratch["louie"] = BotInfo(risk_threshold=0.0, policy=LongLouie, trade_chance=config.scratch["trade_chance"])
    config.scratch["frida"] = BotInfo(policy=ShortSally, trade_chance=config.scratch["trade_chance"])
    config.scratch["random"] = BotInfo(policy=RandomAgent, trade_chance=config.scratch["trade_chance"])
    config.scratch["bot_names"] = {"louie", "frida", "random"}

    config.freeze()
    return config
