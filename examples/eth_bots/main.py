"""Main script for running bots on Hyperdrive."""
from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime

import numpy as np
import requests
from eth_typing import BlockNumber

from elfpy import eth, hyperdrive_interface
from elfpy.bots import DEFAULT_USERNAME
from elfpy.utils import logs
from elfpy.utils.format import format_numeric_string as fmt
from examples.eth_bots.config import agent_config, environment_config
from examples.eth_bots.setup_experiment import setup_experiment
from examples.eth_bots.trade_loop import trade_if_new_block

logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning, module='web3.contract.base_contract')

def main():
    """Entrypoint to load all configurations and run agents."""
    web3, hyperdrive_contract, agent_accounts = setup_experiment(environment_config, agent_config)
    last_executed_block = BlockNumber(0)
    while True:
        last_executed_block = trade_if_new_block(
            web3,
            hyperdrive_contract,
            agent_accounts,
            environment_config.halt_on_errors,
            last_executed_block,
        )


if __name__ == "__main__":
    main()
