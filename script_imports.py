"""Prepare imports and constants."""
# pylint: disable=invalid-name, unused-import
import logging
import os
import asyncio
from pathlib import Path
import time
import json
import warnings
from decimal import Decimal
from dataclasses import fields

from cycler import Cycler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy import size

from agent0 import initialize_accounts
from agent0.accounts_config import AccountKeyConfig
from agent0.base.config import AgentConfig, EnvironmentConfig
from agent0.hyperdrive.exec import create_and_fund_user_account, setup_experiment, async_fund_agents, run_agents
from agent0.hyperdrive.policies.zoo import Zoo
from ethpy import EthConfig
from ethpy.hyperdrive import AssetIdPrefix

from chainsync.analysis.calc_fixed_rate import calc_fixed_rate_df
from chainsync.dashboard.build_ohlcv import build_ohlcv
from chainsync.analysis import calc_spot_price
from chainsync.analysis.calc_pnl import calc_closeout_pnl
from chainsync.analysis.data_to_analysis import get_transactions
from chainsync.db.base import initialize_session
from chainsync.dashboard import plot_fixed_rate, plot_ohlcv
from chainsync.db.hyperdrive import (
    get_pool_config,
    get_pool_info,
    get_wallet_deltas,
    get_current_wallet,
    get_pool_analysis,
)
from dotenv import load_dotenv
from ethpy.eth_config import build_eth_config
from ethpy.hyperdrive import fetch_hyperdrive_address_from_uri
from eth_typing import URI
from fixedpointmath import FixedPoint
