"""Experiment configuration."""
from __future__ import annotations

import logging

from agent0.base.config import AgentConfig, Budget, EnvironmentConfig
from agent0.hyperdrive.policies.zoo import Policies
from fixedpointmath import FixedPoint


def get_eth_bots_config(**config_params) -> tuple[EnvironmentConfig, list[AgentConfig]]:
    """Get the instantiated config objects for the ETH bots demo.

    Arguments
    ---------
    **config_params
        Override parameters passed to the EnvironmentConfig

    Returns
    -------
    tuple[EnvironmentConfig, list[AgentConfig]]
        environment_config : EnvironmentConfig
            Dataclass containing all of the user environment settings
        agent_config : list[BotInfo]
            List containing all of the agent specifications
    """
    trade_list = config_params.pop("trade_list") if "trade_list" in config_params else None
    environment_config = EnvironmentConfig(
        delete_previous_logs = True,
        halt_on_errors = True,
        log_formatter = "%(message)s",
        log_filename = "agent0-bots",
        log_level = logging.DEBUG,
        log_stdout = True,
        random_seed = 1234,
        hyperdrive_abi = "IHyperdrive",
        base_abi = "ERC20Mintable",
        username_register_url = "http://localhost:5002",
        artifacts_url = "http://localhost:8080",
        rpc_url = "http://localhost:8546",
        username = "Mihai",
    )
    # apply overrides for config parameters
    for key, value in config_params.items():
        environment_config[key] = value
    agent_config: list[AgentConfig] = [
        AgentConfig(
            policy=Policies.random_agent,
            number_of_agents=0,
            slippage_tolerance=FixedPoint(0.0001),
            base_budget=Budget(
                mean_wei=int(5_000e18),  # 5k base
                std_wei=int(1_000e18),  # 1k base
                min_wei=1,  # 1 WEI base
                max_wei=int(100_000e18),  # 100k base
            ),
            eth_budget=Budget(min_wei=int(1e18), max_wei=int(1e18)),
            init_kwargs={"trade_chance": FixedPoint(0.8)},
        ),
        AgentConfig(
            policy=Policies.long_louie,
            number_of_agents=0,
            base_budget=Budget(
                mean_wei=int(5_000e18),  # 5k base
                std_wei=int(1_000e18),  # 1k base
                min_wei=1,  # 1 WEI base
                max_wei=int(100_000e18),  # 100k base
            ),
            eth_budget=Budget(min_wei=int(1e18), max_wei=int(1e18)),
            init_kwargs={
                "trade_chance": FixedPoint(0.8),
                "risk_threshold": FixedPoint(0.9),
            },
        ),
        AgentConfig(
            policy=Policies.short_sally,
            number_of_agents=0,
            base_budget=Budget(
                mean_wei=int(5_000e18),  # 5k base
                std_wei=int(1_000e18),  # 1k base
                min_wei=1,  # 1 WEI base
                max_wei=int(100_000e18),  # 100k base
            ),
            eth_budget=Budget(min_wei=int(1e18), max_wei=int(1e18)),
            init_kwargs={
                "trade_chance": FixedPoint(0.8),
                "risk_threshold": FixedPoint(0.8),
            },
        ),
        AgentConfig(
            policy=Policies.deterministic,
            number_of_agents=1,
            base_budget=Budget(min_wei=int(1e9*1e18),max_wei=int(1e9*1e18)),
            init_kwargs={
                # "trade_list": [
                #     ("add_liquidity", 100),
                #     ("open_long", 100),
                #     ("open_short", 100),
                #     ("close_short", 100),
                # ]
                "trade_list": trade_list or [("open_long", 100_000)]*100
                # "trade_list": [("open_long", 10_000)]*100
            },
        ),
        AgentConfig(
            policy=Policies.minimal,
            number_of_agents=0,
        ),
        AgentConfig(
            policy=Policies.oneline,
            number_of_agents=0,
        )
    ]

    return environment_config, agent_config
