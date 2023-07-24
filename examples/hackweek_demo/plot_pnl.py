"""Plots the pnl."""
from __future__ import annotations
import time

import pandas as pd

from elfpy.data import postgres as pg
from extract_data_logs import calculate_spot_price_from_state


def calculate_pnl(
    pool_config: pd.DataFrame,
    pool_info: pd.DataFrame,
    checkpoint_info: pd.DataFrame,
    agent_positions: dict[str, pg.AgentPosition],
) -> dict[str, pg.AgentPosition]:
    """Calculate pnl for all agents.

    Arguments
    ---------
    pool_config : PoolConfig
        Configuration with which the pool was initialized.
    pool_info : pd.DataFrame
        Reserves of the pool at each block.
    checkpoint_info : pd.DataFrame
    agent_positions :
        Dict containing each agent's AgentPosition object.
    """
    position_duration = pool_config.positionDuration.iloc[0]

    # loop across agents since agent_positions is a dict[str, AgentPosition], we extract only the AgentPosition
    for ap in agent_positions.values():  # pylint: disable=invalid-name
        start_time = time.time()
        for block in ap.positions.index:
            # We only calculate pnl up to pool_info
            if block > pool_info.index.max():
                continue
            state = pool_info.loc[block]  # current state of the pool

            # get maturity from current checkpoint
            if block in checkpoint_info.index:
                current_checkpoint = checkpoint_info.loc[block]
                maturity = current_checkpoint["timestamp"] + pd.Timedelta(seconds=position_duration)
            else:
                # it's unclear to me if not finding a current checkpoint is expected behavior that we should handle.
                # if we assume this is the first trade of a new checkpoint,
                # we know the maturity is equal to the current block timestamp plus the position duration.
                maturity = None

            # calculate spot price of the bond, specific to the current maturity
            spot_price = calculate_spot_price_from_state(state, maturity, ap.timestamp[block], position_duration)
            print(f"{spot_price=}")

            # add up the pnl for the agent based on all of their positions.
            # TODO: vectorize this. also store the vector of pnl per position. in postgres?
            ap.pnl.loc[block] = 0
            for position_name in ap.positions.columns:
                if position_name.startswith("LP"):
                    # LP value
                    total_lp_value = state.shareReserves * state.sharePrice + state.bondReserves * spot_price
                    share_of_pool = ap.positions.loc[block, "LP"] / state.lpTotalSupply
                    # TODO this assertion is breaking due to positions having higher LP than supply
                    # assert share_of_pool < 1, "share_of_pool must be less than 1"
                    ap.pnl.loc[block] += share_of_pool * total_lp_value
                    print(f"at block {block} an LP position adds to PNL {share_of_pool * total_lp_value}")
                elif position_name.startswith("LONG"):
                    # LONG value
                    ap.pnl.loc[block] += ap.positions.loc[block, position_name] * spot_price
                    print(f"at block {block} a LONG position adds to PNL {ap.positions.loc[block, position_name] * spot_price}")
                elif position_name.startswith("SHORT"):
                    # SHORT value is calculated as the:
                    # total amount paid for the position (position * 1)
                    # remember this payment is comprised of the spot price (p) and the max loss (1-p) set as margin
                    # minus the closing cost (position * spot_price)
                    # this means the current position value equals position * (1 - spot_price)
                    ap.pnl.loc[block] += ap.positions.loc[block, position_name] * (1 - spot_price)
                    print(f"at block {block} a SHORT position adds to PNL {ap.positions.loc[block, position_name] * (1 - spot_price)}")
                elif position_name.startswith("BASE"):
                    ap.pnl.loc[block] += ap.positions.loc[block, position_name]
                    print(f"at block {block} BASE adds to PNL {ap.positions.loc[block, position_name]}")
        print(f"loop finished in {time.time() - start_time} seconds")
        # ===================== VECTORIZE =====================
        # start_time = time.time()
        # maturities = checkpoint_info.loc[ap.timestamp.index,:]["timestamp"]
        # spot_price_by_maturity_dict = {}
        # for maturity in maturities.unique():
        #     state = pool_info.loc[maturity]
        #     spot_price = calculate_spot_price_from_state(state, maturity, ap.timestamp[maturity], position_duration)
        #     spot_price_by_maturity_dict[maturity] = spot_price
        # # Separate positions into different DataFrames
        # lp_positions = ap.positions.filter(like='LP')
        # long_positions = ap.positions.filter(like='LONG')
        # short_positions = ap.positions.filter(like='SHORT')
        # base_positions = ap.positions.filter(like='BASE')

        # # Calculate PNL for each type
        # lp_pnl = (lp_positions / state.lpTotalSupply) * (state.shareReserves * state.sharePrice + state.bondReserves * spot_price)
        # long_pnl = long_positions * spot_price
        # short_pnl = short_positions * (1 - spot_price)
        # base_pnl = base_positions

        # # Sum PNLs
        # ap.pnl.loc[block] = lp_pnl.sum(axis=1) + long_pnl.sum(axis=1) + short_pnl.sum(axis=1) + base_pnl.sum(axis=1)
        # print(f"loop finished in {time.time() - start_time} seconds")
        # =====================================================
    return agent_positions


def plot_pnl(agent_positions: dict[str, pg.AgentPosition], axes) -> None:
    """Plot the pnl data.

    Arguments
    ---------
    agent_positions : dict[str, pg.AgentPosition]
        Dict containing each agent's AgentPosition object.
    axes : Axes
    Axes object to plot on.
    """
    plot_data = []
    agents = []
    for agent, agent_position in agent_positions.items():
        agents.append(agent)
        plot_data.append(agent_position.pnl)

    if len(plot_data) > 0:
        # TODO see if this concat is slowing things down for plotting
        # Can also plot multiple times
        plot_data = pd.concat(plot_data, axis=1)
        plot_data.columns = agents
    else:
        plot_data = pd.DataFrame([])
        agents = []

    # plot everything in one go
    axes.plot(plot_data.sort_index(), label=agents)

    # change y-axis unit format to #,###.0f
    # TODO this is making the y axis text very large, fix
    # axes.yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, p: format(float(x), ",")))

    # TODO fix these top use axes
    axes.set_xlabel("block timestamp")
    axes.set_ylabel("pnl")
    axes.yaxis.set_label_position("right")
    axes.yaxis.tick_right()
    axes.set_title("pnl over time")

    # %%
