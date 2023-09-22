"""Script to query bot experiment data."""

# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import, bare-except, wrong-import-position, redefined-outer-name, pointless-statement, missing-final-newline
# %%
# setup
import nest_asyncio
nest_asyncio.apply()

# %%
# bot script setup
from script_functions import check_docker
check_docker(restart=True)
from script_setup import *
local_chain = create_hyperdrive_chain("http://127.0.0.1:8546", initial_liquidity=int(20*1e18))
print("\n ==== Pool Config ===")
for k,v in config_data.items():
    print(f"{k:20} | {v}")
print("\n ==== Hyperdrive Addresses ===")
for k,v in local_chain.hyperdrive_contract_addresses.__dict__.items():
    if not k.startswith("_"):
        print(f"{k:20} | {v}")

# %%
# the above loads env_config, agent_config, and account_key_config. let's check them out.
display(env_config)
display(agent_config)
display(account_key_config)

# set private key to the first account in anvil, which is the default LP
# account_key_config.AGENT_KEYS = ['0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80']
# make it withdraw all liquidity
# run_trades(trade_list=[("remove_liquidity", 1e7)])

# %%
# run trades
trade_size = 10_000
n_trades = 108 # max out the market
# n_trades = 4
# run_trades(trade_list=[("open_long", trade_size)]*n_trades + [("close_long", trade_size)]*n_trades)
run_trades(trade_list=[("add_liquidity", trade_size*10)] + [("open_short", trade_size)] + [("remove_liquidity", "all")]+ [("close_short", trade_size)])

# %%
# get data
time.sleep(1)
data, pool_info = get_data(session, config_data)
display(data.iloc[0].T)

# %%
plot_reserves(data, include_buffer=True)

# %%
plot_positions(data)

# %%
plot_rate_price(data)

# %%
ax1, lines1, labels1, _data = plot_rate(data, legend=False)
plot_secondary(_data, "spot_price", ax1, lines1, labels1)

# %%
# plot reserves in zoom mode
ax1, lines1, labels1 = plot_reserves(data, include_buffer=True)
ax1.set_ylim([0,1e8])
ax1.set_title(f"{ax1.get_title()} zoomed in")

# %%

# %%
# plot individual graphs
# fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# plt.tight_layout(pad=0, w_pad=3, h_pad=3)
# plot_positions(data, ax=axes[0,0])
# plot_reserves(data, ax=axes[0,1])
# plot_rate_price(data, ax=axes[1,0])
# plot_buffers(data, ax=axes[1,1])

# %%
# plot stuff
# fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# plt.tight_layout(pad=0, w_pad=3, h_pad=3)
# ax1, lines1, labels1 = plot_positions(data, axes[0,0])
# plot_secondary(data, "fixed_rate", ax1, lines1, labels1)
# ax1, lines1, labels1 = plot_positions(data, axes[1,0])
# plot_secondary(data, "spot_price", ax1, lines1, labels1)
# ax1, lines1, labels1 = plot_reserves(data, axes[0,1])
# plot_secondary(data, "fixed_rate", ax1, lines1, labels1)
# ax1, lines1, labels1 = plot_reserves(data, axes[1,1])
# plot_secondary(data, "spot_price", ax1, lines1, labels1)

# %%
data.columns

# %%
pool_info.columns

# %%