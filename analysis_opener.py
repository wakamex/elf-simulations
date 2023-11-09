"""Script to query bot experiment data."""

# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import, bare-except, wrong-import-position, redefined-outer-name, pointless-statement, missing-final-newline
# %%
# bot script setup
from script_functions import check_docker

# check_docker(restart=True)
from script_setup import *

print("\n ==== Pool Config ===")
for k, v in config_data.items():
    print(f"{k:20} | {v}")

# %%
# set up opener
agent_config = AgentConfig(
    policy=Zoo.opener,
    number_of_agents=1,
    base_budget_wei=FixedPoint(1e9).scaled_value,  # 1 billion base
    eth_budget_wei=FixedPoint(1).scaled_value,  # 1 Eth
    policy_config=Zoo.opener.Config(trade_amount=FixedPoint(1_000_000)),
)
env_config.halt_on_errors = False

# %%
print(f"{agent_config=}")
logging.log(10, "running opener")
run_trades(agent_config=agent_config, env_config=env_config)

# %%
# get data
time.sleep(1)
data, pool_info = get_data(session, config_data)
print(data.iloc[0].T)

# %%
plot_reserves(data, include_buffer=True)
plt.savefig("reserves.png")

# %%
plot_positions(data)
plt.savefig("positions.png")

# %%
ax1, lines1, labels1 = plot_rate_price(data)
plt.savefig("rate_price.png")

# %%
ax1, lines1, labels1, _data = plot_rate(data, legend=False)
plot_secondary(_data, "spot_price", ax1, lines1, labels1)
plt.savefig("rate_price2.png")

# %%
# plot reserves in zoom mode
ax1, lines1, labels1 = plot_reserves(data, include_buffer=True)
# ax1.set_ylim([0,1e8])
ax1.set_title(f"{ax1.get_title()} zoomed in")
plt.savefig("reserves_zoom.png")

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
