# %%
from script_setup import *

print("\n ==== Pool Config ===")
for k, v in config_data.items():
    print(f"{k:20} | {v}")

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