import matplotlib.pyplot as plt
import numpy as np

# Subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

# Price range and strike
price = np.linspace(0, 200, 1000)
K = 80  # strike price
premium = 20  # vertical shift

# Define payoffs with vertical offset (premium)
buy_call  = np.maximum(price - K, 0) - premium   # cost to buy
buy_put   = np.maximum(K - price, 0) - premium
sell_call = -np.maximum(price - K, 0) + premium  # received premium
sell_put  = -np.maximum(K - price, 0) + premium

titles  = ["Buying a Call", "Buying a Put", "Writing a Call", "Writing a Put"]
payoffs = [buy_call, buy_put, sell_call, sell_put]

for ax, title, payoff in zip(axs.flat, titles, payoffs):
    ax.plot(price, payoff, color='black', linewidth=2)

    # Breakeven line at y = 0
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2)

    # Kink dot at x=K, y=0y_val = premium if title in ["Buying a Call", "Writing a Put"] else -premium
    y_val = premium if title in ["Writing a Call", "Writing a Put"] else -premium
    ax.scatter([K], [y_val], color='black', s=200, zorder=5)
    ax.annotate("Strike price", 
            xy=(K, y_val), 
            xytext=(K + 10, y_val + 10),  # offset label position
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
    ax.set_xlabel("Underlying Price", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel("Profit/Loss", fontsize=12, fontweight='bold', labelpad=8)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, 200)
    ax.set_ylim(-100, 100)  # centered around y = 0

plt.show()
