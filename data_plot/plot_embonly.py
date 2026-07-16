import matplotlib.pyplot as plt
import pandas as pd

csv_file = "data_plot/emb_only.csv"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.strip()

plt.figure(figsize=(10, 6))

colors = ["seagreen", "crimson", "royalblue"]
df.plot(x="Step", ax=plt.gca(), linewidth=3, fontsize=20, color=colors)

plt.xlabel("Step", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.legend(fontsize=20, loc="lower right")

plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

plt.savefig("data_plot/emb_only.svg", dpi=300, bbox_inches="tight")
