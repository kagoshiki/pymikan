import matplotlib.pyplot as plt
import pandas as pd

# 1. CSVファイルを読み込む
csv_file = "data_plot/emb_only.csv"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.strip()

# 2. グラフ全体のサイズを設定
plt.figure(figsize=(10, 6))

# 3. プロット（Pandasの強力な自動プロット機能を使用）
colors = ["seagreen", "crimson", "royalblue"]
df.plot(x="Step", ax=plt.gca(), linewidth=3, fontsize=20, color=colors)

# 4. グラフの装飾
# plt.title("Model Accuracy Comparison", fontsize=20)
plt.xlabel("Step", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.legend(fontsize=20, loc="lower right")

# グリッド線（目盛り線）を追加
plt.grid(True, linestyle="--", alpha=0.6)

# レイアウトの自動調整（綺麗に収める）
plt.tight_layout()

plt.savefig("data_plot/emb_only.svg", dpi=300, bbox_inches="tight")
