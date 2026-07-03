import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/Graph Data.csv")

lower_err = df["Mean_Diff"] - df["CI_Lower"]
upper_err = df["CI_Upper"] - df["Mean_Diff"]

yerr = [lower_err, upper_err]

plt.figure(figsize=(8, 4))
plt.bar(
    df["Track"],
    df["Mean_Diff"],
    yerr=yerr,
    capsize=4,
)

plt.axhline(0, linewidth=1)

plt.ylabel("Mean Time Difference (s)")

plt.title("Mean Lap Time Difference by Track (95% CI)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/Mean_Diff_Bar_Chart.png")
plt.show()