from math import log2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_palette("colorblind")

x = np.linspace(0.5, 0.999, 500)
LC = 1 - x
MS = 2 * x - 1
ES = [-el * log2(el) - (1 - el) * log2(1 - el) for el in x]

df = pd.DataFrame({
    "x": x,
    "Least Confidence": LC,
    "Margin Sampling": MS,
    "Entropy Sampling": ES
})

df_melt = df.melt(id_vars=["x"], var_name="Method", value_name="Uncertainty")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melt, x="x", y="Uncertainty", hue="Method", linewidth=2)

opt_LC_x = x[np.argmax(LC)]
opt_LC_y = max(LC)

opt_MS_x = x[np.argmin(np.abs(MS))]
opt_MS_y = MS[np.argmin(np.abs(MS))]

opt_ES_x = x[np.argmax(ES)]
opt_ES_y = max(ES)

colors = sns.color_palette("colorblind")
color_LC = colors[0]
color_MS = colors[1]
color_ES = colors[2]

print(f"Optimal Least Confidence: x={opt_LC_x:.3f}, y={opt_LC_y:.3f}")
print(f"Optimal Margin Sampling: x={opt_MS_x:.3f}, y={opt_MS_y:.3f}")
print(f"Optimal Entropy Sampling: x={opt_ES_x:.3f}, y={opt_ES_y:.3f}")

plt.axvline(opt_LC_x, color="#666666", linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(opt_MS_x, color="#666666", linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(opt_ES_x, color="#666666", linestyle='--', linewidth=2, alpha=0.7)

plt.plot(opt_LC_x, opt_LC_y, 'o', color=color_LC, markersize=15)
plt.plot(opt_MS_x, opt_MS_y, 'o', color=color_MS, markersize=15)
plt.plot(opt_ES_x, opt_ES_y, 'o', color=color_ES, markersize=15)

legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, fontsize=15)

handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]

for handle in handles:
    handle.set_linewidth(3)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel("Maximum Class Likelihood", fontsize=15)
plt.ylabel("Measurement Score", fontsize=15)
plt.tight_layout()

plt.savefig("uncertainty_comparison.pdf", format="pdf")