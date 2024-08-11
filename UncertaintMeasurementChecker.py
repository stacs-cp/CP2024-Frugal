from math import log2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette("colorblind")

x = np.linspace(0.5, 0.99, 100)
LC = x - 1
MS = x - (1 - x)
ES = [el * log2(el) + (1 - el) * log2(1 - el) for el in x]

fig, ax = plt.subplots(figsize=(8, 5))
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.plot(x, LC, color=palette[0], label='Least Confidence')
ax.plot(x, MS, color=palette[1], label='Margin Sampling')
ax.plot(x, ES, color=palette[2], label='Entropy-based Sampling')

ax.set_title("Uncertainty Measurements as a Function of Highest Class Probability")
ax.set_xlabel("Highest Class Probability $x$")
ax.set_ylabel("Uncertainty Measure")

critical_points = [(np.argmin(LC), np.amin(LC)), (np.argmin(MS), np.amin(MS)), (np.argmin(ES), np.amin(ES))]
for idx, (cp_x, cp_y) in enumerate(critical_points):
    ax.plot(x[cp_x], cp_y, 'ko')
    ax.annotate(f'Min Value {cp_y:.2f}', xy=(x[cp_x], cp_y), xytext=(10, -30 if idx==1 else -15),
                textcoords='offset points', arrowprops=dict(arrowstyle='->'))

ax.legend()
plt.tight_layout()
plt.savefig("PLOTS2/uncertainty_function_behaviour.pdf")