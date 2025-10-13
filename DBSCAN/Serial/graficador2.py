import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

n_points = 4000
result = np.loadtxt(f"{n_points}_results.csv", delimiter=",")

plt.figure()
plt.title("Detecting noise with my DBSCAN")
plt.scatter(result.T[0], result.T[1], c=result.T[2], s=50)
plt.xticks([]); plt.yticks([]); plt.box(False)
Path("images").mkdir(exist_ok=True)
plt.savefig(f"images/{n_points}_results_cpp_plot.png", dpi=300, bbox_inches="tight")
plt.close()
