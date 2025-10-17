# -*- coding: utf-8 -*-
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# =========================
# Configuración
# =========================
n_points = 110005
base_dir = Path(__file__).resolve().parent
data_csv = base_dir / f"{n_points}_results.csv"
img_dir  = base_dir / "images"; img_dir.mkdir(exist_ok=True)
out_png  = img_dir / f"{n_points}_results_cpp_plot.png"

# =========================
# 1) Leer resultados del programa C++
# =========================
data = np.loadtxt(data_csv, delimiter=",")
x, y = data[:,0], data[:,1]
labels = data[:,2].astype(int)

# ⚠️ Normalizar: tu C++ usa RUIDO = -2 → sklearn usa -1
labels = np.where(labels == -2, -1, labels)

# =========================
# 2) Graficar (ruido = morado, clúster = amarillo)
# =========================
cidx = (labels == -1).astype(int)  # 1 = ruido, 0 = cluster
cmap = ListedColormap(["#FFD700", "#800080"])

plt.figure(figsize=(6,6))
plt.title("Detecting noise with my DBSCAN (normalized to sklearn)")
plt.scatter(x, y, c=cidx, cmap=cmap, s=50)
plt.xticks([]); plt.yticks([]); plt.box(False)
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

print(f"💾 Imagen guardada en: {out_png}")
