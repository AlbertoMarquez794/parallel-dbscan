# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# =========================
# Configuración
# =========================
n_points   = 4000
centers    = 4
cluster_sd = 0.06
random_st  = 11
center_box = (0.0, 1.0)

# Ajusta estos dos para “balancear” cuántos amarillos/morados quieres:
eps_value  = 0.03     # radio
min_pts    = 14       # subirlo genera más morados (bordes/ruido)

# 📁 Guarda todo en la carpeta Serial/
base_dir = Path(__file__).resolve().parent  # ruta de este script (Serial/)
img_dir = base_dir / "images"
img_dir.mkdir(exist_ok=True)
out_png = img_dir / f"{n_points}_dbscan_core_vs_border_noise.png"
data_csv = base_dir / f"{n_points}_data.csv"  # <--- ahora en Serial/

# =========================
# 1) Datos de prueba
# =========================
points, _ = make_blobs(
    n_samples=n_points,
    centers=centers,
    cluster_std=cluster_sd,
    random_state=random_st,
    center_box=center_box
)
points = np.round(np.abs(points[:, ::-1]), 3)
np.savetxt(data_csv, points, delimiter=",", fmt="%.3f")

# =========================
# 2) DBSCAN
# =========================
db = DBSCAN(eps=eps_value, min_samples=min_pts).fit(points)
labels = db.labels_  # -1 = ruido

is_core = np.zeros(points.shape[0], dtype=bool)
is_core[db.core_sample_indices_] = True

is_border = (labels != -1) & (~is_core)
is_noise  = (labels == -1)

# =========================
# 3) Visualización
# =========================
colors = np.full(points.shape[0], 'purple', dtype=object)
colors[is_core] = 'yellow'

plt.figure()
plt.title("Detecting noise with DBSCAN (core = yellow, border+noise = purple)")
plt.scatter(points[:, 0], points[:, 1], c=colors, s=50)
plt.xticks([]); plt.yticks([]); plt.box(False)

plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"💾 CSV guardado en: {data_csv}")
print(f"💾 Imagen guardada: {out_png}")
plt.close()
