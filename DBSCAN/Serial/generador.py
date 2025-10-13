# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
import numpy as np
from pathlib import Path

# =========================
# Configuración
# =========================
n_points   = 4000
centers    = 4
cluster_sd = 0.06
center_box = (0, 1.0)

# 🔁 Si quieres que cambie cada vez, usa None o una semilla aleatoria
random_seed = None  # o por ejemplo: np.random.randint(0, 10_000)

# =========================
# 1) Generar datos
# =========================
points, _ = make_blobs(
    n_samples=n_points,
    centers=centers,
    cluster_std=cluster_sd,
    center_box=center_box,
    random_state=random_seed
)

# ✅ Solo tomar valores positivos, sin reflejar duplicados
points = np.clip(points, 0, None)

# ✅ No redondeamos mucho, solo al guardar
output_csv = Path(f"{n_points}_data.csv")
np.savetxt(output_csv, points, delimiter=",", fmt="%.6f")

print(f"✅ Datos generados: {n_points} puntos, guardados en {output_csv}")
