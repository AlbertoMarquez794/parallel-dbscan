# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
import numpy as np
from pathlib import Path

# ========= Config =========
n_points      = 200000             # total (clusters + ruido)
centers       = [(0.2,0.2),(0.8,0.2),(0.2,0.8),(0.8,0.8)]
cluster_std   = 0.02
random_state  = 42
noise_ratio   = 0.10                  # 10% del total será ruido
n_noise       = int(n_points * noise_ratio)
n_blobs       = n_points - n_noise

# Radio mínimo para considerar un punto como "ruido alejado" de los centros.
# Guía: con sd=0.02, 3σ≈0.06; usar un margen > 3σ para que quede fuera.
min_dist_from_centers = 0.12          # prueba 0.10–0.15 según tu eps

# ========= 1) Blobs =========
points, labels = make_blobs(
    n_samples=n_blobs,
    centers=centers,
    cluster_std=cluster_std,
    random_state=random_state
)  # labels = 0..3

# ========= 2) Ruido lejos de los centros =========
def gen_noise_far(m, box=(0.0, 1.0), centers=(), min_dist=0.12, seed=None):
    rng = np.random.default_rng(seed)
    noise_accum = []
    need = m
    # Generamos en tandas y filtramos por distancia a TODOS los centros
    while need > 0:
        batch = max(need * 2, 1000)  # over-generate para filtrar
        cand = rng.random((batch, 2)) * (box[1]-box[0]) + box[0]
        if centers:
            c = np.array(centers)  # (k,2)
            # dist^2 a todos los centros y nos quedamos con puntos cuya dist mínima > min_dist
            d2 = ((cand[:, None, :] - c[None, :, :])**2).sum(axis=2)  # (batch,k)
            ok = np.sqrt(d2.min(axis=1)) > min_dist
            cand = cand[ok]
        take = cand[:need]
        noise_accum.append(take)
        need -= take.shape[0]
    return np.vstack(noise_accum)

noise = gen_noise_far(n_noise, centers=centers, min_dist=min_dist_from_centers, seed=random_state)

# Etiquetas de ruido = -1
labels_noise = -1 * np.ones(noise.shape[0], dtype=int)

# ========= 3) Mezclar y guardar =========
all_points = np.vstack([points, noise])
all_labels = np.concatenate([labels, labels_noise])

# Mezcla aleatoria
perm = np.random.default_rng(random_state).permutation(all_points.shape[0])
all_points = all_points[perm]
all_labels = all_labels[perm]

# CSV con etiquetas (x,y,label)
out_csv = Path(f"{n_points}_data_with_noise.csv")
np.savetxt(out_csv, np.column_stack([all_points, all_labels]),
           delimiter=",", fmt=["%.6f","%.6f","%d"])
print(f"✅ Generado: {n_points} puntos (ruido: {n_noise}) → {out_csv}")

# (Opcional) archivo sólo con x,y para tu C++
out_xy = Path(f"{n_points}_data_xy.csv")
np.savetxt(out_xy, all_points, delimiter=",", fmt="%.6f")
print(f"💾 También guardado sólo XY → {out_xy}")
