# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
import numpy as np
from pathlib import Path
import os

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
sizes          = [20000, 40000, 80000, 120000, 140000, 180000, 200000]
centers        = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
cluster_std    = 0.02
random_state   = 42
noise_ratio    = 0.10                  # 10% de ruido
min_dist       = 0.12                  # distancia mínima al centro para ruido
box_ext        = (-0.5, 1.5)           # rango extendido para ruido

# Carpeta destino
out_dir = Path("Serial/Datasets")
os.makedirs(out_dir, exist_ok=True)

# =========================
# FUNCIÓN AUXILIAR
# =========================
def gen_noise_far(m, box=(0.0, 1.0), centers=(), min_dist=0.12, seed=None):
    rng = np.random.default_rng(seed)
    noise_accum = []
    need = m
    while need > 0:
        batch = max(need * 2, 1000)
        cand = rng.random((batch, 2)) * (box[1]-box[0]) + box[0]
        if centers:
            c = np.array(centers)
            d2 = ((cand[:, None, :] - c[None, :, :])**2).sum(axis=2)
            ok = np.sqrt(d2.min(axis=1)) > min_dist
            cand = cand[ok]
        take = cand[:need]
        noise_accum.append(take)
        need -= take.shape[0]
    return np.vstack(noise_accum)

# =========================
# GENERACIÓN MÚLTIPLE
# =========================
for n_points in sizes:
    n_noise = int(n_points * noise_ratio)
    n_blobs = n_points - n_noise

    # 1) Clusters principales
    points, labels = make_blobs(
        n_samples=n_blobs,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    # 2) Ruido alejado
    noise = gen_noise_far(
        n_noise,
        box=box_ext,
        centers=centers,
        min_dist=min_dist,
        seed=random_state
    )
    labels_noise = -1 * np.ones(noise.shape[0], dtype=int)

    # 3) Mezcla y guarda
    all_points = np.vstack([points, noise])
    all_labels = np.concatenate([labels, labels_noise])

    # Barajar
    perm = np.random.default_rng(random_state).permutation(all_points.shape[0])
    all_points = all_points[perm]
    all_labels = all_labels[perm]

    # Archivos de salida
    out_xy   = out_dir / f"{n_points}_data.csv"
    out_full = out_dir / f"{n_points}_result.csv"

    # Guardar archivos
    np.savetxt(out_xy, all_points, delimiter=",", fmt="%.6f")
    np.savetxt(out_full, np.column_stack([all_points, all_labels]),
               delimiter=",", fmt=["%.6f","%.6f","%d"])

    print(f"✅ {n_points} puntos → {out_xy.name} (ruido: {n_noise})")

print(f"\n💾 Todos los datasets fueron guardados en: {out_dir.resolve()}")
