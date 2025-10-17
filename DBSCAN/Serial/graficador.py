# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# =========================
# Configuración de rutas
# =========================
base_dir   = Path(__file__).resolve().parent                 # carpeta donde está este script
in_dir     = base_dir / "Datasets" / "results"               # de donde lee los CSV (######_results.csv)
out_dir    = base_dir / "Serial" / "img" / "cpp"             # a donde guarda las imágenes
out_dir.mkdir(parents=True, exist_ok=True)

# Tamaños / archivos a procesar
sizes = [110005]

# Colores: clusters (amarillo) vs ruido (morado)
cmap = ListedColormap(["#FFD700", "#800080"])

for n_points in sizes:
    in_csv  = in_dir / f"{n_points}_results.csv"
    out_png = out_dir / f"{n_points}_results_cpp_plot.png"

    if not in_csv.exists():
        print(f"⚠️  No encontré {in_csv}. Lo salto.")
        continue

    # =========================
    # 1) Leer resultados C++: x,y,label
    # =========================
    data = np.loadtxt(in_csv, delimiter=",")
    x, y = data[:, 0], data[:, 1]
    labels = data[:, 2].astype(int)

    # Normalización: considera ruido todo label negativo (por compatibilidad)
    is_noise = labels < 0
    cidx = is_noise.astype(int)  # 1 = ruido, 0 = cluster

    # =========================
    # 2) Graficar
    # =========================
    plt.figure(figsize=(6, 6))
    plt.title(f"DBSCAN (C++) — {n_points} puntos")
    # Para grandes N, s pequeño y alpha bajo para rendimiento/legibilidad
    plt.scatter(x, y, c=cidx, cmap=cmap, s=1, alpha=0.8, rasterized=True)
    plt.xticks([]); plt.yticks([]); plt.box(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{n_points}: guardado {out_png}")

print("Listo: imágenes en Serial/img/cpp/")
