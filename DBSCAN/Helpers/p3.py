# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# =========================
# Config
# =========================
N_POINTS     = 4000
CENTERS      = 4
CLUSTER_STD  = 0.06
RANDOM_STATE = 11
CENTER_BOX   = (0.0, 1.0)

DATA_CSV          = f"{N_POINTS}_data.csv"              # input para tu C++
RESULTS_CSV_CPP   = f"{N_POINTS}_results.csv"           # salida de tu C++ (x,y,cluster_id)
RESULTS_CSV_SK    = f"{N_POINTS}_results_sklearn.csv"   # salida scikit-learn (x,y,cluster_id)

IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)

# =========================
# Utilidades
# =========================
def gen_and_save_sample(n_points=N_POINTS):
    points, _ = make_blobs(
        n_samples=n_points,
        centers=CENTERS,
        cluster_std=CLUSTER_STD,
        random_state=RANDOM_STATE,
        center_box=CENTER_BOX
    )
    # mismo post: invertir columnas, abs, 3 decimales
    points = np.round(np.abs(points[:, ::-1]), 3)
    np.savetxt(DATA_CSV, points, delimiter=",", fmt="%.3f")
    return points

def save_plot(fig, title):
    clean = (
        title.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )
    filename = IMG_DIR / f"{N_POINTS}_{clean}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"💾 Imagen guardada: {filename}")

def plot_xyc(x, y, c, title):
    # Mismo estilo del “segundo código”: c = cluster_id directamente.
    fig = plt.figure()
    plt.title(title)
    plt.scatter(x, y, c=c, s=50)
    plt.xticks([]); plt.yticks([]); plt.box(False)
    save_plot(fig, title)
    plt.close(fig)

# =========================
# Flujo
# =========================
def main():
    # 1) Generar sample y guardar CSV
    points = gen_and_save_sample(N_POINTS)

    # 2) DBSCAN scikit-learn -> guardar como x,y,cluster_id (igual que tu C++)
    sk_labels = DBSCAN(eps=0.03, min_samples=10).fit_predict(points)
    sk_out = np.column_stack([points[:, 0], points[:, 1], sk_labels])
    np.savetxt(RESULTS_CSV_SK, sk_out, delimiter=",", fmt=["%.3f","%.3f","%d"])

    # 3) Graficar scikit-learn en el mismo esquema que tu segundo código
    plot_xyc(points[:, 0], points[:, 1], sk_labels, "sklearn_dbscan_clusters")

    # 4) Cargar resultados de tu C++ (mismo formato x,y,cluster_id) y graficar
    cpp_path = Path(RESULTS_CSV_CPP)
    if cpp_path.exists():
        cpp = np.loadtxt(cpp_path.as_posix(), delimiter=",")
        if cpp.ndim == 1:
            cpp = cpp.reshape(1, -1)
        if cpp.shape[1] < 3:
            raise ValueError(f"{RESULTS_CSV_CPP} debe tener 3 columnas: x,y,cluster_id")
        x, y, c = cpp[:, 0], cpp[:, 1], cpp[:, 2].astype(int)
        plot_xyc(x, y, c, "cpp_dbscan_clusters")
    else:
        print(f"⚠️ No encontré {RESULTS_CSV_CPP}. Generé y guardé {RESULTS_CSV_SK} y su imagen.")

if __name__ == "__main__":
    main()
