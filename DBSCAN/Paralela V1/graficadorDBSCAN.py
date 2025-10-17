# -*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Config ===
data_csv = "4000_data.csv"      # el MISMO CSV que usó C++
eps = 0.03
min_samples = 10
img_dir = Path("images"); img_dir.mkdir(exist_ok=True)

# 1) Cargar datos
X = np.loadtxt(data_csv, delimiter=",")

# 2) DBSCAN (sklearn)
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)  # -1 = ruido

# 3) Guardar resultados (x,y,label)
out_csv = "4000_predicted.csv"
np.savetxt(out_csv, np.c_[X, labels], delimiter=",", fmt="%.6f")
print(f"✅ Predicción guardada en: {out_csv}")

# 4) Graficar y guardar imagen
colors = np.where(labels == -1, "#800080", "#FFD700")  # ruido morado, clusters amarillo
plt.figure()
plt.title("DBSCAN (con mi algoritmo de CPP)")
plt.scatter(X[:,0], X[:,1], c=colors, s=10)
plt.xticks([]); plt.yticks([]); plt.box(False)
out_img = img_dir / "4000_prediction_sklearn.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
plt.close()
print(f"🖼️ Imagen guardada en: {out_img}")
