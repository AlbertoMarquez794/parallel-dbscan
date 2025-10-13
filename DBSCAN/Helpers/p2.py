import numpy as np
import matplotlib.pyplot as plt

n_points = 4000
result = np.loadtxt(f"{n_points}_results.csv", delimiter=",")

plt.figure(figsize=(6,6))
plt.title("Detecting noise with my DBSCAN")

# Ruido (-2 en tu código C++) → amarillo, Clusters → morado
plt.scatter(result[:, 0], result[:, 1],
            c=np.where(result[:, 2] == -2, 0, 1),
            s=50)

plt.xticks([])
plt.yticks([])
plt.box(False)

plt.savefig("resultado_dbscan_mio.png", dpi=200)
print("✅ Imagen guardada como resultado_dbscan_mio.png")
