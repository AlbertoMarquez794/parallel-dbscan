#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;

// Estados (compatibles con scikit-learn)
enum {
    SIN_CLASIFICAR = -99,
    RUIDO          = -1
};

struct Punto { double x, y; };

inline double distancia2(const Punto& a, const Punto& b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;
}

int buscar_vecinos(Punto puntos[], int n, int idx, double eps2, int out[]) {
    int c = 0;
    for (int j = 0; j < n; ++j)
        if (distancia2(puntos[idx], puntos[j]) <= eps2)
            out[c++] = j;
    return c;
}

int expandir_cluster(Punto puntos[], int n, int idxCore, int idCluster,
                     double eps2, int minPts, int etiquetas[])
{
    int* vecinos = new int[n];
    int* cola    = new int[n];
    char* enCola = new char[n](); // inicializa en 0

    int numVecinos = buscar_vecinos(puntos, n, idxCore, eps2, vecinos);
    if (numVecinos < minPts) {
        etiquetas[idxCore] = RUIDO;
        delete[] vecinos; delete[] cola; delete[] enCola;
        return 0;
    }

    etiquetas[idxCore] = idCluster;

    int inicio = 0, fin = 0;
    for (int i = 0; i < numVecinos; ++i) {
        int q = vecinos[i];
        if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
            etiquetas[q] = idCluster;
            if (!enCola[q]) { cola[fin++] = q; enCola[q] = 1; }
        }
    }

    int* vecinos2 = new int[n];
    while (inicio < fin) {
        int p = cola[inicio++];
        int m = buscar_vecinos(puntos, n, p, eps2, vecinos2);
        if (m >= minPts) {
            for (int k = 0; k < m; ++k) {
                int q = vecinos2[k];
                if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
                    etiquetas[q] = idCluster;
                    if (!enCola[q]) { cola[fin++] = q; enCola[q] = 1; }
                }
            }
        }
    }

    delete[] vecinos; delete[] vecinos2; delete[] cola; delete[] enCola;
    return 1;
}

int dbscan(Punto puntos[], int n, double eps, int minPts, int etiquetas[]) {
    for (int i = 0; i < n; ++i) etiquetas[i] = SIN_CLASIFICAR;

    double eps2 = eps * eps;
    int idCluster = 0;
    for (int i = 0; i < n; ++i) {
        if (etiquetas[i] != SIN_CLASIFICAR) continue;
        if (expandir_cluster(puntos, n, i, idCluster, eps2, minPts, etiquetas))
            ++idCluster;
    }
    return idCluster;
}

// ---- Lectura CSV simple x,y ----
bool leer_csv_xy(const char* ruta, Punto*& puntos, int& N) {
    std::ifstream in(ruta);
    if (!in.is_open()) return false;
    std::string line;
    int cuenta = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.find_first_not_of(" \t\r\n") != std::string::npos)
            ++cuenta;
    }
    in.clear(); in.seekg(0);
    puntos = new Punto[cuenta];
    int i = 0; double x,y;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        for (char& ch : line) if (ch == ',') ch = ' ';
        if (std::sscanf(line.c_str(), "%lf %lf", &x, &y) == 2)
            puntos[i++] = {x,y};
    }
    N = i;
    return (N > 0);
}

// ======= MAIN =======
int main() {
    std::vector<int> tamanos = {110005};
    std::string inDir  = "Datasets/";
    std::string outDir = "Datasets/results/";

    if (!fs::exists(outDir)) fs::create_directories(outDir);

    double eps = 0.03;
    int    minPts = 10;
    const int REP = 2;

    std::string archivoTimings = outDir + "timings.csv";
    bool nuevo = !fs::exists(archivoTimings);
    std::ofstream fout(archivoTimings, std::ios::app);
    if (!fout.is_open()) {
        std::perror("No se pudo crear timings.csv");
        return 1;
    }
    if (nuevo) fout << "n_points,eps,minPts,iteraciones,mean_ms,std_ms\n";

    for (int n_points : tamanos) {
        std::string archivoEntrada = inDir + std::to_string(n_points) + "_data.csv";

        Punto* P = nullptr; int N = 0;
        if (!leer_csv_xy(archivoEntrada.c_str(), P, N)) {
            std::fprintf(stderr, "❌ No pude leer: %s\n", archivoEntrada.c_str());
            continue;
        }

        int* etiquetas = new int[N];
        std::vector<double> tiempos(REP);

        std::printf("Procesando %s (%d puntos)...\n", archivoEntrada.c_str(), N);
        for (int r = 0; r < REP; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            int k = dbscan(P, N, eps, minPts, etiquetas);
            auto t1 = std::chrono::steady_clock::now();
            tiempos[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("  Iteración %2d: %.3f ms (clusters=%d)\n", r+1, tiempos[r], k);
        }

        double sum = std::accumulate(tiempos.begin(), tiempos.end(), 0.0);
        double mean = sum / REP;
        double sq = 0.0;
        for (double t : tiempos) sq += (t - mean)*(t - mean);
        double stdev = std::sqrt(sq / (REP - 1));

        std::printf("➡️ Promedio: %.3f ms | Desv. estándar: %.3f ms\n\n", mean, stdev);

        fout << n_points << "," << std::fixed << std::setprecision(5)
             << eps << "," << minPts << "," << REP << ","
             << std::setprecision(3) << mean << "," << stdev << "\n";

        delete[] etiquetas;
        delete[] P;
    }

    fout.close();
    std::printf("✅ Ejecución completa. Timings guardados en %s\n", archivoTimings.c_str());
    return 0;
}
