// dbscan_parallel_batch.cpp
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

// Estados (compatibles con scikit-learn)
enum { SIN_CLASIFICAR = -99, RUIDO = -1 };

struct Punto { double x, y; };

inline double distancia2(const Punto& a, const Punto& b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;
}

// ============================
// Vecinos dentro de eps (PARALELO, 2 fases, sin std::vector)
// ============================
int buscar_vecinos(Punto puntos[], int n, int idx, double eps2, int out[]) {
    const int T = omp_get_max_threads();
    int* counts  = new int[T];
    for (int t = 0; t < T; ++t) counts[t] = 0;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int local = 0;

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) ++local;
        }
        counts[tid] = local;
    }

    int* offsets = new int[T];
    int total = 0;
    for (int t = 0; t < T; ++t) { offsets[t] = total; total += counts[t]; }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int pos = offsets[tid];

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[pos++] = j;
        }
    }

    delete[] counts;
    delete[] offsets;
    return total;
}

// ============================
// Expansión de clúster (misma estructura que serial)
// ============================
int expandir_cluster(Punto puntos[], int n, int idxCore, int idCluster,
                     double eps2, int minPts, int etiquetas[]) {
    int* vecinos = new int[n];
    int* cola    = new int[n];
    char* enCola = new char[n](); // 0-inicializado

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

// ============================
// DBSCAN (misma estructura; vecindad ya paralela)
// ============================
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

// ============================
// Lectura CSV x,y
// ============================
bool leer_csv_xy(const fs::path& ruta, Punto*& puntos, int& N) {
    std::ifstream in(ruta);
    if (!in.is_open()) return false;

    std::string line;
    int cuenta = 0;
    while (std::getline(in, line)) {
        bool vacia = true;
        for (char ch : line)
            if (ch!=' ' && ch!='\t' && ch!='\r' && ch!='\n') { vacia=false; break; }
        if (!vacia) ++cuenta;
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
    in.close(); N = i;
    return (N > 0);
}

// ============================
// Guardar resultados x,y,label
// ============================
bool guardar_csv(const fs::path& ruta, const Punto* P, const int* etiquetas, int N) {
    std::FILE* f = std::fopen(ruta.string().c_str(), "w");
    if (!f) return false;
    for (int i = 0; i < N; ++i)
        std::fprintf(f, "%.6f,%.6f,%d\n", P[i].x, P[i].y, etiquetas[i]);
    std::fclose(f);
    return true;
}

// ============================
// MAIN
// Uso: ./dbscan_parallel_batch "<inDir>" "<outDir>" <hilos> [eps] [minPts]
//
// Ejemplo:
// ./dbscan_parallel_batch "/home/.../Paralela V1/Datasets" "/home/.../Paralela V1/Results_8h" 8 0.03 10
// ============================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Uso: %s <inDir> <outDir> <hilos> [eps] [minPts]\n", argv[0]);
        return 1;
    }
    fs::path inDir  = fs::path(argv[1]);
    fs::path outDir = fs::path(argv[2]);
    int num_hilos   = std::atoi(argv[3]);
    double eps      = (argc > 4) ? std::atof(argv[4]) : 0.03;
    int    minPts   = (argc > 5) ? std::atoi(argv[5]) : 10;

    omp_set_num_threads(num_hilos);
    fs::create_directories(outDir);

    // Tamaños fijos solicitados
    const int sizes[] = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};
    const int NSIZES  = sizeof(sizes)/sizeof(sizes[0]);

    for (int idx = 0; idx < NSIZES; ++idx) {
        int npts = sizes[idx];
        fs::path inFile  = inDir  / (std::to_string(npts) + "_data.csv");
        fs::path outFile = outDir / (std::to_string(npts) + "_results_" + std::to_string(num_hilos) + ".csv");

        Punto* P = nullptr; int N = 0;
        if (!leer_csv_xy(inFile, P, N)) {
            std::fprintf(stderr, "❌ No pude leer %s\n", inFile.string().c_str());
            continue;
        }

        int* etiquetas = new int[N];

        double t0 = omp_get_wtime();
        int k = dbscan(P, N, eps, minPts, etiquetas);
        double t1 = omp_get_wtime();

        std::printf("N=%d | hilos=%d | eps=%.3f | minPts=%d | clusters=%d | tiempo=%.2f ms\n",
                    N, num_hilos, eps, minPts, k, (t1 - t0)*1000.0);

        if (!guardar_csv(outFile, P, etiquetas, N)) {
            std::fprintf(stderr, "⚠️  No se pudo escribir %s\n", outFile.string().c_str());
        } else {
            std::printf("💾 %s\n", outFile.string().c_str());
        }

        delete[] etiquetas;
        delete[] P;
    }

    std::puts("✅ Lote completado.");
    return 0;
}
