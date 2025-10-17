// dbscan_parallel_bench.cpp
#include <cstdio>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <climits>
#include <iomanip>
#include <omp.h>

namespace fs = std::filesystem;

// Estados (compatibles con scikit-learn)
enum { SIN_CLASIFICAR = -99, RUIDO = -1 };

struct Punto { double x, y; };

// ============================
// Vecinos (paralelo, 2 mitades, UNA región, orden [mitad0 | mitad1])
// ============================
int buscar_vecinos(Punto puntos[], int n, int idx, double eps2, int out[]) {
    const int T   = omp_get_max_threads(); // respeta omp_set_num_threads
    const int mid = n / 2;
    const double x0 = puntos[idx].x, y0 = puntos[idx].y;

    // Buffers por hilo (dos mitades)
    std::vector<int> counts0(T, 0), counts1(T, 0);
    std::vector<int> offs0(T, 0),   offs1(T, 0);
    int total0 = 0, total1 = 0;

    // Única región paralela
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int c0 = 0, c1 = 0;

        // ===== Fase 1: conteo por mitad =====
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < mid; ++j) {
            double dx = x0 - puntos[j].x;
            double dy = y0 - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) ++c0;
        }
        #pragma omp for schedule(static)
        for (int j = mid; j < n; ++j) {
            double dx = x0 - puntos[j].x;
            double dy = y0 - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) ++c1;
        }

        counts0[tid] = c0;
        counts1[tid] = c1;

        #pragma omp barrier

        // ===== Prefijos (offsets) =====
        #pragma omp single
        {
            for (int t = 0; t < T; ++t) { offs0[t] = total0; total0 += counts0[t]; }
            for (int t = 0; t < T; ++t) { offs1[t] = total1; total1 += counts1[t]; }
        }

        #pragma omp barrier

        // ===== Fase 2: escritura ordenada [mitad0 | mitad1] =====
        int pos0 = offs0[tid];
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < mid; ++j) {
            double dx = x0 - puntos[j].x;
            double dy = y0 - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[pos0++] = j;
        }

        int pos1 = total0 + offs1[tid];
        #pragma omp for schedule(static)
        for (int j = mid; j < n; ++j) {
            double dx = x0 - puntos[j].x;
            double dy = y0 - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[pos1++] = j;
        }
    }

    return total0 + total1;
}

// ============================
// Expansión de clúster (workspaces reutilizados + marca por época)
// ============================
int expandir_cluster(Punto puntos[], int n, int idxCore, int idCluster,
                     double eps2, int minPts, int etiquetas[],
                     int* vecinos, int* vecinos2, int* cola,
                     int* mark, int& epoch)
{
    int numVecinos = buscar_vecinos(puntos, n, idxCore, eps2, vecinos);
    if (numVecinos < minPts) {
        etiquetas[idxCore] = RUIDO;
        return 0;
    }

    etiquetas[idxCore] = idCluster;

    // Semilla (evita auto-encolar idxCore)
    int inicio = 0, fin = 0;
    for (int i = 0; i < numVecinos; ++i) {
        int q = vecinos[i];
        if (q == idxCore) continue;
        if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
            etiquetas[q] = idCluster;
            if (mark[q] != epoch) { mark[q] = epoch; cola[fin++] = q; }
        }
    }

    // Expansión BFS
    while (inicio < fin) {
        int p = cola[inicio++];
        int m = buscar_vecinos(puntos, n, p, eps2, vecinos2);
        if (m >= minPts) {
            for (int k = 0; k < m; ++k) {
                int q = vecinos2[k];
                if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
                    etiquetas[q] = idCluster;
                    if (mark[q] != epoch) { mark[q] = epoch; cola[fin++] = q; }
                }
            }
        }
    }

    // Avanza época (evita limpiar mark[])
    ++epoch;
    if (epoch == INT_MAX) {
        std::fill(mark, mark + n, 0);
        epoch = 1;
    }
    return 1;
}

// ============================
// DBSCAN (serial en control, paralelo en vecindad)
// ============================
int dbscan(Punto puntos[], int n, double eps, int minPts, int etiquetas[]) {
    std::fill(etiquetas, etiquetas + n, SIN_CLASIFICAR);

    double eps2 = eps * eps;
    int idCluster = 0;

    // Workspaces (reutilizados en toda la corrida)
    std::vector<int> vecinos(n), vecinos2(n), cola(n), mark(n, 0);
    int epoch = 1;

    for (int i = 0; i < n; ++i) {
        if (etiquetas[i] != SIN_CLASIFICAR) continue;
        if (expandir_cluster(puntos, n, i, idCluster, eps2, minPts, etiquetas,
                             vecinos.data(), vecinos2.data(), cola.data(),
                             mark.data(), epoch))
        {
            ++idCluster;
        }
    }
    return idCluster;
}

// ============================
// Lectura CSV simple x,y
// ============================
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
    in.close(); N = i;
    return (N > 0);
}

// ============================
// Utilidades
// ============================
static inline void mean_std_ms(const std::vector<double>& v, double& mean, double& stdev) {
    if (v.empty()) { mean = stdev = 0.0; return; }
    mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double sq = 0.0;
    for (double x : v) { double d = x - mean; sq += d*d; }
    stdev = (v.size() > 1) ? std::sqrt(sq / (v.size() - 1)) : 0.0;
}

// ============================
// MAIN — barrido completo en un solo programa
// ============================
int main(int argc, char** argv) {
    // Tamaños y hilos fijos del experimento
    const std::vector<int> tamanos  = {110005};
    const std::vector<int> hilos    = {12};
    const int REP = 1;

    // Ruta datasets (opcional por CLI, default "Datasets")
    std::string baseIn = (argc > 1) ? std::string(argv[1]) : "Datasets";
    if (baseIn.back() != '/' && baseIn.back() != '\\') baseIn += '/';

    // Hiperparámetros (puedes fijarlos o leerlos de CLI si quieres)
    const double eps    = 0.03;
    const int    minPts = 10;

    // CSV de salida
    std::string baseOut = baseIn + "results/";
    if (!fs::exists(baseOut)) fs::create_directories(baseOut);
    const std::string timingsPath = baseOut + "timings_parallel.csv";
    const bool nuevo = !fs::exists(timingsPath);
    std::ofstream tf(timingsPath, std::ios::app);
    if (!tf.is_open()) {
        std::fprintf(stderr, "No se pudo abrir/crear %s\n", timingsPath.c_str());
        return 1;
    }
    if (nuevo) tf << "n_points,threads,eps,minPts,iteraciones,mean_ms,std_ms\n";

    std::printf("=== Benchmark DBSCAN (dos mitades, 1 región paralela) ===\n");
    std::printf("Datasets: %s | eps=%.5f | minPts=%d | REP=%d\n", baseIn.c_str(), eps, minPts, REP);
    std::printf("Hilos: "); for (size_t i=0;i<hilos.size();++i) std::printf("%d%s", hilos[i], i+1<hilos.size()? ", ":"\n");

    for (int N : tamanos) {
        std::string input = baseIn + std::to_string(N) + "_data.csv";
        if (!fs::exists(input)) {
            std::fprintf(stderr, "⚠️  No existe: %s\n", input.c_str());
            continue;
        }

        Punto* P = nullptr; int n = 0;
        if (!leer_csv_xy(input.c_str(), P, n)) {
            std::fprintf(stderr, "❌ No pude leer/parsing: %s\n", input.c_str());
            continue;
        }

        std::printf("\n📂 Dataset %s (%d puntos)\n", input.c_str(), n);

        for (int th : hilos) {
            omp_set_dynamic(0);          // respeta exactamente 'th'
            omp_set_num_threads(th);

            int* etiquetas = new int[n];
            std::vector<double> tiempos_ms; tiempos_ms.reserve(REP);

            std::printf("  ▶️ Hilos=%d\n", th);
            for (int r = 0; r < REP; ++r) {
                double t0 = omp_get_wtime();
                int k = dbscan(P, n, eps, minPts, etiquetas);
                double t1 = omp_get_wtime();
                double ms = (t1 - t0) * 1000.0;
                tiempos_ms.push_back(ms);
                std::printf("     iter %2d: %.3f ms (clusters=%d)\n", r+1, ms, k);
            }

            double mean, stdev;
            mean_std_ms(tiempos_ms, mean, stdev);
            std::printf("     ➡️ promedio=%.3f ms | std=%.3f ms\n", mean, stdev);

            // Fila por combinación (tamaño × hilos)
            tf << N << "," << th << ","
               << std::fixed << std::setprecision(5) << eps << ","
               << minPts << "," << REP << ","
               << std::setprecision(3) << mean << "," << stdev << "\n";

            delete[] etiquetas;
        }

        delete[] P;
    }

    tf.close();
    std::printf("\n✅ Listo. Timings en: %s\n", timingsPath.c_str());
    return 0;
}
