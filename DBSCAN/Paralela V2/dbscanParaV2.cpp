// dbscanParaV1.cpp
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
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
// Vecinos (rango completo [0..n), N hilos, sin std::vector)
// Fase 1: cada hilo cuenta sus matches en su tramo.
// Fase 2: cada hilo escribe en 'out' usando offsets (sin contención).
// ============================
int buscar_vecinos(Punto* __restrict puntos, int n, int idx,
                   double eps2, int* __restrict out) {
    const int T = omp_get_max_threads();   // número de hilos efectivos

    if (T <= 1) {
        // Fallback secuencial
        int c = 0;
        for (int j = 0; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[c++] = j;
        }
        return c;
    }

    // --- Fase 1: conteo por hilo ---
    int* counts = new int[T];
    for (int t = 0; t < T; ++t) counts[t] = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int local = 0;

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) ++local;
        }
        counts[tid] = local;
    }

    // --- Fase 2: calcular offsets (prefijos) ---
    int* offs = new int[T];
    int total = 0;
    for (int t = 0; t < T; ++t) { offs[t] = total; total += counts[t]; }

    // --- Fase 3: escribir en 'out' sin contención ---
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int pos = offs[tid];

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[pos++] = j;
        }
    }

    delete[] counts;
    delete[] offs;
    return total;
}

// ============================
// Expansión de clúster
// ============================
int expandir_cluster(Punto puntos[], int n, int idxCore, int idCluster,
                     double eps2, int minPts, int etiquetas[]) {
    int* vecinos = new int[n];
    int* cola    = new int[n];
    char* enCola = new char[n](); // inicializado en 0

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
// DBSCAN principal
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
// Lectura CSV (x,y)
// ============================
bool leer_csv_xy(const char* ruta, Punto*& puntos, int& N) {
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
// MAIN batch (procesa todos los tamaños conocidos)
// Uso: ./dbscanParaV1 <hilos> [eps] [minPts] [baseIn]
// ============================
int main(int argc, char** argv) {
    std::vector<int> tamanos = {110005};

    std::string baseIn  = (argc > 4)
        ? std::string(argv[4])
        : "/home/alberto/parallel-dbscan/DBSCAN/Paralela V2/Datasets/";
    if (baseIn.back() != '/' && baseIn.back() != '\\') baseIn += '/';

    std::string baseOut = baseIn + "results/";

    int num_hilos = (argc > 1) ? std::atoi(argv[1]) : omp_get_max_threads();
    double eps    = (argc > 2) ? std::atof(argv[2]) : 0.03;
    int    minPts = (argc > 3) ? std::atoi(argv[3]) : 10;

    omp_set_num_threads(num_hilos);

    if (!fs::exists(baseOut)) {
        if (!fs::create_directories(baseOut)) {
            std::fprintf(stderr, "No se pudo crear el directorio de salida: %s\n", baseOut.c_str());
            return 1;
        }
    }

    std::printf("=== DBSCAN paralelo (rango completo) ===\n");
    std::printf("Usando %d hilos | eps = %.5f | minPts = %d\n", num_hilos, eps, minPts);
    std::printf("Entrada: %s\nSalida:  %s\n", baseIn.c_str(), baseOut.c_str());

    for (int N : tamanos) {
        std::string input  = baseIn  + std::to_string(N) + "_data.csv";
        std::string output = baseOut + std::to_string(N) + "_results_" + std::to_string(num_hilos) + ".csv";

        if (!fs::exists(input)) {
            std::fprintf(stderr, "⚠️  No existe: %s\n", input.c_str());
            continue;
        }

        Punto* P = nullptr; int n = 0;
        if (!leer_csv_xy(input.c_str(), P, n)) {
            std::fprintf(stderr, "❌ No pude leer/parsing: %s\n", input.c_str());
            continue;
        }

        std::printf("\n📂 Procesando %s (%d puntos)...\n", input.c_str(), n);

        int* etiquetas = new int[n];
        double t0 = omp_get_wtime();
        int k = dbscan(P, n, eps, minPts, etiquetas);
        double t1 = omp_get_wtime();

        std::printf("✔ Clústeres: %d | Tiempo: %.3f ms\n", k, (t1 - t0)*1000.0);

        std::FILE* f = std::fopen(output.c_str(), "w");
        if (!f) {
            std::perror("No se pudo crear el archivo de salida");
            delete[] etiquetas; delete[] P;
            continue;
        }

        for (int i = 0; i < n; ++i)
            std::fprintf(f, "%.6f,%.6f,%d\n", P[i].x, P[i].y, etiquetas[i]);
        std::fclose(f);

        std::printf("💾 Guardado → %s\n", output.c_str());

        delete[] etiquetas;
        delete[] P;
    }

    std::printf("\n✅ Procesamiento completo para todos los tamaños.\n");
    return 0;
}
