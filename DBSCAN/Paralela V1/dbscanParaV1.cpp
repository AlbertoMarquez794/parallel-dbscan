// dbscan_parallel.cpp
#include <cstdio>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <cstdlib>
#include <vector>
#include <cstring>   // memcpy
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
// Vecinos (2 mitades, N hilos, sin std::vector)
// Fase 1: cada hilo cuenta vecinos en su porción de cada mitad.
// Fase 2: cada hilo escribe en 'out' con offsets (sin contención).
// Orden final determinista: [mitad0 | mitad1].
// ============================
int buscar_vecinos(Punto puntos[], int n, int idx, double eps2, int out[]) {
    const int T = omp_get_max_threads();   // hilos efectivos (respeta omp_set_num_threads)
    const int mid = n / 2;

    // --- Fase 1: conteo por hilo en cada mitad ---
    int* counts0 = new int[T];  // mitad 0
    int* counts1 = new int[T];  // mitad 1
    for (int t = 0; t < T; ++t) counts0[t] = counts1[t] = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int c0 = 0, c1 = 0;

        // Mitad 0: [0, mid)
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < mid; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) ++c0;
        }

        // Mitad 1: [mid, n)
        #pragma omp for schedule(static)
        for (int j = mid; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) ++c1;
        }

        counts0[tid] = c0;
        counts1[tid] = c1;
    }

    // Prefijos (offsets) para escribir sin contención
    int* offs0 = new int[T];
    int* offs1 = new int[T];
    int total0 = 0, total1 = 0;
    for (int t = 0; t < T; ++t) { offs0[t] = total0; total0 += counts0[t]; }
    for (int t = 0; t < T; ++t) { offs1[t] = total1; total1 += counts1[t]; }
    const int base1 = total0;  // mitad 1 inicia después de mitad 0

    // --- Fase 2: escritura en 'out' usando offsets ---
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int pos0 = offs0[tid];

        // Mitad 0
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < mid; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[pos0++] = j;
        }

        int pos1 = base1 + offs1[tid];

        // Mitad 1
        #pragma omp for schedule(static)
        for (int j = mid; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[pos1++] = j;
        }
    }

    delete[] counts0; delete[] counts1; delete[] offs0; delete[] offs1;
    return total0 + total1;
}

// ============================
// Expansión de clúster (igual estructura que serial)
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
// DBSCAN (estructura serial; vecindad paralela)
// ============================
int dbscan(Punto puntos[], int n, double eps, int minPts, int etiquetas[]) {
    for (int i = 0; i < n; ++i) etiquetas[i] = SIN_CLASIFICAR;

    double eps2 = eps * eps;
    int idCluster = 0; // 0,1,2,...

    // Recorrido serial de semillas; la parte pesada (vecindad) va paralela
    for (int i = 0; i < n; ++i) {
        if (etiquetas[i] != SIN_CLASIFICAR) continue;
        if (expandir_cluster(puntos, n, i, idCluster, eps2, minPts, etiquetas))
            ++idCluster;
    }
    return idCluster;
}

// ============================
// Lectura CSV simple x,y
// ============================
bool leer_csv_xy(const char* ruta, Punto*& puntos, int& N) {
    std::ifstream in(ruta);
    if (!in.is_open()) {
        std::perror(("fopen fallo en '" + std::string(ruta) + "'").c_str());
        return false;
    }
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
    if (N == 0) {
        std::fprintf(stderr, "leer_csv_xy: archivo '%s' abrió OK pero no se parseó ninguna fila\n", ruta);
        return false;
    }
    return true;
}


// ============================
// MAIN
// Uso: ./dbscan_parallel <input.csv> <num_hilos> [eps] [minPts]
// ============================
int main(int argc, char** argv) {
    // --- Configuración ---
    std::vector<int> tamanos = {110005};

    // baseIn por CLI opcional (arg4). Si no se da, usa la ruta por defecto:
    // ./dbscanParaV1 <hilos> [eps] [minPts] [baseIn]
    std::string baseIn  = (argc > 4)
        ? std::string(argv[4])
        : "/home/albertomarquez/parallel-dbscan/DBSCAN/Paralela V1/Datasets/";
    if (baseIn.back() != '/' && baseIn.back() != '\\') baseIn += '/';

    std::string baseOut = baseIn + "results/";

    int num_hilos = (argc > 1) ? std::atoi(argv[1]) : omp_get_max_threads();
    double eps    = (argc > 2) ? std::atof(argv[2]) : 0.03;
    int    minPts = (argc > 3) ? std::atoi(argv[3]) : 10;

    omp_set_num_threads(num_hilos);

    // Crear carpeta de salida (si no existe)
    if (!fs::exists(baseOut)) {
        if (!fs::create_directories(baseOut)) {
            std::fprintf(stderr, "No se pudo crear el directorio de salida: %s\n", baseOut.c_str());
            return 1;
        }
    }

    std::printf("=== DBSCAN paralelo (matriz dividida en 2) ===\n");
    std::printf("Usando %d hilos | eps = %.5f | minPts = %d\n", num_hilos, eps, minPts);
    std::printf("Entrada: %s\nSalida:  %s\n", baseIn.c_str(), baseOut.c_str());

    for (int N : tamanos) {
        std::string input  = baseIn  + std::to_string(N) + "_data.csv";
        std::string output = baseOut + std::to_string(N) + "_results_" + std::to_string(num_hilos) + ".csv";

        // Verifica existencia antes de leer
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

        std::printf("✔ Clústeres: %d | Tiempo: %.3f ms\n", k, (t1 - t0) * 1000.0);

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