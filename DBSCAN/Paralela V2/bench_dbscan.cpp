// bench_dbscan.cpp
// Víctor — Benchmark DBSCAN serial/paralelo (OpenMP), multi-tamaño y multi-hilos.
// Compilar (g++):  g++ -O3 -march=native -fopenmp bench_dbscan.cpp -o bench_dbscan
// Uso:
//   ./bench_dbscan [--eps 0.03] [--minpts 10] [--iters 10]
//                  [--in "/ruta/Datasets/"] [--out "/ruta/Datasets/results/"]
//                  [--threads "1,2,4,6,8"] [--sizes "20000,40000,180000,200000"]
//                  [--save 0|1]
//
// Ejemplo:
//   ./bench_dbscan --eps 0.03 --minpts 10 --iters 10 \
//     --in "/home/alberto/parallel-dbscan/DBSCAN/Paralela V2/Datasets/" \
//     --out "/home/alberto/parallel-dbscan/DBSCAN/Paralela V2/Datasets/results/" \
//     --threads "1,6" --sizes "20000,40000,180000,200000" --save 0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

// ============================
// Config & util
// ============================
enum { SIN_CLASIFICAR = -99, RUIDO = -1 };
struct Punto { double x, y; };

static inline double distancia2(const Punto& a, const Punto& b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;
}

static std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::atoi(tok.c_str()));
    return v;
}

static void ensure_trailing_slash(std::string& p) {
    if (p.empty()) return;
    char c = p.back();
    if (c!='/' && c!='\\') p.push_back('/');
}

// ============================
// Vecinos (paralelo seguro, sin contención)
// ============================
int buscar_vecinos(Punto* __restrict puntos, int n, int idx,
                   double eps2, int* __restrict out) {
    const int T = omp_get_max_threads();

    if (T <= 1) {
        int c = 0;
        for (int j = 0; j < n; ++j) {
            double dx = puntos[idx].x - puntos[j].x;
            double dy = puntos[idx].y - puntos[j].y;
            if (dx*dx + dy*dy <= eps2) out[c++] = j;
        }
        return c;
    }

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

    int* offs = new int[T];
    int total = 0;
    for (int t = 0; t < T; ++t) { offs[t] = total; total += counts[t]; }

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

    delete[] counts; delete[] offs;
    return total;
}

// ============================
// Expansión iterativa (sin recursión)
// ============================
int expandir_cluster(Punto puntos[], int n, int idxCore, int idCluster,
                     double eps2, int minPts, int etiquetas[]) {
    int* vecinos = new int[n];
    int* cola    = new int[n];
    char* enCola = new char[n](); // 0-initialized

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
// IO helpers
// ============================
bool leer_csv_xy(const std::string& ruta, std::vector<Punto>& puntos) {
    std::ifstream in(ruta);
    if (!in.is_open()) return false;
    std::string line;
    puntos.clear();
    puntos.reserve(1024);
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        for (char& ch : line) if (ch == ',') ch = ' ';
        double x, y;
        if (std::sscanf(line.c_str(), "%lf %lf", &x, &y) == 2)
            puntos.push_back({x, y});
    }
    return !puntos.empty();
}

bool guardar_labels_csv(const std::string& ruta, const std::vector<Punto>& P, const std::vector<int>& lab) {
    std::FILE* f = std::fopen(ruta.c_str(), "w");
    if (!f) return false;
    for (size_t i = 0; i < P.size(); ++i)
        std::fprintf(f, "%.6f,%.6f,%d\n", P[i].x, P[i].y, lab[i]);
    std::fclose(f);
    return true;
}

struct Stats { double mean=0, stdev=0, minv=0, maxv=0; };

static Stats stats(const std::vector<double>& v) {
    Stats s; if (v.empty()) return s;
    s.minv = *std::min_element(v.begin(), v.end());
    s.maxv = *std::max_element(v.begin(), v.end());
    double sum=0; for (double x: v) sum+=x;
    s.mean = sum / v.size();
    double var=0; for (double x: v) { double d=x - s.mean; var += d*d; }
    s.stdev = (v.size()>1)? std::sqrt(var/(v.size()-1)) : 0.0;
    return s;
}

// ============================
// CLI parsing
// ============================
struct Args {
    double eps = 0.03;
    int minPts = 10;
    int iters = 10;
    std::string in = "/home/alberto/parallel-dbscan/DBSCAN/Paralela V2/Datasets/";
    std::string out = "/home/alberto/parallel-dbscan/DBSCAN/Paralela V2/Datasets/results/";
    std::vector<int> threads = {1, 2, 4, 6, 8};
    std::vector<int> sizes   = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};
    int save = 0;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;i++) {
        std::string k = argv[i];
        auto need = [&](int more){ if(i+more>=argc){std::fprintf(stderr,"Falta valor para %s\n",k.c_str()); std::exit(2);} };
        if (k=="--eps") { need(1); a.eps = std::atof(argv[++i]); }
        else if (k=="--minpts") { need(1); a.minPts = std::atoi(argv[++i]); }
        else if (k=="--iters") { need(1); a.iters = std::atoi(argv[++i]); }
        else if (k=="--in") { need(1); a.in = argv[++i]; }
        else if (k=="--out") { need(1); a.out = argv[++i]; }
        else if (k=="--threads") { need(1); a.threads = parse_csv_ints(argv[++i]); }
        else if (k=="--sizes") { need(1); a.sizes = parse_csv_ints(argv[++i]); }
        else if (k=="--save") { need(1); a.save = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr,"Parámetro no reconocido: %s\n", k.c_str());
            std::exit(2);
        }
    }
    ensure_trailing_slash(a.in);
    ensure_trailing_slash(a.out);
    return a;
}

// ============================
// MAIN (benchmark)
// ============================
int main(int argc, char** argv) {
    Args A = parse_args(argc, argv);

    if (!fs::exists(A.out)) {
        if (!fs::create_directories(A.out)) {
            std::fprintf(stderr, "❌ No se pudo crear dir salida: %s\n", A.out.c_str());
            return 1;
        }
    }

    // CSV resumen
    const std::string resumen_csv = A.out + "resumen_benchmark.csv";
    std::FILE* fout = std::fopen(resumen_csv.c_str(), "w");
    if (!fout) {
        std::fprintf(stderr, "❌ No puedo crear resumen CSV: %s\n", resumen_csv.c_str());
        return 1;
    }
    std::fprintf(fout, "Puntos,Epsilon,MinPts,Hilos,Iteraciones,Promedio_ms,StdDev_ms,Min_ms,Max_ms,Clusters_ultima\n");

    std::printf("=== Benchmark DBSCAN (iterativo, OpenMP) ===\n");
    std::printf("Datasets en: %s\nResultados en: %s\n", A.in.c_str(), A.out.c_str());
    std::printf("eps=%.5f | minPts=%d | iters=%d | save=%d\n", A.eps, A.minPts, A.iters, A.save);
    std::printf("Hilos: "); for (size_t i=0;i<A.threads.size();++i) std::printf("%d%s", A.threads[i], (i+1<A.threads.size())?", ":"\n");
    std::printf("Tamaños: "); for (size_t i=0;i<A.sizes.size();++i) std::printf("%d%s", A.sizes[i], (i+1<A.sizes.size())?", ":"\n");

    for (int N : A.sizes) {
        const std::string input = A.in + std::to_string(N) + "_data.csv";
        if (!fs::exists(input)) {
            std::fprintf(stderr, "⚠️  No existe dataset: %s\n", input.c_str());
            continue;
        }

        // Cargar dataset
        std::vector<Punto> P;
        if (!leer_csv_xy(input, P)) {
            std::fprintf(stderr, "❌ No pude leer/parsing: %s\n", input.c_str());
            continue;
        }
        const int n = (int)P.size();
        std::printf("\n📂 %s  (%d puntos)\n", input.c_str(), n);

        for (int thr : A.threads) {
            omp_set_num_threads(thr);
            std::vector<double> tiempos_ms; tiempos_ms.reserve(A.iters);
            std::vector<int> etiquetas(n);
            int clusters_ultima = 0;

            // Warm-up (no medir)
            {
                clusters_ultima = dbscan(P.data(), n, A.eps, A.minPts, etiquetas.data());
            }

            // Iteraciones medidas
            for (int it = 0; it < A.iters; ++it) {
                double t0 = omp_get_wtime();
                clusters_ultima = dbscan(P.data(), n, A.eps, A.minPts, etiquetas.data());
                double t1 = omp_get_wtime();
                tiempos_ms.push_back((t1 - t0) * 1000.0);
            }

            // Stats
            Stats S = stats(tiempos_ms);
            std::printf("  [thr=%2d] iters=%d  avg=%.3f ms  sd=%.3f  min=%.3f  max=%.3f  | clusters=%d\n",
                        thr, A.iters, S.mean, S.stdev, S.minv, S.maxv, clusters_ultima);

            // Guardar labels del último run (opcional)
            if (A.save) {
                const std::string out_lbl = A.out + std::to_string(N) + "_results_thr" + std::to_string(thr) + ".csv";
                std::vector<int> last_labels = etiquetas; // ya contiene el último run
                if (guardar_labels_csv(out_lbl, P, last_labels)) {
                    std::printf("    💾 labels → %s\n", out_lbl.c_str());
                } else {
                    std::fprintf(stderr, "    ⚠️ No pude guardar labels: %s\n", out_lbl.c_str());
                }
            }

            // Escribir fila al resumen
            std::fprintf(fout, "%d,%.5f,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%d\n",
                         N, A.eps, A.minPts, thr, A.iters, S.mean, S.stdev, S.minv, S.maxv, clusters_ultima);
            std::fflush(fout);
        }
    }

    std::fclose(fout);
    std::printf("\n✅ Listo. Resumen: %s\n", resumen_csv.c_str());
    return 0;
}
