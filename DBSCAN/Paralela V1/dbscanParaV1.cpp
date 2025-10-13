#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <omp.h>

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

// ---- Lectura CSV simple x,y ----
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
        if (std::sscanf(line.c_str(), "%lf %lf", &x, &y) == 2) puntos[i++] = {x,y};
    }
    in.close(); N = i;
    return (N > 0);
}

// Expansión de clúster usando vecindad precomputada
int expandir_cluster_con_vecinos(const std::vector<std::vector<int>>& vecinos,
                                 int idxCore, int idCluster, int minPts,
                                 int etiquetas[])
{
    const auto& Vcore = vecinos[idxCore];
    if ((int)Vcore.size() < minPts) {
        etiquetas[idxCore] = RUIDO; // no es núcleo
        return 0;
    }

    etiquetas[idxCore] = idCluster;

    // Semillas (BFS)
    const int n = (int)vecinos.size();
    std::vector<int> cola;  cola.reserve(n);
    std::vector<char> enCola(n, 0);

    // Encolar vecinos del núcleo
    for (int q : Vcore) {
        if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
            etiquetas[q] = idCluster;
            if (!enCola[q]) { cola.push_back(q); enCola[q] = 1; }
        }
    }

    // BFS
    for (size_t pos = 0; pos < cola.size(); ++pos) {
        int p = cola[pos];
        const auto& Vp = vecinos[p];
        if ((int)Vp.size() >= minPts) {
            for (int q : Vp) {
                if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
                    etiquetas[q] = idCluster;
                    if (!enCola[q]) { cola.push_back(q); enCola[q] = 1; }
                }
            }
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    // Args:
    //  argv[1] -> archivo CSV (opcional, default "4000_data.csv")
    //  argv[2] -> numero de hilos (opcional, default = omp_get_max_threads())
    const char* archivo = (argc > 1) ? argv[1] : "4000_data.csv";
    int num_hilos = (argc > 2) ? std::atoi(argv[2]) : 0;
    if (num_hilos > 0) omp_set_num_threads(num_hilos);

    // Parámetros DBSCAN (ajústalos si quieres desde CLI también)
    double eps   = 0.03;
    int    minPts = 10;

    Punto* P = nullptr; int N = 0;
    if (!leer_csv_xy(archivo, P, N)) {
        std::fprintf(stderr, "No pude abrir/leer el archivo: %s\n", archivo);
        return 1;
    }

    if (num_hilos <= 0) num_hilos = omp_get_max_threads();
    std::printf("Usando %d hilos (OpenMP)\n", num_hilos);
    std::printf("N = %d, eps = %.5f, minPts = %d\n", N, eps, minPts);

    // ==============================
    // 1) Precomputar vecindades O(n^2) en paralelo
    //    "Matriz indivisible": construimos TODAS las listas de vecinos
    // ==============================
    double eps2 = eps * eps;
    std::vector<std::vector<int>> vecinos(N);

    // Cada i escribe SOLO en vecinos[i], por lo que no hay carreras
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        // Reserva aproximada para evitar realocaciones (opcional)
        // vecinos[i].reserve(64);
        for (int j = 0; j < N; ++j) {
            double dx = P[i].x - P[j].x;
            double dy = P[i].y - P[j].y;
            if (dx*dx + dy*dy <= eps2) {
                vecinos[i].push_back(j);
            }
        }
    }

    // ==============================
    // 2) DBSCAN (expansión secuencial sobre vecindad precomputada)
    // ==============================
    std::vector<int> etiquetas(N, SIN_CLASIFICAR);
    int idCluster = 0;

    for (int i = 0; i < N; ++i) {
        if (etiquetas[i] != SINCLASIFICAR) continue;
        if (expandir_cluster_con_vecinos(vecinos, i, idCluster, minPts, etiquetas.data())) {
            ++idCluster;
        }
    }

    std::printf("Se encontraron %d clústeres\n", idCluster);

    // ==============================
    // 3) Guardar resultados
    // ==============================
    std::string salida = std::to_string(N) + "_results.csv";
    std::FILE* f = std::fopen(salida.c_str(), "w");
    if (!f) {
        std::perror("No se pudo crear el archivo de salida");
        delete[] P;
        return 1;
    }
    for (int i = 0; i < N; ++i)
        std::fprintf(f, "%.6f,%.6f,%d\n", P[i].x, P[i].y, etiquetas[i]);
    std::fclose(f);
    std::printf("Resultados guardados en %s\n", salida.c_str());

    delete[] P;
    return 0;
}
