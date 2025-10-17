#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

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

// Vecinos dentro de eps (incluye al propio punto), usando <= eps^2
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
    char* enCola = new char[n](); // 0-inicializado

    int numVecinos = buscar_vecinos(puntos, n, idxCore, eps2, vecinos);
    if (numVecinos < minPts) {
        etiquetas[idxCore] = RUIDO; // no es núcleo
        delete[] vecinos; delete[] cola; delete[] enCola;
        return 0;
    }

    // Etiquetar núcleo explícitamente
    etiquetas[idxCore] = idCluster;

    // Semilla: etiqueta/encola solo SIN_CLASIFICAR o RUIDO (no pisar clusters)
    int inicio = 0, fin = 0;
    for (int i = 0; i < numVecinos; ++i) {
        int q = vecinos[i];
        if (etiquetas[q] == SIN_CLASIFICAR || etiquetas[q] == RUIDO) {
            etiquetas[q] = idCluster;
            if (!enCola[q]) { cola[fin++] = q; enCola[q] = 1; }
        }
    }

    // Expansión BFS
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
    int idCluster = 0; // clusters 0,1,2,...

    for (int i = 0; i < n; ++i) {
        if (etiquetas[i] != SIN_CLASIFICAR) continue;
        if (expandir_cluster(puntos, n, i, idCluster, eps2, minPts, etiquetas)) {
            ++idCluster;
        }
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

// ======= BATCH MAIN =======
int main() {
    std::vector<int> tamanos = {110005};
    std::string inDir  = "Datasets/";
    std::string outDir = "Datasets/results/";

    if (!fs::exists(outDir)) fs::create_directories(outDir);

    double eps = 0.03;
    int    minPts = 10;

    for (int n_points : tamanos) {
        std::string archivoEntrada = inDir  + std::to_string(n_points) + "_data.csv";
        std::string archivoSalida  = outDir + std::to_string(n_points) + "_results.csv";

        Punto* P = nullptr; int N = 0;
        if (!leer_csv_xy(archivoEntrada.c_str(), P, N)) {
            std::fprintf(stderr, "❌ No pude leer: %s\n", archivoEntrada.c_str());
            continue;
        }

        int* etiquetas = new int[N];
        std::printf("Procesando %s (%d puntos)...\n", archivoEntrada.c_str(), N);
        int k = dbscan(P, N, eps, minPts, etiquetas);
        std::printf("✔ Se encontraron %d clústeres\n", k);

        std::FILE* f = std::fopen(archivoSalida.c_str(), "w");
        if (!f) {
            std::perror("No se pudo crear el archivo de salida");
            delete[] etiquetas; delete[] P;
            continue;
        }
        for (int i = 0; i < N; ++i)
            std::fprintf(f, "%.6f,%.6f,%d\n", P[i].x, P[i].y, etiquetas[i]);
        std::fclose(f);

        std::printf("💾 %s\n\n", archivoSalida.c_str());
        delete[] etiquetas;
        delete[] P;
    }

    std::printf("✅ Procesamiento completo.\n");
    return 0;
}