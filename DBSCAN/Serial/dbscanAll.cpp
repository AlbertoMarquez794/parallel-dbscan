#include <vector>
#include <filesystem>
#include <string>
#include <cstdio>

namespace fs = std::filesystem;

// Declaraciones que asumo tienes en otros .cpp/.h:
struct Punto { double x, y; };
bool leer_csv_xy(const char* path, Punto*& P, int& N);
int dbscan(Punto* P, int N, double eps, int minPts, int* etiquetas);

int main() {
    // Lista de tamaños (sin ñ)
    std::vector<int> tamanos = {110005};

    std::string carpetaEntrada = "Datasets/";
    std::string carpetaSalida  = "Datasets_results/"; // evita espacios en el path

    if (!fs::exists(carpetaSalida))
        fs::create_directories(carpetaSalida);

    for (int n_points : tamanos) {
        std::string archivoEntrada = carpetaEntrada + std::to_string(n_points) + "_data.csv";
        std::string archivoSalida  = carpetaSalida  + std::to_string(n_points) + "_results.csv";

        Punto* P = nullptr; int N = 0;
        if (!leer_csv_xy(archivoEntrada.c_str(), P, N)) {
            std::fprintf(stderr, "❌ No pude leer el archivo: %s\n", archivoEntrada.c_str());
            continue;
        }

        double eps = 0.03;
        int minPts = 10;
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

        std::printf("💾 Resultados guardados en %s\n\n", archivoSalida.c_str());

        delete[] etiquetas;
        delete[] P;
    }

    std::printf("✅ Procesamiento completo.\n");
    return 0;
}
