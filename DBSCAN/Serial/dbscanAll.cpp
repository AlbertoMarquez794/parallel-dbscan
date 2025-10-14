#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

int main() {
    // Lista de tamaños
    std::vector<int> tamaños = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};

    std::string carpetaEntrada = "Datasets/";
    std::string carpetaSalida  = "Datasets results/";

    // Crear carpeta de resultados si no existe
    if (!fs::exists(carpetaSalida))
        fs::create_directories(carpetaSalida);

    for (int n_points : tamaños) {
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
