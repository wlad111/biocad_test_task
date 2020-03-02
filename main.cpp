#include <iostream>
#include <fstream>
#include <vector>
#include <sum.cuh>




int main() {
    std::ifstream charges_file("../data/charges.txt");

    float x;

    std::vector<float> charges;

    while (charges_file >> x) {
        charges.push_back(x);
    }

    std::ifstream bonds_file("../data/bonds.txt");

    size_t n = charges.size();

    float* bonds = new float[n * n];
    atom* atoms = new atom[n];


    //reading atomic bonds
    while (!bonds_file.eof()) {
        int a, b;
        bonds_file >> a >> b;
        bonds[a * n + b] = 1;
        bonds[b * n + a] = 1;
        if (bonds_file.eof()) {
            break;
        }
    }

    std::ofstream out("matrix.csv");

    int globalsum = 0;

    for (size_t i = 0; i < n; i++) {
        int localsum = 0;
        for (size_t j = 0; j < n; j++) {
            if (bonds[i * n + j] > 0) {
                localsum++;
            }
        }
        if (localsum > globalsum) {
            globalsum = localsum;
        }
    }


    std::cout << globalsum << std::endl;




    float sum = 0;

    /*for (size_t i = 0; i < charges.size(); i++) {
        for (size_t j = i + 1; j < charges.size(); j++) {
            sum += charges[i]*charges[j];
        }
    }*/

    //reading atomic coordinates
    std::ifstream atoms_file("../data/atoms.txt");

    for (size_t i = 0; i < n; i++) {
        float a, b, c;
        atoms_file >> a >> b >> c;
        atoms[i] = {a, b, c};
    }

    float* charges_raw = charges.data();

    //launching CUDA routines for computation
    calculate(bonds, n, atoms, charges_raw);

    delete[] atoms;
    delete[] bonds;

    return 0;
}
