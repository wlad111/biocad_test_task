#ifndef SUM_CUH
#define SUM_CUH

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define BLOCK_SIZE_128 128

struct atom {
    float x;
    float y;
    float z;
};

struct Matrix {
    int width;
    int height;
    float* elements;
};

/**
 * ELLPACK Sparse matrix storage format, described in this paper: https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf
 *
 * data and indices are stored in col-major format, for coalesced memory access in kernel
 */

struct ELL_Matrix {
    ELL_Matrix(int nrows, int ncols, int ncols_per_row);

    int nrows;
    int ncols;
    int ncols_per_row;
    int* col_indices;
    float* data;

    ELL_Matrix(const float* values, int nrows, int ncols, int ncols_per_row);

    ELL_Matrix();

    ~ELL_Matrix();
};

__global__ void matrix_mult(Matrix A, Matrix B, Matrix C);

__global__ void ell_mult(ELL_Matrix A, Matrix B, Matrix C, int i);

__global__ void get_matrix_F(Matrix ACube, Matrix ASquare, Matrix A, atom* atoms, Matrix F);

__global__ void get_vector_to_sum(Matrix F, float* q, float* res);


void calculate(float* bonds, size_t n, atom* atom_coords, float* charges);

#endif //SUM_CUH