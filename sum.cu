#include "sum.cuh"
#include <cstdio>
#include <iostream>

const float COEFFICIENT = 1389.38757;

int get_max_cols(Matrix A) {
    int globalsum = 0;
    int n = A.height;

    for (size_t i = 0; i < n; i++) {
        int localsum = 0;
        for (size_t j = 0; j < n; j++) {
            if (A.elements[i * n + j] > 0) {
                localsum++;
            }
        }
        if (localsum > globalsum) {
            globalsum = localsum;
        }
    }
    return globalsum;
}

void fill_ell_matrix(ELL_Matrix ell, Matrix A, int nrows, int ncols, int ncols_per_row) {



    for (size_t i = 0; i < nrows; i++) {
        int colidx = 0;
        for (size_t j = 0; j < ncols; j++) {
            if (A.elements[i * nrows + j] != 0) {
                ell.data[colidx * nrows + i] = A.elements[i * nrows + j];
                ell.col_indices[colidx * nrows + i] = j;
                colidx++;
            }
        }
    }
}

void ell_mallocHost(){

}

void ell_freeHost() {

}

__device__ float sign(float x) {
    float t = x < 0 ? -1.0 : 0.0;
    return x > 0 ? 1.0 : t;
}

__device__ float atom_dist (atom a1, atom a2) {
    float sum =  (a1.x - a2.x) * (a1.x - a2.x) +
            (a1.y - a2.y) * (a1.y - a2.y) +
            (a1.z - a2.z) * (a1.z - a2.z);
    return sqrtf(sum);
}

void calculate(float* bonds, size_t n, atom* atom_coords, float* charges) {
    //TODO hard-coded value! fix it
    size_t ncols_per_row = 4;

    Matrix A;
    A.elements = bonds;
    A.height = n;
    A.width = n;

    size_t size = A.width * A.height * sizeof(float);

    ELL_Matrix ell_A;

    size_t sizefloat = n * ncols_per_row * sizeof(float);
    size_t sizeint = n * ncols_per_row * sizeof(int);

    cudaMallocHost((void**)&ell_A.data, sizefloat);
    cudaMallocHost((void**)&ell_A.col_indices, sizeint);

    ell_A.ncols = n;
    ell_A.nrows = n;
    ell_A.ncols_per_row = ncols_per_row;

    fill_ell_matrix(ell_A, A, n, n, 4);

    ELL_Matrix d_ell_A;
    d_ell_A.ncols_per_row = ell_A.ncols_per_row;
    d_ell_A.ncols = ell_A.ncols;
    d_ell_A.nrows = ell_A.nrows;

    cudaMalloc((void**)&d_ell_A.data, sizefloat);
    cudaMalloc((void**)&d_ell_A.col_indices, sizeint);

    cudaMemcpy(d_ell_A.data, ell_A.data, sizefloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_A.col_indices, ell_A.col_indices, sizeint, cudaMemcpyHostToDevice);

    Matrix Asquared;
    Asquared.width = n;
    Asquared.height = n;
    cudaMallocHost((void**)&Asquared.elements, size);

    /*Matrix Acubed;
    cudaMallocHost((void**)&Acubed.elements, size);
    Acubed.height = n;
    Acubed.width = n;*/

    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    cudaMalloc((void**)&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = A.width;
    d_B.height = A.height;
    cudaMalloc((void**)&d_B.elements, size);
    cudaMemcpy(d_B.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_Asquared;
    d_Asquared.width = A.width;
    d_Asquared.height = A.height;
    cudaMalloc((void**)&d_Asquared.elements, size);



    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 dimGrid(A.width / dimBlock.x + 1, A.height / dimBlock.y + 1);

    dim3 dimBlock1(BLOCK_SIZE);

    dim3 dimGrid1(n / dimBlock1.x + 1);
    //std::cout << "Launching kernel for multiply" << std::endl;

    //matrix_mult<<<dimGrid, dimBlock>>>(d_A, d_B, d_Asquared);
    //cudaDeviceSynchronize();
    //std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;


    for (int i = 0; i < n; i++) {
        ell_mult<<<dimGrid1, dimBlock1>>>(d_ell_A, d_A, d_Asquared, i);
        //cudaDeviceSynchronize();
        //std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(Asquared.elements, d_Asquared.elements, size, cudaMemcpyDeviceToHost);

    int ncols_per_row2 = get_max_cols(Asquared);

    ELL_Matrix ell_Asq;

    size_t sizefloat2 = n * ncols_per_row2 * sizeof(float);
    size_t sizeint2 = n * ncols_per_row2 * sizeof(int);

    cudaMallocHost((void**)&ell_Asq.data, sizefloat2);
    cudaMallocHost((void**)&ell_Asq.col_indices, sizeint2);

    ell_Asq.ncols = n;
    ell_Asq.nrows = n;
    ell_Asq.ncols_per_row = ncols_per_row2;

    fill_ell_matrix(ell_Asq, Asquared, n, n, ncols_per_row2);

    ELL_Matrix d_ell_Asq;
    d_ell_Asq.ncols_per_row = ell_Asq.ncols_per_row;
    d_ell_Asq.ncols = ell_Asq.ncols;
    d_ell_Asq.nrows = ell_Asq.nrows;

    cudaMalloc((void**)&d_ell_Asq.data, sizefloat2);
    cudaMalloc((void**)&d_ell_Asq.col_indices, sizeint2);

    cudaMemcpy(d_ell_Asq.data, ell_Asq.data, sizefloat2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_Asq.col_indices, ell_Asq.col_indices, sizeint2, cudaMemcpyHostToDevice);

    Matrix d_Acubed;
    d_Acubed.width = A.width;
    d_Acubed.height = A.height;
    cudaMalloc((void**)&d_Acubed.elements, size);

    atom* d_atoms;
    cudaMalloc((void**)&d_atoms, n * sizeof(atom));
    cudaMemcpy(d_atoms, atom_coords, n * sizeof(atom), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++) {
        ell_mult<<<dimGrid1, dimBlock1>>>(d_ell_Asq, d_A, d_Acubed, i);
        //cudaDeviceSynchronize();
        //std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    //matrix_mult<<<dimGrid, dimBlock>>>(d_A, d_Asquared, d_Acubed);
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaDeviceSynchronize();
    std::cout << "Finished multiplication" << std::endl;
    //cudaMemcpy(Acubed.elements, d_Acubed.elements, size, cudaMemcpyDeviceToHost);

    Matrix F;
    F.width = n;
    F.height = n;

    cudaMalloc((void**)&F.elements, size);

    get_matrix_F<<<dimGrid, dimBlock>>>(d_Acubed, d_Asquared, d_A, d_atoms, F);
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

/*    Matrix F_host;
    F_host.width = F.width;
    F_host.height = F.width;
    cudaMallocHost((void**)&F_host.elements, size);
    cudaMemcpy(F_host.elements, F.elements, size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n * n; i++) {
        if (F_host.elements[i] != 0) {
            std::cout << F_host.elements[i] << std::endl;
        }
    }*/

    float* res;
    float* d_res;
    float* d_charges;



    cudaMallocHost((void**)&res, n * sizeof(float));
    cudaMalloc((void**)&d_res, n * sizeof(float));

    cudaMalloc((void**)&d_charges, n * sizeof(float));
    cudaMemcpy(d_charges, charges, n * sizeof(float), cudaMemcpyHostToDevice);


  /*  dim3 dimBlock1(BLOCK_SIZE);

    dim3 dimGrid1(n / dimBlock1.x + 1);*/

    get_vector_to_sum<<<dimGrid1, dimBlock1>>>(F, d_charges, d_res);
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;



    cudaMemcpy(res, d_res, n * sizeof(float), cudaMemcpyDeviceToHost);

    float ans = 0;


    for (size_t i = 0; i < n; i++) {
        ans += res[i];
    }

    std::cout << "Answer is " << (COEFFICIENT * ans) / 2 << std::endl;

    cudaFree(d_A.elements);
    cudaFree(d_Asquared.elements);
    cudaFree(d_Acubed.elements);
    cudaFree(d_atoms);
    cudaFree(d_res);
    cudaFree(d_charges);
}




__global__ void matrix_mult(Matrix A, Matrix B, Matrix C){
    float Cvalue = 0;
    if (blockIdx.y * blockDim.y + threadIdx.y < A.width && blockIdx.x * blockDim.x + threadIdx.x < A.width){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.width; ++e) {
            Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
        }
    C.elements[row * C.width + col] = Cvalue;
    }
}

__global__ void get_matrix_F(Matrix ACube, Matrix ASquare, Matrix A, atom* atoms, Matrix F) {
    float value = 0;
    float cube = 0;
    float square = 0;
    float a = 0;
    if (blockIdx.y * blockDim.y + threadIdx.y < A.width && blockIdx.x * blockDim.x + threadIdx.x < A.width){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = row * A.width + col;
        if (row != col) {
            value = 1 - 0.5 * sign(ACube.elements[idx] + ASquare.elements[idx] + A.elements[idx]) -
                              0.5 * sign(ASquare.elements[idx] + A.elements[idx]);
            value /= atom_dist(atoms[row], atoms[col]);
            F.elements[idx] = value;
        }
        else {
            F.elements[idx] = 0.0;
        }
    }
}

__global__ void get_vector_to_sum(Matrix F, float* q, float* res) {
    float value = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < F.width) {
        for (int j = 0; j < F.height; j++) { //TODO maybe permute indices order
            value += F.elements[j * F.height + i] * q[j];
        }
        value *= q[i];
        res[i] = value;
    }
}

/**
 * kernel for ELL matrix-vector multiplication
 *
 */

__global__ void ell_mult(ELL_Matrix A, Matrix B, Matrix C, int i){
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < A.nrows) {
        float dot = 0;

        for (int idx = 0; idx < A.ncols_per_row; idx++) {
            int col = A.col_indices[A.nrows * idx + row];
            float val = A.data[A.nrows * idx + row];

            if (val != 0) {
                dot += val * B.elements[i * B.height + col];
            }
            C.elements[i * C.height + row] = dot;
        }
    }
}





