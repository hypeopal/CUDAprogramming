#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 32
#define MATRIX_SIZE 8192
const int testTimes = 10;


// GPU内核函数定义
__global__ void matrixCopyNaive(const int* matrix, int* copiedMatrix) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        copiedMatrix[(y + i) * MATRIX_SIZE + x] = matrix[(y + i) * MATRIX_SIZE + x];
    }
}

__global__ void matrixCopyUsingSharedMem(const int* matrix, int* copiedMatrix) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ int TILE[TILE_SIZE][TILE_SIZE];
    TILE[threadIdx.y][threadIdx.x] = matrix[(y) * MATRIX_SIZE + x];
    TILE[threadIdx.y + blockDim.y][threadIdx.x] = matrix[(y + blockDim.y)*MATRIX_SIZE + x];
    TILE[threadIdx.y + 2 * blockDim.y][threadIdx.x] = matrix[(y + 2 * blockDim.y) * MATRIX_SIZE + x];
    TILE[threadIdx.y + 3 * blockDim.y][threadIdx.x] = matrix[(y + 3 * blockDim.y) * MATRIX_SIZE + x];
    
    copiedMatrix[(y)*MATRIX_SIZE + x] = TILE[threadIdx.y][threadIdx.x];
    copiedMatrix[(y + blockDim.y) * MATRIX_SIZE + x] = TILE[threadIdx.y + blockDim.y][threadIdx.x];
    copiedMatrix[(y + 2 * blockDim.y) * MATRIX_SIZE + x] = TILE[threadIdx.y + 2 * blockDim.y][threadIdx.x];
    copiedMatrix[(y + 3 * blockDim.y) * MATRIX_SIZE + x] = TILE[threadIdx.y + 3 * blockDim.y][threadIdx.x];
}

__global__ void matrixTransposeNaive(const int* matrix, int* transposedMatrix) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        transposedMatrix[x * MATRIX_SIZE + (y + i)] = matrix[(y + i) * MATRIX_SIZE + x];
    }
}

__global__ void matrixTransposeUsingSharedMem(const int* matrix, int* transposedMatrix) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    __shared__ int Tile[TILE_SIZE][TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        Tile[threadIdx.y + i][threadIdx.x] = matrix[(y + i) * MATRIX_SIZE + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // 块内转置
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        transposedMatrix[(y + i) * MATRIX_SIZE + x] = Tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void matrixTransposeUsingSharedMemWithPadding(const int* matrix, int* transposedMatrix) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ int Tile[(TILE_SIZE + 1) * TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        Tile[(threadIdx.y + i) * (TILE_SIZE + 1) + threadIdx.x] = matrix[(y + i) * MATRIX_SIZE + x];
    }

    __syncthreads();
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // 块内转置
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        transposedMatrix[(y + i) * MATRIX_SIZE + x] = Tile[threadIdx.x * (TILE_SIZE + 1) + (threadIdx.y + i)];
    }
}

cudaError_t cuda_check(const cudaError_t& error_code, int line) {
    if (error_code != cudaSuccess)
    {
        printf("line: %d, error_code: %d, error_name: %s, error_description: %s\n",
            line, error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code));
        exit(EXIT_FAILURE); // 检测到CUDA错误时退出程序
    }
    return error_code;
}

// CPU矩阵运算
void matrixTransposeByCPU(const int* matrix, int* transposedMatrix) {
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int k = 0; k < MATRIX_SIZE; ++k)
            transposedMatrix[k * MATRIX_SIZE + i] = matrix[i * MATRIX_SIZE + k];
}

void matrixCopyByCPU(const int* matrix, int* copiedMatrix) {
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        copiedMatrix[i] = matrix[i];
}

bool checkResult(const int* matrix1, const int* matrix2) {
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        if (matrix1[i] != matrix2[i]) return false;
    return true;
}

// CUDA核函数模板
template <typename Kernel>
void runCudaKernel(const int* matrix, int* resultMatrix, Kernel kernel, const char* label) {
    int size = sizeof(int) * MATRIX_SIZE * MATRIX_SIZE;
    int* cudaOldMatrix;
    int* cudaResultMatrix;
    // 分配GPU内存
    cuda_check(cudaMalloc((void**)&cudaOldMatrix, size), __LINE__);
    cuda_check(cudaMemcpy(cudaOldMatrix, matrix, size, cudaMemcpyHostToDevice), __LINE__);
    cuda_check(cudaMalloc((void**)&cudaResultMatrix, size), __LINE__);

    // 定义线程块和网格
    dim3 dimGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, 8, 1);

    // 创建CUDA计时事件
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), __LINE__);
    cuda_check(cudaEventCreate(&stop), __LINE__);
    float gpuTime = 0;

    // 启动核函数并测量执行时间
    cuda_check(cudaEventRecord(start, 0), __LINE__);
    for (int i = 0; i < testTimes; ++i) {
        kernel <<<dimGrid, dimBlock >>> (cudaOldMatrix, cudaResultMatrix);
        cuda_check(cudaDeviceSynchronize(), __LINE__);
    }
    cuda_check(cudaEventRecord(stop, 0), __LINE__);
    cuda_check(cudaEventSynchronize(stop), __LINE__);
    cuda_check(cudaEventElapsedTime(&gpuTime, start, stop), __LINE__);

    std::cout << label << gpuTime / testTimes << " ms" << std::endl;

    // 将结果复制回主机并释放GPU内存
    cuda_check(cudaMemcpy(resultMatrix, cudaResultMatrix, size, cudaMemcpyDeviceToHost), __LINE__);
    cuda_check(cudaFree(cudaResultMatrix), __LINE__);
    cuda_check(cudaFree(cudaOldMatrix), __LINE__);
}

int main() {
    int* matrix = new int[MATRIX_SIZE * MATRIX_SIZE];  // 原矩阵
    int* transposedMatrix = new int[MATRIX_SIZE * MATRIX_SIZE];  // cpu转置后的矩阵
    int* devTransposedMatrix = new int[MATRIX_SIZE * MATRIX_SIZE];  // gpu转置的矩阵
    int* copiedMatrix = new int[MATRIX_SIZE * MATRIX_SIZE];  // cpu复制的矩阵
    int* devCopiedMatrix = new int[MATRIX_SIZE * MATRIX_SIZE];  // gpu复制的矩阵

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) { // 初始化矩阵
        matrix[i] = i - (MATRIX_SIZE * MATRIX_SIZE / 2);
    }

    // 矩阵转置
    matrixTransposeByCPU(matrix, transposedMatrix);
    runCudaKernel(matrix, devTransposedMatrix, matrixTransposeNaive, 
        "CUDA matrix transpose naive average runtime:");
    std::cout << "Transpose Result: " << (checkResult(transposedMatrix, devTransposedMatrix) ? "correct" : "error") << std::endl;
    runCudaKernel(matrix, devTransposedMatrix, matrixTransposeUsingSharedMem, 
        "CUDA matrix transpose with shared memory average runtime:");
    std::cout << "Transpose Result: " << (checkResult(transposedMatrix, devTransposedMatrix) ? "correct" : "error") << std::endl;
    runCudaKernel(matrix, devTransposedMatrix, matrixTransposeUsingSharedMemWithPadding,
        "CUDA matrix transpose with shared memory and padding average runtime:");
    std::cout << "Transpose Result: " << (checkResult(transposedMatrix, devTransposedMatrix) ? "correct" : "error") << std::endl;
    
    std::cout << std::endl;

    // 矩阵复制
    matrixCopyByCPU(matrix, copiedMatrix);
    runCudaKernel(matrix, devCopiedMatrix, matrixCopyNaive, 
        "CUDA matrix copy naive average runtime:");
    std::cout << "Copy Result: " << (checkResult(copiedMatrix, devCopiedMatrix) ? "correct" : "error") << std::endl;
    runCudaKernel(matrix, devCopiedMatrix, matrixCopyUsingSharedMem, 
        "CUDA matrix copy with shared memory average runtime:");
    std::cout << "Copy Result: " << (checkResult(copiedMatrix, devCopiedMatrix) ? "correct" : "error") << std::endl;

    // 释放内存
    delete[] matrix;
    delete[] transposedMatrix;
    delete[] devTransposedMatrix;
    delete[] copiedMatrix;
    delete[] devCopiedMatrix;
    return 0;
}
