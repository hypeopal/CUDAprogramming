#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 32
const int matrixSize = 1024;
const int tileSize = 32;
const int threadSize = 8;
const int testTimes = 10;
__constant__ int devMatrixSize;

__global__ void matrixCopyInCol(int* newMatrix, const int* oldMatrix, int numPerThread) { //列优先
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y * numPerThread;
	int i = 0;
	while (i < numPerThread) {
		newMatrix[(y + i) * devMatrixSize + x] = oldMatrix[(y + i) * devMatrixSize + x];
		++i;
	}
}

__global__ void matrixTransposeUsingSharedMem(const int* matrix, int* transposedMatrix) {
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y;

	__shared__ int Tile[TILE_SIZE][TILE_SIZE];
	for (int i = 0; i < TILE_SIZE; ++i) {
		Tile[threadIdx.x][threadIdx.y + i] = matrix[(y + i) * devMatrixSize + x];
	}

	__syncthreads();

	for (int i = 0; i < TILE_SIZE; ++i) {
		transposedMatrix[x * devMatrixSize + (y + i)] = Tile[threadIdx.x][threadIdx.y + i];
	}
}

cudaError_t cuda_check(const cudaError_t& error_code, int line)
{
	if (error_code != cudaSuccess)
	{
		printf("line: %d, error_code: %d, error_name: %s, error_description: %s\n",
			line, error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code));
		exit(EXIT_FAILURE); // 检测到CUDA错误时退出程序
	}
	return error_code;
}

void matrixTransposeByCPU(const int* matrix, int* transposedMatrix) {
	for (int i = 0; i < matrixSize; ++i) {
		for (int k = 0; k < matrixSize; ++k) {
			transposedMatrix[k * matrixSize + i] = matrix[i * matrixSize + k];
		}
	}
}

bool checkResult(const int* matrix, const int* transposedMatrix) { //转置正确性检测
	for (int i = 0; i < matrixSize; ++i) {
		for (int k = 0; k < matrixSize; ++k) {
			if (matrix[i * matrixSize + k] != transposedMatrix[i * matrixSize + k])
				return false;
		}
	}
	return true;
}

void runCudaMatrixTranspose(const int* matrix, int* transposedMatrix) {
	//GPU内存分配
	int size = sizeof(int) * matrixSize * matrixSize;
	int* cudaOldMatrix;
	int* cudaTransposedMatrix;
	cuda_check(cudaMalloc((void**)&cudaOldMatrix, size), __LINE__);
	cuda_check(cudaMemcpy(cudaOldMatrix, matrix, size, cudaMemcpyHostToDevice), __LINE__);
	cuda_check(cudaMalloc((void**)&cudaTransposedMatrix, size), __LINE__);
	//GPU常量分配
	cuda_check(cudaMemcpyToSymbol(devMatrixSize, &matrixSize, sizeof(int)), __LINE__);

	dim3 dimGrid(tileSize, tileSize, 1);
	dim3 dimBlock(tileSize, 1, 1);

	// 记录GPU时间
	cudaEvent_t start, stop;
	cuda_check(cudaEventCreate(&start), __LINE__);
	cuda_check(cudaEventCreate(&stop), __LINE__);

	cuda_check(cudaEventRecord(start, 0), __LINE__);
	matrixTransposeUsingSharedMem << <dimGrid, dimBlock >> > (cudaOldMatrix, cudaTransposedMatrix);
	cuda_check(cudaDeviceSynchronize(), __LINE__);
	cuda_check(cudaEventRecord(stop, 0), __LINE__);

	cuda_check(cudaEventSynchronize(stop), __LINE__);
	float gpuTime = 0;
	cuda_check(cudaEventElapsedTime(&gpuTime, start, stop), __LINE__);
	std::cout << "CUDA matrix transpose with shared memory runtime: " << gpuTime << std::endl;

	cuda_check(cudaMemcpy(transposedMatrix, cudaTransposedMatrix, size, cudaMemcpyDeviceToHost), __LINE__);
	cuda_check(cudaFree(cudaTransposedMatrix), __LINE__);
	cuda_check(cudaFree(cudaOldMatrix), __LINE__);
}

int main() {
	int* matrix = new int[matrixSize * matrixSize]; //初始矩阵
	int* transposedMatrix = new int[matrixSize * matrixSize]; //CPU转置的矩阵
	int* devTransposedMatrix = new int[matrixSize * matrixSize]; //GPU转置的矩阵

	//初始化矩阵
	for (int i = 0; i < matrixSize * matrixSize; ++i) {
		matrix[i] = i;
	}
	
	runCudaMatrixTranspose(matrix, devTransposedMatrix); //CUDA转置
	matrixTransposeByCPU(matrix, transposedMatrix); //CPU转置
	
	std::cout << "Result: " << (checkResult(transposedMatrix, devTransposedMatrix) ? "correct" : "error") << std::endl;

	delete[] matrix;
	delete[] transposedMatrix;
	delete[] devTransposedMatrix;
	return 0;
}
