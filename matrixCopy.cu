#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int matrixSize = 1024;
const int tileSize = 32;
const int threadSize = 8;
__constant__ int devMatrixSize;


__global__ void matrixCopy(int* newMatrix, const int* oldMatrix) {
	/*int x = blockIdx.x * gridDim.x + threadIdx.x * 4;
	int y = blockIdx.y * gridDim.y + threadIdx.y;
	int i = 0;
	while (i < 4) {
		newMatrix[y * devMatrixSize + x + i] = oldMatrix[y * devMatrixSize + x + i];
		++i;
	}*/
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y * 4;
	int i = 0;
	while (i < 4) {
		newMatrix[(y + i) * devMatrixSize + x] = oldMatrix[(y + i) * devMatrixSize + x];
		++i;
	}
}

int main() {
	int* matrix = new int[matrixSize * matrixSize];
	int* copiedMatrix = new int[matrixSize * matrixSize];
	for (int i = 0; i < matrixSize; ++i) {
		for (int j = 0; j < matrixSize; ++j) {
			matrix[i * matrixSize + j] = i * matrixSize + j;
		}
	}
	cudaMemcpyToSymbol(devMatrixSize, &matrixSize, sizeof(int));
	int size = sizeof(int) * matrixSize * matrixSize;
	int* cudaNewMatrix;
	int* cudaOldMatrix;
	cudaMalloc((void**)&cudaNewMatrix, size);
	cudaMalloc((void**)&cudaOldMatrix, size);

	cudaMemcpy(cudaOldMatrix, matrix, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(tileSize, tileSize, 1);
	//dim3 dimBlock(threadSize, tileSize, 1);
	dim3 dimBlock(tileSize, threadSize, 1);

	matrixCopy <<<dimGrid, dimBlock >>> (cudaNewMatrix, cudaOldMatrix);
	cudaDeviceSynchronize();

	cudaMemcpy(copiedMatrix, cudaNewMatrix, size, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < matrixSize; ++i) {
		for (int j = 0; j < matrixSize; ++j) {
			if (matrix[i * matrixSize + j] != copiedMatrix[i * matrixSize + j]) {
				correct = false;
				break;
			}
		}
	}
	std::cout << "Result:" << (correct ? "correct" : "mistake") << std::endl;
	
	cudaFree(cudaNewMatrix);
	cudaFree(cudaOldMatrix);
	delete[] matrix;
	delete[] copiedMatrix;
	return 0;
}