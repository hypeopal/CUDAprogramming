#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <numeric>

const int matrixSize = 1024;
const int tileSize = 32;
const int threadSize = 8;
const int testTimes = 10;
__constant__ int devMatrixSize;


__global__ void matrixCopyInRow(int* newMatrix, const int* oldMatrix, int numPerThread) { //行优先
	int x = blockIdx.x * gridDim.x + threadIdx.x * numPerThread;
	int y = blockIdx.y * gridDim.y + threadIdx.y;
	int i = 0;
	while (i < numPerThread) {
		newMatrix[y * devMatrixSize + x + i] = oldMatrix[y * devMatrixSize + x + i];
		++i;
	}
}

__global__ void matrixCopyInCol(int* newMatrix, const int* oldMatrix, int numPerThread) { //列优先
	int x = blockIdx.x * gridDim.x + threadIdx.x;
	int y = blockIdx.y * gridDim.y + threadIdx.y * numPerThread;
	int i = 0;
	while (i < numPerThread) {
		newMatrix[(y + i) * devMatrixSize + x] = oldMatrix[(y + i) * devMatrixSize + x];
		++i;
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

void runMatrixCopy(const int* matrix, int* copiedMatrix, int numPerThread, const std::string& type) {
	std::vector<float> runtimeList; //运行时间
	int size = sizeof(int) * matrixSize * matrixSize;
	int* cudaOldMatrix;
	//__device__ int devNumPerThread;
	//cuda_check(cudaMemcpyToSymbol(devNumPerThread, &numPerThread, sizeof(int)), __LINE__);
	cuda_check(cudaMalloc((void**)&cudaOldMatrix, size), __LINE__);
	cuda_check(cudaMemcpy(cudaOldMatrix, matrix, size, cudaMemcpyHostToDevice), __LINE__);
	for (int n = 0; n < testTimes; ++n) {
		int* cudaNewMatrix;
		cuda_check(cudaMalloc((void**)&cudaNewMatrix, size), __LINE__);

		dim3 dimGrid(tileSize, tileSize, 1);
		dim3 dimBlock;
		if (type == "Row")
			dimBlock = dim3(tileSize / numPerThread, tileSize, 1);
		else
			dimBlock = dim3(tileSize, tileSize / numPerThread, 1);

		// 记录GPU时间
		cudaEvent_t start, stop;
		cuda_check(cudaEventCreate(&start), __LINE__);
		cuda_check(cudaEventCreate(&stop), __LINE__);

		//开始运行
		cuda_check(cudaEventRecord(start, 0), __LINE__);
		if (type == "Row")
			matrixCopyInRow << <dimGrid, dimBlock >> > (cudaNewMatrix, cudaOldMatrix, numPerThread);
		else
			matrixCopyInCol << <dimGrid, dimBlock >> > (cudaNewMatrix, cudaOldMatrix, numPerThread);
		cuda_check(cudaDeviceSynchronize(), __LINE__);
		cuda_check(cudaEventRecord(stop, 0), __LINE__);

		cuda_check(cudaEventSynchronize(stop), __LINE__);
		float gpuTime = 0;
		cuda_check(cudaEventElapsedTime(&gpuTime, start, stop), __LINE__);
		runtimeList.emplace_back(gpuTime);

		cuda_check(cudaMemcpy(copiedMatrix, cudaNewMatrix, size, cudaMemcpyDeviceToHost), __LINE__);
		cuda_check(cudaFree(cudaNewMatrix), __LINE__);
		//_ASSERT(cudaGetLastError() == cudaSuccess);

		//结果检查
		bool correct = true;
		for (int i = 0; i < matrixSize * matrixSize; ++i) {
			if (matrix[i] != copiedMatrix[i]) {
				correct = false;
				break;
			}
		}
		if (!correct) {
			std::cout << "Runtime error!" << std::endl;
			return;
		}
	}
	cuda_check(cudaFree(cudaOldMatrix), __LINE__);
	std::cout << "Result in " << type << " with " << numPerThread <<" nums per thread:correct" << std::endl;
	//计算平均时间
	std::cout << "Average runtime: " << std::accumulate(runtimeList.begin(), runtimeList.end(), 0.0f) / testTimes << std::endl;
}

int main() {
	int* matrix = new int[matrixSize * matrixSize]; //初始矩阵
	int* copiedMatrix = new int[matrixSize * matrixSize]; //复制后的矩阵
	//初始化矩阵
	for (int i = 0; i < matrixSize * matrixSize; ++i) {
		matrix[i] = i;
	}
	cuda_check(cudaMemcpyToSymbol(devMatrixSize, &matrixSize, sizeof(int)), __LINE__);

	int numList[]{1, 2, 4, 8, 16, 32};

	for (auto i : numList) {
		runMatrixCopy(matrix, copiedMatrix, i, "Row");
	}
	std::cout << std::endl;
	for (auto i : numList) {
		runMatrixCopy(matrix, copiedMatrix, i, "Col");
	}

	delete[] matrix;
	delete[] copiedMatrix;
	return 0;
}
