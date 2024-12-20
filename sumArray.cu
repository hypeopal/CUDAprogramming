#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(statement) do {\
cudaError_t code = statement;\
if (code != cudaSuccess)\
{\
	printf("line: %d, error_code: %d, error_name: %s, error_description: %s\n",\
		__LINE__, code, cudaGetErrorName(code), cudaGetErrorString(code));\
	exit(EXIT_FAILURE); \
}\
} while(0) \

#define SIZE 512

__device__ int gpuSum = 0;

__global__ void sumArrayCUDA(const int* arr) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < SIZE)
		atomicAdd(&gpuSum, arr[tid]);
}

int sumArrayCPU(const int* arr) {
	int sum = 0;
	for (int i = 0; i < SIZE; ++i) sum += arr[i];
	return sum;
}


int main() {
	int* array = new int[SIZE];
	for (int i = 0; i < SIZE; ++i) {
		array[i] = i;
	}
	int cpuSum = sumArrayCPU(array);
	int size = sizeof(int) * SIZE;
	int* cudaArr;
	CUDA_CHECK(cudaMalloc((void**)&cudaArr, size));
	CUDA_CHECK(cudaMemcpy(cudaArr, array, size, cudaMemcpyHostToDevice));
	int threadsPerBlock = 256;
	int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
	sumArrayCUDA << <blocksPerGrid, threadsPerBlock >> > (cudaArr);
	CUDA_CHECK(cudaDeviceSynchronize());

	int cudaSum;
	CUDA_CHECK(cudaMemcpyFromSymbol(&cudaSum, gpuSum, sizeof(int), 0, cudaMemcpyDeviceToHost));

	std::cout << (cpuSum == cudaSum ? "correct" : "mistake") << std::endl;
	std::cout << cpuSum << "  " << cudaSum << std::endl;

	delete[] array;
	return 0;
}
