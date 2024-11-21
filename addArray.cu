#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA核函数：每个线程负责相邻的两个元素相加
__global__ void addPairs(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 < n) {
        // 每个线程处理两个相邻的元素
        c[idx * 2] = a[idx * 2] + b[idx * 2];      // 处理第一个元素
        if (idx * 2 + 1 < n) {
            c[idx * 2 + 1] = a[idx * 2 + 1] + b[idx * 2 + 1];  // 处理第二个元素
        }
    }
}

int main() {
    const int n = 1000000; // 数组大小
    const size_t size = n * sizeof(float);

    // 分配主机内存
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    float* cpu_c = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // 分配设备内存
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    int blockSize = 256;
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2); // 每个线程处理两个元素

    // 记录GPU时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // 启动CUDA核函数
    addPairs <<<gridSize, blockSize>>> (d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // 将结果从设备复制到主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 记录CPU时间
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        cpu_c[i] = h_a[i] + h_b[i]; // CPU直接加法
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpu_end - cpu_start;

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (cpu_c[i] != h_c[i]) {
            correct = false;
            break;
        }
    }

    // 输出时间
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "Result:" << (correct ? "correct" : "mistake") << std::endl;

    // 释放设备和主机内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_c);

    return 0;
}
