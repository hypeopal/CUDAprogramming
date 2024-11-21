#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA�˺�����ÿ���̸߳������ڵ�����Ԫ�����
__global__ void addPairs(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 < n) {
        // ÿ���̴߳����������ڵ�Ԫ��
        c[idx * 2] = a[idx * 2] + b[idx * 2];      // �����һ��Ԫ��
        if (idx * 2 + 1 < n) {
            c[idx * 2 + 1] = a[idx * 2 + 1] + b[idx * 2 + 1];  // ����ڶ���Ԫ��
        }
    }
}

int main() {
    const int n = 1000000; // �����С
    const size_t size = n * sizeof(float);

    // ���������ڴ�
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    float* cpu_c = (float*)malloc(size);

    // ��ʼ������
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // �����豸�ڴ�
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // �����ݴ��������Ƶ��豸
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // �����߳̿�������С
    int blockSize = 256;
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2); // ÿ���̴߳�������Ԫ��

    // ��¼GPUʱ��
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // ����CUDA�˺���
    addPairs <<<gridSize, blockSize>>> (d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // ��������豸���Ƶ�����
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // ��¼CPUʱ��
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        cpu_c[i] = h_a[i] + h_b[i]; // CPUֱ�Ӽӷ�
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

    // ���ʱ��
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "Result:" << (correct ? "correct" : "mistake") << std::endl;

    // �ͷ��豸�������ڴ�
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_c);

    return 0;
}
