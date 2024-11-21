#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel函数在设备上运行
__global__ void helloWorldKernel() {
    printf("Hello, World! From GPU thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    // 启动内核函数，每个线程将打印一条消息
    helloWorldKernel <<<2, 5 >>> (); // 启动2个block，每个block中有5个thread

    // 等待GPU完成任务
    cudaDeviceSynchronize();

    std::cout << "Hello, World! From CPU\n";
    return 0;
}
