#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

// 錯誤檢查巨集 (HPC 開發標準配備)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA Kernel: 平行計算每一組的量子產率
// 公式: Unknown_QY = Std_QY * (Unknown_Area / Std_Area)
__global__ void calculate_yield_kernel(const double* d_std_area, 
                                       const double* d_unk_area, 
                                       double* d_results, 
                                       double std_qy_percent, 
                                       int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // 直接在此處進行除法與乘法運算
        double ratio = d_unk_area[idx] / d_std_area[idx];
        d_results[idx] = std_qy_percent * ratio;
    }
}

int main() {
    // 1. 實驗數據準備 (Host Side)
    // Tetracene 積分面積 (0.1, 0.05, 0.03)
    std::vector<double> h_std_area = {3.4522e5, 5.2197e5, 4.9133e5};
    // LD12 積分面積 (對應濃度)
    std::vector<double> h_unk_area = {2.0668e5, 1.3628e5, 1.0089e5};
    
    int n = h_std_area.size();
    size_t bytes = n * sizeof(double);
    double std_qy = 13.00; // Tetracene 已知產率 %

    // 儲存結果的 Host 向量
    std::vector<double> h_results(n);

    // 2. 配置 Device 記憶體
    double *d_std_area, *d_unk_area, *d_results;
    gpuErrchk(cudaMalloc(&d_std_area, bytes));
    gpuErrchk(cudaMalloc(&d_unk_area, bytes));
    gpuErrchk(cudaMalloc(&d_results, bytes));

    // 3. 資料傳輸 Host -> Device
    gpuErrchk(cudaMemcpy(d_std_area, h_std_area.data(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_unk_area, h_unk_area.data(), bytes, cudaMemcpyHostToDevice));

    // 4. 執行 Kernel
    // 因為只有 3 筆資料，用 1 個 Block, 3 個 Threads 即可
    // 對於大規模數據，通常會設 dim3 blockSize(256); dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    calculate_yield_kernel<<<1, n>>>(d_std_area, d_unk_area, d_results, std_qy, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 5. 資料傳輸 Device -> Host
    gpuErrchk(cudaMemcpy(h_results.data(), d_results, bytes, cudaMemcpyDeviceToHost));

    // 6. 統計分析 (由於 N 極小，這部分在 CPU 做比在 GPU 做 Reduction 更快)
    double sum = 0.0;
    std::cout << "Individual Yields Calculation:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "  Sample " << i+1 << " (Abs " << (i==0?0.1:i==1?0.05:0.03) << "): " 
                  << std::fixed << std::setprecision(2) << h_results[i] << " %" << std::endl;
        sum += h_results[i];
    }

    double mean = sum / n;

    double sum_sq_diff = 0.0;
    for (double val : h_results) {
        sum_sq_diff += (val - mean) * (val - mean);
    }
    // 樣本標準偏差 (除以 n-1)
    double std_dev = std::sqrt(sum_sq_diff / (n - 1));

    // 7. 最終輸出格式化
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Result for Question 1:" << std::endl;
    std::cout << "Estimated LD12 Fluorescence Quantum Yield: " 
              << mean << " ± " << std_dev << " %" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // 8. 釋放記憶體
    cudaFree(d_std_area);
    cudaFree(d_unk_area);
    cudaFree(d_results);

    return 0;
}