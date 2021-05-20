#include <assert.h>
#include <stdio.h>

#define NI 1000
#define NJ 2000
#define SI NJ
#define SJ 1

__global__ void add_2d_arrays(const double *a, double *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < NI - 1 && 0 < j && j < NJ - 1)
    {
        int n = i * SI + j * SJ;
        int nil = (i - 1) * SI + j * SJ;
        int nir = (i + 1) * SI + j * SJ;
        int njl = i * SI + (j - 1) * SJ;
        int njr = i * SI + (j + 1) * SJ;

        b[n] = (a[nir] - a[nil]) + (a[njr] - a[njl]);
    }
}

int main()
{
    printf("hello 2D!\n");
    double* host_a = (double*) malloc(NI * NJ * sizeof(double));
    double* host_b = (double*) malloc(NI * NJ * sizeof(double));

    for (int i = 0; i < NI; ++i)
    {
        for (int j = 0; j < NJ; ++j)
        {
            int n = i * SI + j * SJ;
            host_a[n] = i + j;
        }
    }

    double *a;
    double *b;
    cudaMalloc(&a, NI * NJ * sizeof(double));
    cudaMalloc(&b, NI * NJ * sizeof(double));

    cudaMemcpy(a, host_a, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, NI * NJ * sizeof(double), cudaMemcpyHostToDevice);

    int thread_per_dim_i = 32;
    int thread_per_dim_j = 32;
    int blocks_per_dim_i = (NI + thread_per_dim_i - 1) / thread_per_dim_i;
    int blocks_per_dim_j = (NJ + thread_per_dim_j - 1) / thread_per_dim_j;
    dim3 group_size = dim3(blocks_per_dim_i, blocks_per_dim_j);
    dim3 block_size = dim3(thread_per_dim_i, thread_per_dim_j);

    add_2d_arrays<<<group_size, block_size>>>(a, b);

    cudaError_t error = cudaGetLastError();

    if (error != 0)
    {
        printf("%s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(host_b, b, NI * NJ * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 1; i < NI - 1; ++i)
    {
        for (int j = 1; j < NJ - 1; ++j)
        {
            int n = i * SI + j * SJ;
            assert(host_b[n] == 4.0);
        }
    }

    cudaFree(a);
    cudaFree(b);

    free(host_a);
    free(host_b);
    return 0;
}
