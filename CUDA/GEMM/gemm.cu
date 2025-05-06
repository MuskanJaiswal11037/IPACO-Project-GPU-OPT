/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU

#define OTHER_DATA_TYPE double

void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
	int i,j,k;
	
	for (i = 0; i < _PB_NI; i++)
	{
    		for (j = 0; j < _PB_NJ; j++)
    		{
			C[i][j] *= beta;
	
			for (k = 0; k < _PB_NK; ++k)
			{
	  			C[i][j] += alpha * A[i][k] * B[k][j];
			}
      		}
	}
}


void init(int ni, int nj, int nk, DATA_TYPE* alpha, DATA_TYPE* beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nk; j++)
		{
      			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < nk; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			B[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < ni; i++)
	{
    		for (j = 0; j < nj; j++)
		{
      			C[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}
}


void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
	int i, j, fail;
	fail = 0;
	
	// Compare CPU and GPU outputs
	for (i=0; i < ni; i++) 
	{
		for (j=0; j < nj; j++) 
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NJ))
	{	
		c[i * NJ + j] *= beta;
		int k;
		for(k=0; k < _PB_NK; k++)
		{
			c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ +j];
		}
	}
}

__global__ void gemm_kernel_better(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NJ))
	{	
        DATA_TYPE val = 0;
		int k;
		for(k=0; k < _PB_NK; k++)
		{
			val += a[i + k*NJ] * b[k * NJ +j];
		}
        c[i*NJ + j] = c[i*NJ + j]*beta + val*alpha;
	}
}

__global__ void gemm_kernel_tiled(int ni, int nj, int nk, const DATA_TYPE alpha, const DATA_TYPE beta, const DATA_TYPE *__restrict__ a, const DATA_TYPE *__restrict__ b, DATA_TYPE *__restrict__ c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int TILE_SIZE = DIM_THREAD_BLOCK_X;

    __shared__ DATA_TYPE Asub[TILE_SIZE][TILE_SIZE];
    __shared__ DATA_TYPE Bsub[TILE_SIZE][TILE_SIZE];
    DATA_TYPE val = 0;

    for (int t = 0; t < (nk + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Load a tile of A and B into shared memory
        if (i < ni && t * TILE_SIZE + threadIdx.x < nk)
            Asub[threadIdx.y][threadIdx.x] = a[i * nk + (t * TILE_SIZE + threadIdx.x)];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0;

        if (j < nj && t * TILE_SIZE + threadIdx.y < nk)
            Bsub[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * nj + j];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++)
        {
            val += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        __syncthreads();
    }

	if ((i < _PB_NI) && (j < _PB_NJ))
	{	
		// int k;
		// for(k=0; k < _PB_NK; k++)
		// {
		// 	val += a[i + k*NJ] * b[k * NJ +j];
		// }
        c[i*NJ + j] = c[i*NJ + j]*beta + val*alpha;
	}
}

__global__ void gemm_kernel_tiled_32_64(int ni, int nj, int nk, const DATA_TYPE alpha, const DATA_TYPE beta, const DATA_TYPE *__restrict__ a, const DATA_TYPE *__restrict__ b, DATA_TYPE *__restrict__ c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int TILE_SIZE = DIM_THREAD_BLOCK_X;

    __shared__ DATA_TYPE Asub[TILE_SIZE][TILE_SIZE];
    __shared__ DATA_TYPE Bsub[TILE_SIZE][TILE_SIZE];
    __shared__ OTHER_DATA_TYPE Asub64[TILE_SIZE][TILE_SIZE];
    __shared__ OTHER_DATA_TYPE Bsub64[TILE_SIZE][TILE_SIZE];
    if (threadIdx.y >= 2){
        DATA_TYPE val = 0;

        for (int t = 0; t < (nk + TILE_SIZE - 1) / TILE_SIZE; t++)
        {
            // Load a tile of A and B into shared memory
            if (i < ni && t * TILE_SIZE + threadIdx.x < nk){
                Asub[threadIdx.y][threadIdx.x] = a[i * nk + (t * TILE_SIZE + threadIdx.x)];
                Asub64[threadIdx.y][threadIdx.x] = (OTHER_DATA_TYPE)a[i * nk + (t * TILE_SIZE + threadIdx.x)];
            } else {
                Asub[threadIdx.y][threadIdx.x] = 0.0;
                Asub64[threadIdx.y][threadIdx.x] = 0.0;
            }

            if (j < nj && t * TILE_SIZE + threadIdx.y < nk){
                Bsub[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * nj + j];
                Bsub64[threadIdx.y][threadIdx.x] = (OTHER_DATA_TYPE)b[(t * TILE_SIZE + threadIdx.y) * nj + j];
            } else {
                Bsub[threadIdx.y][threadIdx.x] = 0.0;
                Bsub64[threadIdx.y][threadIdx.x] = 0.0;
            }

            __syncthreads();

            // Compute partial product
            for (int k = 0; k < TILE_SIZE; k++)
            {
                val += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
            }

            __syncthreads();
        }

        if ((i < _PB_NI) && (j < _PB_NJ))
        {	
            // int k;
            // for(k=0; k < _PB_NK; k++)
            // {
            // 	val += a[i + k*NJ] * b[k * NJ +j];
            // }
            c[i*NJ + j] = c[i*NJ + j]*beta + val*alpha;
        }
    }
    else {
        OTHER_DATA_TYPE val64 = 0;

        for (int t = 0; t < (nk + TILE_SIZE - 1) / TILE_SIZE; t++)
        {
            if (i < ni && t * TILE_SIZE + threadIdx.x < nk){
                Asub[threadIdx.y][threadIdx.x] = a[i * nk + (t * TILE_SIZE + threadIdx.x)];
                Asub64[threadIdx.y][threadIdx.x] = (OTHER_DATA_TYPE)a[i * nk + (t * TILE_SIZE + threadIdx.x)];
            } else {
                Asub[threadIdx.y][threadIdx.x] = 0.0;
                Asub64[threadIdx.y][threadIdx.x] = 0.0;
            }

            if (j < nj && t * TILE_SIZE + threadIdx.y < nk){
                Bsub[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * nj + j];
                Bsub64[threadIdx.y][threadIdx.x] = (OTHER_DATA_TYPE)b[(t * TILE_SIZE + threadIdx.y) * nj + j];
            } else {
                Bsub[threadIdx.y][threadIdx.x] = 0.0;
                Bsub64[threadIdx.y][threadIdx.x] = 0.0;
            }

            __syncthreads();

            // Compute partial product
            for (int k = 0; k < TILE_SIZE; k++)
            {
                val64 += Asub64[threadIdx.y][k] * Bsub64[k][threadIdx.x];
            }

            __syncthreads();
        }

        if ((i < _PB_NI) && (j < _PB_NJ))
        {	
            // int k;
            // for(k=0; k < _PB_NK; k++)
            // {
            // 	val += a[i + k*NJ] * b[k * NJ +j];
            // }
            c[i*NJ + j] = c[i*NJ + j]*beta + val64*alpha;
        }
    }
}

__global__ void gemm_kernel_32_64(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, OTHER_DATA_TYPE * a64, OTHER_DATA_TYPE *b64)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NJ))
	{	
        c[i * NJ + j] *= beta;
        int k;
        if (blockIdx.x>=2){
            for(k=0; k < _PB_NK; k++)
            {
                c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ +j];
            }
        } else {
            OTHER_DATA_TYPE alpha64 = (OTHER_DATA_TYPE)alpha;
            OTHER_DATA_TYPE temp = 0;
            for(k=0; k < _PB_NK; k++)
            {
                temp += alpha64 * a64[i * NK + k] * b64[k * NJ +j];
            }
            c[i * NJ + j] += temp;
        }
	}
}

void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	/* Start timer. */
  	polybench_start_instruments;

	gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}

void gemmCudaBetter(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	/* Start timer. */
  	polybench_start_instruments;

	gemm_kernel_better<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}

void gemmCudaTiled(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	/* Start timer. */
  	polybench_start_instruments;

	gemm_kernel_tiled<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}

void gemmCudaTiled_32_64(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	/* Start timer. */
  	polybench_start_instruments;

	gemm_kernel_tiled_32_64<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}

void gemmCuda_32_64(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
	DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

    OTHER_DATA_TYPE* A64_gpu;
    OTHER_DATA_TYPE* B64_gpu;

	OTHER_DATA_TYPE *A64= (OTHER_DATA_TYPE* )malloc(sizeof(OTHER_DATA_TYPE) * NI * NK);
	OTHER_DATA_TYPE *B64= (OTHER_DATA_TYPE* )malloc(sizeof(OTHER_DATA_TYPE) * NK * NJ);

    for (int i = 0; i < NI*NK; i++){
        A64[i] = (OTHER_DATA_TYPE)((float*)A)[i];
    }
    for (int i = 0; i < NK*NJ; i++){
        B64[i] = (OTHER_DATA_TYPE)((float*)B)[i];
    }

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

	cudaMalloc((void **)&A64_gpu, sizeof(OTHER_DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B64_gpu, sizeof(OTHER_DATA_TYPE) * NK * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

	cudaMemcpy(A64_gpu, A64, sizeof(OTHER_DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B64_gpu, B64, sizeof(OTHER_DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	/* Start timer. */
  	polybench_start_instruments;

	// gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
	gemm_kernel_32_64<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu, A64_gpu, B64_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

	init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
	
	GPU_argv_init();
	
	// gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
	gemmCudaBetter(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
	gemmCudaTiled(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
	gemmCudaTiled_32_64(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
	// gemmCuda_32_64(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));


	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
		
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);  
	POLYBENCH_FREE_ARRAY(C);  
	POLYBENCH_FREE_ARRAY(C_outputFromGpu); 

    	return 0;
}

#include "../../common/polybench.c"
