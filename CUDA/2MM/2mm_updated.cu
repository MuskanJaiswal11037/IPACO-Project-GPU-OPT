#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include<cuda_fp16.h>

#define POLYBENCH_TIME 1

#include "2mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
using namespace std;
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU


void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), 
		DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj), 
		DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
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
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < nl; i++)
	{
		for (j = 0; j < nj; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{   
			// printf("%d -->\n", percentDiff(D[i][j], D_outputFromGpu[i][j]));
			if (percentDiff(D[i][j], D_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}

__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *A, DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = tid/32;
	// printf("%d", warp_id);

	if((i < _PB_NI) && (j < _PB_NJ))
	{   
		tmp[i * NJ + j] = 0;
		if(warp_id %2 == 0){
			int k;
			for (k = 0; k < _PB_NK; k++)
			{	
				tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
			}
		}
		else{
			double alpha_d = (double)alpha;
			double acc = 0.0;
			for(int k =0; k <nk; ++k){
				acc += (alpha_d) * (double )A[i * NK + k] * (double)B[k * NJ + j];
			}
			tmp[i*NJ+j] = static_cast<float>(acc);
		}
	}
}

__global__ void mm2_kernel2(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *C, DATA_TYPE *D)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = tid/32;

	if ((i < _PB_NI) && (j < _PB_NL))
	{ 
		if(warp_id %2 == 0){
			D[i * NL + j] *= beta;
			int k;
			for(k = 0; k < _PB_NJ; k++)
			{
				D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
			}
		}
		else if(warp_id%2 == 1){
			double res = (double)D[i * NL + j];
			double beta1 = beta;
			res = res*beta1;
			for(int k = 0; k < _PB_NJ; k++){
				res += (double)tmp[i * NJ + k] * (double)C[k * NL + j];
			}
			D[i * NL + j] = static_cast<float>(res);
		}	
	}
}

void mm2_cpu(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
	int i, j, k;
	/* D := alpha*A*B*C + beta*D */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			tmp[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				tmp[i][j] += alpha * A[i][k] * B[k][j];
			}
		}
	}

	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			D[i][j] *= beta;
			for (k = 0; k < _PB_NJ; ++k)
			{
				D[i][j] += tmp[i][k] * C[k][j];
				// cout<<D[i][j]<<"  "<<endl;
			}
		}
	}
}
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl, DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


void mm2Cuda(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), 
	DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), 
	DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu,NI,NL,ni,nl))
{
	DATA_TYPE *tmp_gpu;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;

	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NL * NJ);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NI * NL);
	
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NL * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
		
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

	/* Start timer. */
  	polybench_start_instruments;

	mm2_kernel1<<<grid1,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, A_gpu, B_gpu);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	cudaThreadSynchronize();
	mm2_kernel2<<<grid2,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu);
	cudaThreadSynchronize();

	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

	cudaFree(tmp_gpu);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
	POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
	POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu,DATA_TYPE,NI,NL,ni,nl);
	
	/* Initialize array(s). */
  	init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
	GPU_argv_init();

	mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), 
		POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	#ifdef RUN_ON_CPU
		/* Start timer. */
	  	polybench_start_instruments;

		mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nl, POLYBENCH_ARRAY(D_outputFromGpu));

	#endif //RUN_ON_CPU
	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(D_outputFromGpu);

  	return 0;
}

#include "../../common/polybench.c"
