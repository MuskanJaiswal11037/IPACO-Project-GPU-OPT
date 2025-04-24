/*************************************************************************
 * PolyBench/GPU 2-MM — Tensor-Core version
 *   Computes D = β·D + α·A·B·C   (all row-major)
 *   Only GPU math is changed; everything else is identical to
 *   the original shared-memory–tiled program.
 *************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;                    // wmma::

#define POLYBENCH_TIME 1
#include "2mm.cuh"                         // NI,NJ,NK,NL, DATA_TYPE=float
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

/* ---------------- Tensor-Core WMMA tile ---------------------------- */
#define WM 16
#define WN 16
#define WK 16                  /* 16×16×16 MMA                        */
#define WARPS_PER_BLOCK 32     /* one warp / WMMA tile               */
/* ------------------------------------------------------------------- */

#define GPU_DEVICE 0
#define ERR_THRESH 0.05f       

/* =================================================================== */
/* ----------  ORIGINAL helper functions  ---------------- */
static void init_array(int ni,int nj,int nk,int nl, DATA_TYPE *alpha, DATA_TYPE *beta,
                    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                    DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl)){

    *alpha = 32412.0f;
    *beta  = 2123.0f;

    for(int i=0; i<ni; i++)
        for(int k=0; k<nk; k++) 
            A[i][k] = (DATA_TYPE)(i*k)/NI;

    for(int k=0; k<nk; k++)
        for(int j=0; j<nj; j++) 
            B[k][j] = (DATA_TYPE)(k*(j+1))/NJ;

    for(int l=0; l<nl; l++)
        for(int j=0; j<nj; j++) 
            C[l][j] = (DATA_TYPE)(l*(j+3))/NL;

    for(int i=0; i<ni; i++)
        for(int l=0; l<nl; l++) 
            D[i][l] = (DATA_TYPE)(i*(l+2))/NK;
}

static void compareResults(int ni,int nl, DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(Dg,NI,NL,ni,nl)){

    int fail=0;
    for(int i=0; i<ni; i++)
        for(int l=0; l<nl; l++)
            if(percentDiff(D[i][l],Dg[i][l]) > ERR_THRESH) 
                ++fail;

    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold "
        "(%4.2f%%): %d\n", ERR_THRESH, fail);
}

static void GPU_argv_init()
{
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp,GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

/* -------------------  CPU reference  --------------------- */
static void mm2_cpu(int ni,int nj,int nk,int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                    DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl)){

    for(int i=0; i<ni; i++)
        for(int j=0; j<nj; j++){
            DATA_TYPE acc=0.f;
            for(int k=0; k<nk; k++) 
                acc += alpha*A[i][k]*B[k][j];
            tmp[i][j]=acc;
        }

    for(int i=0; i<ni; i++)
        for(int l=0; l<nl; l++){
            DATA_TYPE acc=beta*D[i][l];
            for(int j=0; j<nj; j++) 
                acc += tmp[i][j]*C[j][l];
            D[i][l]=acc;
        }
}

/* =================================================================== */
/* ---------------------  device helper kernels  --------------------- */
__global__ void f32_to_f16(const float *in,__half *out,int n){

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<n) 
        out[idx] = __float2half(in[idx]);
}

/* tmp = α · A · B   (M=NI, N=NJ, K=NK) */
__global__ void wmma_kernel1(int M,int N,int K,float alpha, const __half *A,const __half *B,float *tmp){

    int tm = blockIdx.y, tn = blockIdx.x;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for(int k=0; k<K; k += WK){
        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> Af;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> Bf;
        wmma::load_matrix_sync(Af, A+(tm*WM)*K+k, K);
        wmma::load_matrix_sync(Bf, B+k*N+tn*WN, N);
        wmma::mma_sync(acc, Af, Bf, acc);
    }

    for(int i=0;i<acc.num_elements;i++) 
        acc.x[i] *= alpha;

    wmma::store_matrix_sync(tmp+(tm*WM)*N+tn*WN, acc, N, wmma::mem_row_major);
}

/* D = β·D + tmp · C   (M=NI, N=NL, K=NJ) */
__global__ void wmma_kernel2(int M,int N,int K,float beta, const __half *tmp_h,const __half *C_h,float *D){

    int tm = blockIdx.y, tn = blockIdx.x;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for(int k=0; k<K; k+=WK){
        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half,wmma::row_major> Af;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> Bf;
        wmma::load_matrix_sync(Af, tmp_h+(tm*WM)*K+k, K);
        wmma::load_matrix_sync(Bf, C_h+k*N+tn*WN, N);
        wmma::mma_sync(acc, Af, Bf, acc);
    }

    /* β scaling with original D */
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> Dold;
    float *dst = D+(tm*WM)*N+tn*WN;

    wmma::load_matrix_sync(Dold, dst, N, wmma::mem_row_major);
    for(int i=0; i<acc.num_elements; i++) 
        acc.x[i] += beta*Dold.x[i];

    wmma::store_matrix_sync(dst, acc, N, wmma::mem_row_major);
}

/* =================================================================== */
/* -----------------------  GPU driver  ------------------------------ */
void mm2Cuda(int ni,int nj,int nk,int nl, DATA_TYPE alpha, DATA_TYPE beta,
            DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
            DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
            DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(Dg,NI,NL,ni,nl)){

    size_t sA = sizeof(float)*NI*NK;
    size_t sB = sizeof(float)*NK*NJ;
    size_t sC = sizeof(float)*NL*NJ; 
    size_t sD = sizeof(float)*NI*NL;
    size_t sT = sizeof(float)*NI*NJ;

    /* FP32 device buffers */
    float *dA,*dB,*dC,*dD,*dTmp;
    cudaMalloc(&dA,sA); 
    cudaMalloc(&dB,sB);
    cudaMalloc(&dC,sC); 
    cudaMalloc(&dD,sD); 
    cudaMalloc(&dTmp,sT);

    cudaMemcpy(dA, &A[0][0], sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &B[0][0], sB, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, &C[0][0], sC, cudaMemcpyHostToDevice);
    cudaMemcpy(dD, &D[0][0], sD, cudaMemcpyHostToDevice);

    /* FP16 copies for WMMA */
    __half *dA_h, *dB_h, *dC_h, *dTmp_h;

    cudaMalloc(&dA_h, sA/2); 
    cudaMalloc(&dB_h, sB/2);
    cudaMalloc(&dC_h, sC/2); 
    cudaMalloc(&dTmp_h, sT/2);

    int threads=256;
    f32_to_f16 <<< (NI*NK+threads-1)/threads, threads >>> (dA, dA_h, NI*NK);
    f32_to_f16 <<< (NK*NJ+threads-1)/threads, threads >>> (dB, dB_h, NK*NJ);
    f32_to_f16 <<< (NL*NJ+threads-1)/threads, threads >>> (dC, dC_h, NL*NJ);

    dim3 block(WARPS_PER_BLOCK,1);
    dim3 grid1(NJ/WM, NI/WM);          /* tmp = αAB   */
    dim3 grid2(NL/WM, NI/WM);          /* D = βD+tmpC */

    polybench_start_instruments;

    wmma_kernel1<<<grid1,block>>> (NI, NJ, NK, alpha, dA_h, dB_h, dTmp);
    f32_to_f16 <<< (NI*NJ+threads-1)/threads,threads >>> (dTmp, dTmp_h, NI*NJ);
    wmma_kernel2<<<grid2,block>>>(NI, NL, NJ, beta, dTmp_h, dC_h, dD);

    cudaDeviceSynchronize();
    printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(&Dg[0][0], dD, sD, cudaMemcpyDeviceToHost);

    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC); 
    cudaFree(dD); 
    cudaFree(dTmp);
    cudaFree(dA_h); 
    cudaFree(dB_h); 
    cudaFree(dC_h); 
    cudaFree(dTmp_h);
}

/* =================================================================== */
/* ------------------------------ main ------------------------------- */
int main()
{
    int ni=NI, nj=NJ, nk=NK, nl=NL;
    DATA_TYPE alpha, beta;

    POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NL, NJ, nl, nj);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NI, NL, ni, nl);
    POLYBENCH_2D_ARRAY_DECL(Dgpu, DATA_TYPE, NI, NL, ni, nl);

    init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
    GPU_argv_init();

    mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(Dgpu));


    polybench_start_instruments;
    
    mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(Dgpu));

    POLYBENCH_FREE_ARRAY(tmp); 
    POLYBENCH_FREE_ARRAY(A); 
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);   
    POLYBENCH_FREE_ARRAY(D); 
    POLYBENCH_FREE_ARRAY(Dgpu);

    return 0;
}

/* bring in PolyBench timer implementation */
#include "../../common/polybench.c"
 