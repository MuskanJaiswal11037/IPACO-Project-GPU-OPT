/*****************************************************************************
 * 2mm_cutlass.cu  –  PolyBench “2 MM” solved with CUTLASS tensor‑core GEMM
 *                    (multistage cp.async pipeline, TF32→FP32 Accum)
 *
 *  D = β·D + α·(A·B)·C
 *
 *  CUTLASS chooses the tuned 128×128×32 / 64×64×32 / 16×8×8 pipeline
 *  described in the Colfax TF32 tutorial and feeds TF32 tensor cores.
 *****************************************************************************/

 #include <cuda_runtime.h>
 #include <cstdio>
 #include <cstdlib>
 
 #define POLYBENCH_TIME 1
 #include "2mm.cuh"
 #include "../../common/polybench.h"
 #include "../../common/polybenchUtilFuncts.h"
 
 /* ---------------- CUTLASS ------------------------------------------------ */
 #include <cutlass/layout/matrix.h>
 #include <cutlass/gemm/device/gemm.h>
 
 using RowMajor = cutlass::layout::RowMajor;
 
 /* ---------- simple alias: let CUTLASS pick the best kernel -------------- */
 using GemmRRR = cutlass::gemm::device::Gemm<
         float, RowMajor,   /* A */
         float, RowMajor,   /* B */
         float, RowMajor,   /* C (also D) */
         float>;            /* accumulator */


 
 /* ---------- small helper ------------------------------------------------ */
 #define CUTLASS_CHECK(status)                                 \
   { if (status != cutlass::Status::kSuccess) {                \
       printf("CUTLASS error: %d\n", int(status));             \
       std::exit(EXIT_FAILURE); }}
 
 /* --------------------------------------------------------------------- */
 /*  GPU version using two CUTLASS GEMMs                                  */
 /* --------------------------------------------------------------------- */
 void mm2Cuda(int ni,int nj,int nk,int nl,
              float alpha, float beta,
              float POLYBENCH_2D(tmp,NI,NJ,ni,nj),
              float POLYBENCH_2D(A,NI,NK,ni,nk),
              float POLYBENCH_2D(B,NK,NJ,nk,nj),
              float POLYBENCH_2D(C,NL,NJ,nl,nj),
              float POLYBENCH_2D(D,NI,NL,ni,nl),
              float POLYBENCH_2D(D_gpu,NI,NL,ni,nl))
 {
     float *dA,*dB,*dC,*dTmp,*dD;
     cudaMalloc(&dA, NI*NK*sizeof(float));
     cudaMalloc(&dB, NK*NJ*sizeof(float));
     cudaMalloc(&dC, NL*NJ*sizeof(float));
     cudaMalloc(&dTmp,NI*NJ*sizeof(float));
     cudaMalloc(&dD, NI*NL*sizeof(float));
 
     cudaMemcpy(dA,A, NI*NK*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(dB,B, NK*NJ*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(dC,C, NL*NJ*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(dD,D, NI*NL*sizeof(float), cudaMemcpyHostToDevice);
 
     /* GEMM #1 : tmp = α · A · B  ------------------------------------ */
     GemmRRR gemmAB;
     GemmRRR::Arguments argsAB(
         {ni, nj, nk},            /* (M,N,K) */
         {dA, nk}, {dB, nj},
         {nullptr, 0},            /* unused C */
         {dTmp, nj},              /* output */
         {alpha, 0.0f});          /* epilogue coefficients */
     CUTLASS_CHECK(gemmAB.initialize(argsAB));
     CUTLASS_CHECK(gemmAB());
 
     /* GEMM #2 : D = 1·tmp·C + β·D  ---------------------------------- */
     GemmRRR gemm2;
     GemmRRR::Arguments args2(
         {ni, nl, nj},
         {dTmp, nj}, {dC, nl},
         {dD, nl},                  /* initial D (beta term) */
         {dD, nl},                  /* output D */
         {1.0f, beta});
     CUTLASS_CHECK(gemm2.initialize(args2));
     CUTLASS_CHECK(gemm2());
 
     cudaMemcpy(D_gpu, dD, NI*NL*sizeof(float), cudaMemcpyDeviceToHost);
 
     cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dTmp); cudaFree(dD);
 }
 
 /* --------------------------------------------------------------------- */
 /*  ------------ ORIGINAL PolyBench UTILITIES (unchanged) ---------------*/
 void init_array(int ni,int nj,int nk,int nl,
                 float *alpha, float *beta,
                 float POLYBENCH_2D(A,NI,NK,ni,nk),
                 float POLYBENCH_2D(B,NK,NJ,nk,nj),
                 float POLYBENCH_2D(C,NL,NJ,nl,nj),
                 float POLYBENCH_2D(D,NI,NL,ni,nl))
 {
   int i,j; *alpha=32412.0f; *beta=2123.0f;
   for(i=0;i<ni;i++) for(j=0;j<nk;j++) A[i][j]=((float)i*j)/NI;
   for(i=0;i<nk;i++) for(j=0;j<nj;j++) B[i][j]=((float)i*(j+1))/NJ;
   for(i=0;i<nl;i++) for(j=0;j<nj;j++) C[i][j]=((float)i*(j+3))/NL;
   for(i=0;i<ni;i++) for(j=0;j<nl;j++) D[i][j]=((float)i*(j+2))/NK;
 }
 
 #define PERCENT_DIFF_ERROR_THRESHOLD 0.05f
 void compareResults(int ni,int nl,
                     float POLYBENCH_2D(D1,NI,NL,ni,nl),
                     float POLYBENCH_2D(D2,NI,NL,ni,nl))
 {
   int fail=0;
   for(int i=0;i<ni;i++) for(int j=0;j<nl;j++)
     if(percentDiff(D1[i][j],D2[i][j])>PERCENT_DIFF_ERROR_THRESHOLD) fail++;
   printf("Non‑matching elements >%.2f%% : %d\n",
          PERCENT_DIFF_ERROR_THRESHOLD, fail);
 }
 
 void GPU_argv_init(){
   cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
   printf("Using GPU 0 : %s\n", p.name);  cudaSetDevice(0);
 }
 
 void mm2_cpu(int ni,int nj,int nk,int nl,
              float alpha,float beta,
              float POLYBENCH_2D(tmp,NI,NJ,ni,nj),
              float POLYBENCH_2D(A,NI,NK,ni,nk),
              float POLYBENCH_2D(B,NK,NJ,nk,nj),
              float POLYBENCH_2D(C,NL,NJ,nl,nj),
              float POLYBENCH_2D(D,NI,NL,ni,nl))
 {
   for(int i=0;i<ni;i++)
     for(int j=0;j<nj;j++){
       tmp[i][j]=0;
       for(int k=0;k<nk;k++)
         tmp[i][j]+=alpha*A[i][k]*B[k][j];
     }
   for(int i=0;i<ni;i++)
     for(int j=0;j<nl;j++){
       D[i][j]*=beta;
       for(int k=0;k<nj;k++)
         D[i][j]+=tmp[i][k]*C[k][j];
     }
 }
 
 /* --------------------------------------------------------------------- */
 int main()
 {
   int ni=NI,nj=NJ,nk=NK,nl=NL;
   float alpha,beta;
 
   POLYBENCH_2D_ARRAY_DECL(tmp,float,NI,NJ,ni,nj);
   POLYBENCH_2D_ARRAY_DECL(A,float,NI,NK,ni,nk);
   POLYBENCH_2D_ARRAY_DECL(B,float,NK,NJ,nk,nj);
   POLYBENCH_2D_ARRAY_DECL(C,float,NL,NJ,nl,nj);
   POLYBENCH_2D_ARRAY_DECL(D,float,NI,NL,ni,nl);
   POLYBENCH_2D_ARRAY_DECL(D_gpu,float,NI,NL,ni,nl);
 
   init_array(ni,nj,nk,nl,&alpha,&beta,
              POLYBENCH_ARRAY(A),POLYBENCH_ARRAY(B),
              POLYBENCH_ARRAY(C),POLYBENCH_ARRAY(D));
 
   GPU_argv_init();
 
   polybench_start_instruments;
   mm2Cuda(ni,nj,nk,nl,alpha,beta,
           POLYBENCH_ARRAY(tmp),POLYBENCH_ARRAY(A),POLYBENCH_ARRAY(B),
           POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_gpu));
   printf("GPU time:\n");
   polybench_stop_instruments; polybench_print_instruments;
 
   polybench_start_instruments;
   mm2_cpu(ni,nj,nk,nl,alpha,beta,
           POLYBENCH_ARRAY(tmp),POLYBENCH_ARRAY(A),POLYBENCH_ARRAY(B),
           POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
   printf("CPU time:\n");
   polybench_stop_instruments; polybench_print_instruments;
 
   compareResults(ni,nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_gpu));

   return 0;
 }
 
 #include "../../common/polybench.c"
 
