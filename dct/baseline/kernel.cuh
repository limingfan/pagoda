#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "dct.h"
///////////////////////////////////////////////////////////////////////////////
// DCT
///////////////////////////////////////////////////////////////////////////////

__host__ __device__ void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{
    float X07P = FirstIn[0*StepIn] + FirstIn[7*StepIn];
    float X16P = FirstIn[1*StepIn] + FirstIn[6*StepIn];
    float X25P = FirstIn[2*StepIn] + FirstIn[5*StepIn];
    float X34P = FirstIn[3*StepIn] + FirstIn[4*StepIn];

    float X07M = FirstIn[0*StepIn] - FirstIn[7*StepIn];
    float X61M = FirstIn[6*StepIn] - FirstIn[1*StepIn];
    float X25M = FirstIn[2*StepIn] - FirstIn[5*StepIn];
    float X43M = FirstIn[4*StepIn] - FirstIn[3*StepIn];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    FirstOut[0*StepOut] = C_norm * (X07P34PP + X16P25PP);
    FirstOut[2*StepOut] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    FirstOut[4*StepOut] = C_norm * (X07P34PP - X16P25PP);
    FirstOut[6*StepOut] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    FirstOut[1*StepOut] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    FirstOut[3*StepOut] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    FirstOut[5*StepOut] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    FirstOut[7*StepOut] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

__host__ __device__ void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{
    float Y04P   = FirstIn[0*StepIn] + FirstIn[4*StepIn];
    float Y2b6eP = C_b * FirstIn[2*StepIn] + C_e * FirstIn[6*StepIn];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * FirstIn[7*StepIn] + C_a * FirstIn[1*StepIn] + C_c * FirstIn[3*StepIn] + C_d * FirstIn[5*StepIn];
    float Y7a1fM3d5cMP = C_a * FirstIn[7*StepIn] - C_f * FirstIn[1*StepIn] + C_d * FirstIn[3*StepIn] - C_c * FirstIn[5*StepIn];

    float Y04M   = FirstIn[0*StepIn] - FirstIn[4*StepIn];
    float Y2e6bM = C_e * FirstIn[2*StepIn] - C_b * FirstIn[6*StepIn];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * FirstIn[1*StepIn] - C_d * FirstIn[7*StepIn] - C_f * FirstIn[3*StepIn] - C_a * FirstIn[5*StepIn];
    float Y1d7cP3a5fMM = C_d * FirstIn[1*StepIn] + C_c * FirstIn[7*StepIn] - C_a * FirstIn[3*StepIn] + C_f * FirstIn[5*StepIn];

    FirstOut[0*StepOut] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    FirstOut[7*StepOut] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    FirstOut[4*StepOut] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    FirstOut[3*StepOut] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    FirstOut[1*StepOut] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    FirstOut[5*StepOut] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    FirstOut[2*StepOut] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    FirstOut[6*StepOut] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

__device__ void d_computeDCT(float *fDst, float *fSrc, int Stride, int SIZE, int TDD_NUM, int BLOCK_SIZE)
{

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int LENGTH = ((SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)/TDD_NUM);
    if(t < TDD_NUM){
        for(int i = 0; i < LENGTH; i++)
        {
            //process rows
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineDCTvector((float *)fSrc + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE)+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1);
            }
            //process columns
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineDCTvector(fDst + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE)) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k), Stride, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE)) * BLOCK_SIZE + k), Stride);
            }
        }
    }
}

__device__ void d_computeIDCT(float *fDst, float *fSrc, int Stride, int SIZE, int TDD_NUM, int BLOCK_SIZE)
{

	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int LENGTH = ((SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)/TDD_NUM);
	if(t < TDD_NUM){
          for(int i = 0; i < LENGTH; i++)
          {
            //process rows
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector((float *)fSrc + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE)+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1);
            }
            //process columns
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector(fDst + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE)) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k), Stride, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE)) * BLOCK_SIZE + k), Stride);
            }
        }
    }
}

void computeDCT(float *fDst, float *fSrc, int SIZE, int TDD_NUM, int BLOCK_SIZE, int Stride)
{

    int LENGTH = ((SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)/TDD_NUM);
    for(int t = 0; t < TDD_NUM; t++){
        for(int i = 0; i < LENGTH; i++)
        {
            //process rows
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineDCTvector((float *)fSrc + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE)+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1);
            }
            //process columns
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineDCTvector(fDst + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE)) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k), Stride, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE)) * BLOCK_SIZE + k), Stride);
            }
        }
    }
}

void computeIDCT(float *fDst, float *fSrc, int SIZE, int TDD_NUM, int BLOCK_SIZE, int Stride)
{
	int LENGTH = ((SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)/TDD_NUM);
        for(int t = 0; t < TDD_NUM; t++){
          for(int i = 0; i < LENGTH; i++)
          {
            //process rows
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector((float *)fSrc + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE)+k) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1);
            }
            //process columns
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                SubroutineIDCTvector(fDst + (((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE)) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k), Stride, fDst + ((((i*TDD_NUM+t)/(SIZE/BLOCK_SIZE))) * BLOCK_SIZE) * Stride + (((i*TDD_NUM+t)%(SIZE/BLOCK_SIZE)) * BLOCK_SIZE + k), Stride);
            }
        }
    }
}

__global__ void d_DCT(float *fSrc, float *fDst, int Stride, int size, int thread, int block){
    d_computeDCT(fSrc, fSrc, Stride, size, thread, block);
    d_computeIDCT(fDst, fSrc, Stride, size, thread, block);
}

void DCT(float *fSrc, float *fDst, int Stride, int size, int thread, int block){
    computeDCT(fSrc, fSrc, size, thread, block, Stride);
    computeIDCT(fDst, fSrc, size, thread, block, Stride);
}

