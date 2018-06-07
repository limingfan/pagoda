#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
__device__ void syncBlock();
#include "filter.h"
extern __device__ int syncID;
extern __device__ int threadNum;

///////////////////////////////////////////////////////////////////////////////
// FILTERBANK
///////////////////////////////////////////////////////////////////////////////

__device__ void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, 
		float *Vect_Up, float *Vect_F, float *F, int N_sim, int TDD_NUM, int baseTid, int barId){
	int tid = baseTid + (threadIdx.x & 0x1f);
  	int j, k;

  	//convolving H
  	if(tid < TDD_NUM){
    		for (j=0; j< (N_sim/TDD_NUM); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*TDD_NUM+tid)-k)>=0){
          				Vect_H[j*TDD_NUM+tid] += (r[(j*TDD_NUM+tid)-k]*H[k]);
        			}
      			}
    		}
  	}

 	__syncthreads_block(barId, threadNum);

  	//Down Sampling
  	if(tid < TDD_NUM)
    		for (j=0; j < N_sim/N_samp/TDD_NUM; j++)
      			Vect_Dn[(j*TDD_NUM+tid)]=Vect_H[(j*TDD_NUM+tid)*N_samp];

  	//Up Sampling
  	if(tid < TDD_NUM)
    		for (j=0; j < N_sim/N_samp/TDD_NUM;j++)
      			Vect_Up[(j*TDD_NUM+tid)*N_samp]=Vect_Dn[(j*TDD_NUM+tid)];

  	__syncthreads_block(barId, threadNum);

  	//convolving F
  	if(tid < TDD_NUM){
    		for (j=0; j< (N_sim/TDD_NUM); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*TDD_NUM+tid)-k)>=0){
          				Vect_F[j*TDD_NUM+tid]+=(F[k]*Vect_Up[(j*TDD_NUM+tid)-k]);
        			}
      			}
    		}
  	}

}

