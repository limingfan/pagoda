#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
__device__ void syncBlock();
#include "filter.h"

///////////////////////////////////////////////////////////////////////////////
// FILTERBANK
///////////////////////////////////////////////////////////////////////////////

__device__ void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, 
		float *Vect_Up, float *Vect_F, float *F, int threads, int size, int baseTid, int barId){
  	int tid = baseTid + (threadIdx.x & 0x1f);
  	int j, k;

  	//convolving H
  	if(tid < threads){
    		for (j=0; j< (size/threads); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*threads+tid)-k)>=0){
          				Vect_H[j*threads+tid] += (r[(j*threads+tid)-k]*H[k]);
        			}
      			}
    		}
  	}
 	__syncthreads_block(barId, threads);

  	//Down Sampling
  	if(tid < threads)
    		for (j=0; j < size/N_samp/threads; j++)
      			Vect_Dn[(j*threads+tid)]=Vect_H[(j*threads+tid)*N_samp];

  	//Up Sampling
  	if(tid < threads)
    		for (j=0; j < size/N_samp/threads;j++)
      			Vect_Up[(j*threads+tid)*N_samp]=Vect_Dn[(j*threads+tid)];

  	__syncthreads_block(barId, threads);

  	//convolving F
  	if(tid < threads){
    		for (j=0; j< (size/threads); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*threads+tid)-k)>=0){
          				Vect_F[j*threads+tid]+=(F[k]*Vect_Up[(j*threads+tid)-k]);
        			}
      			}
    		}
  	}
}

