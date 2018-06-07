#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "bmult.h"
///////////////////////////////////////////////////////////////////////////////
// MATRIX MULTIPLICATION
///////////////////////////////////////////////////////////////////////////////

__device__ void mult_gpu(int *A, int *B, int *C, int tdd_num, int size, int baseTid){
  	int tid = baseTid + (threadIdx.x & 0x1f);
  	int i, k;
  	int sum = 0;
  	if(tid < tdd_num){
    		for(i = 0; i < (size*size/tdd_num); i++){
      			for(k = 0; k < size; k++){
        			sum += A[((i*tdd_num+tid)/size)*size+k] * B[k*size+((i*tdd_num+tid)%size)];
      			}
      		C[((i*tdd_num+tid)/size)*size+((i*tdd_num+tid)%size)] = sum;
      		if(k == size) sum = 0;
    		}	
  	}
}
