#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
///////////////////////////////////////////////////////////////////////////////
// MATRIX MULTIPLICATION
///////////////////////////////////////////////////////////////////////////////

__device__ void mult_gpu(float *A, float *B, float *C, int size, int thread, int baseTid){
	int tid = baseTid + (threadIdx.x & 0x1f);
  	int i, k;
  	float sum = 0;

  	if(tid < thread){
    		for(i = 0; i < (size*size/thread); i++){
      			for(k = 0; k < size; k++){
        			sum += A[((i*thread+tid)/size)*size+k] * B[k*size+((i*thread+tid)%size)];
      		}
      			C[((i*thread+tid)/size)*size+((i*thread+tid)%size)] = sum;
      			if(k == size) sum = 0;
    		}

  	}

}
