#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "beam.h"
///////////////////////////////////////////////////////////////////////////////
// BEAMFORMERING
///////////////////////////////////////////////////////////////////////////////

__device__ void BeamFirFilter_dev(int len,
                        int input_length, int decimation_ratio,
                        float *weight, float *buffer,
                        float *in, float *out, int baseTid)
{
  /* Input must be exactly 2*decimation_ratio long; output must be
   * exactly 2 long. */
//  int tid = threadIdx.x + blockIdx.x *blockDim.x;
  int tid = baseTid + (threadIdx.x & 0x1f);
  int i, j;
  int modPos;
  int mask, mask2;

  //len = data->len;
  mask = len - 1;
  mask2 = 2 * len - 1;
  //for(k = 0; k < TD_NUM; k++){
  if(tid < TDD_NUM){
    for(j = 0; j < (input_length/TDD_NUM); j++){
      float real_curr = 0;
      float imag_curr = 0;
      //modPos = 2*(len - 1 - data->pos);
      modPos = 2*(len - 1 - ((j*TDD_NUM+tid) & mask));
      buffer[modPos] = in[(j*TDD_NUM+tid) * decimation_ratio * 2 ];
      buffer[modPos+1] = in[(j*TDD_NUM+tid) * decimation_ratio * 2 + 1];

      /* Profiling says: this is the single inner loop that matters! */
      for (i = 0; i < 2*len; i+=2) {
        float rd = buffer[modPos];
        float id = buffer[modPos+1];
        float rw = weight[i];
        float iw = weight[i+1];
        float rci = rd * rw + id * iw;
        /* sign error?  this is consistent with StreamIt --dzm */
        float ici = id * rw + rd * iw;
        real_curr += rci;
        imag_curr += ici;
        modPos = (modPos + 2) & mask2;
      }
      //data->pos = (data->pos + 1) & mask;
      out[(j*TDD_NUM+tid) * 2] = real_curr;
      out[(j*TDD_NUM+tid) * 2 + 1] = imag_curr;
    }
  }
}

