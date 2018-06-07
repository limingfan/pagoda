#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "beam.h"
///////////////////////////////////////////////////////////////////////////////
// BEAMFORMERING
///////////////////////////////////////////////////////////////////////////////

__device__ void d_BeamFirFilter(int len,
                        float *weight, float *buffer,
                        float *in, float *out, int thread, int baseTid)
{
  /* Input must be exactly 2*decimation_ratio long; output must be
   * exactly 2 long. */
  int tid = baseTid + (threadIdx.x & 0x1f);
  //if(tid == 0) printf("size:%d, thread:%d\n", len, thread);
  int i, j;
  int modPos;
  int mask, mask2;
  mask = len - 1;
  mask2 = 2 * len - 1;
  if(tid < thread){
    for(j = 0; j < (len/thread); j++){
      float real_curr = 0;
      float imag_curr = 0;
      modPos = 2*(len - 1 - ((j*thread+tid) & mask));
      buffer[modPos] = in[(j*thread+tid) * 2 ];
      buffer[modPos+1] = in[(j*thread+tid) * 2 + 1];
#if 1
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
#endif
      out[(j*thread+tid) * 2] = real_curr;
      out[(j*thread+tid) * 2 + 1] = imag_curr;
    }
  }
}

