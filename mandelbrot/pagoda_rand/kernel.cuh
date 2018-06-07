#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "mand.h"
///////////////////////////////////////////////////////////////////////////////
// MANDELBROT IMAGE
///////////////////////////////////////////////////////////////////////////////

__device__ void explode ( float x, float y, int *value){
  int k;
  float x1;
  float x2;
  float y1;
  float y2;
  //int value;
  *value = 0;

  x1 = x;
  y1 = y;

  for ( k = 1; k <= count_max; k++ )
  {
    x2 = x1 * x1 - y1 * y1 + x;
    y2 = 2.0 * x1 * y1 + y;

    if ( x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2 )
    {
      *value = k;
      //if(k > 1000)
         //printf("k:%d\n", k);
      break;
    }
    x1 = x2;
    y1 = y2;
  }
}

__device__ void get_pixel(int *count, float *index, int threads, int size, int baseTid){

  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = baseTid + (threadIdx.x & 0x1f);
  int i;
  float x, y;
  if(tid < threads){
    for(i = 0; i < (size*size/threads); i++){
      x = ( ( float ) (     (i*threads+tid)%size     ) * (x_max + *index)
          + ( float ) ( size - ((i*threads+tid)%size) - 1 ) * (x_min + *index) )
          / ( float ) ( size     - 1 );

      y = ( ( float ) (     (i*threads+tid)/size     ) * (y_max + *index)
          + ( float ) ( size - ((i*threads+tid)/size) - 1 ) * (y_min + *index) )
          / ( float ) ( size     - 1 );

      explode ( x, y, &count[((i*threads+tid)/size) + ((i*threads+tid)%size) * size] );
    }
  }

}
