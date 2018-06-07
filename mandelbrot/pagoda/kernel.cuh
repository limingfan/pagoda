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
      			break;
    		}
    		x1 = x2;
    		y1 = y2;
  	}
}

__device__ void get_pixel(int *count, float *index, int TDD_NUM, int baseTid){

  	int tid = baseTid + (threadIdx.x & 0x1f);
  	int i;
  	float x, y;
  	if(tid < TDD_NUM){
    		for(i = 0; i < (n*n/TDD_NUM); i++){
      			x = ( ( float ) (     (i*TDD_NUM+tid)%n     ) * (x_max + *index)
          			+ ( float ) ( n - ((i*TDD_NUM+tid)%n) - 1 ) * (x_min + *index) )
          			/ ( float ) ( n     - 1 );

      			y = ( ( float ) (     (i*TDD_NUM+tid)/n     ) * (y_max + *index)
          			+ ( float ) ( n - ((i*TDD_NUM+tid)/n) - 1 ) * (y_min + *index) )
          			/ ( float ) ( n     - 1 );

      			explode ( x, y, &count[((i*TDD_NUM+tid)/n) + ((i*TDD_NUM+tid)%n) * n] );
    		}
  	}

}
