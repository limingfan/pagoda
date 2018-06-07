#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "con.h"

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
__device__ void convolutionRowsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int size, 
    int thread,
    int baseTid
)
{
    int x, k, d;
    int kernelR = KERNEL_RADIUS;
    int y = baseTid + (threadIdx.x & 0x1f);
    //if(y == 0) printf("before h_kernel:%f\n", h_Kernel[0]);
    if(y < thread){
        for (int x = 0; x < (size*size)/thread; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = ((x*thread+y)%size) + k;

                if (d >= 0 && d < size)
                    sum += h_Src[((x*thread+y)/size) * size + d] * h_Kernel[kernelR - k];
            }

            h_Dst[((x*thread+y)/size) * size + ((x*thread+y)%size)] = sum;
        }
    }
}


__device__ void convolutionColumnsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int size,
    int thread,
    int baseTid
)
{
    int x, k, d;
    int kernelR = KERNEL_RADIUS;
    int y = baseTid + (threadIdx.x & 0x1f);
    //if(y == 0) printf("after h_kernel:%f\n", h_Kernel[0]);

    if(y < thread){
        for (int x = 0; x < (size * size)/thread; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = ((x*thread+y)/size) + k;

                if (d >= 0 && d < size)
                    sum += h_Src[d * size + ((x*thread+y)%size)] * h_Kernel[kernelR - k];
            }

            h_Dst[((x*thread+y)/size) * size + ((x*thread+y)%size)] = sum;
        }
    }

}

