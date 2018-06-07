#define __syncthreads_block(blockIndex, thread_num) asm volatile("bar.sync %0, %1;" :: "r"(blockIndex), "r"(thread_num));
#include "con.h"

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
__device__ void convolutionRowsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int	  imageW,
    int   imageH,
    int   TDD_NUM,
    int baseTid
)
{
    int x, k, d;
    int kernelR = KERNEL_RADIUS;
    int y = baseTid + (threadIdx.x & 0x1f);
    //if(y == 0) printf("before h_kernel:%f\n", h_Kernel[0]);
    if(y < TDD_NUM){
        for (int x = 0; x < (imageW*imageH)/TDD_NUM; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = ((x*TDD_NUM+y)%imageW) + k;

                if (d >= 0 && d < imageW)
                    sum += h_Src[((x*TDD_NUM+y)/imageH) * imageW + d] * h_Kernel[kernelR - k];
            }

            h_Dst[((x*TDD_NUM+y)/imageH) * imageW + ((x*TDD_NUM+y)%imageW)] = sum;
        }
    }
}


__device__ void convolutionColumnsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int TDD_NUM,
    int baseTid
)
{
    int x, k, d;
    int kernelR = KERNEL_RADIUS;
    int y = baseTid + (threadIdx.x & 0x1f);
    //if(y == 0) printf("after h_kernel:%f\n", h_Kernel[0]);

    if(y < TDD_NUM){
        for (int x = 0; x < (imageW * imageH)/TDD_NUM; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = ((x*TDD_NUM+y)/imageH) + k;

                if (d >= 0 && d < imageH)
                    sum += h_Src[d * imageW + ((x*TDD_NUM+y)%imageW)] * h_Kernel[kernelR - k];
            }

            h_Dst[((x*TDD_NUM+y)/imageH) * imageW + ((x*TDD_NUM+y)%imageW)] = sum;
        }
    }

}

