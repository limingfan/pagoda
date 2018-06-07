#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "con.h"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

__global__ void convolutionRowsGPU(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int imageW, int imageH, int TD_NUM);
__global__ void convolutionColumnsGPU(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int imageW, int imageH, int TD_NUM);

void convolutionRowCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int imageW, int imageH);
void convolutionColumnCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int imageW, int imageH);

int main(int argc, char *argv[]){

	float **h_Kernel, **h_Input, **d_Buffer, **h_OutputGPU, **h_Buffer, **h_OutputCPU;
	float **d_Output, **d_Kernel, **d_Input;

	int i, j;
	double start_timer, end_timer;

	if(argc < 5){
                printf("Input option: ./convolution imageW imageH #thread #task\n");
                exit(1);
        }
        int imageW = atoi(argv[1]);
        int imageH = atoi(argv[2]);
        int TD_NUM = atoi(argv[3]);
        int NUM_TASK = atoi(argv[4]);

	printf("CUDA baseline Conv: image=%d x %d, #thread:%d, #task:%d\n", imageW, imageH, TD_NUM, NUM_TASK);

	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

	printf("Initializing data...\n");

	cudaStream_t *con_stream;
	con_stream = (cudaStream_t*)malloc(NUM_TASK*sizeof(con_stream));

	for(i = 0; i < NUM_TASK; i++){
    		checkCudaErrors(cudaStreamCreate(&con_stream[i]));
  	}


	h_Kernel    = (float **)malloc(NUM_TASK * sizeof(float*));
	h_Input     = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Buffer    = (float **)malloc(NUM_TASK * sizeof(float*));
	h_OutputGPU = (float **)malloc(NUM_TASK * sizeof(float*));
	h_OutputCPU = (float **)malloc(NUM_TASK * sizeof(float*));
	h_Buffer    = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Output = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Kernel = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Input = (float **)malloc(NUM_TASK * sizeof(float*));

	for(i = 0; i < NUM_TASK; i++){
  		//h_Kernel[i]    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
  		checkCudaErrors(cudaHostAlloc(&h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaHostAllocDefault));
  		//h_Input[i]     = (float *)malloc(imageW * imageH * sizeof(float));
  		checkCudaErrors(cudaHostAlloc(&h_Input[i], imageW * imageH*sizeof(float), cudaHostAllocDefault));
 		// h_Buffer[i]    = (float *)malloc(imageW * imageH * sizeof(float));
  		checkCudaErrors(cudaMalloc(&d_Buffer[i], imageW * imageH * sizeof(float)));
  		//h_OutputGPU[i] = (float *)malloc(imageW * imageH * sizeof(float));
  		checkCudaErrors(cudaHostAlloc(&h_OutputGPU[i], imageW * imageH*sizeof(float), cudaHostAllocDefault));
  		checkCudaErrors(cudaMalloc(&d_Output[i], imageW * imageH * sizeof(float)));
  		checkCudaErrors(cudaMalloc(&d_Input[i], imageW * imageH * sizeof(float)));
  		checkCudaErrors(cudaMalloc(&d_Kernel[i], KERNEL_LENGTH * sizeof(float)));

		h_Buffer[i] = (float*)malloc(imageW * imageH * sizeof(float));
  		h_OutputCPU[i] = (float*)malloc(imageW * imageH * sizeof(float));

	}

	for(i = 0; i < NUM_TASK;i++){
  		for (j = 0; j < KERNEL_LENGTH; j++){
    			h_Kernel[i][j] = (float)j/KERNEL_LENGTH;
  		}
	}

	for(i = 0; i < NUM_TASK;i++){
  		for (j = 0; j < imageW * imageH; j++){
    			//h_Input[i][j] = (float)((j/imageW)%2);
			h_Input[i][j] = rand()%100;

  		}
	}

	//mem. copy
	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaMemcpyAsync(d_Kernel[i], h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaMemcpyHostToDevice, con_stream[i%32]));
  		checkCudaErrors(cudaMemcpyAsync(d_Input[i], h_Input[i], imageW * imageH*sizeof(float), cudaMemcpyHostToDevice, con_stream[i%32]));

	}
	checkCudaErrors(cudaDeviceSynchronize());


	start_timer = my_timer();
	for(i = 0; i < NUM_TASK; i++){
  		convolutionRowsGPU<<<1, TD_NUM, 0, con_stream[i%32]>>>(d_Buffer[i], d_Input[i], d_Kernel[i], KERNEL_RADIUS, imageW, imageH, TD_NUM);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	for(i = 0; i < NUM_TASK; i++){
  		convolutionColumnsGPU<<<1, TD_NUM, 0, con_stream[i%32]>>>(d_Output[i], d_Buffer[i], d_Kernel[i],KERNEL_RADIUS, imageW, imageH, TD_NUM);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	end_timer = my_timer();
	printf("GPU elapsed time:%lf Sec.\n", end_timer - start_timer);

	start_timer = my_timer();
	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaMemcpyAsync(h_OutputGPU[i], d_Output[i], imageW * imageH*sizeof(float), cudaMemcpyDeviceToHost, con_stream[i%32]));

	}
	checkCudaErrors(cudaDeviceSynchronize());

	printf("CPU convolution Start\n");
  	start_timer = my_timer();

  	for(i = 0; i < NUM_TASK; i++){
    		convolutionRowCPU( h_Buffer[i], h_Input[i], h_Kernel[i], KERNEL_RADIUS, imageW, imageH);
  	}
  	for(i = 0; i < NUM_TASK; i++){
    		convolutionColumnCPU( h_OutputCPU[i], h_Buffer[i], h_Kernel[i], KERNEL_RADIUS, imageW, imageH);
  	}
  	end_timer = my_timer();
  	printf("CPU exec.time:%lf Sec.\n", end_timer - start_timer);

	//verification
  	printf("verify\n");
	long long flag = 0;
  	for(i = 0; i < NUM_TASK; i++){
    		for(j = 0; j < imageW * imageH; j++){
        		if(fabs(h_OutputCPU[i][j] - h_OutputGPU[i][j] > 0.1)){
          			printf("Error:%f, %f, %d, %d\n", h_OutputCPU[i][j], h_OutputGPU[i][j]);
          		break;
        		}
			flag ++;
    		}
  	}
	if(flag == (NUM_TASK * imageW * imageH)) printf("verify successfully\n");

	//free mem.
	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaStreamDestroy(con_stream[i]));
	}
	for(i = 0; i < NUM_TASK; i++){
  		//free(h_Buffer[i]);
  		checkCudaErrors(cudaFree(d_Buffer[i]));
  		//free(h_Input[i]);
  		checkCudaErrors(cudaFreeHost(h_Input[i]));
  		//free(h_Kernel[i]);
  		checkCudaErrors(cudaFreeHost(h_Kernel[i]));
  		//free(h_OutputGPU[i]);
  		checkCudaErrors(cudaFreeHost(h_OutputGPU[i]));
  		checkCudaErrors(cudaFree(d_Kernel[i]));
  		checkCudaErrors(cudaFree(d_Output[i]));
  		checkCudaErrors(cudaFree(d_Input[i]));

		free(h_Buffer[i]);
  		free(h_OutputCPU[i]);

	}

	free(d_Buffer);
	free(h_Input);
	free(h_Kernel);
	free(d_Kernel);
	free(d_Output);
	free(d_Input);
	free(h_OutputGPU);
	free(h_Buffer);
	free(h_OutputCPU);

return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int imageW,
    int imageH,
    int TDD_NUM
)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;

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

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int imageW,
    int imageH,
    int TDD_NUM
)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;

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

void convolutionRowCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int imageW, int imageH)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = x + k;

                if (d >= 0 && d < imageW)
                    sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}

void convolutionColumnCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int imageW, int imageH)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = y + k;

                if (d >= 0 && d < imageH)
                    sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}

