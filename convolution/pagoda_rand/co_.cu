#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "con.h"
#include "runtime.cuh"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void convolutionRowCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size);
void convolutionColumnCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size);

int main(){

	float **h_Kernel, **h_Input, **d_Buffer, **h_OutputGPU, **h_Buffer, **h_OutputCPU;
	float **d_Output, **d_Kernel, **d_Input;
	int *num_thread;
  	int *num_size;
  	FILE *fp;


	int i, j;
	double start_timer, end_timer;
	cudaSetDevice(0);
	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

	//printf("Initializing data...\n");

	h_Kernel    = (float **)malloc(NUM_TASK * sizeof(float*));
	h_Input     = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Buffer    = (float **)malloc(NUM_TASK * sizeof(float*));
	h_OutputGPU = (float **)malloc(NUM_TASK * sizeof(float*));
	h_OutputCPU = (float **)malloc(NUM_TASK * sizeof(float*));
	h_Buffer    = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Output = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Kernel = (float **)malloc(NUM_TASK * sizeof(float*));
	d_Input = (float **)malloc(NUM_TASK * sizeof(float*));

	num_thread = (int*)malloc(NUM_TASK * sizeof(int));
 	num_size = (int*)malloc(NUM_TASK * sizeof(int));

	fp = fopen("rand.txt", "r");
  	for(i = 0; i < NUM_TASK; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < NUM_TASK; i++)
    		num_size[i] = num_thread[i]*32;



	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaHostAlloc(&h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaHostAllocDefault));
  		checkCudaErrors(cudaHostAlloc(&h_Input[i], num_size[i]*num_size[i]*sizeof(float), cudaHostAllocDefault));
  		checkCudaErrors(cudaMalloc(&d_Buffer[i], num_size[i]*num_size[i] * sizeof(float)));
  		checkCudaErrors(cudaHostAlloc(&h_OutputGPU[i], num_size[i]*num_size[i]*sizeof(float), cudaHostAllocDefault));
  		checkCudaErrors(cudaMalloc(&d_Output[i], num_size[i]*num_size[i] * sizeof(float)));
  		checkCudaErrors(cudaMalloc(&d_Input[i], num_size[i]*num_size[i] * sizeof(float)));
  		checkCudaErrors(cudaMalloc(&d_Kernel[i], KERNEL_LENGTH * sizeof(float)));

  		h_Buffer[i] = (float*)malloc(num_size[i]*num_size[i] * sizeof(float));
  		h_OutputCPU[i] = (float*)malloc(num_size[i]*num_size[i] * sizeof(float));
	}

	printf("Inputs are generating\n");
	for(i = 0; i < NUM_TASK;i++){
  		for (j = 0; j < KERNEL_LENGTH; j++){
    			h_Kernel[i][j] = (float)j/KERNEL_LENGTH;
  		}
	}

	for(i = 0; i < NUM_TASK;i++){
  		for (j = 0; j < num_size[i]*num_size[i]; j++){
    			h_Input[i][j] = (float)((j/num_size[i])%2);

  		}
	}

	runtime_init();

	//mem. copy

	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaMemcpyAsync(d_Kernel[i], h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
  		checkCudaErrors(cudaMemcpyAsync(d_Input[i], h_Input[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));

	}
	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

	printf("Convolution Pagoda program is running\n");
	start_timer = my_timer();
	for(i = 0; i < NUM_TASK; i++){
  		taskLaunch(10, INT, num_thread[i]*32, INT, 1, INT, 0, INT, 0, INT, 0, FLOAT, d_Buffer[i], FLOAT, d_Input[i], FLOAT, d_Kernel[i], INT, num_size[i], INT, num_thread[i]*32);
	}
	waitAll(NUM_TASK);
	for(i = 0; i < NUM_TASK; i++){
  		taskLaunch(10, INT, num_thread[i]*32, INT, 1, INT, 0, INT, 0, INT, 1, FLOAT, d_Output[i], FLOAT, d_Buffer[i], FLOAT, d_Kernel[i], INT, num_size[i], INT, num_thread[i]*32);
	}
	waitAll(NUM_TASK);
	end_timer = my_timer();
	printf("Convolution Pagoda elapsed Time: %lf Sec.\n", end_timer - start_timer);
	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaMemcpyAsync(h_OutputGPU[i], d_Output[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyDeviceToHost, runtime_stream));
	}
	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

	runtime_destroy();
	runtime_free();

	printf("CPU program running\n");
  	start_timer = my_timer();
  	for(i = 0; i < NUM_TASK; i++){
    		convolutionRowCPU( h_Buffer[i], h_Input[i], h_Kernel[i], KERNEL_RADIUS, num_size[i]);
  	}
  	for(i = 0; i < NUM_TASK; i++){
    		convolutionColumnCPU( h_OutputCPU[i], h_Buffer[i], h_Kernel[i], KERNEL_RADIUS, num_size[i]);
  	}
  	end_timer = my_timer();
  	//printf("CPU exec.time:%lf Sec.\n", end_timer - start_timer);


	//verification
  	printf("verify\n");
	int flag = 0;
  	for(i = 0; i < NUM_TASK; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
        		if(fabs(h_OutputCPU[i][j] - h_OutputGPU[i][j] > 0.1)){
          			printf("Error:%f, %f, %d, %d\n", h_OutputCPU[i][j], h_OutputGPU[i][j], i, j);
				flag = 1;
          			break;
        		}
    		}
  	}
	if(!flag) printf("verify successfully\n");

	//free mem.

	for(i = 0; i < NUM_TASK; i++){
  		checkCudaErrors(cudaFree(d_Buffer[i]));
  		checkCudaErrors(cudaFreeHost(h_Input[i]));
  		checkCudaErrors(cudaFreeHost(h_Kernel[i]));
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
	free(num_size);
	free(num_thread);

return 0;
}

void convolutionRowCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size)
{
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = x + k;

                if (d >= 0 && d < size)
                    sum += h_Src[y * size + d] * h_Kernel[kernelR - k];
            }

            h_Dst[y * size + x] = sum;
        }
}

void convolutionColumnCPU( float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size)
{
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = y + k;

                if (d >= 0 && d < size)
                    sum += h_Src[d * size + x] * h_Kernel[kernelR - k];
            }

            h_Dst[y * size + x] = sum;
        }
}

