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

__global__ void convolutionRowsGPU(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size, int thread);
__global__ void convolutionColumnsGPU(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int size, int thread);
void convolutionRowsCPU(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size);
void convolutionColumnsCPU(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int size);

int main(){

  	float **h_Kernel, **h_Input, **d_Buffer, **h_OutputGPU;
  	float **d_Output, **d_Kernel, **d_Input;
  	float **h_OutputCPU, **h_Buffer;
  	int *num_thread;
  	int *num_size;
  	FILE *fp;


  	int i, j;
  	double start_timer, end_timer;

	cudaSetDevice(0);
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	//printf("Initializing data...\n");

  	cudaStream_t *con_stream;
  	con_stream = (cudaStream_t*)malloc(NUM_TASK*sizeof(con_stream));

  	for(i = 0; i < NUM_TASK; i++){
    		checkCudaErrors(cudaStreamCreate(&con_stream[i]));
  	}

  	h_Kernel    = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_Input     = (float **)malloc(NUM_TASK * sizeof(float*));
  	d_Buffer    = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_OutputGPU = (float **)malloc(NUM_TASK * sizeof(float*));
  	d_Output = (float **)malloc(NUM_TASK * sizeof(float*));
  	d_Kernel = (float **)malloc(NUM_TASK * sizeof(float*));
  	d_Input = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_OutputCPU = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_Buffer = (float **)malloc(NUM_TASK * sizeof(float*));

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
    		h_OutputCPU[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
   	 	h_Buffer[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
  	}

	printf("Inputs are generating\n");
  	for(i = 0; i < NUM_TASK;i++){
    		for (j = 0; j < KERNEL_LENGTH; j++){
      			h_Kernel[i][j] = (float)j/KERNEL_LENGTH;
    		}
  	}

  	for(i = 0; i < NUM_TASK;i++){
    		for (j = 0; j < num_size[i]*num_size[i]; j++){
      			h_Input[i][j] = (float)((j/imageW)%2);

    		}
  	}

  	//mem. copy
  	for(i = 0; i < NUM_TASK; i++){
    		checkCudaErrors(cudaMemcpyAsync(d_Kernel[i], h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaMemcpyHostToDevice, con_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(d_Input[i], h_Input[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyHostToDevice, con_stream[i]));

  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	printf("Convolution CUDA baseline program is running\n");
  	start_timer = my_timer();
  	for(i = 0; i < NUM_TASK; i++){
    		convolutionRowsGPU<<<1, num_thread[i]*32, 0, con_stream[i]>>>(d_Buffer[i], d_Input[i], d_Kernel[i], KERNEL_RADIUS, num_size[i], num_thread[i]*32);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	for(i = 0; i < NUM_TASK; i++){
    		convolutionColumnsGPU<<<1, num_thread[i]*32, 0, con_stream[i]>>>(d_Output[i], d_Buffer[i], d_Kernel[i],KERNEL_RADIUS, num_size[i], num_thread[i]*32);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  	end_timer = my_timer();
  	printf("CUDA baseline Convolution elapsed Time: %lf Sec.\n", end_timer - start_timer);

  	for(i = 0; i < NUM_TASK; i++){
    		checkCudaErrors(cudaMemcpyAsync(h_OutputGPU[i], d_Output[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyDeviceToHost, con_stream[i]));

  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  
	printf("CPU program running\n");
	start_timer = my_timer();
  	for(i = 0; i < NUM_TASK; i++){
    		convolutionRowsCPU(h_Buffer[i], h_Input[i], h_Kernel[i], KERNEL_RADIUS, num_size[i]);
  	}

  	for(i = 0; i < NUM_TASK; i++){
    		convolutionColumnsCPU(h_OutputCPU[i], h_Buffer[i], h_Kernel[i],KERNEL_RADIUS, num_size[i]);
  	}

  	end_timer = my_timer();
  	//printf("CPU elapsed time:%lf\n", end_timer - start_timer);

	/*output result*/
	int flag = 0;
  	for(i = 0; i < NUM_TASK; i++)
    		for(j = 0; j < num_size[i]*num_size[i]; j++)
      			if(abs(h_OutputCPU[i][j]- h_OutputGPU[i][j])> 0.1){
        			printf("Error:%f, %f, %d, %d\n", h_OutputCPU[i][j], h_OutputGPU[i][j], i, j);
				flag = 1;
        			break;
      		}
	if(!flag) printf("Verify Successfully\n");

  	//free mem.
  	for(i = 0; i < NUM_TASK; i++){
    		checkCudaErrors(cudaStreamDestroy(con_stream[i]));
  	}

  	for(i = 0; i < NUM_TASK; i++){
    		checkCudaErrors(cudaFree(d_Buffer[i]));
    		checkCudaErrors(cudaFreeHost(h_Input[i]));
    		checkCudaErrors(cudaFreeHost(h_Kernel[i]));
    		checkCudaErrors(cudaFreeHost(h_OutputGPU[i]));
    		checkCudaErrors(cudaFree(d_Kernel[i]));
    		checkCudaErrors(cudaFree(d_Output[i]));
    		checkCudaErrors(cudaFree(d_Input[i]));
    		free(h_OutputCPU[i]);
    		free(h_Buffer[i]);

  	}

  	free(d_Buffer);
  	free(h_Input);
  	free(h_Kernel);
  	free(d_Kernel);
  	free(d_Output);
  	free(d_Input);
  	free(h_OutputGPU);
  	free(num_thread);
  	free(num_size);
  	free(h_OutputCPU);
  	free(h_Buffer);

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
    int size, 
    int thread
)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;

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

void convolutionRowsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int size
)
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

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsGPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int size, 
    int thread
)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;

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

void convolutionColumnsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int size
)
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
