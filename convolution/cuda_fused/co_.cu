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

__global__ void convolutionRowsGPU(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int *size, int *thread, int index);
__global__ void convolutionColumnsGPU(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int *size, int *thread, int index);
void convolutionRowsCPU(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int *size, int *thread, int index);
void convolutionColumnsCPU(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int *size, int *thread, int index);

int main(){

  	float **h_Kernel, **h_Input, **d_Buffer, **h_OutputGPU;
  	float **d_Output, **d_Kernel, **d_Input;
  	float **h_OutputCPU, **h_Buffer;
  	int *num_thread, *num_thread_dev;
  	int *num_size;
  	int pos_task[BT_NUM][TK_NUM];
  	int *pos_task_dev[BT_NUM];
  	FILE *fp;


  	int i, j;
  	double start_timer, end_timer;
	cudaSetDevice(0);
  	//printf("Initializing data...\n");

  	h_Kernel    = (float **)malloc(BT_NUM * sizeof(float*));
  	h_Input     = (float **)malloc(BT_NUM * sizeof(float*));
  	d_Buffer    = (float **)malloc(BT_NUM * sizeof(float*));
  	h_OutputGPU = (float **)malloc(BT_NUM * sizeof(float*));
  	d_Output = (float **)malloc(BT_NUM * sizeof(float*));
  	d_Kernel = (float **)malloc(BT_NUM * sizeof(float*));
  	d_Input = (float **)malloc(BT_NUM * sizeof(float*));
  	h_OutputCPU = (float **)malloc(BT_NUM * sizeof(float*));
  	h_Buffer = (float **)malloc(BT_NUM * sizeof(float*));

  	num_thread = (int*)malloc(NUM_TASK * sizeof(int));
  	num_size = (int*)malloc(BT_NUM * sizeof(int));

 	fp = fopen("rand.txt", "r");
  	for(i = 0; i < NUM_TASK; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < NUM_TASK; i++)
    		num_thread[i] *= 32;

  	for(i = 0; i < BT_NUM; i++){
    		num_size[i] = 0;
  	}

  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < TK_NUM; j++){
        		num_size[i] += (num_thread[i*TK_NUM+j])*
                        	(num_thread[i*TK_NUM+j]);
        		pos_task[i][j] = 0;
        		if(j > 0) pos_task[i][j] += pos_task[i][j-1] + (num_thread[i*TK_NUM+j-1])*
                        	(num_thread[i*TK_NUM+j-1]);

    		}
  	}


  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaHostAlloc(&h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&h_Input[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_Buffer[i], num_size[i] * sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&h_OutputGPU[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_Output[i], num_size[i] * sizeof(float)));
    		checkCudaErrors(cudaMalloc(&d_Input[i], num_size[i] * sizeof(float)));
    		checkCudaErrors(cudaMalloc(&d_Kernel[i], KERNEL_LENGTH * sizeof(float)));
    		h_OutputCPU[i] = (float*)malloc(num_size[i]*sizeof(float));
    		h_Buffer[i] = (float*)malloc(num_size[i]*sizeof(float));
    		checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM*sizeof(int)));

  	}
  	checkCudaErrors(cudaMalloc(&num_thread_dev, NUM_TASK*sizeof(int)));

	printf("Inputs are generating\n");

  	for(i = 0; i < BT_NUM;i++){
    		for (j = 0; j < KERNEL_LENGTH; j++){
      			h_Kernel[i][j] = (float)j/KERNEL_LENGTH;
    		}
  	}

  	for(i = 0; i < BT_NUM;i++){
    		for (j = 0; j < num_size[i]; j++){
      			h_Input[i][j] = (float)((j/imageW)%2);

    		}
  	}

  	//mem. copy
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(d_Kernel[i], h_Kernel[i], KERNEL_LENGTH*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(d_Input[i], h_Input[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));
  	}
  	checkCudaErrors(cudaMemcpy(num_thread_dev, num_thread, NUM_TASK*sizeof(int), cudaMemcpyHostToDevice));

  	checkCudaErrors(cudaDeviceSynchronize());

  	printf("Convolution CUDA static fusion program is running\n");

  	start_timer = my_timer();

  	for(i = 0; i < BT_NUM; i++){
    		convolutionRowsGPU<<<TK_NUM, TDK_NUM>>>(d_Buffer[i], d_Input[i], d_Kernel[i], KERNEL_RADIUS, pos_task_dev[i], num_thread_dev, i);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	for(i = 0; i < BT_NUM; i++){
    		convolutionColumnsGPU<<<TK_NUM, TDK_NUM>>>(d_Output[i], d_Buffer[i], d_Kernel[i],KERNEL_RADIUS, pos_task_dev[i], num_thread_dev, i);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	end_timer = my_timer();
  	printf("Convolution CUDA static fusion elapsed Time: %lf Sec.\n", end_timer - start_timer);

  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(h_OutputGPU[i], d_Output[i], num_size[i]*sizeof(float), cudaMemcpyDeviceToHost));

  	}
  	checkCudaErrors(cudaDeviceSynchronize());

	printf("CPU program running\n");
  	start_timer = my_timer();
  	for(i = 0; i < BT_NUM; i++){
    		convolutionRowsCPU(h_Buffer[i], h_Input[i], h_Kernel[i], KERNEL_RADIUS, pos_task[i], num_thread, i);
  	}

  	for(i = 0; i < BT_NUM; i++){
    		convolutionColumnsCPU(h_OutputCPU[i], h_Buffer[i], h_Kernel[i],KERNEL_RADIUS, pos_task[i], num_thread, i);
  	}

  	end_timer = my_timer();
  	//printf("CPU elapsed time:%lf\n", end_timer - start_timer);

	/*output result*/
	printf("verifying\n");
	int flag = 0;
  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < num_size[i]; j++){
      			if(abs(h_OutputCPU[i][j]- h_OutputGPU[i][j])> 0.1){
        			printf("Error:%f, %f, %d, %d\n", h_OutputCPU[i][j], h_OutputGPU[i][j], i, j);
				flag = 1;
        			break;
      			}
		}
	}
	if(!flag) printf("verify successfully\n");

  	//free mem.
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaFree(d_Buffer[i]));
    		checkCudaErrors(cudaFreeHost(h_Input[i]));
    		checkCudaErrors(cudaFreeHost(h_Kernel[i]));
    		checkCudaErrors(cudaFreeHost(h_OutputGPU[i]));
    		checkCudaErrors(cudaFree(d_Kernel[i]));
    		checkCudaErrors(cudaFree(d_Output[i]));
    		checkCudaErrors(cudaFree(d_Input[i]));
    		free(h_OutputCPU[i]);
    		free(h_Buffer[i]);
    		checkCudaErrors(cudaFree(pos_task_dev[i]));

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
  	checkCudaErrors(cudaFree(num_thread_dev));


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
    int *size, 
    int *thread, 
    int index
)
{
    int x, k, d;
    float sum;
    int td;
    int y = threadIdx.x;
    int bk = blockIdx.x;
    td = thread[index*TK_NUM+bk];
 
    if(y < td)
        for(x = 0; x < (td*td)/td; x++)
        {
            sum = 0;

            for (k = -kernelR; k <= kernelR; k++)
            {
		d = ((x*td+y)%td) + k;

                if (d >= 0 && d < td)
                    sum += h_Src[((x*td+y)/td) * td + d + size[bk]] * h_Kernel[kernelR - k];

            }

            h_Dst[(x*td+y)/td * td + ((x*td+y)%td) + size[bk]] = sum;
	    
        }
}

void convolutionRowsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int *size,
    int *thread,
    int index
)
{
    int x, k, d, y, t;
    float sum;
    int td;

    for(t = 0; t < TK_NUM; t++){
      td = thread[index*TK_NUM+t];
      for(y = 0; y < td; y++){
        for(x = 0; x < td; x++)
        {
            sum = 0;

            for (k = -kernelR; k <= kernelR; k++)
            {
		d = ((x*td+y)%td) + k;

                if (d >= 0 && d < td)
                	sum += h_Src[((x*td+y)/td) * td + d + size[t]] * h_Kernel[kernelR - k];
            }

            h_Dst[(x*td+y)/td * td + ((x*td+y)%td) + size[t]] = sum;

        }
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
    int *size,
    int *thread,
    int index
)
{
    int x, k, d;
    float sum;
    int y = threadIdx.x;
    int bk = blockIdx.x;
    int td;
#if 1
    td = thread[index*TK_NUM+bk];
    
    if(y < td){
	for(x = 0; x < (td*td)/td; x++)
        {
            sum = 0;

            for (k = -kernelR; k <= kernelR; k++)
            {
		d = ((x*td+y)/td) + k;

                if (d >= 0 && d < td)
                	sum += h_Src[d * td + (x*td+y)%td + size[bk]] * h_Kernel[kernelR - k];
            }

            h_Dst[(x*td+y)/td * td + (x*td+y)%td + size[bk]] = sum;
        }
    }
#endif
}

void convolutionColumnsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int *size,
    int *thread,
    int index
)
{
    int x, k, d, y, t;
    float sum;
    int td;

    for(t = 0; t < TK_NUM; t++){
      td = thread[index*TK_NUM+t];
      for(y = 0; y < td; y++){
        for(x = 0; x< td; x++)
        {
            sum = 0;

            for (k = -kernelR; k <= kernelR; k++)
            {
                d = (x*td+y)/td + k;

		d = ((x*td+y)/td) + k;

                if (d >= 0 && d < td)
                	sum += h_Src[d * td + (x*td+y)%td + size[t]] * h_Kernel[kernelR - k];
            }

            h_Dst[(x*td+y)/td * td + (x*td+y)%td + size[t]] = sum;
        }
    }
  }
}
