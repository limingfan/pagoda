#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "dct.h"
#include "runtime.cuh"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

int main(){

	int i, j;
  	float *A[task], *d_A[task], *d_C[task], *C[task], *D[task];
  	double start_timer, end_timer;
  	int num_thread[task];
        int num_size[task];
        int StrideF[task];
        FILE *fp;
	cudaSetDevice(0);
	fp = fopen("rand.txt", "r");
        for(i = 0; i < task; i++)
                fscanf(fp, "%1d", &num_thread[i]);

        fclose(fp);

	for(i = 0; i < task; i++){
                if(num_thread[i] == 1){
                        num_size[i] = 64;
                }else{
                        num_size[i] = num_thread[i]*32;
                }

                StrideF[i] = ((int)ceil((num_size[i]*sizeof(float))/16.0f))*16 / sizeof(float);
        }


  	for(i = 0; i < task; i++){
		checkCudaErrors(cudaHostAlloc(&A[i], num_size[i]*num_size[i]*sizeof(float), cudaHostAllocDefault));
		checkCudaErrors(cudaMalloc(&d_A[i], num_size[i]*num_size[i]*sizeof(float)));
		checkCudaErrors(cudaMalloc(&d_C[i], num_size[i]*num_size[i]*sizeof(float)));
		checkCudaErrors(cudaHostAlloc(&C[i], num_size[i]*num_size[i]*sizeof(float), cudaHostAllocDefault));
    		D[i] = (float*)malloc(sizeof(float)*num_size[i]*num_size[i]);
  	}
	printf("DCT Pagoda inputs are generating\n");
  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			A[i][j] = 2.0;
			C[i][j] = 0;
      			D[i][j] = 0;
    		}
	}

	runtime_init();
	//transfer data to device
	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(d_A[i], A[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
		checkCudaErrors(cudaMemcpyAsync(d_C[i], C[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
	}
	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

	printf("DCT pagoda is running\n");
	start_timer = my_timer();

	for(i = 0; i < task; i++){
		taskLaunch(10, INT, num_thread[i]*32, INT, 1, INT, 0, INT, 0, INT, 0, FLOAT, d_A[i], FLOAT, d_C[i], INT, StrideF[i], INT, num_size[i], INT, num_thread[i]*32);
	}

	waitAll(task);
	end_timer = my_timer();
  	printf("DCT pagoda Elapsed Time: %lf Sec.\n", end_timer - start_timer);

	for(i = 0; i < task; i++){
		checkCudaErrors(cudaMemcpyAsync(C[i], d_C[i], num_size[i]*num_size[i]*sizeof(float), cudaMemcpyDeviceToHost, runtime_stream));
	}
	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
	
	runtime_destroy();
  	runtime_free();

	printf("CPU program running\n");
  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
   		DCT(A[i], D[i], StrideF[i], num_size[i], num_thread[i]*32);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);
	
	//verify
	printf("verifying\n");
	int flag = 0;
	for(i = 0; i < task; i++){
		for(j = 0; j < num_size[i]*num_size[i]; j++){
			if(C[i][j] != D[i][j]){
				printf("Error:%f, %f, %d, %d\n", C[i][j], D[i][j], i, j);
				flag = 1;
				break;
			}
		}
	}

	if(!flag) printf("Verifying Successfully\n");

	//free memory
	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaFreeHost(A[i]));
		checkCudaErrors(cudaFreeHost(C[i]));
		checkCudaErrors(cudaFree(d_A[i]));
		checkCudaErrors(cudaFree(d_C[i]));
    		free(D[i]);
  	}
	
	return 0;
}

