#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "dct.h"
#include "kernel.cuh"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

int main(int argc, char *argv[]){

	int i, j;
  	double start_timer, end_timer;
  	int StrideF;

	if(argc < 5){
                printf("Error Input: dct size, task, thread, block\n");
                exit(1);
        }
        int SIZE = atoi(argv[1]);
        int task = atoi(argv[2]);
        int TDD_NUM = atoi(argv[3]);
        int block = atoi(argv[4]);

	printf("Baseline DCT: #task:%d, image size:%d, thread:%d\n", task, SIZE * SIZE, TDD_NUM);

        float **A, **d_A, **B, **d_C, **C, **D;

        A = (float**)malloc(task * sizeof(float*));
        B = (float**)malloc(task * sizeof(float*));
        C = (float**)malloc(task * sizeof(float*));
        D = (float**)malloc(task * sizeof(float*));
        d_A = (float**)malloc(task * sizeof(float*));
        d_C = (float**)malloc(task * sizeof(float*));


	cudaStream_t dct_stream[task];
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaStreamCreate(&dct_stream[i]));
  	}


  	for(i = 0; i < task; i++){
		checkCudaErrors(cudaHostAlloc(&A[i], SIZE*SIZE*sizeof(float), cudaHostAllocDefault));
		checkCudaErrors(cudaMalloc(&d_A[i], SIZE*SIZE*sizeof(float)));
		checkCudaErrors(cudaMalloc(&d_C[i], SIZE*SIZE*sizeof(float)));
		checkCudaErrors(cudaHostAlloc(&C[i], SIZE*SIZE*sizeof(float), cudaHostAllocDefault));
    		D[i] = (float*)malloc(sizeof(float)*SIZE*SIZE);
  	}
	
	srand(time(NULL));

  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < SIZE*SIZE; j++){
      			A[i][j] = ((double) rand() / (RAND_MAX)) + 1;
			C[i][j] = 0;
      			D[i][j] = 0;
    		}
	}

  	StrideF = ((int)ceil((SIZE*sizeof(float))/16.0f))*16 / sizeof(float);

	//transfer data to device
	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(d_A[i], A[i], SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice, dct_stream[i]));
		checkCudaErrors(cudaMemcpyAsync(d_C[i], C[i], SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice, dct_stream[i]));
	}
	checkCudaErrors(cudaDeviceSynchronize());
	
	start_timer = my_timer();

	for(i = 0; i < task; i++){
		d_DCT<<<1, TDD_NUM, 0, dct_stream[i]>>>(d_A[i], d_C[i], StrideF, SIZE, TDD_NUM, block);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	end_timer = my_timer();
  	printf("The GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

	start_timer = my_timer();
	for(i = 0; i < task; i++){
		checkCudaErrors(cudaMemcpyAsync(C[i], d_C[i], SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost, dct_stream[i]));
	}

	checkCudaErrors(cudaDeviceSynchronize());

  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
   		DCT(A[i], D[i], StrideF, SIZE, TDD_NUM, block);
  	}
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);	

	//verify
	printf("verify\n");
	long long flag = 0;
	for(i = 0; i < task; i++){
		for(j = 0; j < SIZE*SIZE; j++){
			if(abs(C[i][j] - D[i][j]) > 1e-3){
				printf("Error:%f, %f, %d, %d\n", C[i][j], D[i][j], i, j);
				break;
			}
			flag ++;
		}
	}

	if(flag == SIZE * SIZE * task) printf("verify successfully\n");
	
	//free memory
	for(i = 0; i < task; i++){
		checkCudaErrors(cudaStreamDestroy(dct_stream[i]));
    		checkCudaErrors(cudaFreeHost(A[i]));
		checkCudaErrors(cudaFreeHost(C[i]));
		checkCudaErrors(cudaFree(d_A[i]));
		checkCudaErrors(cudaFree(d_C[i]));
    		free(D[i]);
  	}
	
	return 0;
}

