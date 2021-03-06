#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

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

	float **A, **d_A, **B, **d_C, **C, **D;
	
	A = (float**)malloc(task * sizeof(float*));
	B = (float**)malloc(task * sizeof(float*));
	C = (float**)malloc(task * sizeof(float*));
	D = (float**)malloc(task * sizeof(float*));
	d_A = (float**)malloc(task * sizeof(float*));
	d_C = (float**)malloc(task * sizeof(float*));

	
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

	printf("Pagoda DCT: #task:%d, image size:%d, #thread:%d\n", task, SIZE * SIZE, TDD_NUM);

	runtime_init();
	//transfer data to device
	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(d_A[i], A[i], SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
		checkCudaErrors(cudaMemcpyAsync(d_C[i], C[i], SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
	}
	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

	start_timer = my_timer();

	for(i = 0; i < task; i++){
		taskLaunch(11, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 0, FLOAT, d_A[i], FLOAT, d_C[i], INT, StrideF, INT, SIZE, INT, TDD_NUM, INT, block);
	}

	waitAll(task);
	end_timer = my_timer();
  	printf("The GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

	for(i = 0; i < task; i++){
		checkCudaErrors(cudaMemcpyAsync(C[i], d_C[i], SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost, runtime_stream));
	}
	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
	
	runtime_destroy();
  	runtime_free();

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
    		checkCudaErrors(cudaFreeHost(A[i]));
		checkCudaErrors(cudaFreeHost(C[i]));
		checkCudaErrors(cudaFree(d_A[i]));
		checkCudaErrors(cudaFree(d_C[i]));
    		free(D[i]);
  	}
	
	return 0;
}

