#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "runtime.cuh"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void mult(float *A, float *B, float *C, int thread, int size){
	int i, j, k;
  	float sum = 0;
  	for(j = 0; j < thread; j++){
    		for(i = 0; i < (size*size/thread); i++){
      			for(k = 0; k < size; k++){
        			sum += A[((i*thread+j)/size)*size+k] * B[k*size+((i*thread+j)%size)];
      			}
      			C[((i*thread+j)/size)*size+((i*thread+j)%size)] = sum;
      			if(k == size) sum = 0;
    		}
	}
}

int main(int argc, char *argv[]){
	int i, j;
	int task;
  	double start_timer, end_timer;
	int mCols, mRows, matrixSize;
	int thread;

	if(argc < 4){
		printf("Error input options:./matrixMul matrix_size #task #thread\n");
		exit(1);
	}

	mCols = atoi(argv[1]);
	mRows = atoi(argv[1]);
	matrixSize = mCols * mRows;
	task = atoi(argv[2]);
	thread = atoi(argv[3]);
	printf("Pagoda MatrixMul:matrix size:%d x %d, #thread:%d, #task:%d\n", mCols, mRows, thread, task);

	float **A, **B, **C, **D;
	float **A_dev, **B_dev, **C_dev;

	A = (float**)malloc(task * sizeof(float*));
	B = (float**)malloc(task * sizeof(float*));
	C = (float**)malloc(task * sizeof(float*));
	D = (float**)malloc(task * sizeof(float*));
	A_dev = (float**)malloc(task * sizeof(float*));
	B_dev = (float**)malloc(task * sizeof(float*));
	C_dev = (float**)malloc(task * sizeof(float*));

  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	runtime_init();

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaHostAlloc(&A[i], matrixSize*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&B[i], matrixSize*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&C[i], matrixSize*sizeof(float), cudaHostAllocDefault));
  	}

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMalloc(&A_dev[i], matrixSize*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&B_dev[i], matrixSize*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&C_dev[i], matrixSize*sizeof(float)));
    		D[i] = (float*)malloc(sizeof(float)*matrixSize);
  	}


	srand(time(NULL));
  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < matrixSize; j++){
			A[i][j] = ((double) rand() / (RAND_MAX)) + 1;
			B[i][j] = ((double) rand() / (RAND_MAX)) + 1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}

  	//transfer data to device
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(A_dev[i], A[i], matrixSize*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(B_dev[i], B[i], matrixSize*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
  	}
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  
  	start_timer = my_timer();

  	for(i = 0; i < task; i++){
    		taskLaunch(10, INT, thread, INT, 1, INT, 0, INT, 0, INT, 0, FLOAT, A_dev[i], FLOAT, B_dev[i], FLOAT, C_dev[i], INT, mCols, INT, thread);
  	}

  	waitAll(task);
	end_timer = my_timer();
  	printf("The GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//transfer data back to host
  	for(i = 0; i < task; i++)
    		checkCudaErrors(cudaMemcpyAsync(C[i], C_dev[i], matrixSize*sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  
  	runtime_destroy();
  	runtime_free();

	// CPU comp.
  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult(A[i], B[i], D[i], thread, mCols);
  	}
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	printf("Verify\n");
  	//Verification
	long long flag = 0;
  	for(i = 0; i < task; i++){
    		for(j = 0; j < matrixSize; j++){
			if(abs(C[i][j] - D[i][j]) > 1e-3){
        			printf("Error:%f, %f, %d\n", C[i][j], D[i][j], i);
				break;
      			}
			flag ++;
		}
	}

	if(flag == (task * matrixSize)) printf("Verify Successfully\n");

	// free memory
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaFreeHost(A[i]));
    		checkCudaErrors(cudaFree(A_dev[i]));
    		checkCudaErrors(cudaFreeHost(B[i]));
    		checkCudaErrors(cudaFree(B_dev[i]));
    		checkCudaErrors(cudaFreeHost(C[i]));
    		checkCudaErrors(cudaFree(C_dev[i]));
    		free(D[i]);
  	}

	free(A);
	free(B);
	free(C);
	free(D);
	free(A_dev);
	free(B_dev);
	free(C_dev);
  	if(cudaDeviceReset()== cudaSuccess) printf("Reset successful\n");

	return 0;
}
