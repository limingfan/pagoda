#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "runtime.cuh"

#include "bmult.h"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void mult(int *A, int *B, int *C, int size){
	int i, j, k;
  	int sum = 0;
  	for(j = 0; j < TDK_NUM; j++){
    		for(i = 0; i < (size*size/TDK_NUM); i++){
      			for(k = 0; k < size; k++){
        			sum += A[((i*TDK_NUM+j)/size)*size+k] * B[k*size+((i*TDK_NUM+j)%size)];
      			}
      		C[((i*TDK_NUM+j)/size)*size+((i*TDK_NUM+j)%size)] = sum;
      		if(k == size) sum = 0;
    		}
	}
}

int main(){
  	int i, j;
  	int *A[task], *B[task], *C[task], *D[task];
  	int *A_dev[task], *B_dev[task], *C_dev[task];
  	int num_thread[task];
  	int num_size[task];
  	FILE *fp;

  	double start_timer, end_timer;
	cudaSetDevice(0);
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	fp = fopen("rand.txt", "r");
  	for(i = 0; i < task; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < task; i++)
    		num_size[i] = num_thread[i]*32;

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaHostAlloc(&A[i], num_size[i]*num_size[i]*sizeof(int), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&A_dev[i], num_size[i]*num_size[i]*sizeof(int)));
    		checkCudaErrors(cudaHostAlloc(&B[i], num_size[i]*num_size[i]*sizeof(int), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&B_dev[i], num_size[i]*num_size[i]*sizeof(int)));
    		checkCudaErrors(cudaHostAlloc(&C[i], num_size[i]*num_size[i]*sizeof(int), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&C_dev[i], num_size[i]*num_size[i]*sizeof(int)));
    		D[i] = (int*)malloc(sizeof(int)*num_size[i]*num_size[i]);
  	}

	printf("MM Pagoda inputs are generating\n");
  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			A[i][j] = (i%num_size[i])+1;
      			B[i][j] = (i%num_size[i])+1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}

  	runtime_init();

  	//transfer data to device
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(A_dev[i], A[i], num_size[i]*num_size[i]*sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(B_dev[i], B[i], num_size[i]*num_size[i]*sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  	}
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

	printf("MM Pagoda is running\n");
  	start_timer = my_timer();

  	for(i = 0; i < task; i++){
    		taskLaunch(10, INT, num_thread[i]*32, INT, 1, INT, 0, INT, 0, INT, 0, 
			INT, A_dev[i], INT, B_dev[i], INT, C_dev[i], INT, num_thread[i]*32, INT, num_size[i]);
  	}
  	waitAll(task);

  	end_timer = my_timer();
  	printf("The Pagoda matrixMul Elapsed Time: %lf Sec.\n", end_timer - start_timer);

  	//transfer data back to host
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(C[i], C_dev[i], num_size[i]*num_size[i]*sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
	}
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));


  	runtime_destroy();
  	runtime_free();

	printf("cpu program running\n");
  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult(A[i], B[i], D[i], num_size[i]);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//Verification
	printf("Verifying\n");
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			if(C[i][j] != D[i][j]){
        			printf("Error:%d, %d, %d\n", C[i][j], D[i][j], i);
				break;
      			}
		}
	}
	printf("Verify Successfully\n");

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaFreeHost(A[i]));
    		checkCudaErrors(cudaFree(A_dev[i]));
    		checkCudaErrors(cudaFreeHost(B[i]));
    		checkCudaErrors(cudaFree(B_dev[i]));
    		checkCudaErrors(cudaFreeHost(C[i]));
    		checkCudaErrors(cudaFree(C_dev[i]));
    		free(D[i]);
  	}

  	if(cudaDeviceReset()== cudaSuccess) printf("Reset successful\n");

  return 0;
}
