#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../../common/para.h"
#define task (TK_NUM*BT_NUM)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void mult(int *A, int *B, int *C, int size, int td_num){
	int tid;
	int i, k;
        int sum = 0;
	for(tid = 0; tid < td_num; tid ++){
                for(i = 0; i < (size*size/td_num); i++){
                        for(k = 0; k < size; k++){
                                sum += A[((i*td_num+tid)/size)*size+k] * B[k*size+((i*td_num+tid)%size)];
                        }
                        C[((i*td_num+tid)/size)*size+((i*td_num+tid)%size)] = sum;
                        if(k == size) sum = 0;
                }
        }

}

__global__ void mult_gpu(int *A, int *B, int *C, int size, int td_num){
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int i, k;
  	int sum = 0;
  	if(tid < td_num){
    		for(i = 0; i < (size*size/td_num); i++){
      			for(k = 0; k < size; k++){
				sum += A[((i*td_num+tid)/size)*size+k] * B[k*size+((i*td_num+tid)%size)];
      			}
      			C[((i*td_num+tid)/size)*size+((i*td_num+tid)%size)] = sum;
      			if(k == size) sum = 0;
    		}
  	}
}

int main(){
  	int i, j;
  	int *A[task], *B[task], *C[task], *D[task];
  	int *A_dev[task], *B_dev[task], *C_dev[task];
  	double start_timer, end_timer;
  	int num_thread[task];
  	int num_size[task];
  	FILE *fp;
  	cudaStream_t mult_stream[task];
	cudaSetDevice(0);
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaStreamCreate(&mult_stream[i]));
  	}

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
	printf("MM CUDA baseline inputs are generating\n");
  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			A[i][j] = (i%num_size[i])+1;
      			B[i][j] = (i%num_size[i])+1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}

  	//transfer data to device
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(A_dev[i], A[i], num_size[i]*num_size[i]*sizeof(int), cudaMemcpyHostToDevice, mult_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(B_dev[i], B[i], num_size[i]*num_size[i]*sizeof(int), cudaMemcpyHostToDevice, mult_stream[i]));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

	printf("MM CUDA baseline inputs are running\n");
  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult_gpu<<<1, num_thread[i]*32, 0, mult_stream[i]>>>(A_dev[i], B_dev[i], C_dev[i], num_size[i], num_thread[i]*32);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  	end_timer = my_timer();
  	printf("The CUDA baseline matrixMul Elapsed Time: %lf Sec.\n", end_timer - start_timer);
  
  	//transfer data back to host
  	for(i = 0; i < task; i++)
    		checkCudaErrors(cudaMemcpyAsync(C[i], C_dev[i], num_size[i]*num_size[i]*sizeof(int), cudaMemcpyDeviceToHost, mult_stream[i]));
  	checkCudaErrors(cudaDeviceSynchronize());

  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult(A[i], B[i], D[i], num_size[i], num_thread[i]*32);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//Verification
	printf("Verifying\n");
  	for(i = 0; i < task; i++)
    		for(j = 0; j < num_size[i]*num_size[i]; j++)
      			if(C[i][j] != D[i][j]){
        			printf("Error:%d, %d\n", C[i][j], D[i][j]);
				break;
      		}
	printf("Verifying Successfully\n");

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaStreamDestroy(mult_stream[i]));
    		checkCudaErrors(cudaFreeHost(A[i]));
    		checkCudaErrors(cudaFree(A_dev[i]));
    		checkCudaErrors(cudaFreeHost(B[i]));
    		checkCudaErrors(cudaFree(B_dev[i]));
    		checkCudaErrors(cudaFreeHost(C[i]));
    		checkCudaErrors(cudaFree(C_dev[i]));
    		free(D[i]);
  	}
  	return 0;
}
