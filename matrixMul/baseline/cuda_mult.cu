#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void mult(float *A, float *B, float *C, int size, int TD_NUM){
  	int i, j, k;
  	float sum = 0;
  	for(j = 0; j < TD_NUM; j++)
    		for(i = 0; i < (size*size/TD_NUM); i++){
      			for(k = 0; k < size; k++){
        			sum += A[((i*TD_NUM+j)/size)*size+k] * B[k*size+((i*TD_NUM+j)%size)];
      			}
      			C[((i*TD_NUM+j)/size)*size+((i*TD_NUM+j)%size)] = sum;
      			if(k == size) sum = 0;
    	}
}

__global__ void mult_gpu(float *A, float *B, float *C, int size, int TD_NUM){
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int i, k;
  	float sum = 0;
  	if(tid < TD_NUM){
    		for(i = 0; i < (size*size/TD_NUM); i++){
      			for(k = 0; k < size; k++){
				sum += A[((i*TD_NUM+tid)/size)*size+k] * B[k*size+((i*TD_NUM+tid)%size)];
     			}
      			C[((i*TD_NUM+tid)/size)*size+((i*TD_NUM+tid)%size)] = sum;
      			if(k == size) sum = 0;
    		}
  	}
}

int main(int argc, char *argv[]){
  	int i, j;
  	double start_timer, end_timer;

	float **A, **B, **C, **D;
        float **A_dev, **B_dev, **C_dev;

	int mCols, mRows, MSIZE, task;
        int TD_NUM;

	if(argc < 4){
                printf("Error input options:./matrixMul matrix_size #task #thread\n");
                exit(1);
        }

        mCols = atoi(argv[1]);
        mRows = atoi(argv[1]);
        MSIZE = mCols * mRows;
        task = atoi(argv[2]);
        TD_NUM = atoi(argv[3]);

	printf("CUDA baseline MatrixMul:matrix size:%d x %d, #thread:%d, #task:%d\n", mCols, mRows, TD_NUM, task);
	A = (float**)malloc(task * sizeof(float*));
        B = (float**)malloc(task * sizeof(float*));
        C = (float**)malloc(task * sizeof(float*));
        D = (float**)malloc(task * sizeof(float*));
        A_dev = (float**)malloc(task * sizeof(float*));
        B_dev = (float**)malloc(task * sizeof(float*));
        C_dev = (float**)malloc(task * sizeof(float*));

  	cudaStream_t mult_stream[task];

  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaStreamCreate(&mult_stream[i]));
  	}

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaHostAlloc(&A[i], MSIZE*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&B[i], MSIZE*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&C[i], MSIZE*sizeof(float), cudaHostAllocDefault));
  	}

  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMalloc(&A_dev[i], MSIZE*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&B_dev[i], MSIZE*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&C_dev[i], MSIZE*sizeof(float)));
    		D[i] = (float*)malloc(sizeof(float)*MSIZE);
  	}

	srand(time(NULL));

  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < MSIZE; j++){
      			A[i][j] = ((double) rand() / (RAND_MAX)) + 1;
      			B[i][j] = ((double) rand() / (RAND_MAX)) + 1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}
  	//transfer data to device
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(A_dev[i], A[i], MSIZE*sizeof(float), cudaMemcpyHostToDevice, mult_stream[i%32]));
    		checkCudaErrors(cudaMemcpyAsync(B_dev[i], B[i], MSIZE*sizeof(float), cudaMemcpyHostToDevice, mult_stream[i%32]));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult_gpu<<<1, TD_NUM, 0, mult_stream[i%32]>>>(A_dev[i], B_dev[i], C_dev[i], mCols, TD_NUM);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  	end_timer = my_timer();
  	printf("The GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//transfer data back to host
  	for(i = 0; i < task; i++)
    		checkCudaErrors(cudaMemcpyAsync(C[i], C_dev[i], MSIZE*sizeof(float), cudaMemcpyDeviceToHost, mult_stream[i%32]));
  	checkCudaErrors(cudaDeviceSynchronize());

  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult(A[i], B[i], D[i], mCols, TD_NUM);
  	}
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//Verification
	printf("Verifying\n");
	long long flag = 0;
  	for(i = 0; i < task; i++){
    		for(j = 0; j < MSIZE; j++){
      			if(abs(C[i][j] - D[i][j]) > 1e-3){
        			printf("Error:%d, %d\n", C[i][j], D[i][j]);
				break;
			}
			flag ++;
		}
	}

	if(flag == (task * MSIZE)) printf("Verify Successfully\n");

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
	free(A);
        free(B);
        free(C);
        free(D);
        free(A_dev);
        free(B_dev);
        free(C_dev);

  	return 0;
}
