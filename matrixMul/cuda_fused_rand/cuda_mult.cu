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

void mult(int *A, int *B, int *C, int *size, int *threads, int index){
  	int i, j, k, t;
  	int sum, td;

  	for(t = 0; t < TK_NUM; t++){
    		td = threads[index*TK_NUM+t];
    		sum = 0;
    		for(j = 0; j < td; j++){
      			for(i = 0; i < td; i++){
        			for(k = 0; k < td; k++){
          				sum += A[((i*td+j)/td)*td+k + size[t]] * B[k*td+((i*td+j)%td) + size[t]];
        			}
        			C[((i*td+j)/td)*td+((i*td+j)%td)+size[t]] = sum;
        			if(k == td) sum = 0;
      			}
		}
   	}
}

__global__ void mult_gpu(int *A, int *B, int *C, int *size, int *thread, int index){
  	int tid = threadIdx.x;
  	int i, k;
  	int td;
  	int sum;

  	sum = 0;
  	td = thread[index*TK_NUM+blockIdx.x];
 
  	if(tid < td){
    		for(i = 0; i < td; i++){
      			for(k = 0; k < td; k++){
        			sum += A[((i*td+tid)/td)*td+k+size[blockIdx.x]] * B[k*td+((i*td+tid)%td)+size[blockIdx.x]];
      			}
      			C[((i*td+tid)/td)*td+((i*td+tid)%td)+size[blockIdx.x]] = sum;
      			if(k == td) sum = 0;
    		}
   	}
}

int main(){
  	int i, j;
  	int *A[BT_NUM], *B[BT_NUM], *C[BT_NUM], *D[BT_NUM];
  	int *A_dev[BT_NUM], *B_dev[BT_NUM], *C_dev[BT_NUM];
  	double start_timer, end_timer;
  	int num_thread[task], *num_thread_dev;
  	int num_size[BT_NUM];
  	int pos_task[BT_NUM][TK_NUM];
  	int *pos_task_dev[BT_NUM];
  	FILE *fp;

	cudaSetDevice(0);
  	fp = fopen("rand.txt", "r");
  	for(i = 0; i < task; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < task; i++)
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
    		A[i] = (int*)malloc(sizeof(int)*num_size[i]);
    		checkCudaErrors(cudaMalloc(&A_dev[i], num_size[i]*sizeof(int)));
    		B[i] = (int*)malloc(sizeof(int)*num_size[i]);
    		checkCudaErrors(cudaMalloc(&B_dev[i], num_size[i]*sizeof(int)));
    		C[i] = (int*)malloc(sizeof(int)*num_size[i]);
    		checkCudaErrors(cudaMalloc(&C_dev[i], num_size[i]*sizeof(int)));

    		checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM*sizeof(int)));
    		D[i] = (int*)malloc(sizeof(int)*num_size[i]);
  	}

  	checkCudaErrors(cudaMalloc(&num_thread_dev, task*sizeof(int)));

	printf("MM CUDA static fusion inputs are generating\n");
  	// Init matrix
  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < num_size[i]; j++){
      			A[i][j] = (j%num_size[i])+1;
      			B[i][j] = (j%num_size[i])+1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}
  
	//transfer data to device
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(A_dev[i], A[i], num_size[i]*sizeof(int), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(B_dev[i], B[i], num_size[i]*sizeof(int), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));
  	}
  	checkCudaErrors(cudaMemcpy(num_thread_dev, num_thread, task*sizeof(int), cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaDeviceSynchronize());
  
	printf("MM CUDA static fusion is running\n");
  	start_timer = my_timer();
  	for(i = 0; i < BT_NUM; i++){
    		mult_gpu<<<TK_NUM, TD_NUM>>>(A_dev[i], B_dev[i], C_dev[i], pos_task_dev[i], num_thread_dev, i);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  	end_timer = my_timer();
  	printf("The CUDA static fusion Elapsed Time: %lf Sec.\n", end_timer - start_timer);
  
  	//transfer data back to host
  	for(i = 0; i < BT_NUM; i++)
    		checkCudaErrors(cudaMemcpy(C[i], C_dev[i], num_size[i]*sizeof(int), cudaMemcpyDeviceToHost));
  	checkCudaErrors(cudaDeviceSynchronize());


  	start_timer = my_timer();
  	for(i = 0; i < BT_NUM; i++){
    		mult(A[i], B[i], D[i], pos_task[i], num_thread, i);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);
	
  	//Verification
  	printf("Verify\n");
  	for(i = 0; i < BT_NUM; i++)
    		for(j = 0; j < num_size[i]; j++)
      			if(C[i][j] != D[i][j]){
        			printf("Error:%d, %d, %d\n", C[i][j], D[i][j], i);
				break;
      			}

	printf("Verifying Successfully\n");

  	for(i = 0; i < BT_NUM; i++){
    		free(A[i]);
    		checkCudaErrors(cudaFree(A_dev[i]));
    		free(B[i]);
    		checkCudaErrors(cudaFree(B_dev[i]));
    		checkCudaErrors(cudaFree(pos_task_dev[i]));
    		free(C[i]);
    		checkCudaErrors(cudaFree(C_dev[i]));
    		free(D[i]);
  	}

  	checkCudaErrors(cudaFree(num_thread_dev));
  	return 0;
}
