#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../../common/para.h"

// Num. of sample
#define N_samp 8
#define N_col 64


// Num. of Channel
#define N_ch (TK_NUM * BT_NUM)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

__global__ void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, 
			float *Vect_Up, float *Vect_F, float *F, int *size, 
			int *threads, int index);

void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn,
                        float *Vect_Up, float *Vect_F, float *F, int *size,
                        int *threads, int index);

int main(){

  	float **r;
  	float **r_dev;
  	float **H;
  	float **H_dev;
  	float **F;
  	float **F_dev;

  	float **Vect_H; // output of the F
  	float **Vect_H_dev;
  	float **Vect_Dn; // output of the down sampler
  	float **Vect_Dn_dev;
  	float **Vect_Up; // output of the up sampler
  	float **Vect_Up_dev;
  	float **Vect_F; // this is the output of the
  	float **Vect_F_dev;
  	int num_thread[N_ch], *num_thread_dev;
  	int num_size[BT_NUM];
  	int pos_task[BT_NUM][TK_NUM];
  	int **pos_task_dev;

	float **h_Vect_F;
 	cudaSetDevice(0); 
  	FILE *f;

  	int i, j;
  	double start_timer, end_timer;
  	f = fopen("rand.txt", "r");
  	for(i = 0; i < N_ch; i++)
    		fscanf(f, "%1d", &num_thread[i]);

  	fclose(f);

  	for(i = 0; i < BT_NUM; i++){
    		num_size[i] = 0;
  	}

  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < TK_NUM; j++){
        		num_size[i] += (num_thread[i*TK_NUM+j]*16)*
                        	(num_thread[i*TK_NUM+j]*16);
        		pos_task[i][j] = 0;
        		if(j > 0) pos_task[i][j] += pos_task[i][j-1] + (num_thread[i*TK_NUM+j-1]*16)*
                        	(num_thread[i*TK_NUM+j-1]*16);

    		}
  	}

  	for(i = 0; i < N_ch; i++)
    		num_thread[i] *= 32;

  	r = (float**)malloc(BT_NUM*sizeof(float*));
  	H = (float**)malloc(BT_NUM*sizeof(float*));
  	F = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_H = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_Dn = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_Up = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_F = (float**)malloc(BT_NUM*sizeof(float*));
  	r_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	H_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	F_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_H_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_Dn_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_Up_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	Vect_F_dev = (float**)malloc(BT_NUM*sizeof(float*));
  	pos_task_dev = (int**)malloc(BT_NUM*sizeof(int*));
	h_Vect_F = (float**)malloc(BT_NUM*sizeof(float*));


  	/*Memory allocation*/
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaHostAlloc(&r[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&r_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&H[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&H_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&F[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&F_dev[i], num_size[i]*sizeof(float)));

    		checkCudaErrors(cudaHostAlloc(&Vect_H[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_H_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_Dn[i], (num_size[i]/N_samp)*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_Dn_dev[i], (num_size[i]/N_samp)*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_Up[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_Up_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_F[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_F_dev[i], num_size[i]*sizeof(float)));

    		checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM*sizeof(int)));
		h_Vect_F[i] = (float*)malloc(num_size[i] * sizeof(float));

  	}
  	checkCudaErrors(cudaMalloc(&num_thread_dev, N_ch*sizeof(int)));
	
	printf("Filterbank CUDA static fusion inputs are generating\n");
  	/*init data*/
  	for(i = 0; i < BT_NUM; i++)
    		for(j = 0; j < num_size[i]; j++){
      			r[i][j] = j + 0.0001;
      			Vect_Up[i][j] = 0;
      			Vect_F[i][j] = 0;
      			Vect_H[i][j]=0;
			h_Vect_F[i][j] = 0;
    		}

  	for(i = 0; i < BT_NUM; i++)
    		for(j = 0; j < num_size[i]; j++){
      			H[i][j] = 0.0001;
      			F[i][j] = 0.0001;
    		}

  	// Data transfer to device
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(r_dev[i], r[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(Vect_Up_dev[i], Vect_Up[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(Vect_F_dev[i], Vect_F[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(Vect_H_dev[i], Vect_H[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(H_dev[i], H[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(F_dev[i], F[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));
  	}
  	checkCudaErrors(cudaMemcpy(num_thread_dev, num_thread, N_ch*sizeof(int), cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaDeviceSynchronize());

	printf("Filterbank CUDA static fusion is running\n");
  	// task launch
  	start_timer = my_timer();
  	for(i = 0; i < BT_NUM; i++){
    		FBCore<<<TK_NUM, TDK_NUM>>>(r_dev[i], H_dev[i], Vect_H_dev[i], 
			Vect_Dn_dev[i], Vect_Up_dev[i], Vect_F_dev[i], F_dev[i], pos_task_dev[i], num_thread_dev, i);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	end_timer = my_timer();
  	printf("Filterbank CUDA static fusion Elapsed Time: %f Sec.\n", end_timer - start_timer);
  	start_timer = my_timer();
  	// Data transfer back to host
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(Vect_F[i], Vect_F_dev[i], num_size[i]*sizeof(float), cudaMemcpyDeviceToHost));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());
#if 0
	// CPU task launch
	printf("CPU program running\n");
        start_timer = my_timer();
        for(i = 0; i < BT_NUM; i++){
                h_FBCore(r[i], H[i], Vect_H[i],
                        Vect_Dn[i], Vect_Up[i], h_Vect_F[i], F[i], pos_task[i], num_thread, i);
        }

        end_timer = my_timer();
        //printf("The CPU Elapsed time:%f Sec.\n", end_timer - start_timer);


	/*Verify*/
        printf("Verify\n");
        int flag = 0;
        for(i = 0; i < BT_NUM; i++){
                for(j = 0; j < num_size[i]; j++){
                        if(abs(h_Vect_F[i][j] -  Vect_F[i][j]) > 1e-3){
                                printf("Error:%f, %f, %d\n", h_Vect_F[i][j], Vect_F[i][j], i);
                                flag = 1;
                                break;
                        }
                }
        }

	if(!flag) printf("Verify successfully\n");
#endif
  	/*Free Memory*/
  	for(i = 0; i < BT_NUM; i++){ 
    		checkCudaErrors(cudaFreeHost(r[i]));
    		checkCudaErrors(cudaFree(r_dev[i]));
    		checkCudaErrors(cudaFreeHost(H[i]));
    		checkCudaErrors(cudaFree(H_dev[i]));
    		checkCudaErrors(cudaFreeHost(F[i]));
    		checkCudaErrors(cudaFree(F_dev[i]));

    		checkCudaErrors(cudaFreeHost(Vect_H[i]));
    		checkCudaErrors(cudaFree(Vect_H_dev[i]));
    		checkCudaErrors(cudaFreeHost(Vect_Dn[i]));
    		checkCudaErrors(cudaFree(Vect_Dn_dev[i]));
    		checkCudaErrors(cudaFreeHost(Vect_Up[i]));
    		checkCudaErrors(cudaFree(Vect_Up_dev[i]));
    		checkCudaErrors(cudaFreeHost(Vect_F[i]));

    		checkCudaErrors(cudaFree(pos_task_dev[i]));
  	}
  	checkCudaErrors(cudaFree(num_thread_dev));
  	free(r);
  	free(H);
  	free(F);
  	free(Vect_H);
  	free(Vect_Dn);
 	free(Vect_Up);
  	free(Vect_F);
  	free(r_dev);
  	free(H_dev);
  	free(F_dev);
  	free(Vect_H_dev);
  	free(Vect_Dn_dev);
  	free(Vect_Up_dev);
  	free(Vect_F_dev);
  	free(pos_task_dev);

  	return 0;
}

__global__ void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn,
                        float *Vect_Up, float *Vect_F, float *F, int *size,
                        int *threads, int index){
  	int tid = threadIdx.x;
  	int td;
  	int j, k;

  	td = threads[index*TK_NUM+blockIdx.x];

  	//convolving H
  	if(tid < td){
    		for (j=0; j< ((td*td/4)/td); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*td+tid)-k)>=0){
          				Vect_H[j*td+tid+size[blockIdx.x]] +=
                			(r[(j*td+tid)-k+size[blockIdx.x]]*H[k+size[blockIdx.x]]);
        			}
      			}
    		}
  	}
  	__syncthreads();

  	//Down Sampling
  	if(tid < td)
    		for (j=0; j < (td*td/4)/N_samp/td; j++){
      			Vect_Dn[(j*td+tid)+size[blockIdx.x]]
			=Vect_H[(j*td+tid)*N_samp+size[blockIdx.x]];
		}


  	//Up Sampling
  	if(tid < td)
    		for (j=0; j < (td*td/4)/N_samp/td;j++){
      			Vect_Up[(j*td+tid)*N_samp+size[blockIdx.x]]
			=Vect_Dn[(j*td+tid)+size[blockIdx.x]];
		}
  	__syncthreads();

  	//convolving F
  	if(tid < td){
    		for (j=0; j< ((td*td/4)/td); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*td+tid)-k)>=0){
					Vect_F[j*td+tid+size[blockIdx.x]] += 
					(F[k]*Vect_H[(j*td+tid)-k+size[blockIdx.x]]);
        			}
      			}
    		}
  	}

}

void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn,
                        float *Vect_Up, float *Vect_F, float *F, int *size,
                        int *threads, int index){

	int td, tid;
        int i, j, k;

	for(i = 0; i < TK_NUM; i++){
		td = threads[index*TK_NUM+i];

		//convolving H
		for(tid = 0; tid < td; tid ++){
                	for (j=0; j< ((td*td/4)/td); j++){
                        	for(k = 0; k < N_col; k++){
                                	if(((j*td+tid)-k)>=0){
                                        	Vect_H[j*td+tid+size[i]] +=
                                        	(r[(j*td+tid)-k+size[i]]*H[k+size[i]]);
                                	}
                        	}
                	}
        	}

		//Down Sampling
        	for(tid = 0; tid < td; tid ++)
                	for (j=0; j < (td*td/4)/N_samp/td; j++){
                        	Vect_Dn[(j*td+tid)+size[i]]
				=Vect_H[(j*td+tid)*N_samp+size[i]];
			}

		//Up Sampling
        	for(tid = 0; tid < td; tid ++)
                	for (j=0; j < (td*td/4)/N_samp/td;j++){
                        	Vect_Up[(j*td+tid)*N_samp+size[i]]
				=Vect_Dn[(j*td+tid)+size[i]];
			}

		//convolving F
        	for(tid = 0; tid < td; tid ++){
                	for (j=0; j< ((td*td/4)/td); j++){
                        	for(k = 0; k < N_col; k++){
                                	if(((j*td+tid)-k)>=0){
                                        	Vect_F[j*td+tid+size[i]]+=
						(F[k]*Vect_H[(j*td+tid)-k+size[i]]);
                                	}
                        	}
                	}
        	}

		
	}
}


