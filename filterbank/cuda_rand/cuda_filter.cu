#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../../common/para.h"

// Num. of Channel
#define N_ch (TK_NUM * BT_NUM)
// Num. of sample
#define N_samp 8
#define N_col 64

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

__global__ void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, 
			float *Vect_Up, float *Vect_F, float *F, int threads, int size);
void FBComb(float *y, float *Vect_F, int *num_size, int index);
void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int N_sim);

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
  	int num_thread[N_ch];
  	int num_size[N_ch];

  	float **h_Vect_F;

  
  	FILE *f;
	cudaSetDevice(0);
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	cudaStream_t *filter_stream;
  
  	int i, j;
  	double start_timer, end_timer;
  
  	filter_stream = (cudaStream_t*)malloc(N_ch*sizeof(cudaStream_t));
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaStreamCreate(&filter_stream[i]));
  	}

  	f = fopen("rand.txt", "r");
  	for(i = 0; i < N_ch; i++)
    		fscanf(f, "%1d", &num_thread[i]);

  	fclose(f);

  	for(i = 0; i < N_ch; i++)
    		num_size[i] = (num_thread[i]*16)*(num_thread[i]*16);


  	r = (float**)malloc(N_ch*sizeof(float*));
  	H = (float**)malloc(N_ch*sizeof(float*));
  	F = (float**)malloc(N_ch*sizeof(float*));
  	Vect_H = (float**)malloc(N_ch*sizeof(float*));
  	Vect_Dn = (float**)malloc(N_ch*sizeof(float*));
  	Vect_Up = (float**)malloc(N_ch*sizeof(float*));
  	Vect_F = (float**)malloc(N_ch*sizeof(float*));
  	r_dev = (float**)malloc(N_ch*sizeof(float*));
  	H_dev = (float**)malloc(N_ch*sizeof(float*));
  	F_dev = (float**)malloc(N_ch*sizeof(float*));
  	Vect_H_dev = (float**)malloc(N_ch*sizeof(float*));
  	Vect_Dn_dev = (float**)malloc(N_ch*sizeof(float*));
  	Vect_Up_dev = (float**)malloc(N_ch*sizeof(float*));
  	Vect_F_dev = (float**)malloc(N_ch*sizeof(float*));

  	Vect_F_dev = (float**)malloc(N_ch*sizeof(float*));
  	h_Vect_F = (float**)malloc(N_ch*sizeof(float*));



  	/*Memory allocation*/
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaHostAlloc(&r[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&r_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&H[i], N_col*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&H_dev[i], N_col*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&F[i], N_col*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&F_dev[i], N_col*sizeof(float)));

    		checkCudaErrors(cudaHostAlloc(&Vect_H[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_H_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_Dn[i], (num_size[i]/N_samp)*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_Dn_dev[i], (num_size[i]/N_samp)*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_Up[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_Up_dev[i], num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_F[i], num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_F_dev[i], num_size[i]*sizeof(float)));

    		h_Vect_F[i] = (float*)malloc(num_size[i] * sizeof(float));

  	}

	printf("Filterbank inputs are generating\n");
  	/*init data*/
  	for(i = 0; i < N_ch; i++)
    		for(j = 0; j < num_size[i]; j++){
      			r[i][j] = j + 0.0001;
      			Vect_Up[i][j] = 0;
      			Vect_F[i][j] = 0;
      			Vect_H[i][j]=0;
      			h_Vect_F[i][j] = 0;
    		}

  	for(i = 0; i < N_ch; i++)
    		for(j = 0; j < N_col; j++){
      			H[i][j] = 0.0001;
      			F[i][j] = 0.0001;
    		}

  	// Data transfer to device
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaMemcpyAsync(r_dev[i], r[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice, filter_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(Vect_Up_dev[i], Vect_Up[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice, filter_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(Vect_F_dev[i], Vect_F[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice, filter_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(Vect_H_dev[i], Vect_H[i], num_size[i]*sizeof(float), cudaMemcpyHostToDevice, filter_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(H_dev[i], H[i], N_col*sizeof(float), cudaMemcpyHostToDevice, filter_stream[i]));
    		checkCudaErrors(cudaMemcpyAsync(F_dev[i], F[i], N_col*sizeof(float), cudaMemcpyHostToDevice, filter_stream[i]));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

	printf("Filterbank CUDA baseline is running\n");
  	// task launch
  	start_timer = my_timer();
  	for(i = 0; i < N_ch; i++){
    		FBCore<<<1, num_thread[i]*32, 0, filter_stream[i]>>>(r_dev[i], H_dev[i], Vect_H_dev[i], 
			Vect_Dn_dev[i], Vect_Up_dev[i], Vect_F_dev[i], F_dev[i], num_thread[i]*32, num_size[i]);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	end_timer = my_timer();
  	printf("Filterbank CUDA baseline Elapsed Time: %f Sec.\n", end_timer - start_timer);

  	// Data transfer back to host
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaMemcpyAsync(Vect_F[i], Vect_F_dev[i], num_size[i]*sizeof(float), cudaMemcpyDeviceToHost, filter_stream[i]));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

	/*CPU tasks*/
	printf("CPU program running\n");
        start_timer = my_timer();
        for(i = 0; i < N_ch; i++){
                h_FBCore(r[i], H[i], Vect_H[i], Vect_Dn[i], Vect_Up[i], h_Vect_F[i], F[i], num_size[i]);
        }

        end_timer = my_timer();
        //printf("CPU Elapsed time:%f Sec.\n", end_timer - start_timer);

	/*Verify*/
        printf("Verify\n");
        int flag = 0;
        for(i = 0; i < N_ch; i++){
                for(j = 0; j < num_size[i]; j++){
                        if(abs(h_Vect_F[i][j] -  Vect_F[i][j]) > 1e-3){
                                printf("Error:%f, %f, %d\n", h_Vect_F[i][j], Vect_F[i][j], i);
                                flag = 1;
                                break;
                        }
                }
        }

        if(!flag) printf("Verify successfully\n");

  	/*Free Memory*/

  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaStreamDestroy(filter_stream[i]));
  	}

  	for(i = 0; i < N_ch; i++){ 
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
		checkCudaErrors(cudaFree(Vect_F_dev[i]));

    		free(h_Vect_F[i]);
  	}

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
	free(h_Vect_F);

  return 0;
}

void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int N_sim){
        int j, k, p;

        //convolving H
        for (j=0; j< N_sim; j++)
        {
                for(k = 0; k < N_col; k++){
                        if((j-k)>=0){
                                Vect_H[j] += (r[j-k]*H[k]);
                        }
                }
        }

        //Down Sampling
        for (j=0; j < N_sim/N_samp; j++)
                Vect_Dn[j]=Vect_H[j*N_samp];

        //Up Sampling
        for (j=0; j < N_sim/N_samp;j++)
                Vect_Up[j*N_samp]=Vect_Dn[j];

        //convolving F
        for (j=0; j< N_sim; j++)
        {
                for(k = 0; k < N_col; k++){
                        if((j-k)>=0){
                                Vect_F[j]+=(F[k]*Vect_Up[j-k]);
                        }
                }
        }

}


__global__ void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn,
                        float *Vect_Up, float *Vect_F, float *F, int threads, int size){
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	int j, k;

  	//convolving H
  	if(tid < threads){
    		for (j=0; j< (size/threads); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*threads+tid)-k)>=0){
          				Vect_H[j*threads+tid] += (r[(j*threads+tid)-k]*H[k]);
        			}
      			}
    		}
  	}
  	__syncthreads();

  	//Down Sampling
  	if(tid < threads)
    		for (j=0; j < size/N_samp/threads; j++)
      			Vect_Dn[(j*threads+tid)]=Vect_H[(j*threads+tid)*N_samp];

  	//Up Sampling
  	if(tid < threads)
    		for (j=0; j < size/N_samp/threads;j++)
      			Vect_Up[(j*threads+tid)*N_samp]=Vect_Dn[(j*threads+tid)];
  	__syncthreads();

  	//convolving F
  	if(tid < threads){
    		for (j=0; j< (size/threads); j++){
      			for(k = 0; k < N_col; k++){
        			if(((j*threads+tid)-k)>=0){
          				Vect_F[j*threads+tid]+=(F[k]*Vect_Up[(j*threads+tid)-k]);
        			}
      			}
    		}
  	}
}
