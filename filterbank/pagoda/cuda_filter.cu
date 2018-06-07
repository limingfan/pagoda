#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "runtime.cuh"
#include "filter.h"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int N_sim);

void FBComb(float *y, float *Vect_F, int N_sim){
  int j;
  //adding the results to the y matrix
    for (j=0; j < N_sim; j++)
      y[j]+=Vect_F[j];

}
int main(int argc, char *argv[]){

	float **r;
  	float **r_dev;
  	float *y;
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

  	float **h_Vect_F;
  	float *h_y;

	if(argc < 4){
		printf("Error input: filter length #channel #thread\n");
		exit(1);
	}

	int N_sim = atoi(argv[1]);
   	int N_ch = atoi(argv[2]);
	int TDD_NUM = atoi(argv[3]);
  
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
  
  	int i, j;
  	double start_timer, end_timer;

  	runtime_init();
 
  	printf("Pagoda FilterBank: #task:%d, size:%d, #threads:%d\n", N_ch, N_sim, TDD_NUM); 

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
  	h_Vect_F = (float**)malloc(N_ch*sizeof(float*));
  


  	/*Memory allocation*/
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaHostAlloc(&r[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&r_dev[i], N_sim*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&H[i], N_col*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&H_dev[i], N_col*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&F[i], N_col*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&F_dev[i], N_col*sizeof(float)));

    		checkCudaErrors(cudaHostAlloc(&Vect_H[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_H_dev[i], N_sim*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_Dn[i], (N_sim/N_samp)*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_Dn_dev[i], (N_sim/N_samp)*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_Up[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_Up_dev[i], N_sim*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&Vect_F[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&Vect_F_dev[i], N_sim*sizeof(float)));
    		h_Vect_F[i] = (float*)malloc(N_sim * sizeof(float));
  	}

  	y = (float*)malloc(N_sim*sizeof(float));
  	h_y = (float*)malloc(N_sim*sizeof(float));

	//srand(time(NULL));
  	/*init data*/
  	for(i = 0; i < N_ch; i++)
    		for(j = 0; j < N_sim; j++){
      			r[i][j] = j + 0.0001;
      			y[j] = 0;
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

  	double timer = 0.0;
  	start_timer = my_timer();

  	// Data transfer to device
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaMemcpyAsync(r_dev[i], r[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(Vect_Up_dev[i], Vect_Up[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(Vect_F_dev[i], Vect_F[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(Vect_H_dev[i], Vect_H[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(H_dev[i], H[i], N_col*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(F_dev[i], F[i], N_col*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
  	}

  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  	end_timer = my_timer();
  	timer = (end_timer - start_timer);

  	// task launch
  	start_timer = my_timer();

  	for(i = 0; i < N_ch; i++){
    		taskLaunch(14, INT, TDD_NUM, INT, 1, INT, 0, INT, 1, INT, 0, FLOAT, r_dev[i],
			FLOAT, H_dev[i], FLOAT, Vect_H_dev[i], FLOAT, Vect_Dn_dev[i], FLOAT, Vect_Up_dev[i],
			FLOAT, Vect_F_dev[i], FLOAT, F_dev[i], INT, N_sim, INT, TDD_NUM);
  	}
  	waitAll(N_ch);

  	end_timer = my_timer();
  	printf("The GPU Elapsed Time: %f Sec.\n", end_timer - start_timer);

  	start_timer = my_timer();

  	// Data transfer back to host
  	for(i = 0; i < N_ch; i++){
    		checkCudaErrors(cudaMemcpyAsync(Vect_F[i], Vect_F_dev[i], N_sim*sizeof(float), cudaMemcpyDeviceToHost, runtime_stream));
  	}
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

  	end_timer = my_timer();
  	timer += (end_timer - start_timer);
  	printf("The memory time:%f Sec.\n", timer);
  
  	runtime_destroy();
  	runtime_free();

  	/*Merge process*/
  	for(i = 0; i < N_ch; i++){
    		FBComb(y, Vect_F[i], N_sim);
  	}

  	/*CPU tasks*/
  	start_timer = my_timer();
  	for(i = 0; i < N_ch; i++){
    		h_FBCore(r[i], H[i], Vect_H[i], Vect_Dn[i], Vect_Up[i], h_Vect_F[i], F[i], N_sim);
  	}

  	end_timer = my_timer();
  	printf("CPU Elapsed time:%f Sec.\n", end_timer - start_timer);

  	/*Merge process*/
  	for(i = 0; i < N_ch; i++){
    		FBComb(h_y, h_Vect_F[i], N_sim);
  	}

  	/*Verify*/
  	printf("Verify\n");
	long long flag = 0;
  	for(i = 0; i < N_sim; i++){
    		if(abs(h_y[i] -  y[i]) > 1e-3){
			printf("Error:%f, %f, %d\n", h_y[i], y[i], i);
			break;
    		}
		flag ++;
  	}

	if(flag == N_sim) printf("Verify successfully\n");

  	/*Free Memory*/

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
    		free(h_Vect_F[i]);
  	}

  	free(y);
  	free(h_y);
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
