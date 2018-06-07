#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "kernel.h"
#include "headers.h"
#include "runtime.cuh"

#define TK_NUM 2048 //num. of task in each category
#define task (TK_NUM*4)

#define MAX_PACKETS     100
#define MAX_INDEX       32768  

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void init_matrix(int **A, int **B, int **C, int **D);
void init_filter(float **r, float **Vect_Up, float **Vect_F, 
                float **Vect_H, float **H, float **F, float *y, float **Vect_F_host);
void init_des(unsigned char **packet_in);

int des_main_ks( uint32 *SK, uint8 *key );
int des_set_key( uint32 *esk, uint32 *dsk, uint8 key1[8],
                                uint8 key2[8], uint8 key3[8]);


int main(){

  	int i, j;
  	int *h_A[TK_NUM], *h_B[TK_NUM], *h_C[TK_NUM], *h_D[TK_NUM];
  	int *d_A[TK_NUM], *d_B[TK_NUM], *d_C[TK_NUM];

  	int *h_count[TK_NUM];
  	int *d_count[TK_NUM];
  	int *h_count_host[TK_NUM];
  	float *h_task_indx;
  	float *d_task_indx;

  	float *h_r[TK_NUM],*d_r[TK_NUM];
  	float *y, *h_H[TK_NUM], *d_H[TK_NUM];
  	float *h_F[TK_NUM], *d_F[TK_NUM];

  	float *h_Vect_H[TK_NUM], *d_Vect_H[TK_NUM]; // output of the F
  	float *h_Vect_Dn[TK_NUM], *d_Vect_Dn[TK_NUM]; // output of the down sampler
  	float *h_Vect_Up[TK_NUM], *d_Vect_Up[TK_NUM]; // output of the up sampler
  	float *h_Vect_F[TK_NUM], *d_Vect_F[TK_NUM], *h_Vect_F_host[TK_NUM]; // this is the output of the

  	unsigned char *h_packet_in[TK_NUM], *d_packet_in[TK_NUM];
  	unsigned char *h_packet_out[TK_NUM], *d_packet_out[TK_NUM];
  	unsigned char *h_packet_host[TK_NUM];

  	uint32 *h_des_esk;
  	uint32 *h_des_dsk;

  	uint32 *d_des_esk;
  	uint32 *d_des_dsk;

  	double start_timer, end_timer;

  	//matrix mult.
  	for(i = 0; i < TK_NUM; i++){
    		checkCudaErrors(cudaHostAlloc(&h_A[i], MSIZE*sizeof(int), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&h_B[i], MSIZE*sizeof(int), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&h_C[i], MSIZE*sizeof(int), cudaHostAllocDefault));
  	}

  	for(i = 0; i < TK_NUM; i++){
    		checkCudaErrors(cudaMalloc(&d_A[i], MSIZE*sizeof(int)));
    		checkCudaErrors(cudaMalloc(&d_B[i], MSIZE*sizeof(int)));
    		checkCudaErrors(cudaMalloc(&d_C[i], MSIZE*sizeof(int)));
    		h_D[i] = (int*)malloc(sizeof(int)*MSIZE);
  	}

  	// mandelbrot
  	h_task_indx = (float*)malloc(TK_NUM * sizeof(float));
  	checkCudaErrors(cudaMalloc(&d_task_indx, TK_NUM *sizeof(float)));

  	for(i = 0; i < TK_NUM; i++) {
    		h_task_indx[i] = (float)(i/(TK_NUM/2.0));
    		checkCudaErrors(cudaHostAlloc(&h_count[i], n * n *sizeof(int), NULL));
    		checkCudaErrors(cudaMalloc(&d_count[i], n * n *sizeof(int)));
    		h_count_host[i] = (int*)malloc(n * n * sizeof(int));

  	}

  	//filter bank
  	for(i = 0; i < TK_NUM; i++){
    		checkCudaErrors(cudaHostAlloc(&h_r[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_r[i], N_sim*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&h_H[i], N_col*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_H[i], N_col*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&h_F[i], N_col*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_F[i], N_col*sizeof(float)));

    		checkCudaErrors(cudaHostAlloc(&h_Vect_H[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_Vect_H[i], N_sim*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&h_Vect_Dn[i], (N_sim/N_samp)*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_Vect_Dn[i], (N_sim/N_samp)*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&h_Vect_Up[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_Vect_Up[i], N_sim*sizeof(float)));
    		checkCudaErrors(cudaHostAlloc(&h_Vect_F[i], N_sim*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaMalloc(&d_Vect_F[i], N_sim*sizeof(float)));
    		h_Vect_F_host[i] = (float*)malloc(N_sim*sizeof(float));
  	}
  	y = (float*)malloc(N_sim*sizeof(float));

  	//DES
  	for(i = 0; i < TK_NUM; i++){
      		checkCudaErrors(cudaHostAlloc(&h_packet_in[i], LEN*sizeof(unsigned char), cudaHostAllocDefault));
      		checkCudaErrors(cudaMalloc(&d_packet_in[i], LEN*sizeof(unsigned char)));
      		checkCudaErrors(cudaHostAlloc(&h_packet_out[i], LEN*sizeof(unsigned char), cudaHostAllocDefault));
      		checkCudaErrors(cudaMalloc(&d_packet_out[i], LEN*sizeof(unsigned char)));
      		h_packet_host[i] =  (unsigned char *) malloc (LEN*sizeof(unsigned char));
  	}

  	checkCudaErrors(cudaHostAlloc(&h_des_esk, 96*sizeof(uint32), cudaHostAllocDefault));
  	checkCudaErrors(cudaMalloc(&d_des_esk, 96*sizeof(uint32)));
  	checkCudaErrors(cudaHostAlloc(&h_des_dsk, 96*sizeof(uint32), cudaHostAllocDefault));
  	checkCudaErrors(cudaMalloc(&d_des_dsk, 96*sizeof(uint32)));


   	/*Generate encryption key*/
  	des_set_key(h_des_esk, h_des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);
  

  	//Init.matrix
  	init_matrix(h_A, h_B, h_C, h_D);
  	//Init filter
  	init_filter(h_r, h_Vect_Up, h_Vect_F, 
                h_Vect_H, h_H, h_F, y, h_Vect_F_host);
  	//Init DES
  	init_des(h_packet_in);

  	//Init runtime
  	runtime_init();

  	double timer = 0.0;

  	start_timer = my_timer();

  	//mem copy
  	for(i = 0; i < TK_NUM; i++){
    		checkCudaErrors(cudaMemcpyAsync(d_A[i], h_A[i], MSIZE*sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_B[i], h_B[i], MSIZE*sizeof(int), cudaMemcpyHostToDevice, runtime_stream));

    		checkCudaErrors(cudaMemcpyAsync(d_r[i], h_r[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_Vect_Up[i], h_Vect_Up[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_Vect_F[i], h_Vect_F[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_Vect_H[i], h_Vect_H[i], N_sim*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_H[i], h_H[i], N_col*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_F[i], h_F[i], N_col*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(d_packet_in[i], h_packet_in[i], LEN*sizeof(unsigned char), cudaMemcpyHostToDevice, runtime_stream));

  	}

  	checkCudaErrors(cudaMemcpyAsync(d_task_indx, h_task_indx, TK_NUM*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_des_esk, h_des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice, runtime_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_des_dsk, h_des_dsk, 96*sizeof(uint32), cudaMemcpyHostToDevice, runtime_stream));
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  	end_timer = my_timer();
  	timer = end_timer - start_timer;
  
  	printf("Pagoda MultiWrok: #task:%d, #thread:%d\n", task, TDD_NUM);
  	// compute
  	int mult_c, mand_c, filter_c, des_c;
  	mult_c = 0, mand_c = 0, filter_c = 0, des_c = 0;
  	start_timer = my_timer();

  	for(i = 0; i < task; i++){
    		switch(i%4){
      			case 0:
        			taskLaunch(9, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 0, INT, d_A[mult_c], INT, d_B[mult_c], INT, d_C[mult_c], INT, MROW);
        			mult_c ++;
        			break;
      			case 1:
        			taskLaunch(7, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 1, INT, d_count[mand_c], INT, &d_task_indx[mand_c]);
        			mand_c ++;
        			break;
      			case 2:
        			taskLaunch(12, INT, TDD_NUM, INT, 1, INT, 0, INT, 1, INT, 2, FLOAT, d_r[filter_c],
                			FLOAT, d_H[filter_c], FLOAT, d_Vect_H[filter_c], FLOAT, d_Vect_Dn[filter_c], FLOAT, d_Vect_Up[filter_c],
                			FLOAT, d_Vect_F[filter_c], FLOAT, d_F[filter_c]);
        				filter_c ++;
        			break;
      			case 3:
        			taskLaunch(10, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 3, INT32, d_des_esk, INT32, d_des_esk,
                               		CHAR, d_packet_in[des_c], CHAR, d_packet_out[des_c], INT, LEN/8);

        			des_c ++;
        			break;
    
    		}
  	}
  	waitAll(task);

  	end_timer = my_timer();
  	printf("GPU elapsed time:%lf Sec.\n", end_timer - start_timer);
 
  	start_timer = my_timer(); 

  	// memory copy back
  	for(i = 0; i < TK_NUM; i++){
    		checkCudaErrors(cudaMemcpyAsync(h_C[i],d_C[i], MSIZE*sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(h_count[i], d_count[i], n * n*sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(h_Vect_F[i], d_Vect_F[i], N_sim*sizeof(float), cudaMemcpyDeviceToHost, runtime_stream));
    		checkCudaErrors(cudaMemcpyAsync(h_packet_out[i], d_packet_out[i], LEN*sizeof(unsigned char), cudaMemcpyDeviceToHost, runtime_stream));
  	}
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  	end_timer = my_timer();
  	timer += (end_timer - start_timer);
  	printf("Mem. copy time:%lf Sec.\n", timer);

  	runtime_destroy();
  	runtime_free();

  	mult_c = 0, mand_c = 0, filter_c = 0, des_c = 0;

  	start_timer = my_timer();

  	// cpu compute
  	for(i = 0; i < task; i++){
    		switch(i%4){
      			case 0:
        			mult(h_A[mult_c], h_B[mult_c], h_D[mult_c], MROW);
        			mult_c ++;
        			break;
      			case 1:
        			h_get_pixel(h_count_host[mand_c], h_task_indx[mand_c]);
        			mand_c++;
        			break;

      			case 2:
        			h_FBCore(h_r[filter_c], h_H[filter_c], h_Vect_H[filter_c], 
				h_Vect_Dn[filter_c], h_Vect_Up[filter_c], h_Vect_F_host[filter_c], h_F[filter_c]);
        			filter_c ++;
        			break;

      			case 3:
        			des_encrypt(h_des_esk, h_des_dsk, h_packet_in[des_c], h_packet_host[des_c], LEN/8);
        			des_c ++; 
        			break; 
    		}
  	}
  	end_timer = my_timer();
  	printf("CPU elapsed time:%lf Sec.\n", end_timer - start_timer);


  	//verificiation
  	printf("Verifying\n");
	long long flag = 0;

  	for(i = 0; i < TK_NUM; i++){
    		for(j = 0; j < MSIZE; j++){
      			if(h_C[i][j] != h_D[i][j]){
        			printf("Mult, Error:%d, %d\n", h_C[i][j], h_D[i][j]);
       				 break;
      			}
			flag ++;
    		}

    		for(j = 0; j < N_sim; j++){
      			if(abs(h_Vect_F[i][j]- h_Vect_F_host[i][j]) > 0.1){
        			printf("Filter Error:%f, %f\n", h_Vect_F[i][j], h_Vect_F_host[i][j], i, j);
        			break;
      			}
			flag ++;
    		}

    		for(j = 0; j < LEN; j++){
        		if(h_packet_out[i][j] != h_packet_host[i][j]){
          			printf("DES Error:%u, %u, %d, %d\n", h_packet_out[i][j], h_packet_host[i][j], i, j);
          			break;
      			}
			flag ++;
    		}

  	}
	if(flag == (TK_NUM * MSIZE + TK_NUM * N_sim + TK_NUM * LEN)) printf("Verifying Successfully\n");

  	//free mem.
  	for(i = 0; i < TK_NUM; i++){

    		checkCudaErrors(cudaFreeHost(h_A[i]));
    		checkCudaErrors(cudaFree(d_A[i]));
    		checkCudaErrors(cudaFreeHost(h_B[i]));
    		checkCudaErrors(cudaFree(d_B[i]));
   	 	checkCudaErrors(cudaFreeHost(h_C[i]));
    		checkCudaErrors(cudaFree(d_C[i]));

    		checkCudaErrors(cudaFreeHost(h_count[i]));
    		checkCudaErrors(cudaFree(d_count[i]));

    		checkCudaErrors(cudaFreeHost(h_r[i]));
    		checkCudaErrors(cudaFree(d_r[i]));
    		checkCudaErrors(cudaFreeHost(h_H[i]));
    		checkCudaErrors(cudaFree(d_H[i]));
    		checkCudaErrors(cudaFreeHost(h_F[i]));
    		checkCudaErrors(cudaFree(d_F[i]));

    		checkCudaErrors(cudaFreeHost(h_Vect_H[i]));
    		checkCudaErrors(cudaFree(d_Vect_H[i]));
    		checkCudaErrors(cudaFreeHost(h_Vect_Dn[i]));
    		checkCudaErrors(cudaFree(d_Vect_Dn[i]));
    		checkCudaErrors(cudaFreeHost(h_Vect_Up[i]));
    		checkCudaErrors(cudaFree(d_Vect_Up[i]));
    		checkCudaErrors(cudaFreeHost(h_Vect_F[i]));
    		checkCudaErrors(cudaFree(d_Vect_F[i]));

    		checkCudaErrors(cudaFreeHost(h_packet_in[i]));
    		checkCudaErrors(cudaFree(d_packet_in[i]));
    		checkCudaErrors(cudaFreeHost(h_packet_out[i]));
    		checkCudaErrors(cudaFree(d_packet_out[i]));

    		free(h_packet_host[i]);
    		free(h_count_host[i]);
    		free(h_Vect_F_host[i]);
  	}

  	checkCudaErrors(cudaFree(d_task_indx));

  	checkCudaErrors(cudaFreeHost(h_des_esk));
  	checkCudaErrors(cudaFree(d_des_esk));
  	checkCudaErrors(cudaFreeHost(h_des_dsk));
  	checkCudaErrors(cudaFree(d_des_dsk));

  	free(h_task_indx);
  	free(y);

  	if(cudaDeviceReset()== cudaSuccess) printf("Reset successful\n");

return 0;
}

void init_matrix(int **A, int **B, int **C, int **D){
  int i, j;

  for(i = 0; i < TK_NUM; i++){
    for(j = 0; j < MSIZE; j++){
      A[i][j] = (i%MROW)+1;
      B[i][j] = (i%MCOL)+1;
      C[i][j] = 0;
      D[i][j] = 0;
    }
  }
}

void init_filter(float **r, float **Vect_Up, float **Vect_F, 
		float **Vect_H, float **H, float **F, float *y, float **Vect_F_host){
  int i, j;

  for(i = 0; i < TK_NUM; i++)
    for(j = 0; j < N_sim; j++){
      r[i][j] = j + 0.0001;
      y[j] = 0;
      Vect_Up[i][j] = 0;
      Vect_F[i][j] = 0;
      Vect_H[i][j]=0;
      Vect_F_host[i][j] = 0;
    }

  for(i = 0; i < TK_NUM; i++)
    for(j = 0; j < N_col; j++){
      H[i][j] = 0.0001;
      F[i][j] = 0.0001;
    }

}

void init_des(unsigned char **packet_in){
  int i, j;
  for(i = 0; i < TK_NUM; i++){
      for(j = 0; j < LEN; j++){
          if(j < HEADER_SIZE ){
              packet_in[i][j] = headers[i % MAX_PACKETS][j];
          }else{
              packet_in[i][j] = DES3_init[j%8];
          }
      }
  }
}

