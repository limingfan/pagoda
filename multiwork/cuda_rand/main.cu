#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "kernel.cu"
#include "kernel.h"
#include "headers.h"
//#include "../../common/para.h"
#define TK_NUM 2048 //num. of task in each category
#define task (TK_NUM*4)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void init_matrix(int *A, int *B, int *C, int *D, int size);
void init_filter(float *r, float *Vect_Up, float *Vect_F, 
                float *Vect_H, float *H, float *F, float *Vect_H_host, int size);
void init_des(unsigned char *packet_in, int size, int index);

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

  int num_thread[task];
  int num_size[task];
  FILE *fp;

  cudaStream_t work_stream[TK_NUM];
  cudaSetDevice(0);
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaStreamCreate(&work_stream[i]));
  }

  double start_timer, end_timer;

  fp = fopen("rand.txt", "r");
  for(i = 0; i < task; i++)
    fscanf(fp, "%1d", &num_thread[i]);

  fclose(fp);

  for(i = 0; i < task; i++)
    num_size[i] = num_thread[i]*32;

  //matrix mult.
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaHostAlloc(&h_A[i], num_size[i]*num_size[i]*sizeof(int), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&h_B[i], num_size[i]*num_size[i]*sizeof(int), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&h_C[i], num_size[i]*num_size[i]*sizeof(int), cudaHostAllocDefault));
  }

  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMalloc(&d_A[i], num_size[i]*num_size[i]*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_B[i], num_size[i]*num_size[i]*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_C[i], num_size[i]*num_size[i]*sizeof(int)));
    h_D[i] = (int*)malloc(sizeof(int)*num_size[i]*num_size[i]);
  }
  // mandelbrot
  h_task_indx = (float*)malloc(TK_NUM * sizeof(float));
  checkCudaErrors(cudaMalloc(&d_task_indx, TK_NUM *sizeof(float)));
  for(i = 0; i < TK_NUM; i++) {
    h_task_indx[i] = (float)(i/(TK_NUM/2.0));
    checkCudaErrors(cudaHostAlloc(&h_count[i], num_size[i+TK_NUM] * num_size[i+TK_NUM] *sizeof(int), NULL));
    checkCudaErrors(cudaMalloc(&d_count[i], num_size[i+TK_NUM] * num_size[i+TK_NUM] *sizeof(int)));
    h_count_host[i] = (int*)malloc(num_size[i+TK_NUM] * num_size[i+TK_NUM] * sizeof(int));

  }
  //filter bank
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaHostAlloc(&h_r[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_r[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_H[i], N_col*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_H[i], N_col*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_F[i], N_col*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_F[i], N_col*sizeof(float)));

    checkCudaErrors(cudaHostAlloc(&h_Vect_H[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_H[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_Dn[i], (num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]/N_samp)*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_Dn[i], (num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]/N_samp)*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_Up[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_Up[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_F[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_F[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float)));
    h_Vect_F_host[i] = (float*)malloc(num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float));
  }

  //DES
  for(i = 0; i < TK_NUM; i++){
      checkCudaErrors(cudaHostAlloc(&h_packet_in[i], num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char), cudaHostAllocDefault));
      checkCudaErrors(cudaMalloc(&d_packet_in[i], num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char)));
      checkCudaErrors(cudaHostAlloc(&h_packet_out[i], num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char), cudaHostAllocDefault));
      checkCudaErrors(cudaMalloc(&d_packet_out[i], num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char)));
      h_packet_host[i] =  (unsigned char *) malloc (num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char));
  }
  checkCudaErrors(cudaHostAlloc(&h_des_esk, 96*sizeof(uint32), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&d_des_esk, 96*sizeof(uint32)));
  checkCudaErrors(cudaHostAlloc(&h_des_dsk, 96*sizeof(uint32), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&d_des_dsk, 96*sizeof(uint32)));

  printf("MPE CUDA baseline inputs are generating\n");
   /*Generate encryption key*/
  des_set_key(h_des_esk, h_des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);
  

  //Init.matrix
  for(i = 0; i < TK_NUM; i++){
    init_matrix(h_A[i], h_B[i], h_C[i], h_D[i], num_size[i]);
  }

  //Init filter
  for(i = 0; i < TK_NUM; i++){
    init_filter(h_r[i], h_Vect_Up[i], h_Vect_F[i], 
                h_Vect_H[i], h_H[i], h_F[i], h_Vect_F_host[i], 
		num_size[i+2*TK_NUM]*num_size[i+2*TK_NUM]);
  }

  //Init DES
  for(i = 0; i < TK_NUM; i++){
    init_des(h_packet_in[i], num_size[i+3*TK_NUM]*num_size[i+3*TK_NUM], i);
  }
  
  //mem copy
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(d_A[i], h_A[i], num_size[i] * num_size[i]*sizeof(int), cudaMemcpyHostToDevice, work_stream[i]));
    checkCudaErrors(cudaMemcpyAsync(d_B[i], h_B[i], num_size[i] * num_size[i]*sizeof(int), cudaMemcpyHostToDevice, work_stream[i]));
  }
  for(i = 0; i < TK_NUM; i++){

    checkCudaErrors(cudaMemcpyAsync(d_r[i], h_r[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaMemcpyHostToDevice, work_stream[i]));
    checkCudaErrors(cudaMemcpyAsync(d_Vect_Up[i], h_Vect_Up[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaMemcpyHostToDevice, work_stream[i]));
    checkCudaErrors(cudaMemcpyAsync(d_Vect_F[i], h_Vect_F[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaMemcpyHostToDevice, work_stream[i]));
    checkCudaErrors(cudaMemcpyAsync(d_Vect_H[i], h_Vect_H[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaMemcpyHostToDevice, work_stream[i]));
    checkCudaErrors(cudaMemcpyAsync(d_H[i], h_H[i], N_col*sizeof(float), cudaMemcpyHostToDevice, work_stream[i]));
    checkCudaErrors(cudaMemcpyAsync(d_F[i], h_F[i], N_col*sizeof(float), cudaMemcpyHostToDevice, work_stream[i]));
  }
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(d_packet_in[i], h_packet_in[i], num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char), cudaMemcpyHostToDevice, work_stream[i]));


  }
  checkCudaErrors(cudaMemcpyAsync(d_task_indx, h_task_indx, TK_NUM*sizeof(float), cudaMemcpyHostToDevice, work_stream[0]));
  checkCudaErrors(cudaMemcpyAsync(d_des_esk, h_des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice, work_stream[0]));
  checkCudaErrors(cudaMemcpyAsync(d_des_dsk, h_des_dsk, 96*sizeof(uint32), cudaMemcpyHostToDevice, work_stream[0]));
  checkCudaErrors(cudaDeviceSynchronize());
  
  printf("MPE CUDA baseline is running\n");
  start_timer = my_timer();
  // compute
  int mult_c, mand_c, filter_c, des_c;
  mult_c = 0, mand_c = 0, filter_c = 0, des_c = 0;

  for(i = 0; i < task; i++){
    switch(i%4){
      case 0:
        mult_gpu<<<1, num_thread[mult_c]*32, 0, work_stream[mult_c]>>>(d_A[mult_c], d_B[mult_c], d_C[mult_c], num_size[mult_c], num_thread[mult_c]*32);
        mult_c ++;
        break;
      case 1:
        get_pixel<<<1, num_thread[mand_c+TK_NUM]*32, 0, work_stream[mand_c]>>>(d_count[mand_c], &d_task_indx[mand_c], num_size[mand_c+TK_NUM], num_thread[mand_c+TK_NUM]*32);
        mand_c ++;
        break;
      case 2:
        FBCore<<<1, num_thread[filter_c+2*TK_NUM]*32, 0, work_stream[filter_c]>>>(d_r[filter_c], d_H[filter_c], d_Vect_H[filter_c], 
				d_Vect_Dn[filter_c], d_Vect_Up[filter_c], d_Vect_F[filter_c], d_F[filter_c], 
				num_size[filter_c+2*TK_NUM] * num_size[filter_c+2*TK_NUM], num_thread[filter_c+2*TK_NUM]*32);
        filter_c ++;
        break;
#if 1
      case 3:
        des_encrypt_dev<<<1, num_thread[des_c+3*TK_NUM]*32, 0, work_stream[des_c]>>>(d_des_esk, d_des_dsk, d_packet_in[des_c],
                                        d_packet_out[des_c], (num_size[des_c+3*TK_NUM] * num_size[des_c+3*TK_NUM])/8, 
					num_thread[des_c+3*TK_NUM]*32);
        des_c ++;
        break;
#endif
    
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
  end_timer = my_timer();
  printf("Multiprogramming CUDA baseline elapsed Time: %lf Sec.\n", end_timer - start_timer);
 
  // memory copy back
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(h_C[i],d_C[i], num_size[i] * num_size[i]*sizeof(int), cudaMemcpyDeviceToHost, work_stream[i]));
  }
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(h_count[i], d_count[i], num_size[i+TK_NUM] * num_size[i+TK_NUM]*sizeof(int), cudaMemcpyDeviceToHost, work_stream[i]));
  }
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(h_Vect_F[i], d_Vect_F[i], num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float), cudaMemcpyDeviceToHost, work_stream[i]));
  }
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(h_packet_out[i], d_packet_out[i], num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char), cudaMemcpyDeviceToHost, work_stream[i]));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  mult_c = 0, mand_c = 0, filter_c = 0, des_c = 0;

  start_timer = my_timer();
  // cpu compute
  for(i = 0; i < task; i++){
    switch(i%4){
      case 0:
        mult(h_A[mult_c], h_B[mult_c], h_D[mult_c], num_size[mult_c], num_thread[mult_c]*32);
        mult_c ++;
        break;
      case 1:
        h_get_pixel(h_count_host[mand_c], h_task_indx[mand_c], num_size[mand_c+TK_NUM]);
        mand_c++;
        break;
      case 2:
        h_FBCore(h_r[filter_c], h_H[filter_c], h_Vect_H[filter_c], 
			h_Vect_Dn[filter_c], h_Vect_Up[filter_c], h_Vect_F_host[filter_c], 
			h_F[filter_c], num_size[filter_c+2*TK_NUM] * num_size[filter_c+2*TK_NUM]);
        filter_c ++;
        break;
#if 1
      case 3:
        des_encrypt(h_des_esk, h_des_dsk, h_packet_in[des_c], h_packet_host[des_c], (num_size[des_c+3*TK_NUM] * num_size[des_c+3*TK_NUM])/8);
        des_c ++; 
        break; 
#endif
    }
  }
  end_timer = my_timer();
  //printf("CPU elapsed time:%lf Sec.\n", end_timer - start_timer);


  //verificiation
  printf("verifying\n");
  int flag = 0;
  for(i = 0; i < TK_NUM; i++){
    for(j = 0; j < num_size[i] * num_size[i]; j++){
      if(h_C[i][j] != h_D[i][j]){
        printf("Mult, Error:%d, %d\n", h_C[i][j], h_D[i][j]);
	flag = 1;
        break;
      }
    }
  }

  for(i = 0; i < TK_NUM; i++){
    for(j = 0; j < num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]; j++){
      if(abs(h_Vect_F[i][j]- h_Vect_F_host[i][j]) > 0.1){
        printf("Filter Error:%f, %f\n", h_Vect_F[i][j], h_Vect_F_host[i][j]);
	flag = 1;
        break;
      }
    }
  }
#if 1
  for(i = 0; i < TK_NUM; i++){
    for(j = 0; j < num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]; j++){
        if(h_packet_out[i][j] != h_packet_host[i][j]){
          printf("DES Error:%u, %u, %d, %d\n", h_packet_out[i][j], h_packet_host[i][j], i, j);
	  flag = 1;
          break;
      }
    }
  }
  if(!flag) printf("verify successfully\n");
#endif
  //free mem.
  for(i = 0; i < TK_NUM; i++){
    checkCudaErrors(cudaStreamDestroy(work_stream[i]));

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


return 0;
}

void init_matrix(int *A, int *B, int *C, int *D, int size){
  int i;

  for(i = 0; i < size; i++){
      A[i] = (i%8)+1;
      B[i] = (i%8)+1;
      C[i] = 0;
      D[i] = 0;
  }
}

void init_filter(float *r, float *Vect_Up, float *Vect_F, 
		float *Vect_H, float *H, float *F, 
		float *Vect_F_host, int size){
  int i;

  for(i = 0; i < size; i++){
    r[i] = i + 0.0001;
    Vect_Up[i] = 0;
    Vect_F[i] = 0;
    Vect_H[i]=0;
    Vect_F_host[i] = 0;
  }

  for(i = 0; i < N_col; i++){
    H[i] = 0.0001;
    F[i] = 0.0001;
  }

}

void init_des(unsigned char *packet_in, int size, int index){
  int i;

  for(i = 0; i < size; i++){
    if(i < HEADER_SIZE ){
      packet_in[i] = headers[index % MAX_PACKETS][i];
    }else{
      packet_in[i] = DES3_init[i%8];
    }
   }
}
