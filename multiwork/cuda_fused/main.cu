#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "kernel.cu"
#include "kernel.h"
#include "headers.h"

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

  int i, j, k;
  int *h_A[BT_NUM], *h_B[BT_NUM], *h_C[BT_NUM], *h_D[BT_NUM];
  int *d_A[BT_NUM], *d_B[BT_NUM], *d_C[BT_NUM];

  int *h_count[BT_NUM];
  int *d_count[BT_NUM];
  int *h_count_host[BT_NUM];
  float *h_task_indx;
  float *d_task_indx;

  float *h_r[BT_NUM],*d_r[BT_NUM];
  float *y, *h_H[BT_NUM], *d_H[BT_NUM];
  float *h_F[BT_NUM], *d_F[BT_NUM];

  float *h_Vect_H[BT_NUM], *d_Vect_H[BT_NUM]; // output of the F
  float *h_Vect_Dn[BT_NUM], *d_Vect_Dn[BT_NUM]; // output of the down sampler
  float *h_Vect_Up[BT_NUM], *d_Vect_Up[BT_NUM]; // output of the up sampler
  float *h_Vect_F[BT_NUM], *d_Vect_F[BT_NUM], *h_Vect_F_host[BT_NUM]; // this is the output of the

  unsigned char *h_packet_in[BT_NUM], *d_packet_in[BT_NUM];
  unsigned char *h_packet_out[BT_NUM], *d_packet_out[BT_NUM];
  unsigned char *h_packet_host[BT_NUM];

  uint32 *h_des_esk;
  uint32 *h_des_dsk;

  uint32 *d_des_esk;
  uint32 *d_des_dsk;

  int num_thread[task], *d_num_thread;
  int num_size[BT_NUM*(TK_NUM/SUB_NUM)];
  int pos_task[BT_NUM][TK_NUM];
  int *pos_task_dev[BT_NUM];

  FILE *fp;
  cudaSetDevice(0);
  double start_timer, end_timer;

  fp = fopen("rand.txt", "r");
  for(i = 0; i < task; i++)
    fscanf(fp, "%1d", &num_thread[i]);

  fclose(fp);

  for(i = 0; i < task; i++)
    num_thread[i] *= 32;

  for(i = 0; i < BT_NUM*(TK_NUM/SUB_NUM); i++){
    num_size[i] = 0;
    switch(i/BT_NUM){
      case 0:
	for(j = 0; j < SUB_NUM; j++)
	  num_size[i] += (num_thread[i*TK_NUM+j]*num_thread[i*TK_NUM+j]);
        break;
      case 1:
	for(j = SUB_NUM; j < (2*SUB_NUM); j++)
	  num_size[i] += (num_thread[(i%BT_NUM)*TK_NUM+j]*num_thread[(i%BT_NUM)*TK_NUM+j]);
	break;
      case 2:
        for(j = (2*SUB_NUM); j < (3*SUB_NUM); j++)
          num_size[i] += (num_thread[(i%BT_NUM)*TK_NUM+j]*num_thread[(i%BT_NUM)*TK_NUM+j]);
        break;
      case 3:
        for(j = (3*SUB_NUM); j < (4*SUB_NUM); j++)
          num_size[i] += (num_thread[(i%BT_NUM)*TK_NUM+j]*num_thread[(i%BT_NUM)*TK_NUM+j]);
        break;
    }

  }

  for(i = 0; i < BT_NUM; i++){
    for(j = 0; j < TK_NUM/SUB_NUM; j++){
      switch(j){
        case 0:
          for(k = 0; k < SUB_NUM; k++){
            if(k == 0) pos_task[i][k] = 0;
	    else pos_task[i][k] += pos_task[i][k-1] + (num_thread[i*TK_NUM+k-1]*num_thread[i*TK_NUM+k-1]);
          }
          break;
         case 1:
	   for(k = SUB_NUM; k < 2*SUB_NUM; k++){
             if(k == SUB_NUM) pos_task[i][k] = 0;
             else pos_task[i][k] += pos_task[i][k-1] + (num_thread[i*TK_NUM+k-1]*num_thread[i*TK_NUM+k-1]);
           }
           break;
          case 2:
	   for(k = 2*SUB_NUM; k < 3*SUB_NUM; k++){
             if(k == 2*SUB_NUM) pos_task[i][k] = 0;
             else pos_task[i][k] += pos_task[i][k-1] + (num_thread[i*TK_NUM+k-1]*num_thread[i*TK_NUM+k-1]);
           }
             break;
	  case 3:
	   for(k = 3*SUB_NUM; k < 4*SUB_NUM; k++){
     	      if(k == 3*SUB_NUM) pos_task[i][k] = 0;
              else pos_task[i][k] += pos_task[i][k-1] + (num_thread[i*TK_NUM+k-1]*num_thread[i*TK_NUM+k-1]);
            }
           break;
        }
     }
  }

  checkCudaErrors(cudaMalloc(&d_num_thread, task*sizeof(int)));
  //matrix mult.
  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaHostAlloc(&h_A[i], num_size[i]*sizeof(int), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&h_B[i], num_size[i]*sizeof(int), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&h_C[i], num_size[i]*sizeof(int), cudaHostAllocDefault));
  }

  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMalloc(&d_A[i], num_size[i]*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_B[i], num_size[i]*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_C[i], num_size[i]*sizeof(int)));
    checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM*sizeof(int)));
    h_D[i] = (int*)malloc(sizeof(int)*num_size[i]);
  }

  // mandelbrot
  h_task_indx = (float*)malloc(task * sizeof(float));
  checkCudaErrors(cudaMalloc(&d_task_indx, task*sizeof(float)));

  for(i = 0; i < task; i++){
    h_task_indx[i] = (float)(i/(task/2.0));
  }
  for(i = 0; i < BT_NUM; i++) {
    checkCudaErrors(cudaHostAlloc(&h_count[i], num_size[i+BT_NUM] *sizeof(int), NULL));
    checkCudaErrors(cudaMalloc(&d_count[i], num_size[i+BT_NUM] *sizeof(int)));
    h_count_host[i] = (int*)malloc(num_size[i+BT_NUM] * sizeof(int));
  }

  //filter bank
  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaHostAlloc(&h_r[i], num_size[i+2*BT_NUM] *sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_r[i], num_size[i+2*BT_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_H[i], N_col*SUB_NUM*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_H[i], N_col*SUB_NUM*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_F[i], N_col*SUB_NUM*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_F[i], N_col*SUB_NUM*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_H[i], num_size[i+2*BT_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_H[i], num_size[i+2*BT_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_Dn[i], num_size[i+2*BT_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_Dn[i], num_size[i+2*BT_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_Up[i], num_size[i+2*BT_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_Up[i], num_size[i+2*BT_NUM]*sizeof(float)));
    checkCudaErrors(cudaHostAlloc(&h_Vect_F[i], num_size[i+2*BT_NUM]*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&d_Vect_F[i], num_size[i+2*BT_NUM]*sizeof(float)));
    h_Vect_F_host[i] = (float*)malloc(num_size[i+2*BT_NUM]*sizeof(float));
  }
  //DES
  for(i = 0; i < BT_NUM; i++){
      checkCudaErrors(cudaHostAlloc(&h_packet_in[i], num_size[i+3*BT_NUM]*sizeof(unsigned char), cudaHostAllocDefault));
      checkCudaErrors(cudaMalloc(&d_packet_in[i], num_size[i+3*BT_NUM]*sizeof(unsigned char)));
      checkCudaErrors(cudaHostAlloc(&h_packet_out[i], num_size[i+3*BT_NUM]*sizeof(unsigned char), cudaHostAllocDefault));
      checkCudaErrors(cudaMalloc(&d_packet_out[i], num_size[i+3*BT_NUM]*sizeof(unsigned char)));
      h_packet_host[i] =  (unsigned char *) malloc (num_size[i+3*BT_NUM]*sizeof(unsigned char));
  }
  checkCudaErrors(cudaHostAlloc(&h_des_esk, 96*sizeof(uint32), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&d_des_esk, 96*sizeof(uint32)));
  checkCudaErrors(cudaHostAlloc(&h_des_dsk, 96*sizeof(uint32), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&d_des_dsk, 96*sizeof(uint32)));

  printf("MPE CUDA static fusion inputs are generating\n");
   /*Generate encryption key*/
  des_set_key(h_des_esk, h_des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);

  //Init.matrix
  for(i = 0; i < BT_NUM; i++){
    init_matrix(h_A[i], h_B[i], h_C[i], h_D[i], num_size[i]);
  }

  //Init filter
  for(i = 0; i < BT_NUM; i++){
    init_filter(h_r[i], h_Vect_Up[i], h_Vect_F[i], 
                h_Vect_H[i], h_H[i], h_F[i], h_Vect_F_host[i], 
		num_size[i+2*BT_NUM]);
  }


  //Init DES
  for(i = 0; i < BT_NUM; i++){
    init_des(h_packet_in[i], num_size[i+3*SUB_NUM], i);
  }
#if 1  
  //mem copy
  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMemcpy(d_A[i], h_A[i], num_size[i] *sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B[i], h_B[i], num_size[i] *sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM *sizeof(int), cudaMemcpyHostToDevice));

  }

  checkCudaErrors(cudaMemcpy(d_task_indx, h_task_indx, task*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_num_thread, num_thread, task*sizeof(float), cudaMemcpyHostToDevice));


  for(i = 0; i < BT_NUM; i++){

    checkCudaErrors(cudaMemcpy(d_r[i], h_r[i], num_size[i+2*BT_NUM]*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Vect_Up[i], h_Vect_Up[i], num_size[i+2*BT_NUM]*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Vect_F[i], h_Vect_F[i], num_size[i+2*BT_NUM]*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Vect_H[i], h_Vect_H[i], num_size[i+2*BT_NUM]*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_H[i], h_H[i], N_col*SUB_NUM*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_F[i], h_F[i], N_col*SUB_NUM*sizeof(float), cudaMemcpyHostToDevice));
  }

  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMemcpy(d_packet_in[i], h_packet_in[i], num_size[i+3*BT_NUM]*sizeof(unsigned char), cudaMemcpyHostToDevice));

  }
  checkCudaErrors(cudaMemcpy(d_des_esk, h_des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_des_dsk, h_des_dsk, 96*sizeof(uint32), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  printf("MPE CUDA static fusion is running\n");
  start_timer = my_timer();
  // cpu compute
#if 1
  for(int k = 0; k < 2; k++){
  for(i = 0; i < BT_NUM; i++){
    d_fused_kernel<<<TK_NUM, TDK_NUM>>>(d_A[i], d_B[i], d_C[i], d_count[i], d_task_indx, d_r[i],
                d_H[i], d_Vect_H[i], d_Vect_Dn[i], d_Vect_Up[i], d_Vect_F[i],
                d_F[i], d_des_esk, d_des_dsk, d_packet_in[i], d_packet_out[i],
                d_num_thread, pos_task_dev[i], i);
  }
  }
  checkCudaErrors(cudaDeviceSynchronize());
#endif
  end_timer = my_timer();
  printf("Multiprogramming CUDA static fusion elapsed Time: %lf Sec.\n", end_timer - start_timer);

  // memory copy back
  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMemcpy(h_C[i],d_C[i], num_size[i]*sizeof(int), cudaMemcpyDeviceToHost));
  }

  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMemcpyAsync(h_count[i], d_count[i], num_size[i+BT_NUM]*sizeof(int), cudaMemcpyDeviceToHost));
  }
  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMemcpy(h_Vect_F[i], d_Vect_F[i], num_size[i+2*BT_NUM]*sizeof(float), cudaMemcpyDeviceToHost));
  }
  for(i = 0; i < BT_NUM; i++){
    checkCudaErrors(cudaMemcpy(h_packet_out[i], d_packet_out[i], num_size[i+3*BT_NUM]*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  //printf("cpu running\n");
  start_timer = my_timer();
  // cpu compute
#if 1
  for(int k = 0; k < 2; k++){
  for(i = 0; i < BT_NUM; i++){
    fused_kernel(h_A[i], h_B[i], h_D[i], h_count_host[i], h_task_indx, h_r[i], 
		h_H[i], h_Vect_H[i], h_Vect_Dn[i], h_Vect_Up[i], h_Vect_F_host[i],
		h_F[i], h_des_esk, h_des_dsk, h_packet_in[i], h_packet_host[i],
		num_thread, pos_task[i], i);
  }
  }
#endif
  end_timer = my_timer();
  //printf("CPU elapsed time:%lf Sec.\n", end_timer - start_timer);

#if 1
  printf("verifying\n");
  int flag = 0;
  //verificiation
  for(i = 0; i < BT_NUM; i++){
    for(j = 0; j < num_size[i]; j++){
      if(h_C[i][j] != h_D[i][j]){
        printf("Mult, Error:%d, %d, %d, %d\n", h_C[i][j], h_D[i][j], i, j);
	flag = 1;
        break;
      }
    }
  }

  for(i = 0; i < BT_NUM; i++){
    for(j = 0; j < num_size[i+2*BT_NUM]; j++){
      if(abs(h_Vect_F[i][j]- h_Vect_F_host[i][j]) > 0.1){
        printf("Filter Error:%f, %f\n", h_Vect_F[i][j], h_Vect_F_host[i][j]);
	flag = 1;
        break;
      }
    }
  }
  
  for(i = 0; i < BT_NUM; i++){
    for(j = 0; j < num_size[i+3*BT_NUM]; j++){
        if(h_packet_out[i][j] != h_packet_host[i][j]){
          printf("DES Error:%u, %u, %d, %d\n", h_packet_out[i][j], h_packet_host[i][j], i, j);
	  flag = 1;
          break;
      }
    }
  }
  if(!flag) printf("verify succesffully\n");
#endif
#endif
  //free mem.
  for(i = 0; i < BT_NUM; i++){

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
    checkCudaErrors(cudaFree(pos_task_dev[i]));


    free(h_packet_host[i]);
    free(h_count_host[i]);
    free(h_Vect_F_host[i]);
  }
  checkCudaErrors(cudaFree(d_task_indx));

  checkCudaErrors(cudaFreeHost(h_des_esk));
  checkCudaErrors(cudaFree(d_des_esk));
  checkCudaErrors(cudaFreeHost(h_des_dsk));
  checkCudaErrors(cudaFree(d_des_dsk));
  checkCudaErrors(cudaFree(d_num_thread));


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

  for(i = 0; i < N_col*SUB_NUM; i++){
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
