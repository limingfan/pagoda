#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../../common/para.h"

#define TKK_NUM 19200
#define BTT_NUM (TKK_NUM/TK_NUM)
#define NUM_CHAN (TK_NUM * BTT_NUM)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void BeamFirSetup(float *weight, float *buffer, int n);
void InputGenerate(float *input, int n);

__global__ void d_BeamFirFilter(int *len,
                        float *weight, float *buffer,
                        float *in, float *out, int *size, int *thread, int index);

void BeamFirFilter(int *len,
                   float *weight, float *buffer,
                   float *in, float *out, int *size);

int main(){

  	int i, j;
  	float **h_coarse_weight, **h_coarse_buffer;
  	float **d_coarse_weight, **d_coarse_buffer;

  	float **h_inputs, **h_predec, **h_postdec;
  	float **d_inputs, **d_predec, **d_postdec;
  	float **hh_postdec;
  	int *d_len[BTT_NUM];
  	int *d_num_thread;

  	int num_thread[NUM_CHAN];
  	int num_size[BTT_NUM];
  	int pos_task[BTT_NUM][TK_NUM];
  	int *pos_task_dev[BTT_NUM];
  	int len[BTT_NUM][TK_NUM];
  	FILE *f;
  	double start_timer, end_timer;

	cudaSetDevice(0);
  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);


  	f = fopen("rand4.txt", "r");
  	for(i = 0; i < NUM_CHAN; i++)
    		fscanf(f, "%1d", &num_thread[i]);

  	fclose(f);

  	for(i = 0; i < BTT_NUM; i++){
    		num_size[i] = 0;
    //printf("num_size:%d\n", num_size[i]);
  	}

  	for(i = 0; i < BTT_NUM; i++){
    		for(j = 0; j < TK_NUM; j++){
        		num_size[i] += (num_thread[i*TK_NUM+j] * 16)*
                        	(num_thread[i*TK_NUM+j] * 16);
        		len[i][j] = (num_thread[i*TK_NUM+j] * 16)*
                        	(num_thread[i*TK_NUM+j] * 16);
        		pos_task[i][j] = 0;
        		if(j > 0) pos_task[i][j] += pos_task[i][j-1] + (num_thread[i*TK_NUM+j-1] * 16)*
                        	(num_thread[i*TK_NUM+j-1] * 16);

    	}
  	}

  	for(i = 0; i < NUM_CHAN; i++)
    		num_thread[i] *= 32;

  	d_coarse_weight = (float**)malloc(BTT_NUM * sizeof(float *));
  	d_coarse_buffer = (float**)malloc(BTT_NUM * sizeof(float *));
  	h_coarse_weight = (float**)malloc(BTT_NUM * sizeof(float *));
  	h_coarse_buffer = (float**)malloc(BTT_NUM * sizeof(float *));

  	h_inputs = (float**)malloc(BTT_NUM * sizeof(float *));
  	h_predec = (float**)malloc(BTT_NUM * sizeof(float *));
  	h_postdec = (float**)malloc(BTT_NUM * sizeof(float *));
  	d_inputs = (float**)malloc(BTT_NUM * sizeof(float *));
  	d_predec = (float**)malloc(BTT_NUM * sizeof(float *));
  	d_postdec = (float**)malloc(BTT_NUM * sizeof(float *));
  	hh_postdec = (float**)malloc(BTT_NUM * sizeof(float *));

  	for(i = 0; i < BTT_NUM; i++){
    		checkCudaErrors(cudaHostAlloc(&h_inputs[i], 2*num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&h_postdec[i], 2*num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&h_coarse_weight[i], 2*num_size[i]*sizeof(float), cudaHostAllocDefault));
    		checkCudaErrors(cudaHostAlloc(&h_coarse_buffer[i], 2*num_size[i]*sizeof(float), cudaHostAllocDefault));

    		checkCudaErrors(cudaMalloc(&d_inputs[i], 2* num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&d_predec[i], 2* num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&d_postdec[i], 2* num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&d_coarse_weight[i], 2* num_size[i]*sizeof(float)));
    		checkCudaErrors(cudaMalloc(&d_coarse_buffer[i], 2* num_size[i]*sizeof(float)));
    		h_predec[i] = (float*)malloc(2*num_size[i]*sizeof(float));
    		hh_postdec[i] = (float*)malloc(2*num_size[i]*sizeof(float));
    		checkCudaErrors(cudaMalloc(&d_len[i], TK_NUM*sizeof(int)));
    		checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM*sizeof(int)));
  	}
  	checkCudaErrors(cudaMalloc(&d_num_thread, NUM_CHAN*sizeof(int)));

	printf("Inputs are generating\n");
  	// init data
  	for(i = 0; i < BTT_NUM; i++){
    		BeamFirSetup(h_coarse_weight[i], h_coarse_buffer[i], num_size[i]);
    		InputGenerate(h_inputs[i], num_size[i]);
  	}

  	// input transfer
  	start_timer = my_timer();
  	for(i = 0; i < BTT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(d_inputs[i], h_inputs[i], 2*num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(d_coarse_weight[i], h_coarse_weight[i], 2*num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(d_coarse_buffer[i], h_coarse_buffer[i], 2*num_size[i]*sizeof(float), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(d_len[i], len[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));
    		checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));
  	}
  	checkCudaErrors(cudaMemcpy(d_num_thread, num_thread, NUM_CHAN*sizeof(int), cudaMemcpyHostToDevice));

  	checkCudaErrors(cudaDeviceSynchronize());

	printf("GPU program is running\n");
  	// task running
  	start_timer = my_timer();
  	for(i = 0; i < BTT_NUM; i++){
    		d_BeamFirFilter<<<TK_NUM, TDK_NUM>>>(d_len[i],
                        d_coarse_weight[i], d_coarse_buffer[i],
                        d_inputs[i], d_predec[i], pos_task_dev[i], d_num_thread, i);

 	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	for(i = 0; i < BTT_NUM; i++){
    		d_BeamFirFilter<<<TK_NUM, TDK_NUM>>>(d_len[i],
                        d_coarse_weight[i], d_coarse_buffer[i],
                        d_predec[i], d_postdec[i], pos_task_dev[i], d_num_thread, i);

  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  	end_timer = my_timer();
  	printf("Beamformer CUDA static fusion elapsed Time: %lf sec.\n", end_timer - start_timer);

  	// copy back
  	start_timer = my_timer();
  	for (i = 0; i < BTT_NUM; i++) {
    		checkCudaErrors(cudaMemcpyAsync(h_postdec[i], d_postdec[i], 2*num_size[i]*sizeof(float), cudaMemcpyDeviceToHost));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

#if 0
  	//host task running
  	start_timer = my_timer();
  	for(i = 0; i < BTT_NUM; i++){
    		BeamFirFilter(len[i],
                   h_coarse_weight[i], h_coarse_buffer[i],
                   h_inputs[i], h_predec[i], pos_task[i]);
  	}
  	for(i = 0; i < BTT_NUM; i++){
    		BeamFirFilter(len[i],
                   h_coarse_weight[i], h_coarse_buffer[i],
                   h_predec[i], hh_postdec[i], pos_task[i]);
  	}
  	end_timer = my_timer();
  	printf("CPU exec. time:%lf\n", end_timer - start_timer);


  	//verifiy
  	for(i = 0; i < 1; i++){
    		for(j = 0; j < num_size[i]; j++){
      			if(abs(h_postdec[i][j] - hh_postdec[i][j]) > 0.1){
        			printf("Error:%f, %f, %d, %d\n", h_postdec[i][j], hh_postdec[i][j], i, j);
        			break;
      			}
    		}
  	}

#endif
  	//free mem
  	for(i = 0; i < BTT_NUM; i++){

    		checkCudaErrors(cudaFreeHost(h_inputs[i]));
   	 	checkCudaErrors(cudaFreeHost(h_postdec[i]));
    		checkCudaErrors(cudaFreeHost(h_coarse_weight[i]));
    		checkCudaErrors(cudaFreeHost(h_coarse_buffer[i]));

    		checkCudaErrors(cudaFree(d_inputs[i]));
    		checkCudaErrors(cudaFree(d_predec[i]));
    		checkCudaErrors(cudaFree(d_postdec[i]));
    		checkCudaErrors(cudaFree(d_coarse_weight[i]));
    		checkCudaErrors(cudaFree(d_coarse_buffer[i]));
    		free(h_predec[i]);
    		free(hh_postdec[i]);
    		checkCudaErrors(cudaFree(d_len[i]));
    		checkCudaErrors(cudaFree(pos_task_dev[i]));

  	}

  	free(d_coarse_weight);
  	free(d_coarse_buffer);
  	free(h_coarse_weight);
  	free(h_coarse_buffer);

  	free(h_inputs);
  	free(h_predec);
  	free(h_postdec);
  	free(d_inputs);
  	free(d_predec);
  	free(d_postdec);
  	free(hh_postdec);
  	checkCudaErrors(cudaFree(d_num_thread));

	return 0;
}

void BeamFirSetup(float *weight, float *buffer, int n){

  int i;
  for(i = 0; i < n; i++){
    int idx = i + 1;
    //weight[i*2] = sin(idx) / ((float)idx);
    //weight[i*2+1] = cos(idx) / ((float)idx);
    weight[i*2] = 0.001;
    weight[i*2+1] = 0.002;
    buffer[i*2] = 0.0;
    buffer[i*2+1] = 0.0;
  }
}

void InputGenerate(float *input, int n){
  int i;
  for(i = 0; i < n; i++){
    //input[2*i] = sqrt(i);
    //input[2*i+1] = sqrt(i) + 1;
    input[2*i] = 0.01;
    input[2*i+1] = 0.02;
  }
}

void BeamFirFilter(int *len,
		   float *weight, float *buffer,
                   float *in, float *out, int *size)
{
  /* Input must be exactly 2*decimation_ratio long; output must be
   * exactly 2 long. */
  int i, j, t;
  int modPos;
  int mask, mask2;
  for(t = 0; t < TK_NUM; t++){
    mask = len[t] - 1;
    mask2 = 2 * len[t] - 1;
    for(j = 0; j < len[t]; j++){
      float real_curr = 0;
      float imag_curr = 0;
      modPos = 2*(len[t] - 1 - (j & mask));
      buffer[modPos + size[t]] = in[j * 2 + size[t]];
      buffer[modPos+1 + size[t]] = in[j * 2 + 1 + size[t]];

      /* Profiling says: this is the single inner loop that matters! */
      for (i = 0; i < 2*len[t]; i+=2) {

        float rd = buffer[modPos+size[t]];
        float id = buffer[modPos+1+size[t]];
        float rw = weight[i+size[t]];
        float iw = weight[i+1+size[t]];
        float rci = rd * rw + id * iw;
        /* sign error?  this is consistent with StreamIt --dzm */
        float ici = id * rw + rd * iw;
        
#if 1
        real_curr += rci;
        imag_curr += ici;
#endif
        modPos = (modPos + 2) & mask2;
      }

      out[j * 2+size[t]] = real_curr;
      out[j * 2 + 1+size[t]] = imag_curr;
    }
  }
}

__global__ void d_BeamFirFilter(int *len,
                        float *weight, float *buffer,
                        float *in, float *out, int *size, int *thread, int index)
{
  /* Input must be exactly 2*decimation_ratio long; output must be
   * exactly 2 long. */
  int tid = threadIdx.x;
  int i, j;
  int modPos;
  int mask, mask2;
  int bk = blockIdx.x;
  int td;
  td = thread[index*TK_NUM+bk];
  mask = len[bk] - 1;
  mask2 = 2 * len[bk] - 1;
  //for(k = 0; k < TD_NUM; k++){
  if(tid < td){
    for(j = 0; j < (len[bk]/td); j++){
      float real_curr = 0;
      float imag_curr = 0;
      modPos = 2*(len[bk] - 1 - ((j*td+tid) & mask));
      buffer[modPos + size[bk]] = in[(j*td+tid) * 2 + size[bk]];
      buffer[modPos+1 + size[bk]] = in[(j*td+tid)* 2 + 1 + size[bk]];

      /* Profiling says: this is the single inner loop that matters! */
      for (i = 0; i < 2*len[bk]; i+=2) {
        float rd = buffer[modPos + size[bk]];
        float id = buffer[modPos+1 + size[bk]];
        float rw = weight[i + size[bk]];
        float iw = weight[i+1 + size[bk]];
        float rci = rd * rw + id * iw;
        /* sign error?  this is consistent with StreamIt --dzm */
        float ici = id * rw + rd * iw;
        real_curr += rci;
        imag_curr += ici;
        modPos = (modPos + 2) & mask2;
      }
      //out[(j*td+tid) * 2 + size[bk]] = bk;
      //out[(j*td+tid) * 2 + 1 + size[bk]] = 1.0;

      out[(j*td+tid) * 2 + size[bk]] = real_curr;
      out[(j*td+tid) * 2 + 1 + size[bk]] = imag_curr;
    }
  }
}

