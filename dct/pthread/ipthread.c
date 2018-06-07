#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include "../../common/para.h"

// the number of thread
#define BLOCK_SIZE 8

#define LOOP_NUM (BT_NUM)
#define sub_task (TK_NUM)
#define task (LOOP_NUM*sub_task)
#define THREADSTACK  65536


#define C_norm  (0.3535533905932737) // 1 / (8^0.5)
#define C_a  	(1.387039845322148) //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b 	(1.306562964876377) //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c  	(1.175875602419359) //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d     (0.785694958387102) //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e     (0.541196100146197) //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f 	(0.275899379282943) //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.


double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void DCT(float *fSrc, float *fDst, int Stride, int SIZE, int thread);
void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut);
void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut);
void computeDCT(float *fSrc, float *fDst, int Stride, int SIZE, int thread);
void computeIDCT(float *fSrc, float *fDst, int Stride, int SIZE, int thread);

typedef struct
{
  float  **a, **b;
  int c, d, e;
} parm;

void * worker(void *arg)
{
  parm           *p = (parm *) arg;
  DCT(*(p->a), *(p->b), p->c, p->d, p->e);
}


int main(){
  	int i, j, k;
  	float *A[task], *C[task], *D[task];
  	float *A_OpenMP[task], *C_OpenMP[task];

  	int num_thread[task];
  	int num_size[task];
  	int StrideF[task];
  	FILE *fp;

  	double start_timer, end_timer;

  	parm           *arg;
  	pthread_t      *threads;
 	pthread_attr_t  attrs;


  	pthread_attr_init(&attrs);
  	pthread_setconcurrency(16);
  	pthread_attr_setstacksize(&attrs, THREADSTACK);

  	threads = (pthread_t *) malloc(task * sizeof(pthread_t));
  	arg=(parm *)malloc(sizeof(parm)*task);

  	fp = fopen("rand.txt", "r");
  	for(i = 0; i < task; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < task; i++){
		if(num_thread[i] == 1){
        		num_size[i] = 64;
        	}else{
      			num_size[i] = num_thread[i]*32;
        	}
		StrideF[i] = ((int)ceil((num_size[i]*sizeof(float))/16.0f))*16 / sizeof(float);
  	}

  	for(i = 0; i < task; i++){
    		A[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
    		A_OpenMP[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
    		C[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
    		C_OpenMP[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
    		D[i] = (float*)malloc(num_size[i]*num_size[i]*sizeof(float));
  	}

  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			A[i][j] = (i%num_size[i])+1;
      			A_OpenMP[i][j] = (i%num_size[i])+1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}

  	start_timer = my_timer();

  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = &A_OpenMP[k*sub_task+i];
        		arg[i].b = &C[k*sub_task+i];
			arg[i].c = StrideF[k*sub_task+i];
        		arg[i].d = num_size[k*sub_task+i];
        		arg[i].e = num_thread[k*sub_task+i]*32;
        		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));
    		}

    		pthread_attr_destroy(&attrs);

    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

  	}

  	end_timer = my_timer();
  	printf("The DCT pthread Elapsed Time: %lf Sec.\n", end_timer - start_timer);
  
	printf("CPU program running\n");
  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		DCT(A[i], D[i], StrideF[i], num_size[i], num_thread[i]*32);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);
  
	//Verification
	printf("verifying\n");
	int flag = 0;
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			if(C[i][j] != D[i][j]){
        			printf("Error:%f, %f, %d, %d\n", C[i][j], D[i][j], i, j);
				flag = 1;
				break;
      			}
		}
	}
  	if(!flag) printf("verify successfully\n");

  	for(i = 0; i < task; i++){
    		free(A[i]);
    		free(A_OpenMP[i]);
    		free(C[i]);
    		free(C_OpenMP[i]);
    		free(D[i]);
  	}

  	free(arg);
  	return 0;
}

void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{
    float X07P = FirstIn[0*StepIn] + FirstIn[7*StepIn];
    float X16P = FirstIn[1*StepIn] + FirstIn[6*StepIn];
    float X25P = FirstIn[2*StepIn] + FirstIn[5*StepIn];
    float X34P = FirstIn[3*StepIn] + FirstIn[4*StepIn];

    float X07M = FirstIn[0*StepIn] - FirstIn[7*StepIn];
    float X61M = FirstIn[6*StepIn] - FirstIn[1*StepIn];
    float X25M = FirstIn[2*StepIn] - FirstIn[5*StepIn];
    float X43M = FirstIn[4*StepIn] - FirstIn[3*StepIn];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    FirstOut[0*StepOut] = C_norm * (X07P34PP + X16P25PP);
    FirstOut[2*StepOut] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    FirstOut[4*StepOut] = C_norm * (X07P34PP - X16P25PP);
    FirstOut[6*StepOut] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    FirstOut[1*StepOut] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    FirstOut[3*StepOut] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    FirstOut[5*StepOut] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    FirstOut[7*StepOut] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut)
{
    float Y04P   = FirstIn[0*StepIn] + FirstIn[4*StepIn];
    float Y2b6eP = C_b * FirstIn[2*StepIn] + C_e * FirstIn[6*StepIn];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * FirstIn[7*StepIn] + C_a * FirstIn[1*StepIn] + C_c * FirstIn[3*StepIn] + C_d * FirstIn[5*StepIn];
    float Y7a1fM3d5cMP = C_a * FirstIn[7*StepIn] - C_f * FirstIn[1*StepIn] + C_d * FirstIn[3*StepIn] - C_c * FirstIn[5*StepIn];

    float Y04M   = FirstIn[0*StepIn] - FirstIn[4*StepIn];
    float Y2e6bM = C_e * FirstIn[2*StepIn] - C_b * FirstIn[6*StepIn];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * FirstIn[1*StepIn] - C_d * FirstIn[7*StepIn] - C_f * FirstIn[3*StepIn] - C_a * FirstIn[5*StepIn];
    float Y1d7cP3a5fMM = C_d * FirstIn[1*StepIn] + C_c * FirstIn[7*StepIn] - C_a * FirstIn[3*StepIn] + C_f * FirstIn[5*StepIn];

    FirstOut[0*StepOut] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    FirstOut[7*StepOut] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    FirstOut[4*StepOut] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    FirstOut[3*StepOut] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    FirstOut[1*StepOut] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    FirstOut[5*StepOut] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    FirstOut[2*StepOut] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    FirstOut[6*StepOut] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void computeDCT(float *fSrc, float *fDst, int Stride, int SIZE, int thread){
	int i, t, k;

        for(t = 0; t < thread; t++){
                for(i = 0; i < ((SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)/thread); i++)
                {
                        for (k = 0; k < BLOCK_SIZE; k++)
                        {
                                SubroutineDCTvector((float *)fSrc + (((i*thread+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1, fDst + ((((i*thread+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE)+k) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1);
                        }
                        for (k = 0; k < BLOCK_SIZE; k++)
                        {
                                SubroutineDCTvector(fDst + (((i*thread+t)/(SIZE/BLOCK_SIZE)) * BLOCK_SIZE) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k), Stride, fDst + ((((i*thread+t)/(SIZE/BLOCK_SIZE))) * BLOCK_SIZE) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE)) * BLOCK_SIZE + k), Stride);
                        }
                }
        }

}

void computeIDCT(float *fSrc, float *fDst, int Stride, int SIZE, int thread){
	 int i, t, k;

        for(t = 0; t < thread; t++){
                for(i = 0; i < ((SIZE/BLOCK_SIZE)*(SIZE/BLOCK_SIZE)/thread); i++)
                {
                        for (k = 0; k < BLOCK_SIZE; k++)
                        {
                                SubroutineIDCTvector((float *)fSrc + (((i*thread+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1, fDst + ((((i*thread+t)/(SIZE/BLOCK_SIZE))*BLOCK_SIZE)+k) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE), 1);
                        }
                        for (k = 0; k < BLOCK_SIZE; k++)
                        {
                                SubroutineIDCTvector(fDst + (((i*thread+t)/(SIZE/BLOCK_SIZE)) * BLOCK_SIZE) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE))*BLOCK_SIZE+k), Stride, fDst + ((((i*thread+t)/(SIZE/BLOCK_SIZE))) * BLOCK_SIZE) * Stride + (((i*thread+t)%(SIZE/BLOCK_SIZE)) * BLOCK_SIZE + k), Stride);
                        }
                }
        }

}

void DCT(float *fSrc, float *fDst, int Stride, int SIZE, int thread){
    computeDCT(fSrc, fDst, Stride, SIZE, thread);
    //computeIDCT(fDst, fSrc, Stride, SIZE, thread);
}
