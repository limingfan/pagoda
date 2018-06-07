#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#include "../../common/para.h"

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define imageW          128
#define imageH          128

#define LOOP_NUM (BT_NUM)
#define sub_task (TK_NUM)
#define THREADSTACK  65536
#define NUM_TASK        (LOOP_NUM*sub_task)


double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

typedef struct
{
  float **a, **b, **c;
  int d, e, f;
} parm;

void convolutionRowsCPU(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size);
void convolutionColumnsCPU(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int size);
void convolutionRowsOpenMP(float *h_Dst, float *h_Src, float *h_Kernel, int kernelR, int size);
void convolutionColumnsOpenMP(float *h_Dst, float *h_Src, float *h_Kernel,int kernelR, int size);

void * worker(void *arg)
{
  parm           *p = (parm *) arg;
  convolutionRowsOpenMP(*(p->a), *(p->b), *(p->c), p->d, p->e);
}

void * worker1(void *arg)
{
  parm           *p = (parm *) arg;
  convolutionColumnsOpenMP(*(p->a), *(p->b), *(p->c), p->d, p->e);
}


int main(){

  	float **h_Kernel, **h_Input, **h_Buffer, **h_OutputCPU;
  	float **h_OutputOpenMP, **h_BufferOpenMP;

  	int *num_thread;
  	int *num_size;
  	FILE *fp;

  	int i, j, k;
  	double start_timer, end_timer;

  	parm           *arg;
  	pthread_t      *threads;
  	pthread_attr_t  attrs;


  	pthread_attr_init(&attrs);
  	pthread_setconcurrency(16);
  	pthread_attr_setstacksize(&attrs, THREADSTACK);

  	threads = (pthread_t *) malloc(NUM_TASK * sizeof(pthread_t));
  	arg=(parm *)malloc(sizeof(parm)*NUM_TASK);

  	num_thread = (int*)malloc(NUM_TASK * sizeof(int));
  	num_size = (int*)malloc(NUM_TASK * sizeof(int));

  	fp = fopen("rand.txt", "r");
  	for(i = 0; i < NUM_TASK; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < NUM_TASK; i++)
    		num_size[i] = num_thread[i]*32;


  	//printf("Initializing data...\n");

  	h_Kernel    = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_Input     = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_Buffer    = (float **)malloc(NUM_TASK * sizeof(float*));
 	h_OutputCPU = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_OutputOpenMP = (float **)malloc(NUM_TASK * sizeof(float*));
  	h_BufferOpenMP = (float **)malloc(NUM_TASK * sizeof(float*));

  	for(i = 0; i < NUM_TASK; i++){
    		h_Kernel[i]    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    		h_Input[i]     = (float *)malloc(num_size[i]*num_size[i] * sizeof(float));
    		h_Buffer[i]    = (float *)malloc(num_size[i]*num_size[i] * sizeof(float));
    		h_OutputCPU[i] = (float *)malloc(num_size[i]*num_size[i] * sizeof(float));
    		h_OutputOpenMP[i] = (float *)malloc(num_size[i]*num_size[i] * sizeof(float));
    		h_BufferOpenMP[i] = (float *)malloc(num_size[i]*num_size[i] * sizeof(float));
  	}

  	for(i = 0; i < NUM_TASK;i++){
    		for (j = 0; j < KERNEL_LENGTH; j++){
      			h_Kernel[i][j] = (float)j/KERNEL_LENGTH;
    		}
  	}

  	for(i = 0; i < NUM_TASK;i++){
    		for (j = 0; j < num_size[i] * num_size[i]; j++){
      			h_Input[i][j] = (float)((j/imageW)%2);
    		}
  	}

  	start_timer = my_timer();

  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = &h_BufferOpenMP[k*sub_task+i];
        		arg[i].b = &h_Input[k*sub_task+i];
        		arg[i].c = &h_Kernel[k*sub_task+i];
        		arg[i].d = KERNEL_RADIUS;
			arg[i].e = num_size[k*sub_task+i];
       	 		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

    		for(i = 0; i < sub_task; i++){
        		arg[i].a = &h_OutputOpenMP[k*sub_task+i];
        		arg[i].b = &h_BufferOpenMP[k*sub_task+i];
        		arg[i].c = &h_Kernel[k*sub_task+i];
        		arg[i].d = KERNEL_RADIUS;
        		arg[i].e = num_size[k*sub_task+i];
        		pthread_create(&threads[i], &attrs, worker1, (void *)(arg+i));

    		}
    		pthread_attr_destroy(&attrs);

    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}


  	}
  	end_timer = my_timer();
  	printf("Convolution pthread elapsed Time: %lf Sec.\n", end_timer - start_timer);

  	printf("CPU program running\n");
  	start_timer = my_timer();
  	for(i = 0; i < NUM_TASK; i++){
    		convolutionRowsCPU(h_Buffer[i], h_Input[i], h_Kernel[i], KERNEL_RADIUS, num_size[i]);
  	}

  	for(i = 0; i < NUM_TASK; i++){
    		convolutionColumnsCPU(h_OutputCPU[i], h_Buffer[i], h_Kernel[i],KERNEL_RADIUS, num_size[i]);
  	}
  	end_timer = my_timer();
  	//printf("CPU elapsed time:%lf Sec.\n", end_timer - start_timer);


	printf("verifying\n");
	int flag = 0;
  	for(i = 0; i < NUM_TASK; i++){
    		for(j = 0; j < num_size[i] * num_size[i]; j++){
      			if(h_OutputCPU[i][j] != h_OutputOpenMP[i][j]){
        			printf("Error:%f, %f, %d, %d\n", h_OutputCPU[i][j], h_OutputOpenMP[i][j], i, j);
				flag = 1;
        			break;
      			}
    		}
  	}
	if(!flag) printf("verify successfully\n");

  	//free mem.

  	for(i = 0; i < NUM_TASK; i++){
    		free(h_Buffer[i]);
    		free(h_Input[i]);
    		free(h_Kernel[i]);
    		free(h_OutputCPU[i]);
    		free(h_OutputOpenMP[i]);
    		free(h_BufferOpenMP[i]);
  	}

  	free(h_Buffer);
  	free(h_Input);
  	free(h_Kernel);
  	free(h_OutputCPU);
  	free(h_OutputOpenMP);
  	free(h_BufferOpenMP);

  	free(arg);
  	return 0;
  }

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowsOpenMP(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR, 
    int size
)
{
	int x, y, k;
	for (y = 0; y < size; y++)
        	for (x = 0; x < size; x++)
        	{
            		float sum = 0;

            		for (k = -kernelR; k <= kernelR; k++)
            		{
                		int d = x + k;

                		if (d >= 0 && d < size)
                    			sum += h_Src[y * size + d] * h_Kernel[kernelR - k];
            		}

            		h_Dst[y * size + x] = sum;
        	}	

}

void convolutionRowsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int size
)
{
	int x, y, k;

	for (y = 0; y < size; y++)
                for (x = 0; x < size; x++)
                {
                        float sum = 0;

                        for (k = -kernelR; k <= kernelR; k++)
                        {
                                int d = x + k;

                                if (d >= 0 && d < size)
                                        sum += h_Src[y * size + d] * h_Kernel[kernelR - k];
                        }

                        h_Dst[y * size + x] = sum;
                }

}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnsOpenMP(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR, 
    int size
)
{
	int x, y, k;

	for (y = 0; y < size; y++)
        	for (x = 0; x < size; x++)
        	{
            		float sum = 0;

            		for (k = -kernelR; k <= kernelR; k++)
            		{
                		int d = y + k;

                		if (d >= 0 && d < size)
                    			sum += h_Src[d * size + x] * h_Kernel[kernelR - k];
            		}

            		h_Dst[y * size + x] = sum;
        	}

}

void convolutionColumnsCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int kernelR,
    int size
)
{
	int x, y, k;
	for (y = 0; y < size; y++)
                for (x = 0; x < size; x++)
                {
                        float sum = 0;

                        for (k = -kernelR; k <= kernelR; k++)
                        {
                                int d = y + k;

                                if (d >= 0 && d < size)
                                        sum += h_Src[d * size + x] * h_Kernel[kernelR - k];
                        }

                        h_Dst[y * size + x] = sum;
                }

}
