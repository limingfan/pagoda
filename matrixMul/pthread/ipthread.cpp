#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <pthread.h>

#include "../../common/para.h"

// the number of thread
#define LOOP_NUM (BT_NUM)
#define sub_task (TK_NUM)
#define task (LOOP_NUM*sub_task)
#define THREADSTACK  65536


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
	int **a, **b, **c;
	int d, e;
}               parm;



void  mult(int *A, int *B, int *C, int size, int td_num){
  int i, j, k;
  for(j = 0; j < td_num; j++)
    for(i = 0; i < (size*size/td_num); i++){
      for(k = 0; k < size; k++){
        C[((i*td_num+j)/size)*size+((i*td_num+j)%size)] += 
		A[((i*td_num+j)/size)*size+k] * B[k*size+((i*td_num+j)%size)];
      }
    }
}

void * worker(void *arg)
{
	parm           *p = (parm *) arg;
	mult(*(p->a), *(p->b), *(p->c), p->d, p->e);
}


int main(){
  	int i, j, k;
  	int *A[task], *B[task], *C[task], *D[task];
  	int num_thread[task];
  	int num_size[task];
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

  	for(i = 0; i < task; i++)
    		num_size[i] = num_thread[i]*32;

  	for(i = 0; i < task; i++){
    		A[i] = (int*)malloc(sizeof(int)*num_size[i]*num_size[i]);
    		B[i] = (int*)malloc(sizeof(int)*num_size[i]*num_size[i]);
    		C[i] = (int*)malloc(sizeof(int)*num_size[i]*num_size[i]);
    		D[i] = (int*)malloc(sizeof(int)*num_size[i]*num_size[i]);
  	}

  	// Init matrix
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]; j++){
      			A[i][j] = (i%num_size[i])+1;
      			B[i][j] = (i%num_size[i])+1;
      			C[i][j] = 0;
      			D[i][j] = 0;
    		}
  	}

  	start_timer = my_timer();
  	// task launch
  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
			arg[i].a = &A[k*sub_task+i];
			arg[i].b = &B[k*sub_task+i];
			arg[i].c = &C[k*sub_task+i];
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
  	printf("pthread Elapsed Time: %lf Sec.\n", end_timer - start_timer);

  	start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		mult(A[i], B[i], D[i], num_size[i], num_thread[i]*32);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);


  	//verification
	printf("verifying\n");
 	int flag = 0;
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i]*num_size[i]; j++){
      			if(C[i][j] != D[i][j]){
				printf("Error:%d, %d\n", C[i][j], D[i][j]);
				flag = 1;
				break;
      			}
		}
	}
	if(!flag) printf("verify successfully\n");

  	// clean memory
  	for(i = 0; i < task; i++){
    		free(A[i]);
    		free(B[i]);
    		free(C[i]);
    		free(D[i]);
  	}
  	free(arg);
  	return 0;
}
