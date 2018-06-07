#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

#include "../../common/para.h"

#define LOOP_NUM (BT_NUM)
#define sub_task (TK_NUM)
#define THREADSTACK  65536
#define N_ch (LOOP_NUM*sub_task)
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

void FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int size){
  int j, k, p;
  //convolving H
  for (j=0; j< size; j++)
  {
      for(k = 0; k < N_col; k++){
        if((j-k)>=0){
	  Vect_H[j] += (r[j-k]*H[k]);
        }
      }
  }

  //Down Sampling
  for (j=0; j < size/N_samp; j++)
    Vect_Dn[j]=Vect_H[j*N_samp];

  //Up Sampling
  for (j=0; j < size/N_samp;j++)
    Vect_Up[j*N_samp]=Vect_Dn[j];

  //convolving F
  for (j=0; j< size; j++)
  {
      //for (k=0; ((k<N_col) & ((j-k)>=0)); k++)
      for(k = 0; k < N_col; k++){
        if((j-k)>=0){
        //Vect_H[j]+=H[k]*r[j-k];
          Vect_F[j]+=(F[k]*Vect_Up[j-k]);
        }
      }
  }

}

void FBCore_OpenMP(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int size){
  int j, k, p;
  //convolving H
  for (j=0; j< size; j++)
  {
      for(k = 0; k < N_col; k++){
        if((j-k)>=0){
          Vect_H[j] += (r[j-k]*H[k]);
        }
      }
  }

  //Down Sampling
  for (j=0; j < size/N_samp; j++)
    Vect_Dn[j]=Vect_H[j*N_samp];

  //Up Sampling
  for (j=0; j < size/N_samp;j++)
    Vect_Up[j*N_samp]=Vect_Dn[j];

  //convolving F
  for (j=0; j< size; j++)
  {
      for(k = 0; k < N_col; k++){
        if((j-k)>=0){
          Vect_F[j]+=(F[k]*Vect_Up[j-k]);
        }
      }
  }
}

typedef struct
{
  float  **a, **b, **c, **d, **e, **f, **g;
  int h;
} parm;

void * worker(void *arg)
{
        parm           *p = (parm *) arg;
        FBCore_OpenMP(*(p->a), *(p->b), *(p->c), *(p->d), *(p->e), *(p->f), *(p->g), p->h);
}

int main(){

  	float **r;
  	float **H;
  	float **F;

  	float **Vect_H; // output of the F
  	float **Vect_Dn; // output of the down sampler
  	float **Vect_Up; // output of the up sampler
  	float **Vect_F; // this is the output of the
  	float **Vect_F_OpenMP;

  	int num_thread[N_ch];
  	int num_size[N_ch];

  	FILE *f;


  	int i, j, k;

  	double start_timer, end_timer;

  	parm           *arg;
  	pthread_t      *threads;
  	pthread_attr_t  attrs;


  	pthread_attr_init(&attrs);
  	pthread_setconcurrency(16);
  	pthread_attr_setstacksize(&attrs, THREADSTACK);

  	threads = (pthread_t *) malloc(N_ch * sizeof(pthread_t));
  	arg=(parm *)malloc(sizeof(parm)*N_ch);

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
  	Vect_F_OpenMP = (float**)malloc(N_ch*sizeof(float*));

  	/*Memory allocation*/
  	for(i = 0; i < N_ch; i++){
    		r[i] = (float*)malloc(num_size[i]*sizeof(float));
    		H[i] = (float*)malloc(N_col*sizeof(float));
    		F[i] = (float*)malloc(N_col*sizeof(float));

    		Vect_H[i] = (float*)malloc(num_size[i]*sizeof(float));
    		Vect_Dn[i] = (float*)malloc((int)(num_size[i]/N_samp)*sizeof(float));
    		Vect_Up[i] = (float*)malloc(num_size[i]*sizeof(float));
    		Vect_F[i] = (float*)malloc(num_size[i]*sizeof(float));
    		Vect_F_OpenMP[i] = (float*)malloc(num_size[i]*sizeof(float));

  	}

  	// Init data for OpenMP
  	for(i = 0; i < N_ch; i++)
    		for(j = 0; j < num_size[i]; j++){
      			r[i][j] = j + 0.001;
      			Vect_Up[i][j] = 0;
      			Vect_F_OpenMP[i][j] = 0;
      			Vect_H[i][j]=0;
    		}

  	for(i = 0; i < N_ch; i++)
    		for(j = 0; j < N_col; j++){
      			H[i][j] = 0.001;
      			F[i][j] = 0.001;
    		}

  	start_timer = my_timer();

  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = &r[k*sub_task+i];
        		arg[i].b = &H[k*sub_task+i];
        		arg[i].c = &Vect_H[k*sub_task+i];
			arg[i].d = &Vect_Dn[k*sub_task+i];
        		arg[i].e = &Vect_Up[k*sub_task+i];
        		arg[i].f = &Vect_F_OpenMP[k*sub_task+i];
  			arg[i].g = &F[k*sub_task+i];
			arg[i].h = num_size[k*sub_task+i];
        		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		pthread_attr_destroy(&attrs);
    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

  	}

  	end_timer = my_timer();
  	printf("Filterbank pthread Elapsed Time: %f Sec.\n", end_timer - start_timer);

	/*init data*/
        for(i = 0; i < N_ch; i++)
                for(j = 0; j < num_size[i]; j++){
                        r[i][j] = j + 0.001;
                        Vect_Up[i][j] = 0;
                        Vect_F[i][j] = 0;
                        Vect_H[i][j]=0;
                }

        for(i = 0; i < N_ch; i++)
                for(j = 0; j < N_col; j++){
                        H[i][j] = 0.001;
                        F[i][j] = 0.001;
                }
  
	printf("CPU program running\n");
    	start_timer = my_timer();
  	for(i = 0; i < N_ch; i++){
    		FBCore(r[i], H[i], Vect_H[i], Vect_Dn[i], Vect_Up[i], Vect_F[i], F[i], num_size[i]);
  	}
  	end_timer = my_timer();
  	//printf("The CPU Elapsed time:%f Sec.\n", end_timer - start_timer);


	printf("verifying\n");
	int flag = 0;
  	for(i = 0; i < N_ch; i++){
    		for(j = 0; j < num_size[i]; j++){
      			if(Vect_F[i][j] != Vect_F_OpenMP[i][j]){
        			printf("Error:%f, %f, %d, %d\n", Vect_F[i][j], Vect_F_OpenMP[i][j], i, j);
				flag = 1;
        			break;
      			}
    		}
  	}
	if(!flag) printf("verify successfully\n");

  	/*Free Memory*/
  	for(i = 0; i < N_ch; i++){ 
    		free(r[i]);
    		free(H[i]);
    		free(F[i]);

    		free(Vect_H[i]);
    		free(Vect_Dn[i]);
    		free(Vect_Up[i]);
    		free(Vect_F[i]);
    		free(Vect_F_OpenMP[i]);
  	}

  	free(r);
  	free(H);
  	free(Vect_H);
  	free(Vect_Dn);
  	free(Vect_Up);
  	free(Vect_F);
  	free(Vect_F_OpenMP);

  	free(arg);
  	return 0;
}
