#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include "../../common/para.h"
#define x_max (1.25)
#define x_min (-2.25)
#define y_max (1.75)
#define y_min (-1.75)
#define count_max 400
#define n 64
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
  int **a;
  float b;
  int c;
} parm;


void explode ( float x, float y, int *value);
void get_pixel_CPU(int *count, float index, int size);
void get_pixel_OpenMP(int *count, float index, int size);

void * worker(void *arg)
{
        parm           *p = (parm *) arg;
        get_pixel_OpenMP(*(p->a), (p->b), p->c);
}

int main(){
  	int i, j, k;
  	int *c_max;
  	int **count;
  	int **countOpenMP;

  	parm           *arg;
  	pthread_t      *threads;
  	pthread_attr_t  attrs;

  	int num_thread[task];
  	int num_size[task];
  	FILE *fp;

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


  	double start_timer, end_timer;
  		count =  malloc(task * sizeof(int *));
  		countOpenMP =  malloc(task * sizeof(int *));

  	for(i = 0; i < task; i++){
    		count[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
    		countOpenMP[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );

  	}
  	float index;
  	start_timer = my_timer(); 

  	//Carry out the iteration for each pixel, determining COUNT.
 
  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		index = (k*sub_task+i)/(task/2.0);
        		arg[i].a = &countOpenMP[k*sub_task+i];
        		arg[i].b = index;
        		arg[i].c = num_size[k*sub_task+i];
        		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		pthread_attr_destroy(&attrs);
    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

  	}
  end_timer = my_timer();
  printf("Mandelbrot pthread Elapsed Time: %lf Sec.\n", end_timer - start_timer);

#if 0
  	start_timer = my_timer();
  	//Carry out the iteration for each pixel, determining COUNT.

  	for(i = 0 ; i < task ; i++){
    		get_pixel_CPU(count[i], (float)(i/(task/2.0)), num_size[i]);
  	}
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//Verification
  	for(i = 0; i < task; i++){
    		for(j = 0; j < num_size[i] * num_size[i]; j++){
      			if(count[i][j] != countOpenMP[i][j]){
        			printf("Error:%d, %d, %d, %d\n", count[i][j], countOpenMP[i][j], i, j);
        			break;
      			}
    		}
  	}

#endif

  	/*clean up*/
  	for(i = 0; i < task; i++){
    		free(count[i]);
    		free(countOpenMP[i]);
  	}
  	free(count);
  	free(countOpenMP); 
  	return 0;
}

void explode ( float x, float y, int *value){
  int k;
  float x1;
  float x2;
  float y1;
  float y2;
  //int value;
  *value = 0;

  x1 = x;
  y1 = y;

  for ( k = 1; k <= count_max; k++ )
  {
    x2 = x1 * x1 - y1 * y1 + x;
    y2 = 2.0 * x1 * y1 + y;

    if ( x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2 )
    {
      *value = k;
      //if(k > 1000)
         //printf("k:%d\n", k);
      break;
    }
    x1 = x2;
    y1 = y2;
  }
}

 /*
  Carry out the iteration for each pixel, determining COUNT.
*/

void get_pixel_CPU(int *count, float index, int size){

  int i, j;
  float x, y;
  for ( i = 0; i < size; i++ )
  {
    for ( j = 0; j < size; j++ )
    {
      x = ( ( float ) (     j     ) * (x_max + index)
          + ( float ) ( size - j - 1 ) * (x_min + index) )
          / ( float ) ( size     - 1 );

      y = ( ( float ) (     i     ) * (y_max + index)
          + ( float ) ( size - i - 1 ) * (y_min + index) )
          / ( float ) ( size     - 1 );

      explode ( x, y, &count[i + j * size] );
    }
  }

}

void get_pixel_OpenMP(int *count, float index, int size){

  int i, j;
  float x, y;

  for ( i = 0; i < size; i++ )
  {
    for ( j = 0; j < size; j++ )
    {
      x = ( ( float ) (     j     ) * (x_max + index)
          + ( float ) ( size - j - 1 ) * (x_min + index) )
          / ( float ) ( size     - 1 );

      y = ( ( float ) (     i     ) * (y_max + index)
          + ( float ) ( size - i - 1 ) * (y_min + index) )
          / ( float ) ( size     - 1 );

      explode ( x, y, &count[i + j * size] );
    }
  }
}
