# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../../common/para.h"
#define x_max (1.25)
#define x_min (-2.25)
#define y_max (1.75)
#define y_min (-1.75)
#define count_max 400
#define task (TK_NUM * BT_NUM)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

__device__ void explode ( float x, float y, int *value){
  	int k;
  	float x1;
  	float x2;
  	float y1;
  	float y2;
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
      			break;
    		}
    		x1 = x2;
    		y1 = y2;
  	}
}

 /*
  Carry out the iteration for each pixel, determining COUNT.
*/

__global__ void get_pixel(int *count, int *size, int *threads, float *task_indx, int ii){

  	int tid = threadIdx.x;
  	int i;
  	float x, y;
  	int td;
  	float index;
  	td = threads[ii*TK_NUM+blockIdx.x];
  	index = task_indx[ii*TK_NUM+blockIdx.x];
  	if(tid < td){
    		for(i = 0; i < td; i++){
      		x = ( ( float ) (     (i*td+tid)%td     ) * (x_max + index)
          		+ ( float ) ( td - ((i*td+tid)%td) - 1 ) * (x_min + index) )
          		/ ( float ) ( td     - 1 );

      		y = ( ( float ) (     (i*td+tid)/(td*td)     ) * (y_max + index)
         	 	+ ( float ) ( td - ((i*td+tid)/td) - 1 ) * (y_min + index) )
          		/ ( float ) ( td     - 1 );

      		explode ( x, y, &count[((i*td+tid)/td) + ((i*td+tid)%td)*(td) + size[blockIdx.x]] );
    		}
  	}
}

/*
  Determine the coloring of each pixel.
*/

int det_pixel(int *c_max, int *count, int size){

  int i, j;

  *c_max = 0;
  for ( j = 0; j < size; j++ )
  {
    for ( i = 0; i < size; i++ )
    {
      if ( *c_max < count[i+j*size] )
      {
        *c_max = count[i+j*size];
      }
    }
  }
}

/*
  Set the image data.
*/

void set_img(int *r, int *g, int *b, int *count, int c_max, int size){
  int i, j;
  int c;

  for ( i = 0; i < size; i++ )
  {
    for ( j = 0; j < size; j++ )
    {
      if ( count[i+j*size] % 2 == 1 )
      {
        r[i+j*size] = 255;
        g[i+j*size] = 255;
        b[i+j*size] = 255;
      }
      else
      {
        c = ( int ) ( 255.0 * sqrt ( sqrt ( sqrt (
          ( ( double ) ( count[i+j*size] ) / ( double ) ( c_max ) ) ) ) ) );
        r[i+j*size] = 3 * c / 5;
        g[i+j*size] = 3 * c / 5;
        b[i+j*size] = c;
      }
    }
  }  
}

int main(){
  	int i, j, k;
  	int **r, **g, **b;
  	int *c_max;
  	int **count;
  	int **count_dev;
  	float *task_indx;
  	int num_thread[task];
  	int num_size[BT_NUM];
  	int pos_task[BT_NUM][TK_NUM];
  	int img_size[BT_NUM][TK_NUM];
  	int *pos_task_dev[BT_NUM];
  	int *num_thread_dev;
  	float *task_indx_dev;
	cudaSetDevice(0);
  	FILE *f;

  	double start_timer, end_timer;
  	count =  (int**)malloc(BT_NUM * sizeof(int *));
  	count_dev = (int**)malloc(BT_NUM * sizeof(int *));
  	r =  (int**)malloc(task * sizeof(int *));
  	g =  (int** )malloc(task * sizeof(int *));
  	b =  (int**)malloc(task * sizeof(int *));
  	c_max = (int*)malloc(task * sizeof(int));
  	task_indx = (float*)malloc(task * sizeof(float));
  	checkCudaErrors(cudaMalloc(&num_thread_dev, task *sizeof(int)));
  	checkCudaErrors(cudaMalloc(&task_indx_dev, task *sizeof(float)));
  	for(i = 0; i < task; i++) task_indx[i] = (float)(i/(task/2.0));

  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
  	f = fopen("rand.txt", "r");
  	for(i = 0; i < task; i++)
    		fscanf(f, "%1d", &num_thread[i]);

  	fclose(f);

  	for(i = 0; i < task; i++)
    		num_thread[i] *= 32;

  	for(i = 0; i < BT_NUM; i++){
    		num_size[i] = 0;
  	}

  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < TK_NUM; j++){
        		num_size[i] += (num_thread[i*TK_NUM+j])*
                        	(num_thread[i*TK_NUM+j]);
        		img_size[i][j] = (num_thread[i*TK_NUM+j])*
                        	(num_thread[i*TK_NUM+j]);
        		pos_task[i][j] = 0;
       	 		if(j > 0) pos_task[i][j] += pos_task[i][j-1] + (num_thread[i*TK_NUM+j-1])*
                        	(num_thread[i*TK_NUM+j-1]);

    		}
  	}
 
  
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaHostAlloc(&count[i], num_size[i] *sizeof(int), NULL));
    		checkCudaErrors(cudaMalloc(&count_dev[i], num_size[i] *sizeof(int)));
    		checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM *sizeof(int)));
  	}

  	checkCudaErrors(cudaMemcpy(num_thread_dev, num_thread, task*sizeof(int), cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaMemcpy(task_indx_dev, task_indx, task*sizeof(float), cudaMemcpyHostToDevice));
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));
  	}	
  	checkCudaErrors(cudaDeviceSynchronize());
	printf("MB CUDA static fusion is running\n");
  	start_timer = my_timer(); 
  	//Carry out the iteration for each pixel, determining COUNT.
  	for(i = 0 ; i < BT_NUM; i++){
    		get_pixel<<<TK_NUM, TDK_NUM>>>(count_dev[i], pos_task_dev[i], num_thread_dev, task_indx_dev, i);
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	end_timer = my_timer();
  	printf("Mandelbrot CUDA static fusion Elapsed Time: %lf Sec.\n", end_timer - start_timer);

  	//transfer back to host
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaMemcpy(count[i], count_dev[i], num_size[i] *sizeof(int), cudaMemcpyDeviceToHost));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

  	/*clean up*/
  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaFreeHost(count[i]));
    		checkCudaErrors(cudaFree(count_dev[i]));
    		free(r[i]);
    		free(g[i]);
    		free(b[i]);
  	}

  	free(c_max);
  	free(r);
 	free(g);
  	free(b);
  	free(count);
  	free(count_dev);
  	free(task_indx);
  	checkCudaErrors(cudaFree(num_thread_dev));
  	checkCudaErrors(cudaFree(task_indx_dev));
  	return 0;

}
