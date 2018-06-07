# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "runtime.cuh"
#include "mand.h"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
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
  	float *task_indx_dev;
  	int num_thread[task];
  	int num_size[task];
  	FILE *f;
	cudaSetDevice(0);
  	double start_timer, end_timer;
  	count =  (int**)malloc(task * sizeof(int *));
  	count_dev = (int**)malloc(task * sizeof(int *));
  	r =  (int**)malloc(task * sizeof(int *));
  	g =  (int** )malloc(task * sizeof(int *));
  	b =  (int**)malloc(task * sizeof(int *));
  	c_max = (int*)malloc(task * sizeof(int));
  	//task_indx = (float*)malloc(task * sizeof(float));
	checkCudaErrors(cudaHostAlloc(&task_indx, task *sizeof(float), NULL));
  	checkCudaErrors(cudaMalloc(&task_indx_dev, task *sizeof(float)));

  	f = fopen("rand.txt", "r");
  	for(i = 0; i < task; i++)
    		fscanf(f, "%1d", &num_thread[i]);

  	fclose(f);

  	for(i = 0; i < task; i++){
    		num_size[i] = num_thread[i]*32;
    		task_indx[i] = (float)(i/(task/2.0));
  	}

  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	runtime_init();

  	//start_timer = my_timer();
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaHostAlloc(&count[i], num_size[i] * num_size[i] *sizeof(int), NULL));
    		checkCudaErrors(cudaMalloc(&count_dev[i], num_size[i] * num_size[i] *sizeof(int)));
    		r[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
    		g[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
    		b[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
  	}
  
  	checkCudaErrors(cudaMemcpyAsync(task_indx_dev, task_indx, task *sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));
	printf("MB Pagoda is running\n");
  	start_timer = my_timer(); 
  	//Carry out the iteration for each pixel, determining COUNT.

  	for(i = 0 ; i < task ; i++){
    		taskLaunch(9, INT, num_thread[i]*32, INT, 1, INT, 0, INT, 0, INT, 0, 
			INT, count_dev[i], FLOAT, &task_indx_dev[i], INT, num_thread[i]*32, INT, num_size[i]);
  	}
  	waitAll(task);
  	end_timer = my_timer();
  	printf("Mandelbrot pagoda Elapsed Time: %lf Sec.\n", end_timer - start_timer);
  
	//transfer back to host
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(count[i], count_dev[i], num_size[i] * num_size[i] *sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
  	}

  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

  	runtime_destroy();
  	runtime_free();

  	//Determine the coloring of each pixel.
  	for(i = 0; i < task; i++)
    		det_pixel(&c_max[i], count[i], num_size[i]);

  	//Set the image data.
  	for(i = 0; i < task ; i++)
    		set_img(r[i], g[i], b[i], count[i], c_max[i], num_size[i]);

  	/*clean up*/
  	for(i = 0; i < task; i++){
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
  	//free(task_indx);
	checkCudaErrors(cudaFreeHost(task_indx));
  	checkCudaErrors(cudaFree(task_indx_dev));

  	return 0;
}
