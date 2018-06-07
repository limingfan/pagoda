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

void explodeCPU ( float x, float y, int *value){
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

void get_pixelCPU(int *count, float index){

        int i, j;
        float x, y;
        for ( i = 0; i < n; i++ )
        {
                for ( j = 0; j < n; j++ )
                {
                        x = ( ( float ) (     j     ) * (x_max + index)
                                + ( float ) ( n - j - 1 ) * (x_min + index) )
                                / ( float ) ( n     - 1 );

                        y = ( ( float ) (     i     ) * (y_max + index)
                                + ( float ) ( n - i - 1 ) * (y_min + index) )
                                / ( float ) ( n     - 1 );

                        explodeCPU ( x, y, &count[i + j * n] );
                }
        }

}
int det_pixel(int *c_max, int *count){

  int i, j;

  *c_max = 0;
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      if ( *c_max < count[i+j*n] )
      {
        *c_max = count[i+j*n];
      }
    }
  }
}

/*
  Set the image data.
*/

void set_img(int *r, int *g, int *b, int *count, int c_max){
  int i, j;
  int c;

  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < n; j++ )
    {
      if ( count[i+j*n] % 2 == 1 )
      {
        r[i+j*n] = 255;
        g[i+j*n] = 255;
        b[i+j*n] = 255;
      }
      else
      {
        c = ( int ) ( 255.0 * sqrt ( sqrt ( sqrt (
          ( ( double ) ( count[i+j*n] ) / ( double ) ( c_max ) ) ) ) ) );
        r[i+j*n] = 3 * c / 5;
        g[i+j*n] = 3 * c / 5;
        b[i+j*n] = c;
      }
    }
  }  
}

int main(int argc, char *argv[]){
  	int i, j, k;
  	int **r, **g, **b;
  	int *c_max;
  	int **count;
	int **countCPU;
  	int **count_dev;
  	float *task_indx;
  	float *task_indx_dev;

  	double start_timer, end_timer;

	if(argc < 3){
		printf("Error input options:./mand #thread #task");
		exit(1);
	}
	int TDD_NUM = atoi(argv[1]);
	int task = atoi(argv[2]);
	printf("Pagoda Mandelbrot:image size: %d x %d, #thread:%d, #task:%d\n", n, n, TDD_NUM, task);

  	count =  (int**)malloc(task * sizeof(int *));
	countCPU =  (int**)malloc(task * sizeof(int *));
  	count_dev = (int**)malloc(task * sizeof(int *));
  	r =  (int**)malloc(task * sizeof(int *));
  	g =  (int** )malloc(task * sizeof(int *));
  	b =  (int**)malloc(task * sizeof(int *));
  	c_max = (int*)malloc(task * sizeof(int));
  	task_indx = (float*)malloc(task * sizeof(float));
  	checkCudaErrors(cudaMalloc(&task_indx_dev, task *sizeof(float)));
  	for(i = 0; i < task; i++) task_indx[i] = (float)(i/(task/2.0));

  	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

  	runtime_init();

  	for(i = 0; i < task; i++){
		countCPU[i] = ( int * ) malloc ( n * n * sizeof ( int ) );
    		checkCudaErrors(cudaHostAlloc(&count[i], n * n *sizeof(int), NULL));
    		checkCudaErrors(cudaMalloc(&count_dev[i], n * n *sizeof(int)));
    		r[i] = ( int * ) malloc ( n * n * sizeof ( int ) );
    		g[i] = ( int * ) malloc ( n * n * sizeof ( int ) );
    		b[i] = ( int * ) malloc ( n * n * sizeof ( int ) );
  	}
  
  	checkCudaErrors(cudaMemcpyAsync(task_indx_dev, task_indx, task*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

  	start_timer = my_timer(); 
  	//Carry out the iteration for each pixel, determining COUNT.
  	for(i = 0 ; i < task ; i++){
    		taskLaunch(8, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 0, INT, count_dev[i], FLOAT, &task_indx_dev[i], INT, TDD_NUM);
  	}
 	waitAll(task);
  	end_timer = my_timer();

  	printf("GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);
  
	//transfer back to host
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaMemcpyAsync(count[i], count_dev[i], n * n*sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
  	}

  	checkCudaErrors(cudaStreamSynchronize(runtime_stream));

  	runtime_destroy();
  	runtime_free();

	// cpu code
	start_timer = my_timer();
        for(i = 0; i < task; i++){
                get_pixelCPU(countCPU[i], task_indx[i]);
        }
        end_timer = my_timer();
        printf("CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);


  	//Determine the coloring of each pixel.
  	for(i = 0; i < task; i++)
    		det_pixel(&c_max[i], count[i]);

  	//Set the image data.
  	for(i = 0; i < task ; i++)
    		set_img(r[i], g[i], b[i], count[i], c_max[i]);

  	/*clean up*/
  	for(i = 0; i < task; i++){
    		checkCudaErrors(cudaFreeHost(count[i]));
    		checkCudaErrors(cudaFree(count_dev[i]));
    		free(r[i]);
    		free(g[i]);
    		free(b[i]);
		free(countCPU[i]);
  	}
  	free(c_max);
  	free(r);
  	free(g);
  	free(b);
  	free(count);
	free(countCPU);
  	free(count_dev);
  	free(task_indx);
  	checkCudaErrors(cudaFree(task_indx_dev));

  return 0;
}
