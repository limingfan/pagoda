# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <sys/time.h>

#define x_max (1.25)
#define x_min (-2.25)
#define y_max (1.75)
#define y_min (-1.75)
#define count_max 400
#define n 64
#define task 32768

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
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

void get_pixel(int *count, float index, int size){

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

  int num_thread[task];
  int num_size[task];
  FILE *f;

  f = fopen("rand.txt", "r");
  for(i = 0; i < task; i++)
    fscanf(f, "%1d", &num_thread[i]);

  fclose(f);

  for(i = 0; i < task; i++)
    num_size[i] = num_thread[i]*32;


  double start_timer, end_timer;
  count =  (int**)malloc(task * sizeof(int *));
  r =  (int**)malloc(task * sizeof(int *));
  g =  (int** )malloc(task * sizeof(int *));
  b =  (int**)malloc(task * sizeof(int *));
  c_max = (int*)malloc(task * sizeof(int));

  for(i = 0; i < task; i++){
    count[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
    r[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
    g[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
    b[i] = ( int * ) malloc ( num_size[i] * num_size[i] * sizeof ( int ) );
  }
  start_timer = my_timer(); 
  //Carry out the iteration for each pixel, determining COUNT.
  for(i = 0 ; i < task ; i++)
    get_pixel(count[i], (float)(i/(task/2.0)), num_size[i]);

  end_timer = my_timer();
  printf("Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  //Determine the coloring of each pixel.
  for(i = 0; i < task; i++)
    det_pixel(&c_max[i], count[i], num_size[i]);

  //Set the image data.
  for(i = 0; i < task ; i++)
    set_img(r[i], g[i], b[i], count[i], c_max[i], num_size[i]);


#if 0
  // output results
  FILE *fp;
  fp = fopen("output1.txt", "w+");
  for(i = 0; i < task; i++)
    for(j = 0; j < n * n; j++)
        fprintf(fp, "%d, %d, %d\n", r[i][j], g[i][j], b[i][j]);
  fclose(fp);
#endif
  /*clean up*/
  for(i = 0; i < task; i++){
    free(count[i]);
    free(r[i]);
    free(g[i]);
    free(b[i]);
  }
  free(c_max);
  free(r);
  free(g);
  free(b);

  return 0;
}
