#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "kernel.h"
#include "packet.h"
#include "headers.h"

#define TK_NUM 2048 //num. of task in each category
#define LOOP_NUM 4
#define task (LOOP_NUM*TK_NUM)
#define THREADSTACK  65536
#define LEN2 (LEN/8)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void mult(int *A, int *B, int *C, int size, int thread);
void mult_omp(int *A, int *B, int *C, int size, int thread);
void h_explode ( float x, float y, int *value);
void h_get_pixel(int *count, float index, int size);
void get_pixel_omp(int *count, float index, int size);
void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int size);
void FBCore_omp(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int size);
int des_main_ks( uint32 *SK, uint8 *key );
int des_set_key( uint32 *esk, uint32 *dsk, uint8 key1[8],
                                uint8 key2[8], uint8 key3[8]);
void des_encrypt( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int len);
void des_crypt_omp( uint32 *SK, uint8 *input, uint8 *output, int len);

void init_matrix(int *A, int *B, int *C, int *D, int size);
void init_filter(float *r, float *Vect_Up, float *Vect_F,
                float *Vect_H, float *H, float *F, float *Vect_H_host, int size);
void init_des(unsigned char *packet_in, unsigned char *o_packet_in, int size);

typedef struct
{
  int **a, **b, **c;
  int d, e;
} parm;

typedef struct
{
  int **a;
  int b;
  int c;
} parm1;

typedef struct
{
  float **a, **b, **c, **d, **e, **f, **g;
  int h;
} parm2;

typedef struct
{
  uint32 **a;
  unsigned char **b, **c;
  int d;
} parm3;

void * worker(void *arg)
{
  parm	*p = (parm *) arg;
  mult_omp(*(p->a), *(p->b), *(p->c), p->d, p->e);
}

void * worker1(void *arg)
{
  parm1  *p = (parm1 *) arg;
  get_pixel_omp(*(p->a), p->b, p->c);
}

void * worker2(void *arg)
{
  parm2  *p = (parm2 *) arg;
  FBCore_omp(*(p->a), *(p->b), *(p->c), *(p->d), *(p->e), *(p->f), *(p->g), p->h);
}

void * worker3(void *arg)
{
  parm3  *p = (parm3 *) arg;
  des_crypt_omp(*(p->a), *(p->b), *(p->c), p->d);
}

int main(){

  int i, j;
  int *h_A[TK_NUM], *h_B[TK_NUM], *h_C[TK_NUM], *h_D[TK_NUM];

  int *h_count[TK_NUM];
  int *o_count_host[TK_NUM];
  float *h_task_indx;

  float *h_r[TK_NUM];
  float *y, *h_H[TK_NUM];
  float *h_F[TK_NUM];

  float *h_Vect_H[TK_NUM]; // output of the F
  float *h_Vect_Dn[TK_NUM]; // output of the down sampler
  float *h_Vect_Up[TK_NUM]; // output of the up sampler
  float *h_Vect_F[TK_NUM], *o_Vect_F_host[TK_NUM]; // this is the output of the

  unsigned char *h_packet_in[TK_NUM];
  unsigned char *h_packet_out[TK_NUM];
  unsigned char *o_packet_in[TK_NUM];
  unsigned char *o_packet_host[TK_NUM];

  uint32 *h_des_esk;
  uint32 *h_des_dsk;

  int num_thread[task];
  int num_size[task];
  FILE *fp;

  fp = fopen("rand.txt", "r");
  for(i = 0; i < task; i++)
    fscanf(fp, "%1d", &num_thread[i]);

  fclose(fp);

  for(i = 0; i < task; i++)
    num_size[i] = num_thread[i]*32;


  int mult_c, mand_c, filter_c, des_c;
  double start_timer, end_timer;

  uint32 *encrypt[TK_NUM];

  parm           *arg;
  parm1          *arg1;
  parm2		 *arg2;
  parm3		 *arg3;

  pthread_t      *threads;
  pthread_attr_t  attrs;


  pthread_attr_init(&attrs);
  pthread_setconcurrency(16);
  pthread_attr_setstacksize(&attrs, THREADSTACK);

  threads = (pthread_t *) malloc(TK_NUM * sizeof(pthread_t));
  arg=(parm *)malloc(sizeof(parm)*TK_NUM);
  arg1=(parm1 *)malloc(sizeof(parm1)*TK_NUM);
  arg2=(parm2 *)malloc(sizeof(parm2)*TK_NUM);
  arg3=(parm3 *)malloc(sizeof(parm3)*TK_NUM);
  

  //matrix mult.
  for(i = 0; i < TK_NUM; i++){
    h_A[i] = (int*)malloc(num_size[i]*num_size[i]*sizeof(int));
    h_B[i] = (int*)malloc(num_size[i]*num_size[i]*sizeof(int));
    h_C[i] = (int*)malloc(num_size[i]*num_size[i]*sizeof(int));
    h_D[i] = (int*)malloc(num_size[i]*num_size[i]*sizeof(int));

  }
 
  // mandelbrot
  h_task_indx = (float*)malloc(TK_NUM * sizeof(float));
  for(i = 0; i < TK_NUM; i++) {
    h_task_indx[i] = (float)(i/(TK_NUM/2.0));
    h_count[i] = (int*)malloc(num_size[i+TK_NUM] * num_size[i+TK_NUM] * sizeof(int));
    o_count_host[i] = (int*)malloc(num_size[i+TK_NUM] * num_size[i+TK_NUM] * sizeof(int));

  }

    //filter bank
  for(i = 0; i < TK_NUM; i++){

    h_r[i] = (float*)malloc(num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float));
    h_H[i] = (float*)malloc(N_col*sizeof(float));
    h_F[i] = (float*)malloc(N_col*sizeof(float));

    h_Vect_H[i] = (float*)malloc(num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float));
    h_Vect_Dn[i] = (float*)malloc((num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]/N_samp)*sizeof(float));
    h_Vect_Up[i] = (float*)malloc(num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float));
    h_Vect_F[i] = (float*)malloc(num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float));
    o_Vect_F_host[i] = (float*)malloc(num_size[i+2*TK_NUM] * num_size[i+2*TK_NUM]*sizeof(float));
  }

  //DES
  for(i = 0; i < TK_NUM; i++){
      h_packet_in[i] = (unsigned char*)malloc(num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char));
      h_packet_out[i] = (unsigned char*)malloc(num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char));
      o_packet_host[i] =  (unsigned char *) malloc (num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char));
      o_packet_in[i] =  (unsigned char *) malloc (num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]*sizeof(unsigned char));
      encrypt[i] =  (uint32 *) malloc (96*sizeof(uint32));

  }
  h_des_esk = (uint32*)malloc(96*sizeof(uint32));
  h_des_dsk = (uint32*)malloc(96*sizeof(uint32));

   /*Generate encryption key*/
  des_set_key(h_des_esk, h_des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);

  for(i = 0; i < TK_NUM; i++){
    for(j = 0; j < 96; j++)
      encrypt[i][j] = h_des_esk[j];
  }
 
  //Init.matrix
  for(i = 0; i < TK_NUM; i++)
    init_matrix(h_A[i], h_B[i], h_C[i], h_D[i], num_size[i]);

  //Init filter
  for(i = 0; i < TK_NUM; i++)
    init_filter(h_r[i], h_Vect_Up[i], h_Vect_F[i],
                h_Vect_H[i], h_H[i], h_F[i], o_Vect_F_host[i], num_size[i+2*TK_NUM]);
  //Init DES
  for(i = 0; i < TK_NUM; i++)
    init_des(h_packet_in[i], o_packet_in[i], num_size[i+3*TK_NUM]*num_size[i+3*TK_NUM]);

  mult_c = 0, mand_c = 0, filter_c = 0, des_c = 0;
#if 1
  start_timer = my_timer();
  // cpu compute
  for(i = 0; i < task; i++){
    switch(i%4){
      case 0:
        mult(h_A[mult_c], h_B[mult_c], h_C[mult_c], num_size[mult_c], num_thread[mult_c]*32);
        mult_c ++;
        break;
      case 1:
        h_get_pixel(h_count[mand_c], h_task_indx[mand_c], num_size[mand_c+TK_NUM]);
        mand_c++;
        break;
      case 2:
        h_FBCore(h_r[filter_c], h_H[filter_c], h_Vect_H[filter_c],
                        h_Vect_Dn[filter_c], h_Vect_Up[filter_c], h_Vect_F[filter_c], 
			h_F[filter_c], num_size[filter_c+2*TK_NUM] * num_size[filter_c+2*TK_NUM]);
        filter_c ++;
        break;
      case 3:
        des_encrypt(h_des_esk, h_des_dsk, h_packet_in[des_c], h_packet_out[des_c], (num_size[des_c+3*TK_NUM] * num_size[des_c+3*TK_NUM])/8);
        des_c ++;
        break;
    }
  }
  end_timer = my_timer();
  //printf("CPU elapsed time:%lf Sec.\n", end_timer - start_timer);

  mult_c = 0, mand_c = 0, filter_c = 0, des_c = 0;

  start_timer = my_timer();
  // cpu compute

  for(i = 0; i < TK_NUM; i++){
        arg[i].a = &h_A[i];
        arg[i].b = &h_B[i];
        arg[i].c = &h_D[i];
        arg[i].d = num_size[i];
        arg[i].e = num_thread[i]*32;
        pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));
    }

  for (i = 0; i < TK_NUM; i++){
      pthread_join(threads[i], NULL);
  }

  for(i = 0; i < TK_NUM; i++){
        arg1[i].a = &o_count_host[i];
        arg1[i].b = h_task_indx[i];
	arg1[i].c = num_size[i+TK_NUM];
        pthread_create(&threads[i], &attrs, worker1, (void *)(arg1+i));
    }

  for (i = 0; i < TK_NUM; i++){
      pthread_join(threads[i], NULL);
  }

  for(i = 0; i < TK_NUM; i++){
        arg2[i].a = &h_r[i];
        arg2[i].b = &h_H[i];
        arg2[i].c = &h_Vect_H[i];
        arg2[i].d = &h_Vect_Dn[i];
	arg2[i].e = &h_Vect_Up[i];
        arg2[i].f = &o_Vect_F_host[i];
        arg2[i].g = &h_F[i];
	arg2[i].h = num_size[i+2*TK_NUM];

        pthread_create(&threads[i], &attrs, worker2, (void *)(arg2+i));
  }

  for (i = 0; i < TK_NUM; i++){
      pthread_join(threads[i], NULL);
  }

  for(i = 0; i < TK_NUM; i++){
        arg3[i].a = &encrypt[i];
        arg3[i].b = &o_packet_in[i];
        arg3[i].c = &o_packet_host[i];
        arg3[i].d = (num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM])/8;
        pthread_create(&threads[i], &attrs, worker3, (void *)(arg3+i));
    }

  for (i = 0; i < TK_NUM; i++){
      pthread_join(threads[i], NULL);
  }

#if 0
  for(i = 0; i < TK_NUM; i++){
    mult_omp(h_A[i], h_B[i], h_D[i], MROW);
       
  }

  for(i = 0; i < TK_NUM; i++){
    get_pixel_omp(o_count_host[i], h_task_indx[i]);

  }

  for(i = 0; i < TK_NUM; i++){
    FBCore_omp(h_r[i], h_H[i], h_Vect_H[i],
                        h_Vect_Dn[i], h_Vect_Up[i], o_Vect_F_host[i], h_F[i]);
  }

  for(i = 0; i < TK_NUM; i++){
    des_crypt_omp(h_des_esk, o_packet_in[i], o_packet_host[i], LEN2);
  }
#endif

  end_timer = my_timer();
  printf("Multiprogramming pthread elapsed Time: %lf Sec.\n", end_timer - start_timer);


  //verificiation
  printf("verifying\n");
  int flag = 0;
  for(i = 0; i < TK_NUM; i++){
    for(j = 0; j < num_size[i]*num_size[i]; j++){
      if(h_C[i][j] != h_D[i][j]){
        printf("Mult, Error:%d, %d\n", h_C[i][j], h_D[i][j]);
	flag = 1;
        break;
      }
    }

    for(j = 0; j < num_size[i+2*TK_NUM]*num_size[i+2*TK_NUM]; j++){
      if(abs(h_Vect_F[i][j]- o_Vect_F_host[i][j]) > 0.1){
        printf("Filter Error:%f, %f\n", h_Vect_F[i][j], o_Vect_F_host[i][j]);
	flag = 1;
        break;
      }
    }
    for(j = 0; j < (num_size[i+3*TK_NUM] * num_size[i+3*TK_NUM]); j++){
        if(h_packet_out[i][j] != o_packet_host[i][j]){
          printf("DES Error:%u, %u, %d, %d\n", h_packet_out[i][j], o_packet_host[i][j], i, j);
	  flag = 1;
          break;
      }
    }
  }
  if(!flag) printf("verify successfully\n");


  // mem free

  for(i = 0; i < TK_NUM; i++){
    free(h_A[i]);
    free(h_B[i]);
    free(h_C[i]);
    free(h_D[i]);

    free(h_count[i]);
    free(o_count_host[i]);

    free(h_Vect_H[i]);
    free(h_Vect_Dn[i]);
    free(h_Vect_Up[i]);
    free(h_Vect_F[i]);
    free(o_Vect_F_host[i]);
    
    free(h_packet_in[i]);
    free(h_packet_out[i]);
    free(o_packet_host[i]);
    free(o_packet_in[i]);
  }
#endif
  free(h_task_indx);
  free(h_des_esk);
  free(h_des_dsk);
  free(y);
  free(arg);
  free(arg1);
  free(arg2);
  free(arg3);
  return 0;
}


void mult_omp(int *A, int *B, int *C, int size, int thread){
  int i, j, k;
  for(j = 0; j < thread; j++)
    for(i = 0; i < (size*size/thread); i++){
      for(k = 0; k < size; k++){
        C[((i*thread+j)/size)*size+((i*thread+j)%size)] +=
                A[((i*thread+j)/size)*size+k] * B[k*size+((i*thread+j)%size)];
      }
    }
}

void mult(int *A, int *B, int *C, int size, int thread){
  int i, j, k;
  for(j = 0; j < thread; j++)
    for(i = 0; i < (size*size/thread); i++){
      for(k = 0; k < size; k++){
        C[((i*thread+j)/size)*size+((i*thread+j)%size)] +=
                A[((i*thread+j)/size)*size+k] * B[k*size+((i*thread+j)%size)];
      }
    }

}
void h_explode ( float x, float y, int *value){
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

void h_get_pixel(int *count, float index, int size){

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

      h_explode ( x, y, &count[i + j * size] );
    }
  }

}

void get_pixel_omp(int *count, float index, int size){

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

      h_explode ( x, y, &count[i + j * size] );
    }
  }
}

void h_FBCore(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int size){
  int j, k, p;
  //convolving H
  for (j=0; j< size; j++)
  {
      //for (k=0; ((k<N_col) & ((j-k)>=0)); k++)
      for(k = 0; k < N_col; k++){
        if((j-k)>=0){
        //Vect_H[j]+=H[k]*r[j-k];
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

void FBCore_omp(float *r, float *H, float *Vect_H, float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F, int size){
  int j, k, p;
  //convolving H
  for (j=0; j< size; j++)
  {
      //for (k=0; ((k<N_col) & ((j-k)>=0)); k++)
      for(k = 0; k < N_col; k++){
        if((j-k)>=0){
        //Vect_H[j]+=H[k]*r[j-k];
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

/* DES key schedule */

int des_main_ks( uint32 *SK, uint8 *key )
{
    int i;
    uint32 X, Y, T;

    GET_UINT32( X, key, 0 );
    GET_UINT32( Y, key, 4 );

    /* Permuted Choice 1 */

    T =  ((Y >>  4) ^ X) & 0x0F0F0F0F;  X ^= T; Y ^= (T <<  4);
    T =  ((Y      ) ^ X) & 0x10101010;  X ^= T; Y ^= (T      );

    X =   (LHs[ (X      ) & 0xF] << 3) | (LHs[ (X >>  8) & 0xF ] << 2)
        | (LHs[ (X >> 16) & 0xF] << 1) | (LHs[ (X >> 24) & 0xF ]     )
        | (LHs[ (X >>  5) & 0xF] << 7) | (LHs[ (X >> 13) & 0xF ] << 6)
        | (LHs[ (X >> 21) & 0xF] << 5) | (LHs[ (X >> 29) & 0xF ] << 4);

    Y =   (RHs[ (Y >>  1) & 0xF] << 3) | (RHs[ (Y >>  9) & 0xF ] << 2)
        | (RHs[ (Y >> 17) & 0xF] << 1) | (RHs[ (Y >> 25) & 0xF ]     )
        | (RHs[ (Y >>  4) & 0xF] << 7) | (RHs[ (Y >> 12) & 0xF ] << 6)
        | (RHs[ (Y >> 20) & 0xF] << 5) | (RHs[ (Y >> 28) & 0xF ] << 4);

    X &= 0x0FFFFFFF;
    Y &= 0x0FFFFFFF;

    /* calculate subkeys */

    for( i = 0; i < 16; i++ )
    {
        if( i < 2 || i == 8 || i == 15 )
        {
            X = ((X <<  1) | (X >> 27)) & 0x0FFFFFFF;
            Y = ((Y <<  1) | (Y >> 27)) & 0x0FFFFFFF;
        }
        else
        {
            X = ((X <<  2) | (X >> 26)) & 0x0FFFFFFF;
            Y = ((Y <<  2) | (Y >> 26)) & 0x0FFFFFFF;
        }
	*SK++ =   ((X <<  4) & 0x24000000) | ((X << 28) & 0x10000000)
                | ((X << 14) & 0x08000000) | ((X << 18) & 0x02080000)
                | ((X <<  6) & 0x01000000) | ((X <<  9) & 0x00200000)
                | ((X >>  1) & 0x00100000) | ((X << 10) & 0x00040000)
                | ((X <<  2) & 0x00020000) | ((X >> 10) & 0x00010000)
                | ((Y >> 13) & 0x00002000) | ((Y >>  4) & 0x00001000)
                | ((Y <<  6) & 0x00000800) | ((Y >>  1) & 0x00000400)
                | ((Y >> 14) & 0x00000200) | ((Y      ) & 0x00000100)
                | ((Y >>  5) & 0x00000020) | ((Y >> 10) & 0x00000010)
                | ((Y >>  3) & 0x00000008) | ((Y >> 18) & 0x00000004)
                | ((Y >> 26) & 0x00000002) | ((Y >> 24) & 0x00000001);

        *SK++ =   ((X << 15) & 0x20000000) | ((X << 17) & 0x10000000)
                | ((X << 10) & 0x08000000) | ((X << 22) & 0x04000000)
                | ((X >>  2) & 0x02000000) | ((X <<  1) & 0x01000000)
                | ((X << 16) & 0x00200000) | ((X << 11) & 0x00100000)
                | ((X <<  3) & 0x00080000) | ((X >>  6) & 0x00040000)
                | ((X << 15) & 0x00020000) | ((X >>  4) & 0x00010000)
                | ((Y >>  2) & 0x00002000) | ((Y <<  8) & 0x00001000)
                | ((Y >> 14) & 0x00000808) | ((Y >>  9) & 0x00000400)
                | ((Y      ) & 0x00000200) | ((Y <<  7) & 0x00000100)
                | ((Y >>  7) & 0x00000020) | ((Y >>  3) & 0x00000011)
                | ((Y <<  2) & 0x00000004) | ((Y >> 21) & 0x00000002);
    }

    return( 0 );
}

int des_set_key( uint32 *esk, uint32 *dsk, uint8 key1[8],
                                uint8 key2[8], uint8 key3[8])
{
    int i;

    /* setup encryption subkeys */

    des_main_ks( esk, key1 );
    des_main_ks( dsk + 32, key2 );
    des_main_ks( esk + 64, key3 );



    /* setup decryption subkeys */

    for( i = 0; i < 32; i += 2 )
    {
        dsk[i] = esk[94 - i];
        dsk[i + 1] = esk[95 - i];
        esk[i + 32] = dsk[62 - i];
        esk[i + 33] = dsk[63 - i];
        dsk[i + 64] = esk[30 - i];
        dsk[i + 65] = esk[31 - i];

    }

    return( 0 );
}

void DES_ROUND(uint32 *SK, uint32 X, uint32 Y)
{
    uint32 T;

    T = *SK ^ X;
    Y ^= SB8[ (T      ) & 0x3F ] ^
         SB6[ (T >>  8) & 0x3F ] ^
         SB4[ (T >> 16) & 0x3F ] ^
         SB2[ (T >> 24) & 0x3F ];

    T = *SK++ ^ ((X << 28) | (X >> 4));
    Y ^= SB7[ (T      ) & 0x3F ] ^
         SB5[ (T >>  8) & 0x3F ] ^
         SB3[ (T >> 16) & 0x3F ] ^
         SB1[ (T >> 24) & 0x3F ];
}

/* DES 64-bit block encryption/decryption */

void des_crypt( uint32 *SK, uint8 *input, uint8 *output, int len)
{
    int i;
    uint32 X, Y, T;

    for(i = 0; i < len; i++){
      X = ( (uint32) *(input + i*8) << 24 )
        | ( (uint32) *(input + (i*8) + 1) << 16 )
        | ( (uint32) *(input + (i*8) + 2) <<  8 )
        | ( (uint32) *(input + (i*8) + 3)       );

      Y = ( (uint32) *(input + (i*8) + 4) << 24 )
        | ( (uint32) *(input + (i*8) + 5) << 16 )
        | ( (uint32) *(input + (i*8) + 6) <<  8 )
        | ( (uint32) *(input + (i*8) + 7)       );

      DES_IP( X, Y );

      DES_ROUND( SK, Y, X );  DES_ROUND( (SK + 2), X, Y );
      DES_ROUND( (SK + 4), Y, X );  DES_ROUND( (SK + 6), X, Y );
      DES_ROUND( (SK + 8), Y, X );  DES_ROUND( (SK + 10), X, Y );
      DES_ROUND( (SK + 12), Y, X );  DES_ROUND( (SK + 14), X, Y );
      DES_ROUND( (SK + 16), Y, X );  DES_ROUND( (SK + 18), X, Y );
      DES_ROUND( (SK + 20), Y, X );  DES_ROUND( (SK + 22), X, Y );
      DES_ROUND( (SK + 24), Y, X );  DES_ROUND( (SK + 26), X, Y );
      DES_ROUND( (SK + 28), Y, X );  DES_ROUND( (SK + 30), X, Y );

      DES_ROUND( (SK + 32), X, Y );  DES_ROUND( (SK + 34), Y, X );
      DES_ROUND( (SK + 36), X, Y );  DES_ROUND( (SK + 38), Y, X );
      DES_ROUND( (SK + 40), X, Y );  DES_ROUND( (SK + 42), Y, X );
      DES_ROUND( (SK + 44), X, Y );  DES_ROUND( (SK + 46), Y, X );
      DES_ROUND( (SK + 48), X, Y );  DES_ROUND( (SK + 50), Y, X );
      DES_ROUND( (SK + 52), X, Y );  DES_ROUND( (SK + 54), Y, X );
      DES_ROUND( (SK + 56), X, Y );  DES_ROUND( (SK + 58), Y, X );
      DES_ROUND( (SK + 60), X, Y );  DES_ROUND( (SK + 62), Y, X );

      DES_ROUND( (SK + 64), Y, X );  DES_ROUND( (SK + 66), X, Y );
      DES_ROUND( (SK + 68), Y, X );  DES_ROUND( (SK + 70), X, Y );
      DES_ROUND( (SK + 72), Y, X );  DES_ROUND( (SK + 74), X, Y );
      DES_ROUND( (SK + 76), Y, X );  DES_ROUND( (SK + 78), X, Y );
      DES_ROUND( (SK + 80), Y, X );  DES_ROUND( (SK + 82), X, Y );
      DES_ROUND( (SK + 84), Y, X );  DES_ROUND( (SK + 86), X, Y );
      DES_ROUND( (SK + 88), Y, X );  DES_ROUND( (SK + 90), X, Y );
      DES_ROUND( (SK + 92), Y, X );  DES_ROUND( (SK + 94), X, Y );

      DES_FP( Y, X );

      *(output + i*8) = (uint8) ( (Y) >> 24 );
      *(output + (i*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + (i*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + (i*8) + 3) = (uint8) ( (Y)       );
      *(output + (i*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + (i*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + (i*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + (i*8) + 7) = (uint8) ( (X)       );

    }
}

void des_crypt_omp( uint32 *SK, uint8 *input, uint8 *output, int len)
{
    int i;
    uint32 X, Y, T;

    for(i = 0; i < len; i++){
      X = ( (uint32) *(input + i*8) << 24 )
        | ( (uint32) *(input + (i*8) + 1) << 16 )
        | ( (uint32) *(input + (i*8) + 2) <<  8 )
        | ( (uint32) *(input + (i*8) + 3)       );

      Y = ( (uint32) *(input + (i*8) + 4) << 24 )
        | ( (uint32) *(input + (i*8) + 5) << 16 )
        | ( (uint32) *(input + (i*8) + 6) <<  8 )
        | ( (uint32) *(input + (i*8) + 7)       );

      DES_IP( X, Y );

      DES_ROUND( SK, Y, X );  DES_ROUND( (SK + 2), X, Y );
      DES_ROUND( (SK + 4), Y, X );  DES_ROUND( (SK + 6), X, Y );
      DES_ROUND( (SK + 8), Y, X );  DES_ROUND( (SK + 10), X, Y );
      DES_ROUND( (SK + 12), Y, X );  DES_ROUND( (SK + 14), X, Y );
      DES_ROUND( (SK + 16), Y, X );  DES_ROUND( (SK + 18), X, Y );
      DES_ROUND( (SK + 20), Y, X );  DES_ROUND( (SK + 22), X, Y );
      DES_ROUND( (SK + 24), Y, X );  DES_ROUND( (SK + 26), X, Y );
      DES_ROUND( (SK + 28), Y, X );  DES_ROUND( (SK + 30), X, Y );

      DES_ROUND( (SK + 32), X, Y );  DES_ROUND( (SK + 34), Y, X );
      DES_ROUND( (SK + 36), X, Y );  DES_ROUND( (SK + 38), Y, X );
      DES_ROUND( (SK + 40), X, Y );  DES_ROUND( (SK + 42), Y, X );
      DES_ROUND( (SK + 44), X, Y );  DES_ROUND( (SK + 46), Y, X );
      DES_ROUND( (SK + 48), X, Y );  DES_ROUND( (SK + 50), Y, X );
      DES_ROUND( (SK + 52), X, Y );  DES_ROUND( (SK + 54), Y, X );
      DES_ROUND( (SK + 56), X, Y );  DES_ROUND( (SK + 58), Y, X );
      DES_ROUND( (SK + 60), X, Y );  DES_ROUND( (SK + 62), Y, X );

      DES_ROUND( (SK + 64), Y, X );  DES_ROUND( (SK + 66), X, Y );
      DES_ROUND( (SK + 68), Y, X );  DES_ROUND( (SK + 70), X, Y );
      DES_ROUND( (SK + 72), Y, X );  DES_ROUND( (SK + 74), X, Y );
      DES_ROUND( (SK + 76), Y, X );  DES_ROUND( (SK + 78), X, Y );
      DES_ROUND( (SK + 80), Y, X );  DES_ROUND( (SK + 82), X, Y );
      DES_ROUND( (SK + 84), Y, X );  DES_ROUND( (SK + 86), X, Y );
      DES_ROUND( (SK + 88), Y, X );  DES_ROUND( (SK + 90), X, Y );
      DES_ROUND( (SK + 92), Y, X );  DES_ROUND( (SK + 94), X, Y );
      
      DES_FP( Y, X );
      *(output + i*8) = (uint8) ( (Y) >> 24 );
      *(output + (i*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + (i*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + (i*8) + 3) = (uint8) ( (Y)       );
      *(output + (i*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + (i*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + (i*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + (i*8) + 7) = (uint8) ( (X)       );

    }

}

void des_encrypt( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int len)
{
    des_crypt( esk, input, output, len);
    //des_crypt( dsk, input, output, len);
}

void init_matrix(int *A, int *B, int *C, int *D, int size){
  int i;

    for(i = 0; i < size*size; i++){
      A[i] = (i%size)+1;
      B[i] = (i%size)+1;
      C[i] = 0;
      D[i] = 0;
    }
}

void init_filter(float *r, float *Vect_Up, float *Vect_F,
                float *Vect_H, float *H, float *F, float *Vect_F_host, int size){
  int i;

    for(i = 0; i < size; i++){
      r[i] = i + 0.0001;
      Vect_Up[i] = 0;
      Vect_F[i] = 0;
      Vect_H[i]=0;
      Vect_F_host[i] = 0;
    }

    for(i = 0; i < N_col; i++){
      H[i] = 0.0001;
      F[i] = 0.0001;
    }

}

void init_des(unsigned char *packet_in, unsigned char *o_packet_in, int size){
  int i;
      for(i = 0; i < size; i++){
          if(i < HEADER_SIZE ){
              packet_in[i] = headers[i % MAX_PACKETS][i];
 	      o_packet_in[i] = headers[i % MAX_PACKETS][i];
          }else{
              packet_in[i] = DES3_init[i%8];
	      o_packet_in[i] = DES3_init[i%8];
          }
      }
}

