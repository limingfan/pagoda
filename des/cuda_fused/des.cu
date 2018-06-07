#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "packet_lengths.h"
#include "headers.h"

#include "packet.h"
#include "../../common/para.h"

#define HEADER_SIZE 36

#define numpackets (TK_NUM * BT_NUM)

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
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
__device__ void DES_ROUND_dev(uint32 *SK, uint32 X, uint32 Y)
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

/* DES 64-bit block encryption/decryption */

void des_crypt( uint32 *SK, uint8 *input, uint8 *output, int *size, int *threads, int index)
{
    int i, t, k;
    uint32 X, Y, T;
    int td, tt;

    for(t = 0; t < TK_NUM; t++){
      td = threads[index*TK_NUM+t];
      for(k = 0; k < td; k++){
        tt = td/8;
        for(i = 0; i < tt; i++){
          X = ( (uint32) *(input + size[t] + (i*td+k)*8) << 24 )
            | ( (uint32) *(input + size[t] + ((i*td+k)*8) + 1) << 16 )
            | ( (uint32) *(input + size[t] + ((i*td+k)*8) + 2) <<  8 )
            | ( (uint32) *(input + size[t] + ((i*td+k)*8) + 3)       );

          Y = ( (uint32) *(input + size[t] + ((i*td+k)*8) + 4) << 24 )
            | ( (uint32) *(input + size[t] + ((i*td+k)*8) + 5) << 16 )
            | ( (uint32) *(input + size[t] + ((i*td+k)*8) + 6) <<  8 )
            | ( (uint32) *(input + size[t] + ((i*td+k)*8) + 7)       );

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

        *(output + size[t] + (i*td+k)*8) = (uint8) ( (Y) >> 24 );
        *(output + size[t] + ((i*td+k)*8) + 1) = (uint8) ( (Y) >> 16 );
        *(output + size[t] + ((i*td+k)*8) + 2) = (uint8) ( (Y) >>  8 );
        *(output + size[t] + ((i*td+k)*8) + 3) = (uint8) ( (Y)       );
        *(output + size[t] + ((i*td+k)*8) + 4) = (uint8) ( (X) >> 24 );
        *(output + size[t] + ((i*td+k)*8) + 5) = (uint8) ( (X) >> 16 );
        *(output + size[t] + ((i*td+k)*8) + 6) = (uint8) ( (X) >>  8 );
        *(output + size[t] + ((i*td+k)*8) + 7) = (uint8) ( (X)       );
	
        } // End i
      } // End k
    } // End t
}

__device__ void des_crypt_dev( uint32 *SK, uint8 *input, uint8 *output, int *size, int *thread, int index)
{
    int i;
    uint32 X, Y, T;
    int tid = threadIdx.x;
    int bk = blockIdx.x;
    int td, tt;
    td = thread[index*TK_NUM+bk];
    tt = td/8;

    if(tid < td){
      for(i = 0; i < tt; i++){
      
	X = ( (uint32) *(input + size[bk] + (i*td+tid)*8) << 24 )
        | ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 1) << 16 )
        | ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 2) <<  8 )
        | ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 3)       );

        Y = ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 4) << 24 )
        | ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 5) << 16 )
        | ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 6) <<  8 )
        | ( (uint32) *(input + size[bk] + ((i*td+tid)*8) + 7)       );


        DES_IP( X, Y );

        DES_ROUND_dev( SK, Y, X );  DES_ROUND_dev( (SK + 2), X, Y );
        DES_ROUND_dev( (SK + 4), Y, X );  DES_ROUND_dev( (SK + 6), X, Y );
        DES_ROUND_dev( (SK + 8), Y, X );  DES_ROUND_dev( (SK + 10), X, Y );
        DES_ROUND_dev( (SK + 12), Y, X );  DES_ROUND_dev( (SK + 14), X, Y );
        DES_ROUND_dev( (SK + 16), Y, X );  DES_ROUND_dev( (SK + 18), X, Y );
        DES_ROUND_dev( (SK + 20), Y, X );  DES_ROUND_dev( (SK + 22), X, Y );
        DES_ROUND_dev( (SK + 24), Y, X );  DES_ROUND_dev( (SK + 26), X, Y );
        DES_ROUND_dev( (SK + 28), Y, X );  DES_ROUND_dev( (SK + 30), X, Y );

        DES_ROUND_dev( (SK + 32), X, Y );  DES_ROUND_dev( (SK + 34), Y, X );
        DES_ROUND_dev( (SK + 36), X, Y );  DES_ROUND_dev( (SK + 38), Y, X );
        DES_ROUND_dev( (SK + 40), X, Y );  DES_ROUND_dev( (SK + 42), Y, X );
        DES_ROUND_dev( (SK + 44), X, Y );  DES_ROUND_dev( (SK + 46), Y, X );
        DES_ROUND_dev( (SK + 48), X, Y );  DES_ROUND_dev( (SK + 50), Y, X );
        DES_ROUND_dev( (SK + 52), X, Y );  DES_ROUND_dev( (SK + 54), Y, X );
        DES_ROUND_dev( (SK + 56), X, Y );  DES_ROUND_dev( (SK + 58), Y, X );
        DES_ROUND_dev( (SK + 60), X, Y );  DES_ROUND_dev( (SK + 62), Y, X );

        DES_ROUND_dev( (SK + 64), Y, X );  DES_ROUND_dev( (SK + 66), X, Y );
        DES_ROUND_dev( (SK + 68), Y, X );  DES_ROUND_dev( (SK + 70), X, Y );
        DES_ROUND_dev( (SK + 72), Y, X );  DES_ROUND_dev( (SK + 74), X, Y );
        DES_ROUND_dev( (SK + 76), Y, X );  DES_ROUND_dev( (SK + 78), X, Y );
        DES_ROUND_dev( (SK + 80), Y, X );  DES_ROUND_dev( (SK + 82), X, Y );
        DES_ROUND_dev( (SK + 84), Y, X );  DES_ROUND_dev( (SK + 86), X, Y );
        DES_ROUND_dev( (SK + 88), Y, X );  DES_ROUND_dev( (SK + 90), X, Y );
        DES_ROUND_dev( (SK + 92), Y, X );  DES_ROUND_dev( (SK + 94), X, Y );

        DES_FP( Y, X );

      *(output + size[bk] + (i*td+tid)*8) = (uint8) ( (Y) >> 24 );
      *(output + size[bk] + ((i*td+tid)*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + size[bk] + ((i*td+tid)*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + size[bk] + ((i*td+tid)*8) + 3) = (uint8) ( (Y)       );
      *(output + size[bk] + ((i*td+tid)*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + size[bk] + ((i*td+tid)*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + size[bk] + ((i*td+tid)*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + size[bk] + ((i*td+tid)*8) + 7) = (uint8) ( (X)       );

      }
    }
}
__global__ void des_encrypt_dev( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int *size, int *thread, int index)
{
    des_crypt_dev( esk, input, input, size, thread, index);
    des_crypt_dev( dsk, input, output, size, thread, index);
}

void des_encrypt( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int *size, int *thread, int index)
{
    des_crypt( esk, input, input, size, thread, index);
    des_crypt( dsk, input, output, size, thread, index);
}


/* For debugging purposes, to get the size of the packet_numberth packet */
unsigned int
packet_size (unsigned int packet_number)
{
  packet_number = packet_number % MAX_INDEX;
  return (packet_lengths[packet_number]);
}


int main (int argc, char **argv)
{
  	int i, j;
  	unsigned char **packet_in, **packet_in_dev, 
		**packet_out, **packet_out_dev, **packet_open;
  	int num_thread[numpackets], *num_thread_dev;
  	int num_size[BT_NUM];
  	int pos_task[BT_NUM][TK_NUM];
  	int *pos_task_dev[BT_NUM];
  	FILE *fp;
	cudaSetDevice(0);

  	uint32 *des_esk;
  	uint32 *des_dsk;

  	uint32 *des_esk_dev;
  	uint32 *des_dsk_dev;


  	double start_timer, end_timer;
 
  	packet_in = (unsigned char**)malloc(BT_NUM*sizeof(unsigned char*)); 
  	packet_in_dev = (unsigned char**)malloc(BT_NUM*sizeof(unsigned char*));
  	packet_out = (unsigned char**)malloc(BT_NUM*sizeof(unsigned char*)); 
  	packet_out_dev = (unsigned char**)malloc(BT_NUM*sizeof(unsigned char*));
  	packet_open = (unsigned char**)malloc(BT_NUM*sizeof(unsigned char*));
 

  	checkCudaErrors(cudaHostAlloc(&des_esk, 96*sizeof(uint32), cudaHostAllocDefault));
  	checkCudaErrors(cudaMalloc(&des_esk_dev, 96*sizeof(uint32)));
  	checkCudaErrors(cudaHostAlloc(&des_dsk, 96*sizeof(uint32), cudaHostAllocDefault));
  	checkCudaErrors(cudaMalloc(&des_dsk_dev, 96*sizeof(uint32)));

  	fp = fopen("rand.txt", "r");
  	for(i = 0; i < numpackets; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < numpackets; i++)
    		num_thread[i] *= 32;

  	for(i = 0; i < BT_NUM; i++){
    		num_size[i] = 0;
  	}

  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < TK_NUM; j++){
        		num_size[i] += (num_thread[i*TK_NUM+j])*
                        	(num_thread[i*TK_NUM+j]);
        		pos_task[i][j] = 0;
        		if(j > 0) pos_task[i][j] += pos_task[i][j-1] + (num_thread[i*TK_NUM+j-1])*
                        	(num_thread[i*TK_NUM+j-1]);

    		}
  	}


  	/*Generate encryption key*/
  	des_set_key(des_esk, des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);

  	//memory allocation for packet
  	for(i = 0; i < BT_NUM; i++){    
      		checkCudaErrors(cudaHostAlloc(&packet_in[i], num_size[i]*sizeof(unsigned char), cudaHostAllocDefault));
      		checkCudaErrors(cudaMalloc(&packet_in_dev[i], num_size[i]*sizeof(unsigned char)));
      		checkCudaErrors(cudaHostAlloc(&packet_out[i], num_size[i]*sizeof(unsigned char), cudaHostAllocDefault));
      		checkCudaErrors(cudaMalloc(&packet_out_dev[i], num_size[i]*sizeof(unsigned char)));
      		packet_open[i] =  (unsigned char *) malloc (num_size[i]*sizeof(unsigned char));
      		checkCudaErrors(cudaMalloc(&pos_task_dev[i], TK_NUM*sizeof(int)));

  	}
  	checkCudaErrors(cudaMalloc(&num_thread_dev, numpackets*sizeof(int)));

	printf("DES CUDA static fusion inputs are generating\n");
  	//generate packet
  	for(i = 0; i < BT_NUM; i++){
      		for(j = 0; j < num_size[i]; j++){
          		if(j < HEADER_SIZE ){
              			packet_in[i][j] = headers[i % MAX_PACKETS][j];
          		}else{
              			packet_in[i][j] = DES3_init[j%8];
          		}
      		}
  	}

  	// copy data to GPU
  	for(i = 0; i < BT_NUM; i++){
     		checkCudaErrors(cudaMemcpy(packet_in_dev[i], packet_in[i], num_size[i]*sizeof(unsigned char), cudaMemcpyHostToDevice));
     		checkCudaErrors(cudaMemcpy(pos_task_dev[i], pos_task[i], TK_NUM*sizeof(int), cudaMemcpyHostToDevice));

  	}
  	checkCudaErrors(cudaMemcpy(des_esk_dev, des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaMemcpy(des_dsk_dev, des_dsk, 96*sizeof(uint32), cudaMemcpyHostToDevice));
  	checkCudaErrors(cudaMemcpy(num_thread_dev, num_thread, numpackets*sizeof(int), cudaMemcpyHostToDevice));

  	checkCudaErrors(cudaDeviceSynchronize());

	printf("DES CUDA static fusion is running\n");
  	start_timer = my_timer();
  	// run des
  	for(i = 0; i < BT_NUM; i++){
		des_encrypt_dev<<<TK_NUM, TDK_NUM>>>( des_esk_dev, des_esk_dev, packet_in_dev[i], 
					packet_out_dev[i], pos_task_dev[i], num_thread_dev, i);

  	}
  	checkCudaErrors(cudaDeviceSynchronize());
  	end_timer = my_timer();
  	printf("DES CUDA static fusion Time: %lf Sec.\n", end_timer - start_timer);

  	for(i = 0; i < BT_NUM; i++){
     		checkCudaErrors(cudaMemcpy(packet_out[i], packet_out_dev[i], num_size[i]*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  	}
  	checkCudaErrors(cudaDeviceSynchronize());

	printf("CPU program running\n");
  	start_timer = my_timer();
  	// run des
  	for(i = 0; i < BT_NUM; i++){
        	des_encrypt(des_esk, des_dsk, packet_in[i], packet_open[i], pos_task[i], num_thread, i);
  	}
  	end_timer = my_timer();
  	//printf("CPU time:%lf Sec.\n", end_timer - start_timer);

	/*Verification*/
	printf("verifying\n");
	int flag = 0;
  	for(i = 0; i < BT_NUM; i++){
    		for(j = 0; j < num_size[i]; j++){
      			if(packet_out[i][j] != packet_open[i][j]){
        			printf("Error:%u, %u, %d, %d\n", packet_out[i][j], packet_open[i][j], i, j);
				flag = 1;
        			break;
      			}
    		}
  	}
	if(!flag) printf("verify successfully\n");

  	for(i = 0; i < BT_NUM; i++){
    		checkCudaErrors(cudaFreeHost(packet_in[i]));
    		checkCudaErrors(cudaFree(packet_in_dev[i]));
    		checkCudaErrors(cudaFreeHost(packet_out[i]));
   		checkCudaErrors(cudaFree(packet_out_dev[i]));
    		checkCudaErrors(cudaFree(pos_task_dev[i]));
    		free(packet_open[i]);
  	}

  	checkCudaErrors(cudaFreeHost(des_esk));
  	checkCudaErrors(cudaFree(des_esk_dev));
  	checkCudaErrors(cudaFreeHost(des_dsk));
  	checkCudaErrors(cudaFree(des_dsk_dev));
  	checkCudaErrors(cudaFree(num_thread_dev));


  	free(packet_in);
  	free(packet_in_dev);
  	free(packet_out);
  	free(packet_out_dev);
  	free(packet_open);

  	if(cudaDeviceReset()== cudaSuccess) printf("Reset successful\n");

  
  	return 0;
}
