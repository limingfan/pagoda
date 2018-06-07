#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include "packet_lengths.h"
#include "headers.h"

#include "packet.h"
#include "../../common/para.h"

#define HEADER_SIZE 36

#define LOOP_NUM (BT_NUM)
#define sub_task (TK_NUM)
#define numpackets (LOOP_NUM*sub_task)
#define THREADSTACK  65536

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void GET_UINT32(uint32 n,uint8 *b,int i)
{
    (n) = ( (uint32) *(b + i) << 24 )
        | ( (uint32) *(b + i + 1) << 16 )
        | ( (uint32) *(b + i + 2) <<  8 )
        | ( (uint32) *(b + i + 3)       );
}

void PUT_UINT32(uint32 n,uint8 *b,int i)
{
    *(b + i) = (uint8) ( (n) >> 24 );
    *(b + i + 1) = (uint8) ( (n) >> 16 );
    *(b + i + 2) = (uint8) ( (n) >>  8 );
    *(b + i + 3) = (uint8) ( (n)       );
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

void des_crypt( uint32 *SK, uint8 *input, uint8 *output, int len)
{
    int i;
    uint32 X, Y, T;

    for(i = 0; i < len; i++){
#if 0
      GET_UINT32( X, input, i*8 );
      GET_UINT32( Y, input, i*8+4 );
#endif
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
#if 0
      PUT_UINT32( Y, output, i*8 );
      PUT_UINT32( X, output, i*8+4 );
#endif
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

/* DES 64-bit block encryption/decryption */

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

void des_encrypt_omp( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int len)
{
    des_crypt_omp( esk, input, output, len);
    //des_crypt_omp( dsk, input, output, len);
}

/* For debugging purposes, to get the size of the packet_numberth packet */
unsigned int
packet_size (unsigned int packet_number)
{
  packet_number = packet_number % MAX_INDEX;
  return (packet_lengths[packet_number]);
}

typedef struct
{
  uint32 **a;
  unsigned char  **b, **c;
  int d;
} parm;

void * worker(void *arg)
{
  parm  *p = (parm *) arg;
  des_crypt_omp(*(p->a), *(p->b), *(p->c), p->d);
}


int main (int argc, char **argv)
{
  	int i, j, k;
  	unsigned char **packet_in, **packet_out, **packet_openout, **packet_openin;
  	int num_thread[numpackets];
  	int num_size[numpackets];
  	FILE *fp;
  	int *packet_length;

  	parm           *arg;
  	pthread_t      *threads;
  	pthread_attr_t  attrs;


  	pthread_attr_init(&attrs);
  	pthread_setconcurrency(16);
  	pthread_attr_setstacksize(&attrs, THREADSTACK);

  	threads = (pthread_t *) malloc(numpackets * sizeof(pthread_t));
  	arg=(parm *)malloc(sizeof(parm)*numpackets);

  	uint32 des_esk[96];
  	uint32 des_dsk[96];
  	uint32 **encrypt;
  	uint32 **decrypt;

  	double start_timer, end_timer;
 
  	packet_in = (unsigned char**)malloc(numpackets*sizeof(unsigned char*)); 
  	packet_out = (unsigned char**)malloc(numpackets*sizeof(unsigned char*)); 
  	packet_openout = (unsigned char**)malloc(numpackets*sizeof(unsigned char*));
  	packet_openin = (unsigned char**)malloc(numpackets*sizeof(unsigned char*));
  	packet_length = (int*)malloc(numpackets*sizeof(int));
  	encrypt = (uint32**)malloc(numpackets*sizeof(uint32*)); 
  	decrypt = (uint32**)malloc(numpackets*sizeof(uint32*)); 
  
  	fp = fopen("rand.txt", "r");
  	for(i = 0; i < numpackets; i++)
    		fscanf(fp, "%1d", &num_thread[i]);

  	fclose(fp);

  	for(i = 0; i < numpackets; i++)
    		num_size[i] = num_thread[i]*32;


  	/*Generate encryption key*/
  	des_set_key(des_esk, des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);

  	//memory allocation for packet
  	for(i = 0; i < numpackets; i++){
     
    		if(num_size[i] == 32){
			packet_in[i] =  (unsigned char *) malloc (2*num_size[i]*num_size[i]);
      			packet_out[i] =  (unsigned char *) malloc (2*num_size[i]*num_size[i]);
      			packet_openout[i] =  (unsigned char *) malloc (2*num_size[i]*num_size[i]);
      			packet_openin[i] =  (unsigned char *) malloc (2*num_size[i]*num_size[i]);

    		} else{
      			packet_in[i] =  (unsigned char *) malloc (num_size[i]*num_size[i]);
      			packet_out[i] =  (unsigned char *) malloc (num_size[i]*num_size[i]);
      			packet_openout[i] =  (unsigned char *) malloc (num_size[i]*num_size[i]);
      			packet_openin[i] =  (unsigned char *) malloc (num_size[i]*num_size[i]);
    		}
      		encrypt[i] =  (uint32 *) malloc (96*sizeof(uint32));
      		decrypt[i] =  (uint32 *) malloc (96*sizeof(uint32));

  	}

  	//generate packet
  	for(i = 0; i < numpackets; i++){
      		if(num_size[i] == 32){
        		for(j = 0; j < 2* num_size[i]*num_size[i]; j++){
          			if(j < HEADER_SIZE ){
              				packet_in[i][j] = headers[i % MAX_PACKETS][j];
	      				packet_openin[i][j] = headers[i % MAX_PACKETS][j];
          			}else{
              				packet_in[i][j] = DES3_init[j%8];
	      				packet_openin[i][j] = DES3_init[j%8];
          			}
        		}
      		}else{
        		for(j = 0; j < num_size[i]*num_size[i]; j++){
          			if(j < HEADER_SIZE ){
              				packet_in[i][j] = headers[i % MAX_PACKETS][j];
	      				packet_openin[i][j] = headers[i % MAX_PACKETS][j];
          			}else{
              				packet_in[i][j] = DES3_init[j%8];
	      				packet_openin[i][j] = DES3_init[j%8];

          			}
        		}
      		}

      		for(j = 0; j < 96; j++){
			encrypt[i][j] = des_esk[j];
        		decrypt[i][j] = des_dsk[j];
      		}
  	}

  	start_timer = my_timer();
  	// run des
  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = &encrypt[k*sub_task+i];
        		arg[i].b = &packet_in[k*sub_task+i];
        		arg[i].c = &packet_out[k*sub_task+i];
			if(num_size[k*sub_task+i] == 32)
          			arg[i].d = 2*num_size[k*sub_task+i]*num_size[k*sub_task+i]/8;
        		else
          			arg[i].d = num_size[k*sub_task+i]*num_size[k*sub_task+i]/8;

        		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}
    
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = &decrypt[k*sub_task+i];
        		arg[i].b = &packet_out[k*sub_task+i];
        		arg[i].c = &packet_in[k*sub_task+i];
        		if(num_size[k*sub_task+i] == 32)
          			arg[i].d = 2*num_size[k*sub_task+i]*num_size[k*sub_task+i]/8;
        		else
          			arg[i].d = num_size[k*sub_task+i]*num_size[k*sub_task+i]/8;

        		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		pthread_attr_destroy(&attrs);
    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

  	}
  	end_timer = my_timer();
  	printf("DES pthread Execution Time: %lf Sec.\n", end_timer - start_timer);

	printf("CPU program running\n");
  	start_timer = my_timer();
  	// run des
  	for(i = 0; i < numpackets; i++){
    		if(num_size[i] == 32)
    			des_crypt(des_esk, packet_openin[i], packet_openout[i], 2*num_size[i]*num_size[i]/8);
    		else
        		des_crypt(des_esk, packet_openin[i], packet_openout[i], num_size[i]*num_size[i]/8);
  	}

  	for(i = 0; i < numpackets; i++){
    		if(num_size[i] == 32)
    			des_crypt(des_dsk, packet_openout[i], packet_openin[i], 2*num_size[i]*num_size[i]/8);
    		else
			des_crypt(des_dsk, packet_openout[i], packet_openin[i], num_size[i]*num_size[i]/8);
  	}
  	end_timer = my_timer();
  	//printf("CPU time:%lf Sec.\n", end_timer - start_timer);\


	/*Verification*/
	printf("verifying\n");
	int flag = 0;
  	for(i = 0; i < numpackets; i++){
    		if(num_size[i] == 32){
      			for(j = 0; j < 2*num_size[i]*num_size[i]; j++){
        			if(packet_out[i][j] != packet_openout[i][j]){
          				printf("Error:%u, %u, %d, %d\n", packet_out[i][j], packet_openout[i][j], i, j);
					flag = 1;
          				break;
        			}
      			}
    		}else{
			for(j = 0; j < num_size[i]*num_size[i]; j++){
          			if(packet_out[i][j] != packet_openout[i][j]){
            				printf("Error:%u, %u, %d, %d\n", packet_out[i][j], packet_openout[i][j], i, j);
					flag = 1;
            				break;
          			}
        		}

    		}
  	}
	if(!flag) printf("verify successfully\n");

  	for(i = 0; i < numpackets; i++){
    		free(packet_in[i]);
    		free(packet_out[i]);
    		free(packet_openin[i]);
    		free(packet_openout[i]);
    		free(encrypt[i]);
    		free(decrypt[i]);
  	}

  	free(packet_in);
  	free(packet_out);
  	free(packet_openin);
  	free(packet_openout);
  	free(packet_length);
  	free(encrypt);
  	free(decrypt);

  	free(arg);  

  	return 0;
}
