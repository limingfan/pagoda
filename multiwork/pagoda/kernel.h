#ifndef _GPU_MULTIWORK_H_
#define _GPU_MULTIWORK_H_

#ifndef uint8
#define uint8  unsigned char
#endif

#ifndef uint32
#define uint32 unsigned long int
#endif

#define x_max (1.25)
#define x_min (-2.25)
#define y_max (1.75)
#define y_min (-1.75)
#define count_max 400
#define n 64

// Size of inputs
#define N_sim 2048
// Num. of sample
#define N_samp 8
#define N_col 64

#define MROW 64
#define MCOL 64
#define MSIZE (MROW*MCOL)

#define HEADER_SIZE 36
#define LEN 8192

#define TDD_NUM 256

extern void mult(int *A, int *B, int *C, int size);
extern void h_FBCore(float *r, float *H, float *Vect_H, 
			float *Vect_Dn, float *Vect_Up, float *Vect_F, float *F);
extern int des_set_key( uint32 *esk, uint32 *dsk, uint8 key1[8],
                                uint8 key2[8], uint8 key3[8]);
extern void h_get_pixel(int *count, float index);
extern void des_encrypt( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int len);

/* For testing...*/
unsigned int packet_size (unsigned int);

static unsigned char DES3_keys[3][8] =
{
    { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF },
    { 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01 },
    { 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23 }
};

static unsigned char DES3_init[8] =
{
    0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74
};


#endif
