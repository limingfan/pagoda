#ifndef _GPU_DCT_H_
#define _GPU_DCT_H_

#include "../../common/para.h"
#define BLOCK_SIZE 8
#define task (TK_NUM*(BT_NUM-50))

#define __MUL24_FASTER_THAN_ASTERIX

#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x,y)   (__mul24(x,y))
#else
#define FMUL(x,y)   ((x)*(y))
#endif

#define C_norm  (0.3535533905932737) // 1 / (8^0.5)
#define C_a     (1.387039845322148) //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b     (1.306562964876377) //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c     (1.175875602419359) //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d     (0.785694958387102) //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e     (0.541196100146197) //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f     (0.275899379282943) //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.

extern void SubroutineDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut);
extern void SubroutineIDCTvector(float *FirstIn, int StepIn, float *FirstOut, int StepOut);
extern void DCT(float *fSrc, float *fDst, int Stride, int size, int thread);
extern void computeDCT(float *fSrc, float *fDst, int Stride, int size, int thread);
extern void computeIDCT(float *fSrc, float *fDst, int Stride, int size, int thread);
#endif
