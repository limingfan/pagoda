#ifndef _GPU_MULTIWORK_H_
#define _GPU_MULTIWORK_H_

#include "../../common/para.h"
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

// the number of thread
//#define TD_NUM 256 
#define task (32768)
// the number of tasks in a batch
//#define TK_NUM 120
// the number of batches 
#define BT_NUM ((4096/TK_NUM) + 1) 
// the number of sub-tasks in a batch
#define SUB_NUM (TK_NUM/4)

#endif
