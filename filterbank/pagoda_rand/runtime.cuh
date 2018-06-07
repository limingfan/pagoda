#ifndef RUNTIME_H
#define RUNTIME_H

void runtime_init();
int taskLaunch(int paraN, ...);
void waitAll(int num_tasks);
void runtime_destroy();
void runtime_free();
int getTaskID();

extern cudaStream_t runtime_stream;
enum mytypes {LONG, INT, FLOAT, DOUBLE };
//extern __device__ int syncID;
//extern __device__ int threadNum; 
#endif
