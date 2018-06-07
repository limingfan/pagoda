#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../../common/para.h"
#include "../../common/para.cuh"
#include "omp.h"
#include <stdarg.h>
//#include "hash/dict.h"

#include "runtime.cuh"

__device__ int syncID;
__device__ int threadNum;
//double time_counter = 0.0;
int *done, *doneDev;
//int *exec, *execDev;
//int *totalExecTasks, *totalExecTasksDev;
int *totalScheTasks, *totalScheTasksDev;

cTaskStruct *ccTaskPool;
gTaskStruct *ggTaskPool;

int *readyFlagArray, *readyFlagArrayDev;
int *jobIDTable;
int *initReFlagValue;
int *initTkFlagValue;
int *initDoFlagValue;
//int *barIDArray;
//enum mytypes {LONG, INT, FLOAT, DOUBLE };
cudaStream_t master_kernel_stream;
cudaStream_t runtime_stream;

//extern __global__ void masterKernel(volatile int *done, volatile int *totalScheTasks, volatile void **gTaskPool, 
//                                	struct params_dev *warpPool);
__global__ void masterKernel(volatile int *done, volatile int *totalScheTasks, volatile gTaskStruct *gTaskPool);

int get_JobID(){
  static int i = 0;
  while(1){
    if(!jobIDTable[i]){
      jobIDTable[i] = 1;
      return i;
    }
    i++;
    if(i == WP_SIZE) i = 0;
  }
  
}

void free_JobID(int jobID){
  jobIDTable[jobID] = 0;
}
void runtime_init(){
  int i;

  //cudaStream_t s1;
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
  checkCudaErrors(cudaStreamCreate(&runtime_stream));
  checkCudaErrors(cudaStreamCreate(&master_kernel_stream));
    
  // done flag to interrupt runtime
  checkCudaErrors(cudaHostAlloc(&done, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&doneDev, sizeof(int)));
  // host task buffer
  checkCudaErrors(cudaHostAlloc(&ccTaskPool, (BK_NUM*BP_NUM)*sizeof(cTaskStruct), cudaHostAllocDefault));

  // device task buffer
  checkCudaErrors(cudaMalloc(&ggTaskPool, (BK_NUM*BP_NUM)*sizeof(gTaskStruct)));
  // totalScheTasks: 
  checkCudaErrors(cudaHostAlloc(&totalScheTasks, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&totalScheTasksDev, sizeof(int)));

  jobIDTable = (int*)malloc(WP_SIZE*sizeof(int));
  //checkCudaErrors(cudaMalloc(&barIDArray, syncNum*sizeof(int*)));

  for(i = 0; i < (BK_NUM*BP_NUM); i++) {
    jobIDTable[i] = 0;
    ccTaskPool[i].ready = 0;
    ccTaskPool[i].done = -1;
    ccTaskPool[i].taskId = 0;
  }
  
// runtime variables copy
  *done = 0;
  *totalScheTasks = 0;
  checkCudaErrors(cudaMemcpyAsync(doneDev, done, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(totalScheTasksDev, totalScheTasks, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(ggTaskPool, ccTaskPool, (BK_NUM*BP_NUM)*sizeof(gTaskStruct), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  
  //MasterKernel
  masterKernel<<<BK_NUM, TD_NUM, SH_MEM_SIZE, master_kernel_stream>>>(doneDev, totalScheTasksDev, ggTaskPool);
}

int taskLaunch(int paraN, ...){
  static int i = 0;
  static int sync_counter = 0;
  int j, k;
  int terminate = 1;
  int taskId;
  int test;
  va_list ap;
  va_start(ap,paraN);

  while(i < (BK_NUM*BP_NUM) && terminate == 1){
    if(ccTaskPool[i].ready == 0){
        // **Add here**: renew task table, set the bit of task ID on
        // **Add here**: get_ID()
 	ccTaskPool[i].ready = 1;
	ccTaskPool[i].taskId = i+1;
	taskId = i;
        //printf("taskId:%d\n", taskId);
	ccTaskPool[i].done = 1;
	for(j = 0; j < paraN; j++){ // set parameters
	  int type = va_arg(ap, enum mytypes);
	    switch(type){
	      case INT:
		if(j == 0) ccTaskPool[i].thread = va_arg(ap, int);
		if(j == 1) ccTaskPool[i].block = va_arg(ap, int);
		if(j == 2) ccTaskPool[i].sharemem = va_arg(ap, int);
                if(j == 3) ccTaskPool[i].sync = va_arg(ap, int);
		if(j == 4) ccTaskPool[i].funcId = va_arg(ap, int);
		if(j > 4)  ccTaskPool[i].para[j-inParaNum] = va_arg(ap, int*);
	        break;
	      case FLOAT:
	        ccTaskPool[i].para[j-inParaNum] = va_arg(ap, float*);
	        break;
	      case DOUBLE:
	        ccTaskPool[i].para[j-inParaNum] = va_arg(ap, double*);
	        break;
              case CHAR:
                ccTaskPool[i].para[j-inParaNum] = va_arg(ap, unsigned char*);
                break;
	      case INT32:
                ccTaskPool[i].para[j-inParaNum] = va_arg(ap, unsigned long int*);
                break;
	      default:
	        break;
	    } // End switch
//	  } // End else
	} // End for paraN
	//printf("runtime addres:%x, %x,%x\n", ccTaskPool[i].para[0], ccTaskPool[i].para[1], ccTaskPool[i].para[2]);
	checkCudaErrors(cudaMemcpyAsync(ggTaskPool+i, ccTaskPool+i, 
				sizeof(gTaskStruct), cudaMemcpyHostToDevice, runtime_stream)); 
        terminate = 0; 
     } // end if cTaskPool
     i++;

     if(i == (BK_NUM*BP_NUM)){
          checkCudaErrors(cudaMemcpyAsync(ccTaskPool, ggTaskPool, (BK_NUM*BP_NUM)*sizeof(gTaskStruct), cudaMemcpyDeviceToHost, runtime_stream));
          checkCudaErrors(cudaStreamSynchronize(runtime_stream));
          i = 0;
     }
  } // end while i < BK_NUM*BP_NUM

  va_end(ap);
  return taskId;
}

void waitAll(int num_tasks){
	*totalScheTasks = 0;
	int i;
	while(*totalScheTasks < num_tasks){
		checkCudaErrors(cudaMemcpyAsync(totalScheTasks, totalScheTasksDev, sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
		checkCudaErrors(cudaStreamSynchronize(runtime_stream));
		if(*totalScheTasks > 22760) i++;
		if(i == 15000){
		  printf("runtime task:%d\n", *totalScheTasks);
		  break;
		} 
	}

}

void runtime_destroy(){

  *done = 1;
  checkCudaErrors(cudaMemcpyAsync(doneDev, done, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));

  checkCudaErrors(cudaStreamSynchronize(runtime_stream));


}
void runtime_free(){

  checkCudaErrors(cudaStreamDestroy(master_kernel_stream));
  checkCudaErrors(cudaStreamDestroy(runtime_stream));

  checkCudaErrors(cudaFreeHost(done));
  checkCudaErrors(cudaFreeHost(ccTaskPool));
  checkCudaErrors(cudaFreeHost(totalScheTasks));

  checkCudaErrors(cudaFree(doneDev));
  checkCudaErrors(cudaFree(ggTaskPool));
  checkCudaErrors(cudaFree(totalScheTasksDev));

}
