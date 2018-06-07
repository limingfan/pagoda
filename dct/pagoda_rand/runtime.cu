#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../../common/para.h"
#include "../../common/para.cuh"
#include <stdarg.h>

#include "runtime.cuh"

__device__ int syncID;
__device__ int threadNum;
int *done, *doneDev;
int *totalScheTasks, *totalScheTasksDev;

cTaskStruct *ccTaskPool;
gTaskStruct *ggTaskPool;

int *readyFlagArray, *readyFlagArrayDev;
int *jobIDTable;
int *initReFlagValue;
int *initTkFlagValue;
int *initDoFlagValue;

static int taskId = 0;
static int lastEmptyTask = 0;
static int round_count = 0;
static int taskIndex = 0;

static int barrierCount = 0;
cudaStream_t master_kernel_stream;
cudaStream_t runtime_stream;

__global__ void masterKernel(volatile int *done, volatile int *totalScheTasks, volatile gTaskStruct *gTaskPool);

void runtime_init(){
  	int i;

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

  	for(i = 0; i < (BK_NUM*BP_NUM); i++) {
    		ccTaskPool[i].ready = 0;
    		ccTaskPool[i].done = -1;
    		ccTaskPool[i].taskId = 0;
    		ccTaskPool[i].readyId = -1;
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
  	int j, k;
  	int terminate = 1;
  	va_list ap;
  	va_start(ap,paraN);

  	while(taskIndex < (BK_NUM*BP_NUM) && terminate == 1){
    		if(ccTaskPool[taskIndex].ready == 0 && ccTaskPool[taskIndex].readyId == -1){
        		// **Add here**: renew task table, set the bit of task ID on
        		// **Add here**: get_ID()
 			ccTaskPool[taskIndex].ready = 1;
			ccTaskPool[taskIndex].taskId = taskIndex+1;
			ccTaskPool[taskIndex].done = 1;

			if(round_count > 0) {
                		ccTaskPool[taskIndex].readyId = taskId;
        		}else{
                		lastEmptyTask = taskIndex;
        		}
        		round_count ++;

			taskId = taskIndex;

			for(j = 0; j < paraN; j++){ // set parameters
	  			int type = va_arg(ap, enum mytypes);
	    			switch(type){
	      				case INT:
						if(j == 0) ccTaskPool[taskIndex].thread = va_arg(ap, int);
						if(j == 1) ccTaskPool[taskIndex].block = va_arg(ap, int);
						if(j == 2) ccTaskPool[taskIndex].sharemem = va_arg(ap, int);
                				if(j == 3) ccTaskPool[taskIndex].sync = va_arg(ap, int);
						if(j == 4) ccTaskPool[taskIndex].funcId = va_arg(ap, int);
						if(j > 4)  ccTaskPool[taskIndex].para[j-inParaNum] = va_arg(ap, int*);
	        				break;
	      				case FLOAT:
	        				ccTaskPool[taskIndex].para[j-inParaNum] = va_arg(ap, float*);
	        				break;
	      				case DOUBLE:
	        				ccTaskPool[taskIndex].para[j-inParaNum] = va_arg(ap, double*);
	        				break;
	      				case LONG:
	        				ccTaskPool[taskIndex].para[j-inParaNum] = va_arg(ap, long*);
	        				break;
	      				default:
	       	 				break;
	    			} // End switch
//	  } // End else
			} // End for paraN
				checkCudaErrors(cudaMemcpyAsync(ggTaskPool+taskIndex, ccTaskPool+taskIndex, 
					sizeof(gTaskStruct), cudaMemcpyHostToDevice, runtime_stream)); 
        			terminate = 0; 
     		} // end if cTaskPool
     		taskIndex++;

     		if(taskIndex == (BK_NUM*BP_NUM) && round_count > 0){
	  		ccTaskPool[lastEmptyTask].readyId = taskId;
          		checkCudaErrors(cudaMemcpyAsync((int*)&ggTaskPool[lastEmptyTask].readyId, 
				(int*)&ccTaskPool[lastEmptyTask].readyId,sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
	  		checkCudaErrors(cudaStreamSynchronize(runtime_stream));
	  		barrierCount ++;
	  		round_count = 0;
     		}
     		if(taskIndex == (BK_NUM*BP_NUM)){
          		checkCudaErrors(cudaMemcpyAsync(ccTaskPool, ggTaskPool, (BK_NUM*BP_NUM)*sizeof(gTaskStruct), 
				cudaMemcpyDeviceToHost, runtime_stream));
          		checkCudaErrors(cudaStreamSynchronize(runtime_stream));
          		taskIndex = 0;
     		}
  	} // end while i < BK_NUM*BP_NUM

  	va_end(ap);
  	return taskId;
}

void waitAll(int num_tasks){
	*totalScheTasks = 0;

	ccTaskPool[lastEmptyTask].readyId = taskId;
        checkCudaErrors(cudaMemcpyAsync((int*)&ggTaskPool[lastEmptyTask].readyId, (int*)&ccTaskPool[lastEmptyTask].readyId,
                                sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
        checkCudaErrors(cudaStreamSynchronize(runtime_stream));
	round_count = 0;

	int i;
	while(*totalScheTasks < num_tasks){
		checkCudaErrors(cudaMemcpyAsync(totalScheTasks, totalScheTasksDev, sizeof(int), cudaMemcpyDeviceToHost, runtime_stream));
		checkCudaErrors(cudaStreamSynchronize(runtime_stream));
	}
        *totalScheTasks = 0;
	checkCudaErrors(cudaMemcpyAsync(totalScheTasksDev, totalScheTasks, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
        checkCudaErrors(cudaStreamSynchronize(runtime_stream));

        taskIndex = 0;
	taskId = 0;
	lastEmptyTask = 0;

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
