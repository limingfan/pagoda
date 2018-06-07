#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../../common/para.h"
#include "../../common/para.cuh"
#include "kernel.cuh"
#include "runtime.cuh"

__global__ void masterKernel(volatile int *done, volatile int *totalScheTasks, volatile gTaskStruct *gTaskPool)
{
	int warpIdxx = (threadIdx.x/warpSize);
	__shared__ volatile int barID; // the ID for bar.sync
	__shared__ volatile int smStartIndx;  // the start pointer for free memory region of shared memory
	__shared__ volatile int doneCtr[BP_NUM]; // number of warp in a task
	__shared__ volatile gWarpStruct warpPoolDev[BP_NUM]; // warpPool
	int taskPointer; //local pointer of task table
  	int taskStartP; //global pointer of task table
  	__shared__ volatile int barIDArray[syncNum]; 
  	__shared__ volatile int sharedTree[SH_TREE_SIZE]; //shared mem data structure
  	__shared__ volatile int warpCtr;
  	__shared__ volatile int warpId;
  	__shared__ volatile int exit;
  	extern __shared__ volatile int shared_mem[];
  	int i;
  	int threadDone;

  	// Init warp pool  
  	if((threadIdx.x & 0x1f) != 0)
    		warpPoolDev[(threadIdx.x & 0x1f)].exec = 0;
  	else
    		warpPoolDev[(threadIdx.x & 0x1f)].exec = -1;

  	taskPointer = 0;
  	exit = 0;
  	doneCtr[(threadIdx.x & 0x1f)] = 0;

   	__threadfence_block();
	  
  	if(threadIdx.x < warpSize){
    		while(!(*done)){

      			taskStartP = (taskPointer * BK_NUM + blockIdx.x);
       			__threadfence_block();
      			if(gTaskPool[taskStartP].readyId != -1 && doneCtr[taskPointer] == 0){
      				if(gTaskPool[gTaskPool[taskStartP].readyId].done == 1){
        				barID = -1;
					smStartIndx = -1;
					doneCtr[taskPointer] = gTaskPool[gTaskPool[taskStartP].readyId].thread * 
							gTaskPool[gTaskPool[taskStartP].readyId].block/warpSize;
					warpCtr = doneCtr[taskPointer];
        				warpId = 0;
					//parallel scheduling
					while(1){
          					threadDone = 1;
          					if(threadIdx.x > 0) {
            						threadDone = 0;
            						if(warpPoolDev[threadIdx.x].exec == 0){
              							if(atomicSub((int*)&warpCtr, 1) > 0){
                							warpPoolDev[threadIdx.x].warpId = atomicAdd((int*)&warpId, 1)*warpSize;
									// global pointer of task table
                							warpPoolDev[threadIdx.x].bufferNum = gTaskPool[taskStartP].readyId;
                							warpPoolDev[threadIdx.x].SMindex = smStartIndx; // shared mem. index
                							warpPoolDev[threadIdx.x].barId = barID; // index of threadblock
									// num. of thread
                							warpPoolDev[threadIdx.x].threadNum = gTaskPool[gTaskPool[taskStartP].readyId].thread;
                							warpPoolDev[threadIdx.x].taskId = taskPointer; // local pointer of task table
                							__threadfence_block(); // To make sure the exec. is worked after fence
                							warpPoolDev[threadIdx.x].exec = 1;
                							__threadfence_block(); 
              							} // End atomic 
            						}
          					}
          					if(warpCtr <= 0) threadDone = 1;
          					if(__all(threadDone == 1) != 0){
           					 	break;
          					}

       					}// End while(1)

					//gTaskPool[taskStartP-1].done = 0; // reset flag whenever task scheduling has been done
					gTaskPool[gTaskPool[taskStartP].readyId].done = 0;
					gTaskPool[taskStartP].readyId = -1;
								
      				} // End if ready flag
      			}			

      			taskPointer++; // renew the local pointer of task table
      			if(taskPointer == BP_NUM)
				taskPointer = 0;
			
      		}// End while done
      		exit = 1;
      		__threadfence_block();
    	}// End if thread < 32

#if 1
    	else{
      	//while(!(*done)){
      		while(!exit){
			if(warpPoolDev[warpIdxx].exec == 1){
			// kernel running here
			mult_gpu((int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[0],
                  		(int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[1],
                  		(int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[2],
                  		(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[3],
                  		(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[4],
                  		warpPoolDev[warpIdxx].warpId);


				if((threadIdx.x & 0x1f) == 0){
	  				if(atomicSub((int*)&doneCtr[warpPoolDev[warpIdxx].taskId], 1) == 1){ // when all warps in a task have been done
	    					__threadfence_system();
	    					gTaskPool[warpPoolDev[warpIdxx].bufferNum].ready = 0; //unset the ready flag
	    
	    					atomicAdd((int*)totalScheTasks,1); //update the global task counter
          				}

	  				warpPoolDev[warpIdxx].exec = 0;
	  				__threadfence_block();

				} // End if threadIdx.x
      			} // End if exec
    		} // End while done
  	} // End else
#endif
}
