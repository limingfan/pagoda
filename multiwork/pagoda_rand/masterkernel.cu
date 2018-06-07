#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../../common/para.h"
#include "../../common/para.cuh"
#include "kernel.cuh"
#include "runtime.cuh"

__device__ void syncBlock(){
    asm volatile("bar.sync %0, %1;" :: "r"(syncID), "r"(threadNum));
}

__global__ void masterKernel(volatile int *done, volatile int *totalScheTasks, volatile gTaskStruct *gTaskPool){

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
  __shared__ volatile int doneBarId; // thread block counter
  __shared__ volatile int exit;
  extern __shared__ volatile int shared_mem[];
  int i;
  int threadDone;

  // Init warp pool  
  if((threadIdx.x & 0x1f) != 0)
    warpPoolDev[(threadIdx.x & 0x1f)].exec = 0;
  else
    warpPoolDev[(threadIdx.x & 0x1f)].exec = -1;

  barIDArray[(threadIdx.x & 0x0f)] = 0;

  taskPointer = 0;
  exit = 0;
   __threadfence_block();
	  
  if(threadIdx.x < warpSize){
    while(!(*done)){

      taskStartP = (taskPointer * BK_NUM + blockIdx.x);
       __threadfence_block();

      if(gTaskPool[taskStartP].done == 1){ // Checking ready flag for task scheduling
        barID = -1;
	smStartIndx = -1;
	doneCtr[taskPointer] = gTaskPool[taskStartP].thread*gTaskPool[taskStartP].block/warpSize;
	warpCtr = doneCtr[taskPointer];
        warpId = 0;

        if(gTaskPool[taskStartP].sync == 0){
	  //parallel scheduling
	  while(1){
            threadDone = 1;
            if(threadIdx.x > 0) {
              threadDone = 0;
              if(warpPoolDev[threadIdx.x].exec == 0){
                if(atomicSub((int*)&warpCtr, 1) > 0){
                  warpPoolDev[threadIdx.x].warpId = atomicAdd((int*)&warpId, 1)*warpSize;
                  warpPoolDev[threadIdx.x].bufferNum = taskStartP; // global pointer of task table
		  //printf("bufferNum:%d, %d\n", taskStartP, threadIdx.x);
                  warpPoolDev[threadIdx.x].SMindex = smStartIndx; // shared mem. index
                  warpPoolDev[threadIdx.x].barId = barID; // index of threadblock
                  warpPoolDev[threadIdx.x].threadNum = gTaskPool[taskStartP].thread; // num. of thread
                  warpPoolDev[threadIdx.x].taskId = taskPointer; // local pointer of task table
                  __threadfence_block(); // To make sure the exec. is worked after fence
                  warpPoolDev[threadIdx.x].exec = 1;
                  __threadfence_block(); //
                } // End atomic 
              }
            }
            if(warpCtr <= 0) threadDone = 1;
            if(__all(threadDone == 1) != 0){
              break;
            }

         }// End while(1)
       } // End if sync flag

       if(gTaskPool[taskStartP].sync != 0 && gTaskPool[taskStartP].sharemem == 0){ // sync bit is on
         for(i = 0; i < gTaskPool[taskStartP].block; i++){ // Schedule block by block
           // get barId
           doneBarId = 1;
           while(1){
             threadDone = 1;
             if(threadIdx.x < syncNum){
               threadDone = 0;
               if(barIDArray[threadIdx.x] == 0){
                 if(atomicSub((int*)&doneBarId, 1) > 0){
                   barIDArray[threadIdx.x] = gTaskPool[taskStartP].thread/warpSize; //num. of thread in one block
                   barID =  threadIdx.x;
                   //printf("before barId:%d\n", barID);
                   __threadfence_block();
                  } // End atomicSub
                } // End if barIDArray
              } // End if threadIdx
              if(doneBarId <= 0) threadDone = 1;
              if(__all(threadDone == 1) != 0){
                break;
              } // End if all
            } // End while 1

            // parallel warp scheduling
            warpCtr = gTaskPool[taskStartP].thread/warpSize;
            warpId = i*(gTaskPool[taskStartP].thread/warpSize);
            while(1){
              threadDone = 1;
              if(threadIdx.x > 0) {
                threadDone = 0;
                if(warpPoolDev[threadIdx.x].exec == 0){
                  if(atomicSub((int*)&warpCtr, 1) > 0){
                    warpPoolDev[threadIdx.x].warpId = atomicAdd((int*)&warpId, 1)*warpSize;
                    warpPoolDev[threadIdx.x].bufferNum = taskStartP; // global pointer of task table
                    warpPoolDev[threadIdx.x].SMindex = smStartIndx; // shared mem. index
                    warpPoolDev[threadIdx.x].barId = barID; // index of threadblock
                    //printf("after barId:%d\n", barID);
                    warpPoolDev[threadIdx.x].threadNum = gTaskPool[taskStartP].thread; // num. of thread
                    warpPoolDev[threadIdx.x].taskId = taskPointer; // local pointer of task table
                    __threadfence_block(); // To make sure the exec. is worked after fence
                    warpPoolDev[threadIdx.x].exec = 1;
                    __threadfence_block(); //
                  } // End atomic 
                } // End if warpPoolDev
              } // End if threadIdx
              if(warpCtr <= 0) threadDone = 1;
              if(__all(threadDone == 1) != 0){
                break;
              }
            }// End while(1)
          } // End for
        } // End if sync flag
	gTaskPool[taskStartP].done = 0; // reset flag whenever task scheduling has been done
								
      } // End if ready flag
			
      taskPointer++; // renew the local pointer of task table
      if(taskPointer == BP_NUM)
	taskPointer = 0;
			
      }// End while done
      exit = 1;
        __threadfence_block();

    }// End if thread < 32

#if 1
    else{
      while(!exit){
      //while(!(*done)){
	if(warpPoolDev[warpIdxx].exec == 1){
#if 0
	  if(gTaskPool[warpPoolDev[warpIdxx].bufferNum].sync){
	    syncID = warpPool[warpIdxx].barId;
	    threadNum = warpPool[warpIdxx].threadNum;
	    //syncBlock();
	  } // End sync if
#endif	
	// kernel running here
#if 1
	 switch(gTaskPool[warpPoolDev[warpIdxx].bufferNum].funcId){
	   case 0:
	     mult_gpu((int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[0], 
			(int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[1], 
			(int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[2], 
			(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[3],
			(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[4],
			warpPoolDev[warpIdxx].warpId);
	     break;  
           case 1:
             get_pixel((int*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[0], 
                        (float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[1],
			(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[2],
                        (int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[3], 
                        warpPoolDev[warpIdxx].warpId);
             break;
#if 1   
	   case 2:
             FBCore((float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[0], 
                        (float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[1], 
                        (float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[2], 
                        (float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[3],
  			(float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[4],
                        (float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[5],
                        (float*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[6],
			(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[7],
                        (int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[8],
                        warpPoolDev[warpIdxx].warpId, warpPoolDev[warpIdxx].barId);
             break;   
#endif
	  case 3:
	     des_encrypt_dev((uint32*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[0],
                        (uint32*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[1],
                        (uint8*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[2],
                        (uint8*)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[3],
                        (int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[4],
			(int)gTaskPool[warpPoolDev[warpIdxx].bufferNum].para[5],
                        warpPoolDev[warpIdxx].warpId);
             break;

 
         }
#endif
	if((threadIdx.x & 0x1f) == 0){

	  // release barId
          if(gTaskPool[warpPoolDev[warpIdxx].bufferNum].sync != 0){
                atomicSub((int*)&barIDArray[warpPoolDev[warpIdxx].barId], 1);
          }

//	  printf("device:%d, %d\n", warpPoolDev[warpIdxx].bufferNum, warpIdxx);
	  if(atomicSub((int*)&doneCtr[warpPoolDev[warpIdxx].taskId], 1) == 1){ // when all warps in a task have been done
	    gTaskPool[warpPoolDev[warpIdxx].bufferNum].ready = 0; //unset the ready flag
	    atomicAdd((int*)totalScheTasks,1); //update the global task counter
          }

	  warpPoolDev[warpIdxx].exec = 0;
	  __threadfence_block();

	} // End if exec
      } // End if threadIdx.x
    } // End while done
  } // End else
#endif
}
