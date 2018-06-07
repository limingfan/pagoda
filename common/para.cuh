typedef struct{
  volatile int exec;
  volatile int warpId;
  volatile int taskId;

  volatile int bufferNum;
  volatile int SMindex;
  volatile int barId;
  volatile int threadNum;
}gWarpStruct;

typedef struct{
  volatile int ready;
  volatile int taskId;
  volatile int done;
  volatile int thread;
  volatile int block;
  volatile int sharemem;
  volatile int sync;
  volatile int funcId;
  volatile int readyId;
  volatile void *para[paraNum];
}cTaskStruct;

typedef struct{
  volatile int ready;
  volatile int taskId;
  volatile int done;
  volatile int thread;
  volatile int block;
  volatile int sharemem;
  volatile int sync;
  volatile int funcId;
  volatile int readyId;
  volatile void *para[paraNum];
}gTaskStruct;
