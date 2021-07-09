#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if(col<numCColumns && row<numCRows){
    float v=0.0f;
    for(int i=0; i<numAColumns; i++){
      v += A[row*numAColumns + i] * B[col+(numBColumns*i)];
    }
    
    C[row*numCColumns + col] = v;
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *) malloc(numCRows * numCColumns*sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA,numARows * numAColumns*sizeof(float));
  cudaMalloc(&deviceB,numBRows * numBColumns*sizeof(float));
  cudaMalloc(&deviceC,numCRows * numCColumns*sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,numARows * numAColumns*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,numBRows * numBColumns*sizeof(float),cudaMemcpyHostToDevice);
  
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  
  ////////////////////////////////////////////////
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  wbLog(TRACE, "Device ", 0, " name: ", deviceProp.name);
    wbLog(TRACE, " Computational Capabilities: ", deviceProp.major, ".",
          deviceProp.minor);
    wbLog(TRACE, " Maximum global memory size: ",
          deviceProp.totalGlobalMem);
    wbLog(TRACE, " Maximum constant memory size: ",
          deviceProp.totalConstMem);
    wbLog(TRACE, " Maximum shared memory size per block: ",
          deviceProp.sharedMemPerBlock);
    wbLog(TRACE, " Maximum block dimensions: ",
          deviceProp.maxThreadsDim[0], " x ", deviceProp.maxThreadsDim[1],
          " x ", deviceProp.maxThreadsDim[2]);
    wbLog(TRACE, " Maximum grid dimensions: ", deviceProp.maxGridSize[0],
          " x ", deviceProp.maxGridSize[1], " x ",
          deviceProp.maxGridSize[2]);
    wbLog(TRACE, " Warp size: ", deviceProp.warpSize);
  ///////////////////////////////////
  //@@ Initialize the grid and block dimensions here
  dim3 gridDim((int)(numCColumns-1)/32+1,(int)(numCColumns-1)/32+1,1);
  //dim3 blockDim(1024,1024,1);
  //dim3 gridDim(1,1,1);
  dim3 blockDim(32,32,1);
  
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<gridDim,blockDim>>>(deviceA,deviceB,deviceC, numARows,
                               numAColumns, numBRows,
                               numBColumns, numCRows,
                               numCColumns);

  cudaDeviceSynchronize();
  wbCheck(cudaGetLastError());
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC, numCRows*numCColumns*sizeof(float),cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying output memory to the CPU");
  


  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
