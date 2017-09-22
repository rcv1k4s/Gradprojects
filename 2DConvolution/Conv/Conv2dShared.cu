#include <iostream>
#include <chrono>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#define TILESIZE 3

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

inline  void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void convolutionGPU(
float *d_Result,
float *d_Data,
int dataW,
int dataH
)
{
// Data cache: threadIdx.x , threadIdx.y
__shared__ float data[TILE_W + KERNEL_RADIUS * 2][TILE_W + KERNEL_RADIUS * 2];

// global mem address of this thread
const int gLoc = threadIdx.x +
IMUL(blockIdx.x, blockDim.x) +
IMUL(threadIdx.y, dataW) +
IMUL(blockIdx.y, blockDim.y) * dataW;

// load cache (32x32 shared memory, 16x16 threads blocks)
// each threads loads four values from global memory into shared mem
// if in image area, get value in global mem, else 0
int x, y; // image based coordinate

// original image based coordinate
const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);

// case1: upper left
x = x0 - KERNEL_RADIUS;
y = y0 - KERNEL_RADIUS;
if ( x < 0 || y < 0 )
data[threadIdx.x][threadIdx.y] = 0;
else
data[threadIdx.x][threadIdx.y] = d_Data[ gLoc - KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];

// case2: upper right
x = x0 + KERNEL_RADIUS;
y = y0 - KERNEL_RADIUS;
if ( x > dataW-1 || y < 0 )
data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
else
data[threadIdx.x + blockDim.x][threadIdx.y] = d_Data[gLoc + KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];

// case3: lower left
x = x0 - KERNEL_RADIUS;
y = y0 + KERNEL_RADIUS;
if (x < 0 || y > dataH-1)
data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
else
data[threadIdx.x][threadIdx.y + blockDim.y] = d_Data[gLoc - KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];

// case4: lower right
x = x0 + KERNEL_RADIUS;
y = y0 + KERNEL_RADIUS;
if ( x > dataW-1 || y > dataH-1)
data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
else
data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = d_Data[gLoc + KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];

__syncthreads();

// convolution
float sum = 0;
x = KERNEL_RADIUS + threadIdx.x;
y = KERNEL_RADIUS + threadIdx.y;
for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
sum += data[x + i][y + j] * d_Kernel[KERNEL_RADIUS + j] * d_Kernel[KERNEL_RADIUS + i];

d_Result[gLoc] = sum;
}


int main (int argc, char* argv[]) {

/***************************************************************Initialization********************************************************************/

  if(argc!=5)
  {
        printf("Not enough arguments"); 
        return 0;
  }

  int m = atoi(argv[1]);
  printf("M is  %d\n",m);

  int n = atoi(argv[2]); //TODO: atoi is an unsafe function
  printf("N is  %d\n",n);

 
  int k = atoi(argv[3]);
  printf("The kernel size is %d\n",k);

  int threadsperblock=atoi(argv[4]);
  printf("Threads per block are %d\n",threadsperblock);

  int totalblocks= (int)(m*n)/threadsperblock;

  int total_pixels= m*n;

  if((total_pixels)%threadsperblock!=0)
  {
        totalblocks= totalblocks+1;

  }

  int i,j;

  //printf("Total blocks are %d\n",totalblocks);

/*******************************************************Allocating Memory to the arrays***************************************************************/
  


 // int *image= new int [total_pixels];
 // int *kernel= new int[k*k];
  int *output_image= new int[total_pixels];

  int *d_image;
  int *d_outimage;
  int *d_kernel;



int image[numCols*numRows]={1,6, 3, 8, 13,
				2,7,4,9,14,
				3,8,5,10,15,
				4,9,6,11,16,
				5,10,7,12,17};

//int filter[filterWidth]={1/16,1/8,1/16};
int kernel[k*k]={1,0,1,0,0,0,1,0,1};

  uint32_t BLOCKSIZE= (TILESIZE);


// Memory Allocation//
std::chrono::time_point<std::chrono::system_clock> begin, end;

cudaMalloc((void**)&d_image, (total_pixels)*sizeof(int));
cudaMalloc((void**)&d_kernel, (k*k)*sizeof(int));
cudaMalloc((void**)&d_outimage, (total_pixels)*sizeof(int));


cudaMemcpy(d_outimage,output_image,(total_pixels)*sizeof(int),cudaMemcpyHostToDevice);


cudaMemcpy(d_image,image,(total_pixels)*sizeof(int),cudaMemcpyHostToDevice);

//cudaDeviceSynchronize();

cudaMemcpy(d_kernel,kernel,(k*k)*sizeof(int),cudaMemcpyHostToDevice);


const dim3 blocksize(BLOCKSIZE,BLOCKSIZE);
// Launch Kernel

const dim3 gridsize(n/TILESIZE +1,m/TILESIZE+1);
//printf("gridsize.x=%d, gridsize.y=%d\n",gridsize.x,gridsize.y);
begin = std::chrono::system_clock::now();

convolution<<<gridsize,blocksize>>>(d_image,d_kernel,n,m,k,d_outimage);

cudaDeviceSynchronize();

end = std::chrono::system_clock::now();

cudaMemcpy(output_image, d_outimage, (m*n)*sizeof(int),cudaMemcpyDeviceToHost);


int p=0;
for(i=0;i<total_pixels;i++)
{
	
	if(p==m)
	{
	printf("\n");
	p=0;
	}
	p++;
	printf("%f ", output_image[i]);
}
 std::chrono::duration<double> totaltime = (end-begin);

std::cout<<std::fixed<<" For array size " << n <<" The time required for convolution is "<<(totaltime.count())<<std::endl;

cudaFree(d_image);
cudaFree(d_kernel);
cudaFree(d_outimage);
free(image);
free(kernel);
free(output_image);

}

