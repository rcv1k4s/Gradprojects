#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "timer.h"
#include "utils.h"



__global__ void gaussian_blur(int* inputChannel,
                   int* outputChannel,
                   int numRows, int numCols,
                   int* filter, const int filterWidth,const int s, int oRows, int oCols)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //


   int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_1D_pos = y * oCols + x;

   if ( x >=oCols ||y >= oRows )
   {
       return;
   }

   int sum=0;
//printf("gloc=%d =>threadId.x=%d,threadId.y=%d,blockId.x=%d,blocId.y=%d,position.x=%d,position.y=%d\n",thread_1D_pos,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,threadIdx.x +(blockIdx.x* blockDim.x),threadIdx.y +(blockIdx.y* blockDim.y));
int kidx=0;

   for(int r=0; r<filterWidth;++r){
        for(int c=0; c<filterWidth;++c){
        
            int idx=(y*s+r)*numCols+x*s+c;
            
        int filter_value=filter[kidx++];
        sum+=filter_value*inputChannel[idx];
   
        }
    }
    outputChannel[thread_1D_pos]=sum;
    
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}




__global__
void gaussian_blur_row(  int*  inputChannel,
                    int*  outputChannel,
                   int numRows, int numCols,
                    int*  filter, const int filterWidth )
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
 
    
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
    int absolute_image_position_x=thread_2D_pos.x;
    int absolute_image_position_y=thread_2D_pos.y;
   if ( absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows )
   {
       return;
   }
   
   int sum=0;
   int i=-filterWidth/2;
   int j=filterWidth/2;
   if(thread_2D_pos.x < filterWidth/2)
        i=-thread_2D_pos.x;
    if((thread_2D_pos.x+filterWidth/2)>(numCols-1))
        j=numCols-1-thread_2D_pos.x;
   for(int c=i; c<=j;++c){
       // for(int c=-filterWidth/2; c<=filterWidth/2;++c){
            int xIdx=absolute_image_position_x+c;
            int idx=(thread_2D_pos.y)*numCols+xIdx;
            int filter_value=filter[c+filterWidth/2];
            sum+=filter_value*(inputChannel[idx]);
        //}
    }
    outputChannel[thread_1D_pos]=sum;
    
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}
__global__
void gaussian_blur_col(int*  inputChannel,
                    int*  outputChannel,
                   int numRows, int numCols,
                    int*  filter, const int filterWidth)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
 
    
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
    int absolute_image_position_x=thread_2D_pos.x;
    int absolute_image_position_y=thread_2D_pos.y;
   if ( absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows )
   {
       return;
   }
   
   int sum=0;
   int i=-filterWidth/2;
   int j=filterWidth/2;
   if(thread_2D_pos.y < filterWidth/2)
        i=-thread_2D_pos.y;
    if((thread_2D_pos.y+filterWidth/2)>(numRows-1))
        j=numRows-1-thread_2D_pos.y;
   for(int r=i; r<=j;++r){
            int yIdx=absolute_image_position_y+r;
            int idx=(yIdx)*numCols+thread_2D_pos.x;
            int filter_value=filter[r+filterWidth/2];
            sum+=filter_value*(inputChannel[idx]);
            //if(idx%10==0)
               
    }
   // fflush();
    outputChannel[thread_1D_pos]=sum;
    
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}


int main(int argc, char **argv) {
  int  *d_inputImageRGBA;
  int *h_outputImageRGBA, *d_outputImageRGBA ;
  int *d_filter ;
 
const int numRows=5;
const int numCols=5;
const int filterWidth=3;
/*float img[numCols][numRows]={{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0},
				{9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0},
				{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0},
				{9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0},
				{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0},
				{9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0},
				{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0},
				{9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0},
				{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0}};*/
/*int h_inputImageRGBA[numCols*numRows]={1,6, 3, 8, 13,
				2,7,4,9,14,
				3,8,5,10,15,
				4,9,6,11,16,
				5,10,7,12,17};

//float filter[filterWidth]={1/16,1/8,1/16};
int h_filter[filterWidth*filterWidth]={1,0,1,0,0,0,1,0,1};*/

int h_inputImageRGBA[numCols*numRows]={1,6, 3, 8, 13,
				2,7,4,9,14,
				3,8,5,10,15,
				4,9,6,11,16,
				5,10,7,12,17};

//float filter[filterWidth]={1/16,1/8,1/16};
int h_filter[filterWidth*filterWidth]={1,0,1,0,0,0,1,0,1};


//h_filter=filter;
//h_inputImageRGBA=(int*) malloc ( sizeof(int)* numRows * numCols);
  //h_inputImageRGBA=filter;
 const long numPixels = numRows * numCols;

const int s=1;
const int oCol=(numCols-filterWidth)/s+1;
const int oRow=(numRows-filterWidth)/s+1;
const int oNumPixels=oCol*oRow;
h_outputImageRGBA=(int*) malloc ( sizeof(int)* oNumPixels);
//int test[oNumPixels]={0,0,0,0};
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc((void**)&d_inputImageRGBA, sizeof(int) * numPixels));
  checkCudaErrors(cudaMalloc((void**)&d_outputImageRGBA, sizeof(int) *oNumPixels));
  checkCudaErrors(cudaMemset(d_outputImageRGBA, 0, oNumPixels * sizeof(int))); //make sure no memory is left laying around
//printf("sizeOf d_inputImageRGBA= %d ,d_outputImageRGBA=%d, %d\n",sizeof(d_inputImageRGBA),sizeof(d_outputImageRGBA),numPixels * sizeof(int));
  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(int) * numPixels, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_filter,  sizeof(int) * filterWidth*filterWidth));
  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter,h_filter, sizeof(int) * filterWidth*filterWidth,cudaMemcpyHostToDevice));


  GpuTimer timer;
  timer.Start();
  //call the students' code
  
 //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(1,1);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(oCol/blockSize.x +1,oRow/blockSize.y +1);

 

  //TODO: Call your convolution kernel here 3 times, once for each color channel.

    /* gaussian_blur_row<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                                d_outputImageRGBA,
                                                numRows,
                                                numCols,
                                                d_filter,
                                                filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

     gaussian_blur_col<<<gridSize, blockSize>>>(d_outputImageRGBA,
                                                d_inputImageRGBA,
                                                numRows,
                                                numCols,
                                                d_filter,
                                                filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());*/
gaussian_blur<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                                d_outputImageRGBA,
                                                numRows,
                                                numCols,
                                                d_filter,
                                                filterWidth,s,oRow,oCol);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the blurred image

  //copy the output back to the host
//printf("sizeOf d_inputImageRGBA= %d ,d_outputImageRGBA=%d",sizeof(d_inputImageRGBA),sizeof(d_outputImageRGBA));
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(int) * oNumPixels, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaFree(d_inputImageRGBA));
checkCudaErrors(cudaFree(d_outputImageRGBA));
checkCudaErrors(cudaFree(d_filter));

for(int i=0;i<oNumPixels;i++){
	printf("\t%d",h_outputImageRGBA[i]);
	if(i%6==0)
		printf("\n");
}


  return 0;
}
