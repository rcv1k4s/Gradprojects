
#include "utils.h"
#include <stdio.h>

#define TILE_SIZE    8
#define MASK_WIDTH   9



__global__
void gaussian_blur_row(  const unsigned char* const N,
                    unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
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
    int col=thread_2D_pos.x;
    int row=thread_2D_pos.y;
   /*if ( absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows )
   {
       return;
   }*/
   
int n = MASK_WIDTH/2; 
  __shared__ float  N_ds[TILE_SIZE][TILE_SIZE + MASK_WIDTH - 1]; 

//loading left halo
int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x; 
  if (threadIdx.x >= blockDim.x - n) {
	
		N_ds[threadIdx.y][threadIdx.x - (blockDim.x - n)] =(halo_index_left < 0) ? 0 : static_cast<float>(N[thread_2D_pos.y * numCols+halo_index_left]); 
  } 

// loading internal elements
if(row<numRows&&col<numCols)
	N_ds[threadIdx.y][n + threadIdx.x] = static_cast<float>(N[thread_1D_pos]); 

//loading right halo
int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x; 
  if (threadIdx.x < n) { 
	
		N_ds[threadIdx.y][n + blockDim.x + threadIdx.x] =(halo_index_right >= numCols) ? 0 : static_cast<float>(N[thread_2D_pos.y * numCols+halo_index_right]); 
  } 
  __syncthreads(); 
   float sum=0.0f;

   for(int i=0;i<filterWidth;i++){
	//int idx=(thread_2D_pos.y)*numCols+xIdx;
	//int filter_value=filter[c+radius];
	sum+=N_ds[threadIdx.y][threadIdx.x+i]*filter[i];

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
void gaussian_blur_col(const unsigned char* const N,
                    unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
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
    int col=thread_2D_pos.x;
    int row=thread_2D_pos.y;
   /*if ( absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows )
   {
       return;
   }*/
   
   float sum=0.0f;


int n = MASK_WIDTH/2; 
  __shared__ float  N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE ]; 

//loading top halo
int halo_index_top = (blockIdx.y - 1)*blockDim.y + threadIdx.y; 
  if (threadIdx.y >= blockDim.y - n) { 
	N_ds[threadIdx.y - (blockDim.y - n)][threadIdx.x] =(halo_index_top < 0) ? 0 : static_cast<float>(N[halo_index_top * numCols+thread_2D_pos.x]); 
  } 

// loading internal elements
if(row<numRows&&col<numCols)
	N_ds[n + threadIdx.y][threadIdx.x] = static_cast<float>(N[thread_1D_pos]); 

//loading bottom halo
int halo_index_bottom = (blockIdx.y + 1)*blockDim.y + threadIdx.y; 
  if (threadIdx.y < n) { 
	N_ds[n + blockDim.y +threadIdx.y][threadIdx.x] =(halo_index_bottom >= numRows) ? 0 : static_cast<float>(N[halo_index_bottom * numCols+thread_2D_pos.x]); 
  } 
  __syncthreads(); 
   // fflush();


	for(int i=0;i<filterWidth;i++){
		//int idx=(thread_2D_pos.y)*numCols+xIdx;
		//int filter_value=filter[c+radius];
		sum+=N_ds[threadIdx.y+i][threadIdx.x]*filter[i];

	}


    outputChannel[thread_1D_pos]=sum;
    
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}



//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  const int absolute_image_position_x = thread_2D_pos.x;
  const int absolute_image_position_y = thread_2D_pos.y;
  if ( absolute_image_position_x >= numCols ||
       absolute_image_position_y >= numRows )
  {
      return;
  }
  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
//if(thread_1D_pos==500)
	//printf("in kernel value.x=%f\t,value.y=%f\tvalue.z=%f",static_cast<float>(outputImageRGBA[500].x),static_cast<float>(outputImageRGBA[500].y),static_cast<float>(outputImageRGBA[500].z));
}



unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof( float) * filterWidth * filterWidth));
  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth, cudaMemcpyHostToDevice));

}


void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(TILE_SIZE, TILE_SIZE);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(numCols/blockSize.x+1,numRows/blockSize.y+1);

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                              numRows,
                                              numCols,
                                              d_red,
                                              d_green,
                                              d_blue);
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur_row<<<gridSize, blockSize>>>(d_red,
                                         d_redBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur_row<<<gridSize, blockSize>>>(d_green,
                                         d_greenBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur_row<<<gridSize, blockSize>>>(d_blue,
                                         d_blueBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
gaussian_blur_col<<<gridSize, blockSize>>>(d_redBlurred,
					d_red,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur_col<<<gridSize, blockSize>>>(d_greenBlurred,d_green,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur_col<<<gridSize, blockSize>>>(d_blueBlurred,
					d_blue,                                         
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);


  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_red,
                                             d_green,
                                             d_blue,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}






