//! [includes]
 #include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdlib.h>
#include "timer.h"
#include "utils.h"

using namespace std;
using namespace cv;
Mat imageInputRGBA;
Mat imageOutputRGBA;
size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }





/*******  DEFINED IN student_func.cu *********/

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth,const int tilesize);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
const float* const h_filter, const size_t filterWidth);



int main(int argc, char **argv) {
  uchar4  *d_inputImageRGBA,*h_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
  float *d_filter,*h_filter ;
 
//const int numRows=1920;
//const int numCols=1080;
 int filterWidth=9;
int tilesize=8;
if( argc > 1)
    {
        filterWidth = atoi(argv[1]);
    }
if( argc > 2)
    {
        tilesize = atoi(argv[2]);
    }
//printf("tilesize=%d",tilesize);
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

String output_file = "Conv_output.jpg";

String imageName( "/media/ubuntu/myData/Conv/DSC_6637.JPG" ); // by default
    if( argc > 3)
    {
        imageName = argv[3];
    }
    //! [load]

    //! [mat]
    Mat image;
    //! [mat]

    //! [imread]
    image = imread( imageName, CV_LOAD_IMAGE_COLOR ); // Read the file
    //! [imread]

    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }




cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

 //allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
}

h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
h_filter = new float[filterWidth];
const size_t numPixels = numRows() * numCols();


const float blurKernelSigma = 2.;
float filterSum = 0.f; //for normalization

  for (int r = -filterWidth/2; r <= filterWidth/2; ++r) {
   // for (int c = -filterWidth/2; c <= filterWidth/2; ++c) {
      float filterValue = expf( -(float)( r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      h_filter[(r + filterWidth/2) ] = filterValue;
      filterSum += filterValue;
    //}
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -filterWidth/2; r <= filterWidth/2; ++r) {
  //  for (int c = -filterWidth/2; c <= filterWidth/2; ++c) {
      h_filter[(r + filterWidth/2) ] *= normalizationFactor;
	//printf("filtervalue =%f for index=%d \t",h_filter[(r + filterWidth/2) * filterWidth + c + filterWidth/2],(r + filterWidth/2) * filterWidth + c + filterWidth/2);
   // }
//printf("\n");
}




//float filter[filterWidth]={1/16,1/8,1/16};
//float h_filter[filterWidth*filterWidth]={1.0,4.0,6.0,4.0,1.0,4.0,16.0,24.0,16.0,4.0,6.0,24.0,36.0,24.0,6.0,4.0,16.0,24.0,16.0,4.0,1.0,4.0,6.0,4.0,1.0};

//h_filter=(int*) malloc ( sizeof(int)* filterWidth);
//h_filter=filter;
//h_inputImageRGBA=(int*) malloc ( sizeof(int)* numRows * numCols);
  //h_inputImageRGBA=filter;
 //const long numPixels = numRows * numCols;
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc((void**)&d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc((void**)&d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(d_outputImageRGBA, 0, sizeof(uchar4) * numPixels)); //make sure no memory is left laying around
//printf("sizeOf d_inputImageRGBA= %d ,d_outputImageRGBA=%d, %d\n",sizeof(d_inputImageRGBA),sizeof(d_outputImageRGBA),numPixels * sizeof(int));
  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(int) * numPixels, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&d_redBlurred,    sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_greenBlurred,  sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_blueBlurred,   sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_blueBlurred, 0, sizeof(unsigned char) * numPixels));


allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

    //checkCudaErrors(cudaMalloc((void**)&d_filter,  sizeof(int) * filterWidth*filterWidth));
  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  //checkCudaErrors(cudaMemcpy(d_filter,h_filter, sizeof(int) * filterWidth*filterWidth,cudaMemcpyHostToDevice));


  GpuTimer timer;
  timer.Start();
  //call the students' code
  your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
				d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth,tilesize);

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
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
//printf("in main value.x=%f\t,value.y=%f\tvalue.z=%f",static_cast<float>(h_outputImageRGBA[500].x),static_cast<float>(h_outputImageRGBA[500].y),static_cast<float>(h_outputImageRGBA[500].z));
cv::Mat output(numRows(), numCols(), CV_8UC4, h_outputImageRGBA);

  cv::Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
  //output the image
cv::imwrite(output_file.c_str(), imageOutputBGR);

checkCudaErrors(cudaFree(d_inputImageRGBA));
checkCudaErrors(cudaFree(d_outputImageRGBA));
checkCudaErrors(cudaFree(d_filter));

/*for(int i=0;i<sizeof(h_inputImageRGBA)/4;i++){
	printf("\t%d",h_inputImageRGBA[i]);
	if(i%6==0)
		printf("\n");
}*/


  return 0;
}
