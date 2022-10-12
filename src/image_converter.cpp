#include "image_converter.h"
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/cudaMappedMemory.h>

// constructor
imageConverter::imageConverter()
{
	mWidth  	  = 0;
	mHeight 	  = 0;
	mSizeInput  = 0;
	mSizeOutput = 0;

	mInputCPU = NULL;
	mInputGPU = NULL;

	mOutputCPU = NULL;
	mOutputGPU = NULL;
}


// destructor
imageConverter::~imageConverter()
{
	Free();	
}


// Free
void imageConverter::Free()
{
	if( mInputCPU != NULL )
	{
		CUDA(cudaFreeHost(mInputCPU));

		mInputCPU = NULL;
		mInputGPU = NULL;
	}

	if( mOutputCPU != NULL )
	{
		CUDA(cudaFreeHost(mOutputCPU));

		mOutputCPU = NULL;
		mOutputGPU = NULL;
	}
}

// Convert
bool imageConverter::Convert( cv::Mat& cvImg, imageFormat format, PixelType* imageGPU )
{
	if( !mInputCPU || !imageGPU || mWidth == 0 || mHeight == 0 || mSizeInput == 0 || mSizeOutput == 0 )
		return false;
	
	// perform colorspace conversion into the desired encoding
	// in this direction, we reverse use of input/output pointers
	if( CUDA_FAILED(cudaConvertColor(imageGPU, InternalFormat, mInputGPU, format, mWidth, mHeight)) )
	{
		fprintf(stderr, "failed to convert %ux%u image (from %s to %s) with CUDA", mWidth, mHeight, imageFormatToStr(InternalFormat), imageFormatToStr(format));
		return false;
	}

	// calculate size of the msg
	const size_t msg_size = imageFormatSize(format, mWidth, mHeight);

	// allocate msg storage
  cvImg = cv::Mat(cv::Size(mWidth, mHeight), CV_8UC3);

  // copy the converted image into the msg
	if (format != IMAGE_GRAY8)
	{
		memcpy(cvImg.data, mInputCPU, msg_size);
	}
	else 
	{
		memcpy(cvImg.data, mOutputCPU, msg_size);
	}
		
	// populate metadata
	cvImg.cols  = mWidth;
	cvImg.rows = mHeight;
	
	return true;
}


// Resize
bool imageConverter::Resize( uint32_t width, uint32_t height, imageFormat inputFormat )
{
	const size_t input_size  = imageFormatSize(inputFormat, width, height);
	const size_t output_size = imageFormatSize(InternalFormat, width, height);

	if( input_size != mSizeInput || output_size != mSizeOutput || mWidth != width || mHeight != height )
	{
		Free();

		if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputGPU, input_size) ||
		    !cudaAllocMapped((void**)&mOutputCPU, (void**)&mOutputGPU, output_size) )
		{
			fprintf(stdout, "failed to allocate memory for %ux%u image conversion", width, height);
			return false;
		}

		fprintf(stdout, "allocated CUDA memory for %ux%u image conversion", width, height);

		mWidth      = width;
		mHeight     = height;
		mSizeInput  = input_size;
		mSizeOutput = output_size;		
	}

	return true;
}