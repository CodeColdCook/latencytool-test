#include "opencv2/opencv.hpp"

#include "interface.h"
#include "image_converter.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <jetson-utils/videoSource.h>
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>

// in [0,1], i.e, what brightness level is the light/dark cutoff
#define THRESHOLD 0.5

imageConverter *image_cvt = NULL;
struct state
{
  // Readout
  // cv::VideoCapture *cap;
  videoSource *stream = NULL;
  cv::Mat graylevel;
  cv::Mat bgrframe;

  enum WhatToDo output_state;
  struct analysis control;
};

// aquire and publish camera frame
bool aquireFrame(state *s_cur)
{
  imageConverter::PixelType *nextFrame = NULL;
  if (!s_cur->stream->Capture(&nextFrame, 1000))
	{
		fprintf(stderr, "failed to capture next frame");
		return false;
	}

  if (!image_cvt->Resize(s_cur->stream->GetWidth(), s_cur->stream->GetHeight(), imageConverter::CVOutputFormat))
	{
		fprintf(stderr, "failed to resize camera image converter");
		return false;
	}

  if (!image_cvt->Convert(s_cur->bgrframe, imageConverter::CVOutputFormat, nextFrame))
	{
		fprintf(stderr, "failed to convert video stream frame to cv::Mat");
		return false;
	}

  // cv::imshow("test", s_cur->bgrframe);
  // cv::waitKey(1);
  
	return true;
}

static void *isetup(int camera)
{
  videoOptions video_options;
  video_options.width = 1920;
  video_options.height = 1280;
  video_options.frameRate = 30;
  video_options.numBuffers = 4;
  std::string resource_str = "csi://0";

  struct state *s = new struct state;
  s->stream = videoSource::Create(resource_str.c_str(), video_options);

  if (!s->stream->Open())
  {
    fprintf(stdout, "Camera loaded with backend failed\n");
    delete s->stream;
    delete s;
    return NULL;
  }

  fprintf(stdout, "Camera #%d loaded with backend \n", camera);

  image_cvt = new imageConverter();

	if (!image_cvt)
	{
		fprintf(stderr, "failed to create imageConverter");
		return 0;
	}


  double fps = 30;
  double width = video_options.frameRate;
  double height = video_options.height;
  fprintf(
      stderr,
      "nominal fps=%.0f width=%.0f height=%.0f \n",
      fps, width, height);

  if (setup_analysis(&s->control) < 0)
  {
    delete s->stream;
    delete s;
    return NULL;
  }
  s->output_state = DisplayLight;
  return s;
}

extern "C"
{

  void *setup_backend(int camera) { return isetup(camera); }

  enum WhatToDo update_backend(void *state)
  {
    struct state *s = (struct state *)state;
    // We return the opposite of the current camera state, and record/print
    // brightness transitions
    if (!aquireFrame(s))
      return s->output_state;

    // Record frame capture time *before* postprocessing, as though it
    // had zero cost.
    struct timespec capture_time;
    clock_gettime(CLOCK_MONOTONIC, &capture_time);

    cv::cvtColor(s->bgrframe, s->graylevel, cv::COLOR_BGR2GRAY);
    double level = cv::mean(s->graylevel)[0] / 255.0;

    s->output_state =
        update_analysis(&s->control, capture_time, level, THRESHOLD);
    return s->output_state;
  }

  void cleanup_backend(void *state)
  {
    if (state)
    {
      struct state *s = (struct state *)state;
      cleanup_analysis(&s->control);

      delete s->stream;
      delete s;
    }
  }
}
