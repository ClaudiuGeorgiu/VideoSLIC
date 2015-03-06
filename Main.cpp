/****************************************************************************/
/*                                                                          */
/* Original algorithm: http://ivrg.epfl.ch/research/superpixels             */
/* Original OpenCV implementation: http://github.com/PSMM/SLIC-Superpixels  */
/*                                                                          */
/* Paper: "Optimizing Superpixel Clustering for Real-Time                   *//*         Egocentric-Vision Applications"                                  */
/*        http://www.isip40.it/resources/papers/2015/SPL_Pietro.pdf         */
/*                                                                          */
/****************************************************************************/

#include "SLIC.h"

/* OpenCV libraries for video and image
   elaborations. */
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

/* Chrono library for measuring time 
   performances. */
#include <boost/chrono.hpp>

#include <vector>

using namespace std;
using namespace cv;

/* Function performing SLIC algorithm on a video sequence. */
int VideoSLIC(
	VideoCapture&        capturedVideo,
	unsigned             superpixelNumber,
	unsigned             spatialDistanceWeight,
	bool                 connectedFrames,
	SLICElaborationMode  SLICMode,
	unsigned             iterationNumber,
	double               errorThreshold,
	VideoElaborationMode videoMode,
	unsigned             keyFramesRatio,
	double               GaussianStdDev
	);

int main(int argc, char *argv[])
{
	/* Video source location. */
	const string videoLocation = (argc == 2) ? argv[1] : "C:\\Users\\Claudiu\\Desktop\\Video Data Set\\EDSH1.avi";

	/* Output window name. */
	const string windowName = "VideoSLIC";

	/* Declare a container for the video and try to import the video from
	   a specific location. */
	VideoCapture capturedVideo(videoLocation);

	/* Check if the video was correctly imported into the program. */
	if (capturedVideo.isOpened() == false)
	{
		/* Send an error message and then close the program if there was
		   an error with the video capture. */
		cout << "\nSorry, there was an error with video capturing.\n";
		return -1;
	}

	/* SLIC algorithm parameters. */
	unsigned spatialDistanceWeight = 30;
	unsigned superpixelNumber      = 1000;
	/* There's no need to set both iterationNumber and errorThreshold,
	   it's enough to set only one of them and set the other one to zero.
	   Which of the two values will be used for video elaboration depends
	   on the value of the field SLICMode. */
	unsigned iterationNumber       = 10;
	double   errorThreshold        = 0.25;

	/* Compute SLIC algorithm step for later use in generating Gaussian noise. */
	const unsigned videoWidth  = static_cast<unsigned>(capturedVideo.get(CV_CAP_PROP_FRAME_WIDTH));
	const unsigned videoHeight = static_cast<unsigned>(capturedVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	unsigned       stepSLIC    = static_cast<unsigned>(sqrt((videoHeight * videoWidth) / superpixelNumber) + 0.5);

	/* Parameters used when applying SLIC algorithm to video sequences. */
	/* Decide if frames are to be processed independently or the result
	   obtained after elaborating one frame should be used to initialize 
	   the next frame (more details are found in the paper). */
	bool                 connectedFrames = true;
	SLICElaborationMode  SLICMode        = ERROR_THRESHOLD;
	VideoElaborationMode VideoMode       = KEY_FRAMES_NOISE;
	/* Use a key frame every keyFramesRatio frames. */
	unsigned             keyFramesRatio  = 30;
	/* Standard deviation of the Gaussian noise. */
	double               GaussianStdDev  = static_cast<double>(stepSLIC / 5);

	/* Call function to perform SLIC algorithm operations on video. */
	VideoSLIC(
		capturedVideo,
		superpixelNumber,
		spatialDistanceWeight,
		connectedFrames,
		SLICMode,
		iterationNumber,
		errorThreshold,
		VideoMode,
		keyFramesRatio,
		GaussianStdDev);

	return 0;
}

int VideoSLIC(
	VideoCapture&        capturedVideo,
	unsigned             superpixelNumber,
	unsigned             spatialDistanceWeight,
	bool                 connectedFrames,
	SLICElaborationMode  SLICMode,
	unsigned             iterationNumber,
	double               errorThreshold,
	VideoElaborationMode videoMode,
	unsigned             keyFramesRatio,
	double               GaussianStdDev
	)
{
	/* Get video width and height. */
	const unsigned videoWidth  = static_cast<unsigned>(capturedVideo.get(CV_CAP_PROP_FRAME_WIDTH));
	const unsigned videoHeight = static_cast<unsigned>(capturedVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

	/* Compute the sampling step and round to the nearest integer. */
	unsigned stepSLIC = static_cast<unsigned>(sqrt((videoHeight * videoWidth) / superpixelNumber) + 0.5);

	/* Create an object for SLIC algorithm operations. */
	SLIC* SLICFrame = new SLIC();

	/* A container which will hold a video frame for
	   the time necessary for its elaboration. */
	Mat currentFrame;

	/* Video frames counter. */
	unsigned framesNumber = 0;

	/* Debug data. */
	double totalTime    = 0;
	double totalTime2   = 0;
	double avgTime      = 0;
	double stdDeviation = 0;

	/* Output window name. */
	const string windowName = "VideoSLIC";

	/* Open a new window where to play the video. */
	namedWindow(windowName, CV_WINDOW_AUTOSIZE);

	/* Enter an infinite cycle to elaborate the video until its last frame. */
	while (true)
	{
		/* Measure time before processing a video frame. */
		boost::chrono::high_resolution_clock::time_point startPoint =
			boost::chrono::high_resolution_clock::now();

		/* Take the next frame from the video. */
		capturedVideo >> currentFrame;

		/* If there are no more frames in the video, break the loop because
		   the video has reached its end or there was an error. */
		if (currentFrame.data == NULL)
			break;

		/* Convert the frame from RGB to LAB color space
		   before SLIC elaboration. */
		cvtColor(currentFrame, currentFrame, CV_BGR2Lab);

		/* Perform the SLIC algorithm operations. */
		SLICFrame->createSuperpixels(
			currentFrame, stepSLIC, spatialDistanceWeight, iterationNumber, errorThreshold,
			SLICMode, videoMode, keyFramesRatio, GaussianStdDev, connectedFrames);
		//SLICFrame->enforceConnectivity(currentFrame);
		//SLICFrame->colorSuperpixels(currentFrame);

		/* Convert frame back to RGB. */
		cvtColor(currentFrame, currentFrame, CV_Lab2BGR);

		SLICFrame->drawClusterContours(currentFrame, Vec3b(0, 0, 255)/*, Rect(videoWidth / 2, 0, videoWidth / 2, videoHeight)*/);
		//SLICFrame->drawClusterCentres(currentFrame, Scalar(0, 0, 255));

		/* Measure time after processing a video frame. */
		boost::chrono::high_resolution_clock::time_point endPoint =
			boost::chrono::high_resolution_clock::now();

		/* Get frame processing time in milliseconds. */
		boost::chrono::duration<int, boost::milli> elapsedTime =
			boost::chrono::duration_cast<boost::chrono::milliseconds>(endPoint - startPoint);

		++framesNumber;
		//SLICFrame->drawInformation(currentFrame, framesNumber, elapsedTime.count());

		/* Show frame in the window. */
		imshow(windowName, currentFrame);

		/* Compute some statistics and print them on screen. */
		totalTime  += elapsedTime.count();
		totalTime2 += elapsedTime.count() * elapsedTime.count();
		avgTime     = totalTime / framesNumber;

		stdDeviation = sqrt(framesNumber * totalTime2 - totalTime * totalTime) / framesNumber;

		cout << "Frame: " << framesNumber << "   ex. time now: " << elapsedTime.count() << "   average ex. time: "
			<< avgTime << "   stdDev: " << stdDeviation << endl << endl;

		/* End program on ESC press. */
		if (cvWaitKey(1) == 27)
			break;
	}

	/* Free used memory before closing the program. */
	if (SLICFrame != NULL)
		delete SLICFrame;

	/* Close window after video processing. */
	cv::destroyAllWindows();

	return 0;
}