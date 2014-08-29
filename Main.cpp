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

int main(int argc, char *argv[])
{
	/* Video source location. */
	const string videoLocation = (argc == 2) ? argv[1] : "C:\\Users\\Claudiu\\Desktop\\Video Data Set\\EDSH1.avi";

	/* Output window name. */
	const string windowName = "Captured video";

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

	/* Get video width and height. */
	const unsigned videoWidth  = static_cast<unsigned>(capturedVideo.get(CV_CAP_PROP_FRAME_WIDTH));
	const unsigned videoHeight = static_cast<unsigned>(capturedVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

	/* A container which will hold a video frame for
	   the necessary time for its elaboration. */
	Mat currentFrame;

	/* SLIC algorithm parameters. */
	unsigned spatialDistanceWeight = 30;
	unsigned superpixelNumber      = 1000;

	/* Round the sampling step to the nearest integer. */
	unsigned stepSLIC = static_cast<unsigned>(sqrt((videoHeight * videoWidth) / superpixelNumber) + 0.5);

	/* Video frames counter. */
	unsigned framesNumber = 0;

	/* Open a new window where to play the imported video. */
	namedWindow(windowName, CV_WINDOW_AUTOSIZE);

	SLIC* SLICVideoElaboration = new SLIC();

	/* Debug data. */
	double totalTime    = 0;
	double totalTime2   = 0;
	double avgTime      = 0;
	double stdDeviation = 0;

	/* Enter an infinite cycle to elaborate the video until its last frame. */
	while (true)
	{
		boost::chrono::high_resolution_clock::time_point startPoint =
			boost::chrono::high_resolution_clock::now();

		/* Re-initialize SLIC each 30 frames. */
		if ((framesNumber != 0) && (framesNumber % 30 == 0))
		{
			if (SLICVideoElaboration != NULL)
				delete SLICVideoElaboration;
			SLICVideoElaboration = new SLIC();
		}

		/* Take the next frame from the video. */
		capturedVideo >> currentFrame;

		/* If there are no more frames in the video, break the loop because
		   the video has reached its end or there was an error. */
		if (currentFrame.data == NULL)
			break;

		/* Convert the frame from RGB to YCrCb color space
		   before SLIC elaboration. */
		cvtColor(currentFrame, currentFrame, CV_BGR2YCrCb);

		/* Perform the SLIC algorithm operations. */
		SLICVideoElaboration->createSuperpixels(
			currentFrame, stepSLIC, spatialDistanceWeight, false);
		SLICVideoElaboration->enforceConnectivity(currentFrame);
		//SLICVideoElaboration->colorSuperpixels(currentFrame);

		/* Convert frame back to RGB. */
		cvtColor(currentFrame, currentFrame, CV_YCrCb2BGR);

		SLICVideoElaboration->drawClusterContours(currentFrame, Vec3b(0, 0, 255)/*, Rect(videoWidth / 2, 0, videoWidth / 2, videoHeight)*/);

		boost::chrono::high_resolution_clock::time_point endPoint =
			boost::chrono::high_resolution_clock::now();

		boost::chrono::duration<int, boost::milli> elapsedTime =
			boost::chrono::duration_cast<boost::chrono::milliseconds>(endPoint - startPoint);

		++framesNumber;
		//SLICVideoElaboration->drawInformation(currentFrame, framesNumber, elapsedTime.count());

		/* Show frame in the window. */
		imshow(windowName, currentFrame);

		totalTime  += elapsedTime.count();
		totalTime2 += elapsedTime.count() * elapsedTime.count();
		avgTime     = totalTime / framesNumber;

		stdDeviation = sqrt(framesNumber * totalTime2 - totalTime * totalTime) / framesNumber;

		cout << "Frame: " << framesNumber << "   ora: " << elapsedTime.count() << "   tempo medio: "
			<< avgTime << "   stdDev: " << stdDeviation << endl << endl;

		/* End program on ESC press. */
		if (cvWaitKey(1) == 27)
			break;
	}

	/* Free used memory before closing the program. */
	if (SLICVideoElaboration != NULL)
		delete SLICVideoElaboration;

	/* Close window after video processing. */
	cv::destroyAllWindows();

	return 0;
}