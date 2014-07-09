#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "SLIC.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <tbb/tbb.h>
#include <vector>

using namespace std;
using namespace cv;

// class SLICProva : public SLIC 
// {
//     private:
// 		double totalAverageDiff;
// 
// 	public:
// 
// 		void colorSuperpixels(
// 		cv::Mat  LABImage,
// 		cv::Rect areaToColor = cv::Rect(0, 0, INT_MAX, INT_MAX))
// 		{
// 			/* Verify that area to color is within image boundaries,
// 			   otherwise reset area to the entire image. */
// 			if (areaToColor.x < 0 || areaToColor.x > LABImage.cols)
// 				areaToColor.x = 0;
// 			if (areaToColor.y < 0 || areaToColor.y > LABImage.rows)
// 				areaToColor.y = 0;
// 			if (areaToColor.width < 0 || areaToColor.x + areaToColor.width > LABImage.cols)
// 				areaToColor.width = LABImage.cols - areaToColor.x;
// 			if (areaToColor.height < 0 || areaToColor.y + areaToColor.height > LABImage.rows)
// 				areaToColor.height = LABImage.rows - areaToColor.y;
// 
// 			/* Create a matrix which will store the color of each cluster. */
// 			vector<Vec3f> clusterColors(clusterCentres.size());
// 			vector<Vec3f> clusterColorsFromCentres(clusterCentres.size());
// 
// 			for (int n = 0; n < clusterColorsFromCentres.size(); n++)
// 				clusterColorsFromCentres[n] = LABImage.at<Vec3b>(clusterCentres[n][4], clusterCentres[n][3]);
// 
// // 			/* Sum color information of all the pixels in the same cluster
// // 			   for future average color calculation. */
// // 			for (int y = 0; y < LABImage.rows; y++) 
// // 			{
// // 				for (int x = 0; x < LABImage.cols; x++)
// // 				{
// // 					/* Get pixel's cluster. */
// // 					int tempColorIndex = pixelCluster[y * LABImage.cols + x];
// // 
// // 					/* Continue to next loop if the pixel doesn't
// // 					   belong to any cluster. */
// // 					if (tempColorIndex <= -1)
// // 						continue;
// // 
// // 					/* Get pixel color. */
// // 					Vec3f tempColor = LABImage.at<Vec3b>(y, x);
// // 
// // 					// Qui potrei avere accessi simultanei in scrittura e quindi
// // 					// non posso andare in parallelo. Provando con un altro 
// // 					// metodo parallelo, si riescono a guadagnare 2-3 ms.
// // 					clusterColors[tempColorIndex].val[0] += tempColor.val[0];
// // 					clusterColors[tempColorIndex].val[1] += tempColor.val[1];
// // 					clusterColors[tempColorIndex].val[2] += tempColor.val[2];
// // 				}
// // 			}
// // 
// // 			/* Divide by the number of pixels in each cluster to get the
// // 			   average cluster color. */
// // 			tbb::parallel_for(0, static_cast<int>(clusterColors.size()), [&](int n)
// // 			{
// // 				/* Continue to next loop if a cluster is empty. */
// // 				if (pixelsOfSameCluster[n] > 0)
// // 				{
// // 					clusterColors[n].val[0] /= pixelsOfSameCluster[n];
// // 					clusterColors[n].val[1] /= pixelsOfSameCluster[n];
// // 					clusterColors[n].val[2] /= pixelsOfSameCluster[n];
// // 				}
// // 			});
// 
// 			/* Fill in each cluster with its centre color. */
// 			tbb::parallel_for(areaToColor.y, areaToColor.y + areaToColor.height, 1, [&](int y)
// 			{
// 				for (int x = areaToColor.x; x < areaToColor.x + areaToColor.width; x++)
// 					if (pixelCluster[y * LABImage.cols + x] >= 0 &&
// 						pixelCluster[y * LABImage.cols + x] < clusterColorsFromCentres.size())
// 					{
// 						Vec3f tempColor = clusterColorsFromCentres[pixelCluster[y * LABImage.cols + x]];
// 						LABImage.at<Vec3b>(y, x) = tempColor;
// 					}
// 			});
// // 
// // 			Vec3f averageClustersColor(0.0);
// // 			Vec3f averageCentresColor(0.0);
// // 
// // 			for (int n = 0; n < clusterColors.size(); n++)
// // 			{
// // 				averageClustersColor.val[0] += clusterColors[n].val[0];
// // 				averageClustersColor.val[1] += clusterColors[n].val[1];
// // 				averageClustersColor.val[2] += clusterColors[n].val[2];
// // 
// // 				averageCentresColor.val[0] += clusterColorsFromCentres[n].val[0];
// // 				averageCentresColor.val[1] += clusterColorsFromCentres[n].val[1];
// // 				averageCentresColor.val[2] += clusterColorsFromCentres[n].val[2];
// // 			}
// // 
// // 			averageClustersColor.val[0] /= clusterColors.size();
// // 			averageClustersColor.val[1] /= clusterColors.size();
// // 			averageClustersColor.val[2] /= clusterColors.size();
// // 
// // 			averageCentresColor.val[0] /= clusterColorsFromCentres.size();
// // 			averageCentresColor.val[1] /= clusterColorsFromCentres.size();
// // 			averageCentresColor.val[2] /= clusterColorsFromCentres.size();
// // 
// // 			averageClustersColor = (Vec3b)averageClustersColor;
// // 			averageCentresColor = (Vec3b)averageCentresColor;
// // 
// // 			ostringstream stringStream;
// // 
// // 			rectangle(LABImage, Point(0, 0), Point(300, 140), CV_RGB(255, 255, 255), CV_FILLED);
// // 
// // 			stringStream << "               B    G    R";
// // 			putText(LABImage, stringStream.str(), Point(5, 20), 
// // 				FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
// // 
// // 			stringStream.str("");
// // 			stringStream << "Average color: " << averageClustersColor.val[0] << "  " << averageClustersColor.val[1] << "  " << averageClustersColor.val[2];
// // 			putText(LABImage, stringStream.str(), Point(5, 40), 
// // 				FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
// // 
// // 			stringStream.str("");
// // 			stringStream << "Centre color:  " << averageCentresColor.val[0] << "  " << averageCentresColor.val[1] << "  " << averageCentresColor.val[2];
// // 			putText(LABImage, stringStream.str(), Point(5, 60), 
// // 				FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
// // 
// // 			stringStream.str("");
// // 			stringStream << "Difference:     " << abs(averageCentresColor.val[0] - averageClustersColor.val[0]) << "  " <<
// // 				abs(averageCentresColor.val[1] - averageClustersColor.val[1]) << "  " <<
// // 				abs(averageCentresColor.val[2] - averageClustersColor.val[2]);
// // 			putText(LABImage, stringStream.str(), Point(5, 80), 
// // 				FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
// // 
// // 			double avg = (
// // 				abs(averageCentresColor.val[0] - averageClustersColor.val[0]) +
// // 				abs(averageCentresColor.val[1] - averageClustersColor.val[1]) +
// // 				abs(averageCentresColor.val[2] - averageClustersColor.val[2])) / 3;
// // 
// // 			stringStream.str("");
// // 			stringStream << "Average diff.:      " << avg;
// // 			putText(LABImage, stringStream.str(), Point(5, 100), 
// // 				FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
// // 
// // 			if (framesNumber == 0)
// // 				totalAverageDiff = 0;
// // 
// // 			totalAverageDiff += avg;
// // 			framesNumber++;
// // 
// // 			stringStream.str("");
// // 			stringStream << "Total avg. diff.:    " << totalAverageDiff / framesNumber;
// // 			putText(LABImage, stringStream.str(), Point(5, 120), 
// // 				FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
//  		}
// };
// 
// Mat tempFrame;
// vector<Vec3f> sampleColors;
// int colorNumber = 0;
// Vec3f averageColor(0.0);
// 
// void onMouse(int event, int x, int y, int flags, void* param)
// {
// 	if (event == CV_EVENT_LBUTTONDOWN)
// 	{
// 		cvtColor(tempFrame, tempFrame, CV_BGR2YCrCb);
// 		Vec3b color = tempFrame.at<Vec3b>(y,x);
// 		sampleColors.push_back(Vec3f(color.val[0], color.val[1], color.val[2]));
// 		cout << "Y = " << (int)color.val[0] << "   , Cr = " << (int)color.val[1] << "   , Cb = " << (int)color.val[2] << endl;
// 		ofstream myfile;
// 		myfile.open ("C:\\Users\\Claudiu\\Desktop\\Colors.txt", ios::out | ios::app);
// 		myfile << "Y = " << (int)color.val[0] << "   , Cr = " << (int)color.val[1] << "   , Cb = " << (int)color.val[2] << endl;
// 		myfile.close();
// 		colorNumber++;
// 		averageColor.val[0] += color.val[0];
// 		averageColor.val[1] += color.val[1];
// 		averageColor.val[2] += color.val[2];
// 	}
// }

int main()
{
	/* Video source location. */
	const string videoLocation = "C:\\Users\\Claudiu\\Desktop\\Video Data Set\\EDSH1.avi";

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
	const int videoWidth  = static_cast<int>(capturedVideo.get(CV_CAP_PROP_FRAME_WIDTH));
	const int videoHeight = static_cast<int>(capturedVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

	/* A container which will hold a video frame for
	   the necessary time for its elaboration. */
	Mat currentFrame;

	/* SLIC algorithm parameters. */
	int spatialDistanceWeight = 30;
	int superpixelNumber      = 1000;

	/* Round the sampling step to the nearest integer. */
	int stepSLIC = static_cast<int>(sqrt((videoHeight * videoWidth) / superpixelNumber) + 0.5);

	/* Video frames counter. */
	int framesNumber = 0;

	/* Open a new window where to play the imported video. */
	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	namedWindow("ResultYCrCb", CV_WINDOW_AUTOSIZE);
	//namedWindow("ResultLAB", CV_WINDOW_AUTOSIZE);

	SLIC* SLICVideoElaboration = new SLIC();
//	SLICProva* SLICSecond = new SLICProva();

	/* Output video. */
// 	VideoWriter outputVideo(
// 		"C:\\Users\\Claudiu\\Desktop\\Prova.avi",
// 		capturedVideo.get(CV_CAP_PROP_FOURCC), capturedVideo.get(CV_CAP_PROP_FPS),
// 		Size(capturedVideo.get(CV_CAP_PROP_FRAME_WIDTH), capturedVideo.get(CV_CAP_PROP_FRAME_HEIGHT)));

	int Y_MIN = 70;
	int Y_MAX = 240;
	int Cr_MIN = 140;
	int Cr_MAX = 160;
	int Cb_MIN = 100;
	int Cb_MAX = 125;

	createTrackbar("Y_MIN", windowName, &Y_MIN, 255);
	createTrackbar("Cr_MIN", windowName, &Cr_MIN, 255);
	createTrackbar("Cb_MIN", windowName, &Cb_MIN, 255);
	createTrackbar("Y_MAX", windowName, &Y_MAX, 255);
	createTrackbar("Cr_MAX", windowName, &Cr_MAX, 255);
	createTrackbar("Cb_MAX", windowName, &Cb_MAX, 255);

// 	int L_MIN = 30;
// 	int L_MAX = 240;
// 	int A_MIN = 135;
// 	int A_MAX = 150;
// 	int B_MIN = 125;
// 	int B_MAX = 150;
// 
// 	createTrackbar("L_MIN", windowName, &L_MIN, 255);
// 	createTrackbar("A_MIN", windowName, &A_MIN, 255);
// 	createTrackbar("B_MIN", windowName, &B_MIN, 255);
// 	createTrackbar("L_MAX", windowName, &L_MAX, 255);
// 	createTrackbar("A_MAX", windowName, &A_MAX, 255);
// 	createTrackbar("B_MAX", windowName, &B_MAX, 255);

	/* Enter an infinite cycle to elaborate the video until its last frame. */
	while (true)
	{
// 		if (framesNumber >= 2500)
// 			break;

// 		if ((framesNumber != 0) && (framesNumber % 200 == 0))
// 		{
// 			if (SLICVideoElaboration != NULL)
// 				delete SLICVideoElaboration;
// 
// 			SLICVideoElaboration = new SLIC();
// 
// 			if (spatialDistanceWeight > 120)
// 				spatialDistanceWeight = 10;
// 			else
// 				spatialDistanceWeight += 10;
// 
// 			if (framesNumber % 2400 == 0)
// 			{
// 				if (superpixelNumber > 50000)
// 					superpixelNumber  = 200;
// 				else
// 					superpixelNumber *= 2;
// 
// 				stepSLIC = static_cast<int>(sqrt((videoHeight * videoWidth) / superpixelNumber) + 0.5);
// 			}
// 		}

		// per vedere se le performances degradano
		if ((framesNumber != 0) && (framesNumber % 30 == 0))
		{
			if (SLICVideoElaboration != NULL)
				delete SLICVideoElaboration;
//			if (SLICSecond != NULL)
//				delete SLICSecond;
			SLICVideoElaboration = new SLIC();
//			SLICSecond = new SLICProva();
		}

		/* Take the next frame from the video. */
		capturedVideo >> currentFrame;

		/* If there are no more frames in the video, break the loop because
		   the video has reached its end or there was an error. */
		if (currentFrame.data == NULL)
			break;

// 		chrono::high_resolution_clock::time_point startPoint =
// 			chrono::high_resolution_clock::now();

		/* Convert the frame from RGB to LAB color space
		   before SLIC elaboration. */
		//cvtColor(currentFrame, currentFrame, CV_BGR2YCrCb);

		/* Perform the SLIC algorithm operations. */
// 		SLICVideoElaboration->createSuperpixels(
// 			currentFrame, stepSLIC, spatialDistanceWeight, !framesNumber);
		//SLICVideoElaboration.enforceConnectivity(currentFrame);

// 		SLICSecond->createSuperpixels(
// 			currentFrame, stepSLIC, spatialDistanceWeight, !framesNumber);

		/* Convert frame back to RGB. */
		//cvtColor(currentFrame, currentFrame, CV_Lab2BGR);

		//Mat tempFrame = currentFrame.clone();

		//SLICVideoElaboration->colorSuperpixels(currentFrame);
		//SLICVideoElaboration->drawClusterContours(currentFrame, Vec3b(0, 0, 255), Rect(videoWidth / 2, 0, videoWidth / 2, videoHeight));
		//SLICVideoElaboration.drawClusterCentres(currentFrame, CV_RGB(255, 0, 0));

// 		chrono::high_resolution_clock::time_point endPoint =
// 			chrono::high_resolution_clock::now();
// 
// 		chrono::duration<int, std::milli> elapsedTime =
// 			chrono::duration_cast<chrono::milliseconds>(endPoint - startPoint);

		//SLICVideoElaboration->drawInformation(currentFrame, framesNumber, elapsedTime.count());

		//Mat diffFrame;
		//absdiff(currentFrame, tempFrame, diffFrame);

		Mat filteredFrameYCrCb;
		cvtColor(currentFrame, filteredFrameYCrCb, CV_BGR2YCrCb);

		//Mat filteredFrameLAB;
		//cvtColor(currentFrame, filteredFrameLAB, CV_BGR2Lab);

		SLICVideoElaboration->createSuperpixels(
			filteredFrameYCrCb, stepSLIC, spatialDistanceWeight, !framesNumber);
		SLICVideoElaboration->colorSuperpixels(filteredFrameYCrCb);

		//SLICVideoElaboration->createSuperpixels(
		//	filteredFrameLAB, stepSLIC, spatialDistanceWeight, !framesNumber);
		//SLICVideoElaboration->colorSuperpixels(filteredFrameLAB);

		inRange(filteredFrameYCrCb, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), filteredFrameYCrCb);
		//inRange(filteredFrameLAB, Scalar(L_MIN, A_MIN, B_MIN), Scalar(L_MAX, A_MAX, B_MAX), filteredFrameLAB);

		framesNumber++;

		//tempFrame = currentFrame.clone();
		//setMouseCallback(windowName, onMouse, 0);
		/* Show frame in the window and increase frame counter. */
		imshow(windowName, currentFrame);
		//SLICVideoElaboration->recognizerHands(filteredFrameYCrCb);
		imshow("ResultYCrCb", filteredFrameYCrCb);
		//imshow("ResultLAB", filteredFrameLAB);

		//outputVideo.write(currentFrame);

		/* End program on ESC press. */
		if (cvWaitKey(1) == 27)
			break;
	}

	if (SLICVideoElaboration != NULL)
		delete SLICVideoElaboration;
//	if (SLICSecond != NULL)
//		delete SLICSecond;


	/* Close window after video processing. */
	cv::destroyAllWindows();

// 	averageColor.val[0] /= colorNumber;
// 	averageColor.val[1] /= colorNumber;
// 	averageColor.val[2] /= colorNumber;
// 	ofstream myfile;
// 	myfile.open ("C:\\Users\\Claudiu\\Desktop\\Colors.txt", ios::out | ios::app);
// 	myfile << "Average Y = " << (float)averageColor.val[0] << "   , Cr = " << (float)averageColor.val[1] << "   , Cb = " << (float)averageColor.val[2] << endl;
// 	myfile.close();

	return 0;
}