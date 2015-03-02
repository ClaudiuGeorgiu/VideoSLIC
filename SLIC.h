/****************************************************************************/
/*                                                                          */
/* Algorithm: http://ivrg.epfl.ch/research/superpixels                      */
/* Original OpenCV implementation: http://github.com/PSMM/SLIC-Superpixels  */
/*                                                                          */
/****************************************************************************/

/* OpenCV libraries for video and image
   elaborations. */
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>

#ifndef SLIC_H
#define SLIC_H

class SLIC
{
	protected:

		/* Debug data. */
		unsigned framesNumber;
		unsigned minIterations;
		unsigned maxIterations;
		unsigned averageExecutionTime;
		unsigned minExecutionTime;
		unsigned maxExecutionTime;
		double averageError;
		double averageIterations;
		double minError;
		double maxError;

		/* The cluster which each pixel belongs to. */
		std::vector<int> pixelCluster;

		/* The distance of each pixel from the nearest cluster's centre. */
		std::vector<double> distanceFromClusterCentre;

		/* The color and position values of the centres. */
		std::vector<double> clusterCentres;

		/* The color and position values of the centres
		   before centre recalculation (used for calculating
		   the residual error). */
		std::vector<double> previousClusterCentres;

		/* The number of pixels belonging to the same cluster. */
		std::vector<int> pixelsOfSameCluster;

		/* The error between clusters' centre recalculation. */
		std::vector<double> residualError;

		/* The number of iterations performed by the algorithm. */
		unsigned iterationIndex;

		/* The total number of pixel of the image. */
		unsigned pixelsNumber;

		/* The total number of cluster. */
		unsigned clustersNumber;

		/* The sampling step distance. */
		unsigned samplingStep;

		/* The distance weight factor. */
		unsigned spatialDistanceWeight;

		/* spatialDistanceWeight^2 / samplingStep^2. */
		double distanceFactor;

		/* The residual error after clusters' centres
		   recalculation. */
		double totalResidualError;

		/* The maximum residual error allowed. */
		double errorThreshold;

		/* Erase all matrices' elements and reset variables. */
		void clearSLICData();

		/* Initialize matrices' elements and variables. */
		void initializeSLICData(
			const cv::Mat& image,
			const unsigned samplingStep,
			const unsigned spatialDistanceWeight,
			const bool     firstVideoFrame = true);

		/* Find the pixel with the lowest gradient in a 3x3 surrounding. */
		cv::Point findLowestGradient(
			const cv::Mat&   image,
			const cv::Point& centre);

		/* Compute the distance between a cluster's centre and an individual pixel. */
		double computeDistance(
			const int        centreIndex,
			const cv::Point& pixelPosition,
			const cv::Vec3b& pixelColor);

	public:

		/* Class default constructor. */
		SLIC();

		/* Class copy constructor. */
		SLIC(const SLIC& otherSLIC);

		/* Class destructor. */
		virtual ~SLIC();

		/* Generate superpixels for an image. */
		void createSuperpixels(
			const cv::Mat& image,
			const unsigned samplingStep,
			const unsigned spatialDistanceWeight,
			const bool     firstVideoFrame = true);

		/* Enforce superpixel connectivity. */
		void SLIC::enforceConnectivity(const cv::Mat image);

		/* Color each created superpixel in a certain area (by default the
		   entire image) with superpixel's average color. */
		void colorSuperpixels(
			cv::Mat&  image,
			cv::Rect& areaToColor = cv::Rect(0, 0, INT_MAX, INT_MAX));

		/* Draw contours around created superpixels in a certain area
		   (by default the entire image). */
		void drawClusterContours(
			cv::Mat&         image,
			const cv::Vec3b& contourColor,
			cv::Rect&        areaToDraw = cv::Rect(0, 0, INT_MAX, INT_MAX));

		/* Draw superpixels' centres. */
		void drawClusterCentres(
			cv::Mat&          image,
			const cv::Scalar& centreColor);

		/* Draw superpixels' informations. */
		void drawInformation(
			cv::Mat&       image,
			const unsigned totalFrames,
			const unsigned executionTimeInMilliseconds);

		/* Recognize hands in the video. */
		void recognizeHands(cv::Mat& image);
};

#endif