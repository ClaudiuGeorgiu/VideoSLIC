/****************************************************************************/
/*                                                                          */
/* Original algorithm: http://ivrg.epfl.ch/research/superpixels             */
/* Original OpenCV implementation: http://github.com/PSMM/SLIC-Superpixels  */
/*                                                                          */
/* Paper: "Optimizing Superpixel Clustering for Real-Time                   *//*         Egocentric-Vision Applications"                                  */
/*        http://www.isip40.it/resources/papers/2015/SPL_Pietro.pdf         */
/*                                                                          */
/****************************************************************************/

/* OpenCV libraries for video and image
   elaborations. */
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>

#ifndef SLIC_H
#define SLIC_H

/* Choose to run SLIC algorithm for a fixed number of iterations or
   until the residual error converges below a certain threshold. */
enum SLICElaborationMode {
	/* Repeat SLIC operations for a fixed number of times. */
	FIXED_ITERATIONS,
	/* Repeat SLIC operations until error convergence is reached. */
	ERROR_THRESHOLD,
};

/* Choose whether to use key frames and Gaussian noise (as described in the paper)
   or not when running SLIC algorithm for a video sequence. */
enum VideoElaborationMode {
	/* Process frames without using key frames nor adding Gaussian noise. */
	NAIVE,
	/* Use only key frames. */
	KEY_FRAMES,
	/* Add only Gaussian noise. */
	NOISE,
	/* Use key frames and add Gaussian noise. */
	KEY_FRAMES_NOISE,
};

class SLIC
{
	protected:

		/* Debug data for performance analysis. */
		unsigned minIterations;
		unsigned maxIterations;
		unsigned averageExecutionTime;
		unsigned minExecutionTime;
		unsigned maxExecutionTime;
		double   averageError;
		double   averageIterations;
		double   minError;
		double   maxError;

		/* The cluster which each pixel belongs to.
		   pixelCluster[p] = c means that p-th pixel belongs to c-th cluster. */
		std::vector<int> pixelCluster;

		/* The distance of each pixel from the nearest cluster's centre.
		   Suppose c being the nearest cluster to pixel p.
		   distanceFromClusterCentre[p] = d means that the distance between the p-th
		   pixel and the c-th cluster centre is d. */
		std::vector<double> distanceFromClusterCentre;

		/* The color and position values of the centres,
		   stored as [L, A, B, x, y] values. We don't use 2D vectors for
		   the sake of performance. For example, in order to find the 21-th
		   centre's B channel value, one has to write
		   B_channel = clusterCentres[(21 * 5) + 2]. */
		std::vector<double> clusterCentres;

		/* The color and position values of the centres before
		   centre recalculation (used for calculating the residual
		   error). Works the same as clusterCentres vector. */
		std::vector<double> previousClusterCentres;

		/* The number of pixels belonging to the same cluster.
		   pixelsOfSameCluster[c] = n means that the c-th cluster has
		   n pixel inside it. */
		std::vector<int> pixelsOfSameCluster;

		/* The error between clusters' centre recalculation. Using the
		   formula described in the paper, the error is calculated between
		   each element in clusterCentres and previousClusterCentres.
		   residualError[c] = e means that the residual error related to 
		   the c-th cluster is e. */
		std::vector<double> residualError;

		/* The number of iterations performed by the algorithm
		   (for each frame in the video). */
		unsigned iterationIndex;

		/* The total number of pixel of the image or frame. */
		unsigned pixelsNumber;

		/* The total number of cluster. */
		unsigned clustersNumber;

		/* The sampling step distance. */
		unsigned samplingStep;

		/* The distance weight factor. */
		unsigned spatialDistanceWeight;

		/* spatialDistanceWeight^2 / samplingStep^2. */
		double distanceFactor;

		/* The total residual error after clusters' centres
		   recalculation. Computed as a mean among all the 
		   cluster errors. */
		double totalResidualError;

		/* The maximum residual error allowed. When totalResidualError
		   goes below this threshold, we have the convergence of
		   the algorithm. */
		double errorThreshold;

		/* Number of connected frames which have been processed (i.e. we use
		   the results obtained for a certain frame to initialize the algorithm
		   grid of the next frame; more information can be found in the paper). */
		unsigned framesNumber;

		/* Erase all matrices' elements and reset variables. */
		void clearSLICData();

		/* Initialize matrices' elements and variables. */
		void initializeSLICData(
			const cv::Mat&       image,
			const unsigned       samplingStep,
			const unsigned       spatialDistanceWeight,
			const double         errorThreshold,
			VideoElaborationMode videoMode,
			const unsigned       keyFramesRatio,
			const double         GaussianStdDev,
			/* By default we choose to process frames independently. */
			const bool           connectedFrames = false);

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

		/* Generate superpixels for an image/frame. */
		void createSuperpixels(
			const cv::Mat&       image,
			const unsigned       samplingStep,
			const unsigned       spatialDistanceWeight,
			const unsigned       iterationNumber,
			const double         errorThreshold,
			SLICElaborationMode  SLICMode,
			VideoElaborationMode videoMode,
			const unsigned       keyFramesRatio,
			const double         GaussianStdDev,
			/* By default we choose to process frames independently. */
			const bool           connectedFrames = false);

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

		/* Draw superpixels' informations (for debug/analysis purposes). */
		void drawInformation(
			cv::Mat&       image,
			const unsigned totalFrames,
			const unsigned executionTimeInMilliseconds);
};

#endif