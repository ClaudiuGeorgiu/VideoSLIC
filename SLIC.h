/****************************************************************************/
/*                                                                          */
/* Algorithm: http://ivrg.epfl.ch/research/superpixels                      */
/* Original OpevCV implementation: http://github.com/PSMM/SLIC-Superpixels  */
/*                                                                          */
/****************************************************************************/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

#ifndef SLIC_H
#define SLIC_H

#define MatrixOfDouble2D std::vector<std::vector<double>>

/* Algorithm's number of iterations. */
#define ITERATIONS_NUMBER 10

class SLIC
{
	private:

		/* The cluster which each pixel belongs to. */
		std::vector<int> pixelCluster;

		/* The distance of each pixel from the nearest cluster's centre. */
		std::vector<double> distanceFromClusterCentre;

		/* The LAB and position values of the centres. */
		MatrixOfDouble2D clusterCentres;

		/* The number of pixels belonging to the same cluster. */
		std::vector<int> pixelsOfSameCluster;

		/* The total number of pixel of the image. */
		int pixelsNumber;

		/* The sampling step distance. */
		int samplingStep;

		/* The distance weight factor. */
		int spatialDistanceWeight;

		/* Erase all matrices' elements and reset variables. */
		void clearSLICData();

		/* Initialize matrices' elements and variables. */
		void initializeSLICData(
			const cv::Mat          LABImage,
			const int              samplingStep,
			const int              spatialDistanceWeight,
			const MatrixOfDouble2D previousCentreMatrix);

		/* Find the pixel with the lowest gradient in a 3x3 surrounding. */
		cv::Point findLowestGradient(
			const cv::Mat   LABImage,
			const cv::Point centre);

		/* Compute the distance between a cluster's centre and an individual pixel. */
		double computeDistance(
			const int       centreIndex,
			const cv::Point pixelPosition,
			const cv::Vec3b pixelColor);

	public:

		/* Class default constructor. */
		SLIC();

		/* Class copy constructor. */
		SLIC(SLIC& otherSLIC);

		/* Class destructor. */
		virtual ~SLIC();

		/* Generate superpixels for an image. */
		MatrixOfDouble2D createSuperpixels(
			const cv::Mat          LABImage,
			const int              samplingStep,
			const int              spatialDistanceWeight,
			const MatrixOfDouble2D previousCentreMatrix);

		/* Enforce connectivity among the superpixels of an image. */
		void enforceConnectivity(cv::Mat LABImage);

		/* Color each created superpixel with superpixel's average color. */
		void colorSuperpixels(cv::Mat LABImage);

		/* Draw contours around created superpixels. */
		void drawClusterContours(
			cv::Mat         LABImage,
			const cv::Vec3b contourColor);

		/* Draw superpixels' centres. */
		void drawClusterCentres(
			cv::Mat          LABImage,
			const cv::Scalar centreColor);
};

#endif
