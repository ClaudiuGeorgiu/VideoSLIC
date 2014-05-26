#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

#include "SLIC.h"

using namespace cv;

SLIC::SLIC()
{

}

SLIC::SLIC(const SLIC& otherSLIC)
{

}

SLIC::~SLIC()
{
	clearSLICData();
}

void SLIC::clearSLICData()
{
	/* Reset variables. */
	this->pixelsNumber          = 0;
	this->samplingStep          = 0;
	this->spatialDistanceWeight = 0;

	/* Erase all matrices' elements. */
	pixelCluster.clear();
	distanceFromClusterCentre.clear();
	clusterCentres.clear();
	pixelsOfSameCluster.clear();
	residualError.clear();
}

void SLIC::initializeSLICData(
	const cv::Mat LABImage,
	const int     samplingStep,
	const int     spatialDistanceWeight,
	const bool    firstVideoFrame)
{
		/* Initialize total residual error for each frame. */
	totalResidualError = FLT_MAX;

	/* If centres matrix from previous frame is empty,
	   or this is the first frame in the video,
	   initialize data from scratch. Otherwise, use
	   the data from previous frame as initialization.*/
	if (firstVideoFrame == true || clusterCentres.size() <= 0)
	{
		/* Clear previous data before initialization. */
		clearSLICData();

		/* Initialize variables. */
		this->pixelsNumber          = LABImage.rows * LABImage.cols;
		this->samplingStep          = samplingStep;
		this->spatialDistanceWeight = spatialDistanceWeight;
		this->totalResidualError    = FLT_MAX;
		this->errorThreshold        = 0.5;

		/* Initialize the clusters and the distances matrices. */
		for (int n = 0; n < pixelsNumber; n++)
		{
			pixelCluster.push_back(-1);
			distanceFromClusterCentre.push_back(FLT_MAX);
		}

		/* Initialize the centres matrix by sampling the image
		   at a regular step. */
		for (int y = samplingStep; y < LABImage.rows; y += samplingStep)
			for (int x = samplingStep; x < LABImage.cols; x += samplingStep)
			{
				/* Find the pixel with the lowest gradient in a 3x3 surrounding. */
				Point lowestGradientPixel = findLowestGradient(LABImage, Point(x, y));
				Vec3b tempPixelColor = LABImage.at<Vec3b>(y, x);

				/* Create a centre [l, a, b, x, y] and insert it in the centres vector. */
				vector<double> tempCentre;

				/* Generate the centre vector. */
				tempCentre.push_back(tempPixelColor.val[0]);
				tempCentre.push_back(tempPixelColor.val[1]);
				tempCentre.push_back(tempPixelColor.val[2]);
				tempCentre.push_back(lowestGradientPixel.x);
				tempCentre.push_back(lowestGradientPixel.y);

				/* Insert centre in centres matrix. */
				clusterCentres.push_back(tempCentre);
				previousClusterCentres.push_back(tempCentre);
				
				/* Initialize "pixel of same cluster" matrix
				   (with 1 because of the new centre per cluster). */
				pixelsOfSameCluster.push_back(1);

				/* Initialize residual error to be zero for each cluster
				   centre. */
				residualError.push_back(0);
			}	
	}
}

Point SLIC::findLowestGradient(
	const cv::Mat   LABImage,
	const cv::Point centre) 
{
	double lowestGradient = FLT_MAX;
	Point lowestGradientPoint = Point(centre.x, centre.y);

	for (int y = centre.y - 1; y <= centre.y + 1 && y < LABImage.rows - 1; y++) 
		for (int x = centre.x - 1; x <= centre.x + 1 && x < LABImage.cols - 1; x++) 
		{
			/* Exclude pixels on borders. */
			if (x < 1)
				continue;
			if (y < 1)
				continue;

			Vec3b tempPixelUp    = LABImage.at<Vec3b>(y - 1, x);
			Vec3b tempPixelDown  = LABImage.at<Vec3b>(y + 1, x);
			Vec3b tempPixelRight = LABImage.at<Vec3b>(y, x + 1);
			Vec3b tempPixelLeft  = LABImage.at<Vec3b>(y, x - 1);

			/* Compute horizontal and vertical gradients and keep track
			   of the minimum. */
			double tempGradient =
				(tempPixelRight.val[0] - tempPixelLeft.val[0]) *
				(tempPixelRight.val[0] - tempPixelLeft.val[0]) +
				(tempPixelUp.val[0]    - tempPixelDown.val[0]) * 
				(tempPixelUp.val[0]    - tempPixelDown.val[0]);

			if (tempGradient < lowestGradient)
			{
				lowestGradient        = tempGradient;
				lowestGradientPoint.x = x;
				lowestGradientPoint.y = y;
			}
		}

	return lowestGradientPoint;
}

double SLIC::computeDistance(
	const int       centreIndex,
	const cv::Point pixelPosition,
	const cv::Vec3b pixelColor)
{
	/* Compute the color distance between two pixels. */
	double colorDistance =
		(clusterCentres[centreIndex][0] - pixelColor.val[0]) *
		(clusterCentres[centreIndex][0] - pixelColor.val[0]) +
		(clusterCentres[centreIndex][1] - pixelColor.val[1]) *
		(clusterCentres[centreIndex][1] - pixelColor.val[1]) + 
		(clusterCentres[centreIndex][2] - pixelColor.val[2]) *
		(clusterCentres[centreIndex][2] - pixelColor.val[2]);

	/* Compute the spatial distance between two pixels. */
	double spaceDistance =
		(clusterCentres[centreIndex][3] - pixelPosition.x) *
		(clusterCentres[centreIndex][3] - pixelPosition.x) + 
		(clusterCentres[centreIndex][4] - pixelPosition.y) *
		(clusterCentres[centreIndex][4] - pixelPosition.y);

	/* Compute total distance between two pixels using the formula 
	   described by the algorithm. */
	return colorDistance + spaceDistance * spatialDistanceWeight *
		spatialDistanceWeight / (samplingStep * samplingStep);
}

void SLIC::createSuperpixels(
	const cv::Mat LABImage,
	const int     samplingStep,
	const int     spatialDistanceWeight,
	const bool    firstVideoFrame)
{
	/* Initialize algorithm data. */
	initializeSLICData(
		LABImage, samplingStep, spatialDistanceWeight, firstVideoFrame);

	/* Repeat next steps the number of times prescribed by the algorithm. */
	for (int iterationIndex = 0; totalResidualError > errorThreshold; iterationIndex++)
	{
		/* Reset distance values. */
		for (int n = 0; n < pixelsNumber; n++)
			distanceFromClusterCentre[n] = FLT_MAX;
		
		for (int centreIndex = 0; centreIndex < clusterCentres.size(); centreIndex++) 
		{
			/* Look for pixels in a 2 x step by 2 x step region only. */
			for (int y = (int)clusterCentres[centreIndex][4] - samplingStep - 1;
				y < clusterCentres[centreIndex][4] + samplingStep + 1; y++) 
				for (int x = (int)clusterCentres[centreIndex][3] - samplingStep - 1;
					x < clusterCentres[centreIndex][3] + samplingStep + 1; x++)
				{
					/* Verify that neighbor pixel is within the image boundaries. */
					if (x >= 0 && x < LABImage.cols && y >= 0 && y < LABImage.rows)
					{
						Vec3b pixelColor = LABImage.at<Vec3b>(y, x);
						
						double tempDistance =
							computeDistance(centreIndex, Point(x, y), pixelColor);

						/* Update pixel's cluster if this distance is smaller
						   than pixel's previous distance. */
						if (tempDistance < distanceFromClusterCentre[y * LABImage.cols + x])
						{
							distanceFromClusterCentre[y * LABImage.cols + x] = tempDistance;
							pixelCluster[y * LABImage.cols + x]              = centreIndex;
						}
					}
				}
		}

		/* Reset centres values and the number of pixel
		   per cluster to zero. */
		for (int n = 0; n < clusterCentres.size(); n++)
		{
			clusterCentres[n][0] = 0;
			clusterCentres[n][1] = 0;
			clusterCentres[n][2] = 0;
			clusterCentres[n][3] = 0;
			clusterCentres[n][4] = 0;

			pixelsOfSameCluster[n] = 0;
		}

		/* Compute the new cluster centres. */
		for (int y = 0; y < LABImage.rows; y++) 
			for (int x = 0; x < LABImage.cols; x++) 
			{
				int currentPixelCluster = pixelCluster[y * LABImage.cols + x];

				/* Verify if current pixel belongs to a cluster. */
				if (currentPixelCluster != -1)
				{
					/* Sum the information of pixels of the same
					   cluster for future centre recalculation. */
					Vec3b pixelColor = LABImage.at<Vec3b>(y, x);

					clusterCentres[currentPixelCluster][0] += pixelColor.val[0];
					clusterCentres[currentPixelCluster][1] += pixelColor.val[1];
					clusterCentres[currentPixelCluster][2] += pixelColor.val[2];
					clusterCentres[currentPixelCluster][3] += x;
					clusterCentres[currentPixelCluster][4] += y;

					pixelsOfSameCluster[currentPixelCluster] += 1;
				}
			}

		/* Normalize the clusters' centres. */
		for (int centreIndex = 0; centreIndex < clusterCentres.size(); centreIndex++)
		{
			/* Avoid empty clusters. */
			if (pixelsOfSameCluster[centreIndex] == 0)
				continue;

			clusterCentres[centreIndex][0] /= pixelsOfSameCluster[centreIndex];
			clusterCentres[centreIndex][1] /= pixelsOfSameCluster[centreIndex];
			clusterCentres[centreIndex][2] /= pixelsOfSameCluster[centreIndex];
			clusterCentres[centreIndex][3] /= pixelsOfSameCluster[centreIndex];
			clusterCentres[centreIndex][4] /= pixelsOfSameCluster[centreIndex];
		}
		
		/* Skip error calculation if this is the first iteration,
		   meaning this is a new frame in the video. */
		if (iterationIndex == 0)
			for (int centreIndex = 0; centreIndex < previousClusterCentres.size(); centreIndex++)
			{
				/* Update previous centres matrix. */
				previousClusterCentres[centreIndex] = clusterCentres[centreIndex];
			}
		else
		{
			/* Compute residual error. */
			for (int centreIndex = 0; centreIndex < clusterCentres.size(); centreIndex++)
			{
				/* Calculate residual error for each cluster centre. */
				residualError[centreIndex] = sqrt(
					(clusterCentres[centreIndex][4] - previousClusterCentres[centreIndex][4]) *
					(clusterCentres[centreIndex][4] - previousClusterCentres[centreIndex][4]) +
					(clusterCentres[centreIndex][3] - previousClusterCentres[centreIndex][3]) *
					(clusterCentres[centreIndex][3] - previousClusterCentres[centreIndex][3]));

				/* Update previous centres matrix. */
				previousClusterCentres[centreIndex] = clusterCentres[centreIndex];
			}

			/* Compute total residual error by averaging all clusters' errors. */
			totalResidualError = 0;

			for (int centreIndex = 0; centreIndex < residualError.size(); centreIndex++)
				totalResidualError += residualError[centreIndex];

			totalResidualError /= residualError.size();
		}
	}
}

void SLIC::enforceConnectivity(Mat LABImage)
{
// 	int label = 0, adjlabel = 0;
// 	const int clustersAverageSize = (int)(0.5 + pixelsNumber / clusterCentres.size());
// 
// 	const int dx4[4] = { -1, 0, 1, 0 };
// 	const int dy4[4] = { 0, -1, 0, 1 };
// 
// 	/* Initialize the new cluster matrix. */
// 	vector<vector<int>> new_clusters;
// 	for (int i = 0; i < LABImage.cols; i++) {
// 		vector<int> nc;
// 		for (int j = 0; j < LABImage.rows; j++) {
// 			nc.push_back(-1);
// 		}
// 		new_clusters.push_back(nc);
// 	}
// 
// 	for (int i = 0; i < LABImage.cols; i++)
// 		for (int j = 0; j < LABImage.rows; j++)
// 			if (new_clusters[i][j] == -1) {
// 				vector<Point> elements;
// 				elements.push_back(Point(i, j));
// 
// 				/* Find an adjacent label, for possible use later. */
// 				for (int k = 0; k < 4; k++) {
// 					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
// 
// 					if (x >= 0 && x < LABImage.cols && y >= 0 && y < LABImage.rows) {
// 						if (new_clusters[x][y] >= 0) {
// 							adjlabel = new_clusters[x][y];
// 						}
// 					}
// 				}
// 
// 				int count = 1;
// 				for (int c = 0; c < count; c++)
// 					for (int k = 0; k < 4; k++)
// 					{
// 						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
// 
// 						if (x >= 0 && x < LABImage.cols && y >= 0 && y < LABImage.rows) {
// 							if (new_clusters[x][y] == -1 &&
// 								pixelCluster[j * LABImage.cols + i] == pixelCluster[y * LABImage.cols + x]) {
// 								elements.push_back(Point(x, y));
// 								new_clusters[x][y] = label;
// 								count += 1;
// 							}
// 						}
// 					}
// 
// 				/* Use the earlier found adjacent label if a segment size is
// 				smaller than a limit. */
// 				if (count <= clustersAverageSize >> 2) {
// 					for (int c = 0; c < count; c++) {
// 						new_clusters[elements[c].x][elements[c].y] = adjlabel;
// 					}
// 					label -= 1;
// 				}
// 				label += 1;
// 			}
}

void SLIC::colorSuperpixels(Mat LABImage)
{
	/* Create a matrix which will store the color of each cluster. */
	vector<Vec3f> clusterColors(clusterCentres.size());

	/* Sum color information of all the pixels in the same cluster
	   for future average color calculation. */
	for (int y = 0; y < LABImage.rows; y++)
		for (int x = 0; x < LABImage.cols; x++)
		{
			/* Get pixel's cluster. */
			int tempColorIndex = pixelCluster[y * LABImage.cols + x];

			/* Continue to next loop if the pixel doesn't
			   belong to any cluster. */
			if (tempColorIndex <= -1)
				continue;

			/* Get pixel color. */
			Vec3f tempColor = LABImage.at<Vec3b>(y, x);

			clusterColors[tempColorIndex].val[0] += tempColor.val[0];
 			clusterColors[tempColorIndex].val[1] += tempColor.val[1];
 			clusterColors[tempColorIndex].val[2] += tempColor.val[2];
		}

	/* Divide by the number of pixels in each cluster to get the
	   average cluster color. */
	for (int n = 0; n < clusterColors.size(); n++)
	{
		/* Continue to next loop if a cluster is empty. */
		if (pixelsOfSameCluster[n] <= 0)
			continue;

		clusterColors[n].val[0] /= pixelsOfSameCluster[n];
		clusterColors[n].val[1] /= pixelsOfSameCluster[n];
		clusterColors[n].val[2] /= pixelsOfSameCluster[n];
	}

	/* Fill in each cluster with its average color. */
	for (int y = 0; y < LABImage.rows; y++)
		for (int x = 0; x < LABImage.cols; x++)
			if (pixelCluster[y * LABImage.cols + x] >= 0 &&
				pixelCluster[y * LABImage.cols + x] < clusterColors.size())
			{
				Vec3f tempColor = clusterColors[pixelCluster[y * LABImage.cols + x]];
				LABImage.at<Vec3b>(y, x) = tempColor;
			}
}

void SLIC::drawClusterContours(
	cv::Mat		LABImage,
	const cv::Vec3b	contourColor) 
{
	/* Create a matrix which will store contour pixels. */
	vector<Point> contourPixels;

	/* Create a matrix with bool values detailing whether a
	   pixel is a contour or not. */
	vector<bool> isContour;

	/* Initialize all pixels not to be contour pixels. */
	for (int n = 0; n < pixelsNumber; n++)
		isContour.push_back(false);

	/* Scan all the pixels and compare them to neighbor
	   pixels to see if they belong to a different cluster. */
	for (int y = 0; y < LABImage.rows; y++) 
		for (int x = 0; x < LABImage.cols; x++)
		{
			/* Continue to next loop if the selected pixel
			   doesn't belong to any cluster. */
			if (pixelCluster[y * LABImage.cols + x] <= -1)
				continue;

			/* Number of different adjacent clusters to the 
			   selected pixel. */
			int differentAdjacentClusters = 0;

			/* Compare the pixel to its eight neighbor pixels. */
			for (int tempY = y - 1; tempY <= y + 1; tempY++) 
				for (int tempX = x - 1; tempX <= x + 1; tempX++)
				{
					/* Verify that neighbor pixel is within the image boundaries. */
					if (tempX >= 0 && tempX < LABImage.cols && tempY >= 0 && tempY < LABImage.rows) 
						/* Verify that neighbor pixel belongs to a valid different cluster
						   and it's not already a contour pixel. */
						if (pixelCluster[tempY * LABImage.cols + tempX] > -1 &&
							pixelCluster[y * LABImage.cols + x] != pixelCluster[tempY * LABImage.cols + tempX] &&
							isContour[y * LABImage.cols + x] == false) 
							differentAdjacentClusters += 1;
				}

			/* Add the pixel to the contour matrix. */
			if (differentAdjacentClusters > 1)
			{
				contourPixels.push_back(Point(x, y));
				isContour[y * LABImage.cols + x] = true;
			}
		}

	/* Draw the contour pixels. */
	for (int n = 0; n < contourPixels.size(); n++)
		LABImage.at<Vec3b>(contourPixels[n].y, contourPixels[n].x) = contourColor;

// 	/* Mostra pixel orfani. */
// 	for (int y = 0; y < LABImage.rows; y++) 
// 		for (int x = 0; x < LABImage.cols; x++) 
// 			if (pixelCluster[y * LABImage.cols + x] == -1)
// 				//circle(LABImage, Point(x, y), 3, Scalar(0, 255, 0), 2);
// 				std::cout << "Trovato pixel orfano: x = " << x << "   y = " << y << "\n";
}

void SLIC::drawClusterCentres(
	cv::Mat		 LABImage,
	const cv::Scalar centreColor)
{
	for (int i = 0; i < clusterCentres.size(); i++)
		circle(LABImage, Point((int)clusterCentres[i][3], (int)clusterCentres[i][4]), 3, centreColor, 2);
}
