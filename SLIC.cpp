#include "SLIC.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

/* Intel Threading Building Blocks libraries
   for multi-threading. */
#include <tbb/tbb.h>

#include "RandomGen.h"

using namespace cv;

SLIC::SLIC()
{
	clearSLICData();
}

SLIC::SLIC(const SLIC& otherSLIC)
{
	/* Copy variables. */
	this->iterationIndex        = otherSLIC.iterationIndex;
	this->pixelsNumber          = otherSLIC.pixelsNumber;
	this->samplingStep          = otherSLIC.samplingStep;
	this->spatialDistanceWeight = otherSLIC.spatialDistanceWeight;
	this->totalResidualError    = otherSLIC.totalResidualError;
	this->errorThreshold        = otherSLIC.errorThreshold;

	/* Copy matrices. */
	this->pixelCluster.resize(otherSLIC.pixelsNumber);
	this->distanceFromClusterCentre.resize(otherSLIC.pixelsNumber);
	this->clusterCentres.resize(otherSLIC.clusterCentres.size());
	this->previousClusterCentres.resize(otherSLIC.previousClusterCentres.size());
	this->pixelsOfSameCluster.resize(otherSLIC.pixelsOfSameCluster.size());
	this->residualError.resize(otherSLIC.residualError.size());

	for (int n = 0; n < otherSLIC.pixelsNumber; n++)
	{
		this->pixelCluster[n]              = otherSLIC.pixelCluster[n];
		this->distanceFromClusterCentre[n] = otherSLIC.distanceFromClusterCentre[n];
	}

	for (int n = 0; n < otherSLIC.clusterCentres.size(); n++)
	{
		this->clusterCentres[n]         = otherSLIC.clusterCentres[n];
		this->previousClusterCentres[n] = otherSLIC.previousClusterCentres[n];
		this->pixelsOfSameCluster[n]    = otherSLIC.pixelsOfSameCluster[n];
		this->residualError[n]          = otherSLIC.residualError[n];
	}
}

SLIC::~SLIC()
{
	clearSLICData();
}

void SLIC::clearSLICData()
{
	/* Reset debug data. */
	this->framesNumber         = 0;
	this->averageError         = 0;
	this->averageIterations    = 0;
	this->minError             = 0;
	this->minIterations        = 0;
	this->maxError             = 0;
	this->maxIterations        = 0;
	this->averageExecutionTime = 0;
	this->minExecutionTime     = 0;
	this->maxExecutionTime     = 0;

	/* Reset variables. */
	this->iterationIndex        = 0;
	this->pixelsNumber          = 0;
	this->samplingStep          = 0;
	this->spatialDistanceWeight = 0;
	this->totalResidualError    = 0;
	this->errorThreshold        = 0;

	/* Erase all matrices' elements. */
	this->pixelCluster.clear();
	this->distanceFromClusterCentre.clear();
	this->clusterCentres.clear();
	this->previousClusterCentres.clear();
	this->pixelsOfSameCluster.clear();
	this->residualError.clear();
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

		/* Initialize debug data. */
		this->minError          = FLT_MAX;
		this->minIterations     = INT_MAX;
		this->minExecutionTime  = INT_MAX;

		/* Initialize variables. */
		this->pixelsNumber          = LABImage.rows * LABImage.cols;
		this->samplingStep          = samplingStep;
		this->spatialDistanceWeight = spatialDistanceWeight;
		this->totalResidualError    = FLT_MAX;
		this->errorThreshold        = 0.25;

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

		/* Initialize debug data. */
		this->minError          = FLT_MAX;
		this->minIterations     = INT_MAX;
		this->minExecutionTime  = INT_MAX;

		/* Initialize variables. */
		this->pixelsNumber          = LABImage.rows * LABImage.cols;
		this->samplingStep          = samplingStep;
		this->spatialDistanceWeight = spatialDistanceWeight;
		this->totalResidualError    = FLT_MAX;
		this->errorThreshold        = 0.25;

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
	else 
	{

		/*random generator*/
		RandNormal randomGen(0.0, double(samplingStep*0.5)); // std dev could be tuned

		/* add some gaussian noise to position */
		/* colour should be kept equal: we look for a similar colour in the surroundings*/
		for (int i = 0; i < (int)clusterCentres.size() ; i++)
		{
			clusterCentres[i][3] += randomGen();
			clusterCentres[i][4] += randomGen();
		}
	}
}

Point SLIC::findLowestGradient(
	const cv::Mat   LABImage,
	const cv::Point centre) 
{
	double lowestGradient     = FLT_MAX;
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

	/* Repeat next steps until error is lower than the threshold. */
	for (iterationIndex = 0; totalResidualError > errorThreshold; iterationIndex++)
	{
		/* Reset distance values. */
		tbb::parallel_for(0, pixelsNumber, 1, [&](int n)
		{
			distanceFromClusterCentre[n] = FLT_MAX;
		});
		
		tbb::parallel_for(0, static_cast<int>(clusterCentres.size()), 1, [&](int centreIndex)
		// Questa parte funziona, ma andrebbe modificata perché non è thread safe dato che
		// due pixel potrebbero "controllare" due cluster diversi contemporaneamente
		{
			/* Look for pixels in a 2 x step by 2 x step region only. */
			for (int y = static_cast<int>(clusterCentres[centreIndex][4]) - samplingStep - 1;
				y < clusterCentres[centreIndex][4] + samplingStep + 1; y++) 
				for (int x = static_cast<int>(clusterCentres[centreIndex][3]) - samplingStep - 1;
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
		});

		/* Reset centres values and the number of pixel
		   per cluster to zero. */
		tbb::parallel_for(0, static_cast<int>(clusterCentres.size()), 1, [&](int centreIndex)
		{
			clusterCentres[centreIndex][0] = 0;
			clusterCentres[centreIndex][1] = 0;
			clusterCentres[centreIndex][2] = 0;
			clusterCentres[centreIndex][3] = 0;
			clusterCentres[centreIndex][4] = 0;

			pixelsOfSameCluster[centreIndex] = 0;
		});

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
		tbb::parallel_for(0, static_cast<int>(clusterCentres.size()), 1, [&](int centreIndex)
		{
			/* Avoid empty clusters, if there are any. */
			if (pixelsOfSameCluster[centreIndex] != 0)
			{
				clusterCentres[centreIndex][0] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[centreIndex][1] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[centreIndex][2] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[centreIndex][3] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[centreIndex][4] /= pixelsOfSameCluster[centreIndex];
			}
		});

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
			tbb::parallel_for(0, static_cast<int>(clusterCentres.size()), 1, [&](int centreIndex)
			{
				/* Calculate residual error for each cluster centre. */
				residualError[centreIndex] = sqrt(
					(clusterCentres[centreIndex][4] - previousClusterCentres[centreIndex][4]) *
					(clusterCentres[centreIndex][4] - previousClusterCentres[centreIndex][4]) +
					(clusterCentres[centreIndex][3] - previousClusterCentres[centreIndex][3]) *
					(clusterCentres[centreIndex][3] - previousClusterCentres[centreIndex][3]));

				/* Update previous centres matrix. */
				previousClusterCentres[centreIndex] = clusterCentres[centreIndex];
			});

			/* Compute total residual error by averaging all clusters' errors. */
			totalResidualError = 0;

			for (int centreIndex = 0; centreIndex < residualError.size(); centreIndex++)
				totalResidualError += residualError[centreIndex];

			totalResidualError /= residualError.size();
		}
	}
}

void SLIC::enforceConnectivity(cv::Mat LABImage)
{
// 	int label = 0, adjlabel = 0;
// 	const int clustersAverageSize = static_cast<int>(0.5 + pixelsNumber / clusterCentres.size());
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

void SLIC::colorSuperpixels(
	cv::Mat  LABImage,
	cv::Rect areaToColor)
{
	/* Verify that area to color is within image boundaries,
	   otherwise reset area to the entire image. */
	if (areaToColor.x < 0 || areaToColor.x > LABImage.cols)
		areaToColor.x = 0;
	if (areaToColor.y < 0 || areaToColor.y > LABImage.rows)
		areaToColor.y = 0;
	if (areaToColor.width < 0 || areaToColor.x + areaToColor.width > LABImage.cols)
		areaToColor.width = LABImage.cols - areaToColor.x;
	if (areaToColor.height < 0 || areaToColor.y + areaToColor.height > LABImage.rows)
		areaToColor.height = LABImage.rows - areaToColor.y;

	/* Create a matrix which will store the color of each cluster. */
	vector<Vec3f> clusterColors(clusterCentres.size());

	/* Sum color information of all the pixels in the same cluster
	   for future average color calculation. */
	for (int y = 0; y < LABImage.rows; y++) 
	{
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

			// Qui potrei avere accessi simultanei in scrittura e quindi
			// non posso andare in parallelo. Provando con un altro 
			// metodo parallelo, si riescono a guadagnare 2-3 ms.
			clusterColors[tempColorIndex].val[0] += tempColor.val[0];
			clusterColors[tempColorIndex].val[1] += tempColor.val[1];
			clusterColors[tempColorIndex].val[2] += tempColor.val[2];
		}
	}

	/* Divide by the number of pixels in each cluster to get the
	   average cluster color. */
	tbb::parallel_for(0, static_cast<int>(clusterColors.size()), [&](int n)
	{
		/* Continue to next loop if a cluster is empty. */
		if (pixelsOfSameCluster[n] > 0)
		{
			clusterColors[n].val[0] /= pixelsOfSameCluster[n];
			clusterColors[n].val[1] /= pixelsOfSameCluster[n];
			clusterColors[n].val[2] /= pixelsOfSameCluster[n];
		}
	});

	/* Fill in each cluster with its average color. */
	tbb::parallel_for(areaToColor.y, areaToColor.y + areaToColor.height, 1, [&](int y)
	{
		for (int x = areaToColor.x; x < areaToColor.x + areaToColor.width; x++)
			if (pixelCluster[y * LABImage.cols + x] >= 0 &&
				pixelCluster[y * LABImage.cols + x] < clusterColors.size())
			{
				Vec3f tempColor = clusterColors[pixelCluster[y * LABImage.cols + x]];
				LABImage.at<Vec3b>(y, x) = tempColor;
			}
	});
}

void SLIC::drawClusterContours(
	cv::Mat         LABImage,
	const cv::Vec3b	contourColor,
	cv::Rect        areaToDraw) 
{
	/* Verify that area to color is within image boundaries,
	   otherwise reset area to the entire image. */
	if (areaToDraw.x < 0 || areaToDraw.x > LABImage.cols)
		areaToDraw.x = 0;
	if (areaToDraw.y < 0 || areaToDraw.y > LABImage.rows)
		areaToDraw.y = 0;
	if (areaToDraw.width < 0 || areaToDraw.x + areaToDraw.width > LABImage.cols)
		areaToDraw.width = LABImage.cols - areaToDraw.x;
	if (areaToDraw.height < 0 || areaToDraw.y + areaToDraw.height > LABImage.rows)
		areaToDraw.height = LABImage.rows - areaToDraw.y;

	/* Create a matrix with bool values detailing whether a
	   pixel is a contour or not. */
	vector<bool> isContour(pixelsNumber);

	/* Scan all the pixels and compare them to neighbor
	   pixels to see if they belong to a different cluster. */
	tbb::parallel_for(areaToDraw.y, areaToDraw.y + areaToDraw.height, 1, [&](int y)
	{
		for (int x = areaToDraw.x; x < areaToDraw.x + areaToDraw.width; x++)
		{
			/* Continue only if the selected pixel
			   belongs to a cluster. */
			if (pixelCluster[y * LABImage.cols + x] >= 0)
			{
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
							{
								isContour[y * LABImage.cols + x] = true;
								/* Color contour pixel. */
								LABImage.at<Vec3b>(y, x) = contourColor;
							}
					}
			}
		}
	});
}

void SLIC::drawClusterCentres(
	cv::Mat          LABImage,
	const cv::Scalar centreColor)
{
	tbb::parallel_for(0, static_cast<int>(clusterCentres.size()), [&](int n)
	{
		/* Draw a circle on the image for each cluster centre. */
		circle(LABImage, Point(static_cast<int>(clusterCentres[n][3]), static_cast<int>(clusterCentres[n][4])), 3, centreColor, 2);
	});
}

void SLIC::drawInformation(
	cv::Mat LABImage,
	int     totalFrames,
	int     executionTimeInMilliseconds)
{
	framesNumber++;

	std::ostringstream stringStream;

	if (totalResidualError < minError)
		minError = totalResidualError;
	if (totalResidualError > maxError)
		maxError = totalResidualError;
	if (iterationIndex < minIterations)
		minIterations = iterationIndex;
	if (iterationIndex > maxIterations)
		maxIterations = iterationIndex;
	if (executionTimeInMilliseconds < minExecutionTime)
		minExecutionTime = executionTimeInMilliseconds;
	if (executionTimeInMilliseconds > maxExecutionTime)
		maxExecutionTime = executionTimeInMilliseconds;

	rectangle(LABImage, Point(0, 0), Point(260, 320), CV_RGB(255, 255, 255), CV_FILLED);

	stringStream << "Frame: " << framesNumber << " (" << totalFrames << " total)";
	putText(LABImage, stringStream.str(), Point(5, 20), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Superpixels: " << clusterCentres.size();
	putText(LABImage, stringStream.str(), Point(5, 40), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Distance weight: " << spatialDistanceWeight;
	putText(LABImage, stringStream.str(), Point(5, 60), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time now: " << executionTimeInMilliseconds << " ms";
	putText(LABImage, stringStream.str(), Point(5, 80), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time max.: " << maxExecutionTime;
	putText(LABImage, stringStream.str(), Point(5, 100), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time min.: " << minExecutionTime;
	putText(LABImage, stringStream.str(), Point(5, 120), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time avg.: " << (averageExecutionTime += executionTimeInMilliseconds) / framesNumber << " ms";
	putText(LABImage, stringStream.str(), Point(5, 140), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations now: " << iterationIndex;
	putText(LABImage, stringStream.str(), Point(5, 160), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations max.: " << maxIterations;
	putText(LABImage, stringStream.str(), Point(5, 180), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations min.: " << minIterations;
	putText(LABImage, stringStream.str(), Point(5, 200), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations avg.: " << (averageIterations += iterationIndex) / framesNumber;
	putText(LABImage, stringStream.str(), Point(5, 220), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error now: " << totalResidualError;
	putText(LABImage, stringStream.str(), Point(5, 240), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error max.: " << maxError;
	putText(LABImage, stringStream.str(), Point(5, 260), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error min.: " << minError;
	putText(LABImage, stringStream.str(), Point(5, 280), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error avg.: " << (averageError += totalResidualError) / framesNumber;
	putText(LABImage, stringStream.str(), Point(5, 300), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
}

void SLIC::recognizerHands(cv::Mat& YCrCbImage)
{
// 	/* Average hand color in YCrCb color space (experimental). */
// 	const Vec3f averageHandColor(125.3, 150.2, 111.4);
// 	const int probabilityThreshold = 85;
// 
// 	vector<double> clusterProbabilityToBeHand(clusterCentres.size());
// 	vector<double> nearProbabilityToBeHand(clusterCentres.size());
// 	
// 	for (int n = 0; n < clusterCentres.size(); n++)
// 	{
// 		double primo = abs(averageHandColor[1] - clusterCentres[n][1]);
// 		double secondo = abs(averageHandColor[2] - clusterCentres[n][2]);
// 		clusterProbabilityToBeHand[n] = 100.0 - abs(primo + secondo);
// 	}
// 
// 	for (int n = 0; n < clusterCentres.size(); n++)
// 	{
// 		nearProbabilityToBeHand[n] = 0;
// 		for (int i = 0; i < clusterCentres.size(); i++)
// 		{
// 			if (n == i)
// 				continue;
// 			if (abs(clusterCentres[i][3] - clusterCentres[n][3]) <= 1.5 * samplingStep &&
// 				abs(clusterCentres[i][4] - clusterCentres[n][4]) <= 1.5 * samplingStep)
// 			{
// 				if (clusterProbabilityToBeHand[i] > probabilityThreshold &&
// 					clusterProbabilityToBeHand[n] > probabilityThreshold)
// 					nearProbabilityToBeHand[n]++;
// 				else nearProbabilityToBeHand[n]--;
// 			}
// 		}
// 	}
// 
//  	cvtColor(YCrCbImage, YCrCbImage, CV_YCrCb2BGR);
//  	cvtColor(YCrCbImage, YCrCbImage, CV_BGR2GRAY);
//  
// 	/* Fill in each cluster with its average color. */
// 	tbb::parallel_for(0, YCrCbImage.rows, 1, [&](int y)
// 	{
// 		for (int x = 0; x < YCrCbImage.cols; x++)
// 		{
// 			if (clusterProbabilityToBeHand[pixelCluster[y * YCrCbImage.cols + x]] > probabilityThreshold)
// 				YCrCbImage.at<uchar>(y, x) = 0;
// 			if (nearProbabilityToBeHand[pixelCluster[y * YCrCbImage.cols + x]] > 0)
// 				YCrCbImage.at<uchar>(y, x) = 0;
// 		}
// 	});
}