#include "SLIC.h"
#include "RandomGen.h"

/* OpenCV libraries for video and image
   elaborations. */
#include <opencv/cv.h>
#include <opencv/highgui.h>

/* Intel Threading Building Blocks libraries
   for multi-threading. */
#include <tbb/tbb.h>

#include <vector>

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
	this->clustersNumber        = otherSLIC.clustersNumber;
	this->samplingStep          = otherSLIC.samplingStep;
	this->spatialDistanceWeight = otherSLIC.spatialDistanceWeight;
	this->distanceFactor        = otherSLIC.distanceFactor;
	this->totalResidualError    = otherSLIC.totalResidualError;
	this->errorThreshold        = otherSLIC.errorThreshold;

	/* Copy matrices. */
	this->pixelCluster.resize(otherSLIC.pixelsNumber);
	this->distanceFromClusterCentre.resize(otherSLIC.pixelsNumber);
	this->clusterCentres.resize(otherSLIC.clusterCentres.size());
	this->previousClusterCentres.resize(otherSLIC.previousClusterCentres.size());
	this->pixelsOfSameCluster.resize(otherSLIC.pixelsOfSameCluster.size());
	this->residualError.resize(otherSLIC.residualError.size());

	for (int n = 0; n < otherSLIC.pixelsNumber; ++n)
	{
		this->pixelCluster[n]              = otherSLIC.pixelCluster[n];
		this->distanceFromClusterCentre[n] = otherSLIC.distanceFromClusterCentre[n];
	}

	for (int n = 0; n < otherSLIC.clustersNumber; ++n)
	{
		this->pixelsOfSameCluster[n]    = otherSLIC.pixelsOfSameCluster[n];
		this->residualError[n]          = otherSLIC.residualError[n];
	}

	for (int n = 0; n < 5 * otherSLIC.clustersNumber; ++n)
	{
		this->clusterCentres[n]         = otherSLIC.clusterCentres[n];
		this->previousClusterCentres[n] = otherSLIC.previousClusterCentres[n];
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
	this->clustersNumber        = 0;
	this->pixelsNumber          = 0;
	this->samplingStep          = 0;
	this->spatialDistanceWeight = 0;
	this->distanceFactor        = 0;
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
	const cv::Mat image,
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
	if (firstVideoFrame == true || static_cast<int>(clusterCentres.size()) <= 0)
	{
		/* Clear previous data before initialization. */
		clearSLICData();

		/* Initialize debug data. */
		this->minError          = FLT_MAX;
		this->minIterations     = INT_MAX;
		this->minExecutionTime  = INT_MAX;

		/* Initialize variables. */
		this->pixelsNumber          = image.rows * image.cols;
		this->samplingStep          = samplingStep;
		this->spatialDistanceWeight = spatialDistanceWeight;
		this->distanceFactor        =
			spatialDistanceWeight * spatialDistanceWeight / (samplingStep * samplingStep);
		this->totalResidualError    = FLT_MAX;
		this->errorThreshold        = 0.25;

		/* Initialize the clusters and the distances matrices. */
		for (int n = 0; n < pixelsNumber; ++n)
		{
			pixelCluster.push_back(-1);
			distanceFromClusterCentre.push_back(FLT_MAX);
		}

		/* Initialize the centres matrix by sampling the image
		   at a regular step. */
		for (int y = samplingStep; y < image.rows; y += samplingStep)
			for (int x = samplingStep; x < image.cols; x += samplingStep)
			{
				/* Find the pixel with the lowest gradient in a 3x3 surrounding. */
				Point lowestGradientPixel = findLowestGradient(image, Point(x, y));
				Vec3b tempPixelColor      = image.at<Vec3b>(lowestGradientPixel.y, lowestGradientPixel.x);

				/* Insert a [l, a, b, x, y] centre in the centres vector. */
				clusterCentres.push_back(tempPixelColor.val[0]);
				clusterCentres.push_back(tempPixelColor.val[1]);
				clusterCentres.push_back(tempPixelColor.val[2]);
				clusterCentres.push_back(lowestGradientPixel.x);
				clusterCentres.push_back(lowestGradientPixel.y);

				previousClusterCentres.push_back(tempPixelColor.val[0]);
				previousClusterCentres.push_back(tempPixelColor.val[1]);
				previousClusterCentres.push_back(tempPixelColor.val[2]);
				previousClusterCentres.push_back(lowestGradientPixel.x);
				previousClusterCentres.push_back(lowestGradientPixel.y);
				
				/* Initialize "pixel of same cluster" matrix
				   (with 1 because of the new centre per cluster). */
				pixelsOfSameCluster.push_back(1);

				/* Initialize residual error to be zero for each cluster
				   centre. */
				residualError.push_back(0);
			}

		this->clustersNumber = static_cast<int>(pixelsOfSameCluster.size());
	}
	else 
	{
		/* Random noise generator. */
		RandNormal randomGen(0.0, static_cast<double>(samplingStep / 5));

		/* Add some gaussian noise to position. */
		/* Color should be kept equal: we look for a similar color in the surroundings. */
		for (int n = 0; n < clustersNumber; ++n)
		{
			clusterCentres[5 * n + 3] += randomGen();
			clusterCentres[5 * n + 4] += randomGen();
		}
	}
}

Point SLIC::findLowestGradient(
	const cv::Mat   image,
	const cv::Point centre) 
{
	double lowestGradient     = FLT_MAX;
	Point lowestGradientPoint = Point(centre.x, centre.y);

	for (int y = centre.y - 1; y <= centre.y + 1 && y < image.rows - 1; ++y) 
		for (int x = centre.x - 1; x <= centre.x + 1 && x < image.cols - 1; ++x) 
		{
			/* Exclude pixels on borders. */
			if (x < 1)
				continue;
			if (y < 1)
				continue;

			Vec3b tempPixelUp    = image.at<Vec3b>(y - 1, x);
			Vec3b tempPixelDown  = image.at<Vec3b>(y + 1, x);
			Vec3b tempPixelRight = image.at<Vec3b>(y, x + 1);
			Vec3b tempPixelLeft  = image.at<Vec3b>(y, x - 1);

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
		(clusterCentres[5 * centreIndex]     - pixelColor.val[0]) *
		(clusterCentres[5 * centreIndex]     - pixelColor.val[0]) +
		(clusterCentres[5 * centreIndex + 1] - pixelColor.val[1]) *
		(clusterCentres[5 * centreIndex + 1] - pixelColor.val[1]) + 
		(clusterCentres[5 * centreIndex + 2] - pixelColor.val[2]) *
		(clusterCentres[5 * centreIndex + 2] - pixelColor.val[2]);

	/* Compute the spatial distance between two pixels. */
	double spaceDistance =
		(clusterCentres[5 * centreIndex + 3] - pixelPosition.x) *
		(clusterCentres[5 * centreIndex + 3] - pixelPosition.x) + 
		(clusterCentres[5 * centreIndex + 4] - pixelPosition.y) *
		(clusterCentres[5 * centreIndex + 4] - pixelPosition.y);

	/* Compute total distance between two pixels using the formula 
	   described by the algorithm. */
	return colorDistance + spaceDistance * distanceFactor;
}

void SLIC::createSuperpixels(
	const cv::Mat image,
	const int     samplingStep,
	const int     spatialDistanceWeight,
	const bool    firstVideoFrame)
{
	/* Initialize algorithm data. */
	initializeSLICData(
		image, samplingStep, spatialDistanceWeight, firstVideoFrame);

	/* Repeat next steps until error is lower than the threshold. */
	for (iterationIndex = 0; totalResidualError > errorThreshold; ++iterationIndex)
	{
		/* Reset distance values. */
		tbb::parallel_for(0, pixelsNumber, 1, [&](int n)
		{
			distanceFromClusterCentre[n] = FLT_MAX;
		});
		
		tbb::parallel_for(0, clustersNumber, 1, [&](int centreIndex)
		{
			/* Look for pixels in a 2 x step by 2 x step region only. */
			for (int y = static_cast<int>(clusterCentres[5 * centreIndex + 4]) - samplingStep - 1;
				y < clusterCentres[5 * centreIndex + 4] + samplingStep + 1; ++y) 
				for (int x = static_cast<int>(clusterCentres[5 * centreIndex + 3]) - samplingStep - 1;
					x < clusterCentres[5 * centreIndex + 3] + samplingStep + 1; ++x)
				{
					/* Verify that neighbor pixel is within the image boundaries. */
					if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
					{
						Vec3b pixelColor = image.at<Vec3b>(y, x);
						
						double tempDistance =
							computeDistance(centreIndex, Point(x, y), pixelColor);

						/* Update pixel's cluster if this distance is smaller
						   than pixel's previous distance. */
						if (tempDistance < distanceFromClusterCentre[y * image.cols + x])
						{
							distanceFromClusterCentre[y * image.cols + x] = tempDistance;
							pixelCluster[y * image.cols + x]              = centreIndex;
						}
					}
				}
		});

		/* Reset centres values and the number of pixel
		   per cluster to zero. */
		tbb::parallel_for(0, clustersNumber, 1, [&](int centreIndex)
		{
			clusterCentres[5 * centreIndex]     = 0;
			clusterCentres[5 * centreIndex + 1] = 0;
			clusterCentres[5 * centreIndex + 2] = 0;
			clusterCentres[5 * centreIndex + 3] = 0;
			clusterCentres[5 * centreIndex + 4] = 0;

			pixelsOfSameCluster[centreIndex]    = 0;
		});

		/* Compute the new cluster centres. */
		for (int y = 0; y < image.rows; ++y) 
			for (int x = 0; x < image.cols; ++x) 
			{
				int currentPixelCluster = pixelCluster[y * image.cols + x];

				/* Verify if current pixel belongs to a cluster. */
				if (currentPixelCluster != -1)
				{
					/* Sum the information of pixels of the same
					   cluster for future centre recalculation. */
					Vec3b pixelColor = image.at<Vec3b>(y, x);

					clusterCentres[5 * currentPixelCluster]     += pixelColor.val[0];
					clusterCentres[5 * currentPixelCluster + 1] += pixelColor.val[1];
					clusterCentres[5 * currentPixelCluster + 2] += pixelColor.val[2];
					clusterCentres[5 * currentPixelCluster + 3] += x;
					clusterCentres[5 * currentPixelCluster + 4] += y;

					pixelsOfSameCluster[currentPixelCluster]    += 1;
				}
			}

		/* Normalize the clusters' centres. */
		tbb::parallel_for(0, clustersNumber, 1, [&](int centreIndex)
		{
			/* Avoid empty clusters, if there are any. */
			if (pixelsOfSameCluster[centreIndex] != 0)
			{
				clusterCentres[5 * centreIndex]     /= pixelsOfSameCluster[centreIndex];
				clusterCentres[5 * centreIndex + 1] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[5 * centreIndex + 2] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[5 * centreIndex + 3] /= pixelsOfSameCluster[centreIndex];
				clusterCentres[5 * centreIndex + 4] /= pixelsOfSameCluster[centreIndex];
			}
		});

		/* Skip error calculation if this is the first iteration,
		   meaning this is a new frame in the video. */
		if (iterationIndex == 0)
			for (int centreIndex = 0; centreIndex < 5 * clustersNumber; ++centreIndex)
			{
				/* Update previous centres matrix. */
				previousClusterCentres[centreIndex] = clusterCentres[centreIndex];
			}
		else
		{
			/* Compute residual error. */
			tbb::parallel_for(0, clustersNumber, 1, [&](int centreIndex)
			{
				/* Calculate residual error for each cluster centre. */
				residualError[centreIndex] = sqrt(
					(clusterCentres[5 * centreIndex + 4] - previousClusterCentres[5 * centreIndex + 4]) *
					(clusterCentres[5 * centreIndex + 4] - previousClusterCentres[5 * centreIndex + 4]) +
					(clusterCentres[5 * centreIndex + 3] - previousClusterCentres[5 * centreIndex + 3]) *
					(clusterCentres[5 * centreIndex + 3] - previousClusterCentres[5 * centreIndex + 3]));

				/* Update previous centres matrix. */
				previousClusterCentres[5 * centreIndex]     = clusterCentres[5 * centreIndex];
				previousClusterCentres[5 * centreIndex + 1] = clusterCentres[5 * centreIndex + 1];
				previousClusterCentres[5 * centreIndex + 2] = clusterCentres[5 * centreIndex + 2];
				previousClusterCentres[5 * centreIndex + 3] = clusterCentres[5 * centreIndex + 3];
				previousClusterCentres[5 * centreIndex + 4] = clusterCentres[5 * centreIndex + 4];
			});

			/* Compute total residual error by averaging all clusters' errors. */
			totalResidualError = 0;

			for (int centreIndex = 0; centreIndex < clustersNumber; ++centreIndex)
				totalResidualError += residualError[centreIndex];

			totalResidualError /= clustersNumber;
		}
	}
}

void SLIC::colorSuperpixels(
	cv::Mat  image,
	cv::Rect areaToColor)
{
	/* Verify that area to color is within image boundaries,
	   otherwise reset area to the entire image. */
	if (areaToColor.x < 0 || areaToColor.x > image.cols)
		areaToColor.x = 0;
	if (areaToColor.y < 0 || areaToColor.y > image.rows)
		areaToColor.y = 0;
	if (areaToColor.width < 0 || areaToColor.x + areaToColor.width > image.cols)
		areaToColor.width = image.cols - areaToColor.x;
	if (areaToColor.height < 0 || areaToColor.y + areaToColor.height > image.rows)
		areaToColor.height = image.rows - areaToColor.y;

	/* Fill in each cluster with its average color (cluster centre color). */
	tbb::parallel_for(areaToColor.y, areaToColor.y + areaToColor.height, 1, [&](int y)
	{
		for (int x = areaToColor.x; x < areaToColor.x + areaToColor.width; ++x)
			if (pixelCluster[y * image.cols + x] >= 0 &&
				pixelCluster[y * image.cols + x] < clustersNumber)
			{
				image.at<Vec3b>(y, x) = Vec3d(
					clusterCentres[5 * pixelCluster[y * image.cols + x]],
					clusterCentres[5 * pixelCluster[y * image.cols + x] + 1],
					clusterCentres[5 * pixelCluster[y * image.cols + x] + 2]);
			}
	});
}

void SLIC::drawClusterContours(
	cv::Mat         image,
	const cv::Vec3b	contourColor,
	cv::Rect        areaToDraw) 
{
	/* Verify that area to color is within image boundaries,
	   otherwise reset area to the entire image. */
	if (areaToDraw.x < 0 || areaToDraw.x > image.cols)
		areaToDraw.x = 0;
	if (areaToDraw.y < 0 || areaToDraw.y > image.rows)
		areaToDraw.y = 0;
	if (areaToDraw.width < 0 || areaToDraw.x + areaToDraw.width > image.cols)
		areaToDraw.width = image.cols - areaToDraw.x;
	if (areaToDraw.height < 0 || areaToDraw.y + areaToDraw.height > image.rows)
		areaToDraw.height = image.rows - areaToDraw.y;

	/* Create a matrix with bool values detailing whether a
	   pixel is a contour or not. */
	vector<bool> isContour(pixelsNumber);

	/* Scan all the pixels and compare them to neighbor
	   pixels to see if they belong to a different cluster. */
	tbb::parallel_for(areaToDraw.y, areaToDraw.y + areaToDraw.height, 1, [&](int y)
	{
		for (int x = areaToDraw.x; x < areaToDraw.x + areaToDraw.width; ++x)
		{
			/* Continue only if the selected pixel
			   belongs to a cluster. */
			if (pixelCluster[y * image.cols + x] >= 0)
			{
				/* Compare the pixel to its eight neighbor pixels. */
				for (int tempY = y - 1; tempY <= y + 1; ++tempY) 
					for (int tempX = x - 1; tempX <= x + 1; ++tempX)
					{
						/* Verify that neighbor pixel is within the image boundaries. */
						if (tempX >= 0 && tempX < image.cols && tempY >= 0 && tempY < image.rows) 
							/* Verify that neighbor pixel belongs to a valid different cluster
							and it's not already a contour pixel. */
							if (pixelCluster[tempY * image.cols + tempX] > -1 &&
								pixelCluster[y * image.cols + x] != pixelCluster[tempY * image.cols + tempX] &&
								isContour[y * image.cols + x] == false)
							{
								isContour[y * image.cols + x] = true;
								/* Color contour pixel. */
								image.at<Vec3b>(y, x)         = contourColor;
							}
					}
			}
		}
	});
}

void SLIC::drawClusterCentres(
	cv::Mat          image,
	const cv::Scalar centreColor)
{
	tbb::parallel_for(0, clustersNumber, [&](int n)
	{
		/* Draw a circle on the image for each cluster centre. */
		circle(image, Point(static_cast<int>(clusterCentres[5 * n + 3]),
			static_cast<int>(clusterCentres[5 * n + 4])), 3, centreColor, 2);
	});
}

void SLIC::drawInformation(
	cv::Mat image,
	int     totalFrames,
	int     executionTimeInMilliseconds)
{
	++framesNumber;

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

	rectangle(image, Point(0, 0), Point(260, 320), CV_RGB(255, 255, 255), CV_FILLED);

	stringStream << "Frame: " << framesNumber << " (" << totalFrames << " total)";
	putText(image, stringStream.str(), Point(5, 20), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Superpixels: " << clustersNumber;
	putText(image, stringStream.str(), Point(5, 40), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Distance weight: " << spatialDistanceWeight;
	putText(image, stringStream.str(), Point(5, 60), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time now: " << executionTimeInMilliseconds << " ms";
	putText(image, stringStream.str(), Point(5, 80), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time max.: " << maxExecutionTime;
	putText(image, stringStream.str(), Point(5, 100), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time min.: " << minExecutionTime;
	putText(image, stringStream.str(), Point(5, 120), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Exe. time avg.: " << (averageExecutionTime += executionTimeInMilliseconds) / framesNumber << " ms";
	putText(image, stringStream.str(), Point(5, 140), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations now: " << iterationIndex;
	putText(image, stringStream.str(), Point(5, 160), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations max.: " << maxIterations;
	putText(image, stringStream.str(), Point(5, 180), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations min.: " << minIterations;
	putText(image, stringStream.str(), Point(5, 200), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Iterations avg.: " << (averageIterations += iterationIndex) / framesNumber;
	putText(image, stringStream.str(), Point(5, 220), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error now: " << totalResidualError;
	putText(image, stringStream.str(), Point(5, 240), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error max.: " << maxError;
	putText(image, stringStream.str(), Point(5, 260), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error min.: " << minError;
	putText(image, stringStream.str(), Point(5, 280), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);

	stringStream.str("");
	stringStream << "Error avg.: " << (averageError += totalResidualError) / framesNumber;
	putText(image, stringStream.str(), Point(5, 300), 
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(0, 0, 0), 1, CV_AA);
}

void SLIC::recognizeHands(cv::Mat image)
{

}