/****************************************************************************/
/*                                                                          */
/* Created:        2011/12/01                                               */
/*                                                                          */
/* Filename:       RandomGen.cpp                                            */
/*                                                                          */
/* File base:      RandomGen                                                */
/* File extension: cpp                                                      */
/* Author:         Pietro Morerio                                           */
/*                                                                          */
/* Purpose:        generation of random numbers                             */
/*                 with Gaussian distribution                               */
/*                                                                          */
/****************************************************************************/

#include "RandomGen.h"

/****************************************************************************/
/*                      Normal (Gaussian) Distribution                      */
/****************************************************************************/
RandNormal::RandNormal(double mean, double stddev)
	: m_rnd_gen(static_cast<unsigned>(time(0))), m_normal_gen(m_rnd_gen, boost::normal_distribution<>(mean, stddev))
{
	m_dMean = mean;
	m_dStdDev = stddev;
}

double RandNormal::GetMean()
{
	return m_dMean;
}

double RandNormal::GetStdDev()
{
	return m_dStdDev;
}

double RandNormal::GetVariance()
{
	return m_dStdDev * m_dStdDev;
}

double RandNormal::operator()()
{
	return m_normal_gen();
}