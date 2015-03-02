/****************************************************************************/
/*                                                                          */
/* Created:        2011/12/01                                               */
/*                                                                          */
/* Filename:       RandomGen.h                                              */
/*                                                                          */
/* File base:      RandomGen                                                */
/* File extension: cpp                                                      */
/* Author:         Pietro Morerio                                           */
/*                                                                          */
/* Purpose:        generation of random numbers                             */
/*                 (with specific distribution)                             */
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