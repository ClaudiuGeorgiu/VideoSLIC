/****************************************************************************/
/*                                                                          */
/* Created:        2011/12/01                                               */
/*                                                                          */
/* Filename:       RandomGen.h                                              */
/*                                                                          */
/* File base:      RandomGen                                                */
/* File extension: h                                                        */
/* Author:         Pietro Morerio                                           */
/*                                                                          */
/* Purpose:        generation of random numbers                             */
/*                 with Gaussian distribution                               */
/*                                                                          */
/****************************************************************************/

#ifndef RANDOMGEN_H
#define RANDOMGEN_H

#include <boost/random/mersenne_twister.hpp> 
#include <boost/random/variate_generator.hpp> 
#include <boost/random/normal_distribution.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <time.h>

/****************************************************************************/
/*                      Normal (Gaussian) Distribution                      */
/****************************************************************************/
class RandNormal
{
	private:

		double	m_dMean;
		double	m_dStdDev;

		/* Mersenne Twister generator. */
		boost::mt19937 m_rnd_gen;

		typedef boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> rnd_normal_gen;
		rnd_normal_gen m_normal_gen;

	public:

		RandNormal(double mean, double stddev);

		double GetMean();

		double GetStdDev();

		double GetVariance();

		double operator()();
};

#endif


