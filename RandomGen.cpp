/********************************************************************
created:	2011/12/01

filename: src/math/Random.cpp

file base:	Random
file ext:	cpp
author:		Pietro Morerio

purpose:	generation of random numbers (with specific distributions)
*********************************************************************/
#include "RandomGen.h"


/*** Poisson Distribution ***/
RandPoisson::RandPoisson(int mean) : m_rnd_gen(static_cast<unsigned> (time(0))),
	m_poisson_gen(m_rnd_gen, boost::poisson_distribution<>( mean ))
{
	m_iMean = mean;
	m_iVariance = mean;
}
/*** Binomial Distribution ***/
RandBinomial::RandBinomial(int n, double p) : m_rnd_gen(static_cast<unsigned> (time(0))),
	m_binomial_gen(m_rnd_gen, boost::binomial_distribution<>(n, p))
{
	m_iN = n;
	m_dP = p;
	m_dMean = p*n;
	m_dVariance = p*n*(1-p);
}
/*** Normal Distribution ***/
RandNormal::RandNormal(double mean, double stddev) : m_rnd_gen(static_cast<unsigned> (time(0))),
	m_normal_gen(m_rnd_gen, boost::normal_distribution<>(mean, stddev))
{
	m_dMean = mean;
	m_dStdDev = stddev;

}