/********************************************************************
created:	2011/12/01

filename: src/math/Random.h

file base:	Random
file ext:	h
author:		Pietro Morerio

purpose:	generation of random numbers (with specific distributions)
*********************************************************************/
#ifndef _RANDOM_H__
#define _RANDOM_H__

//#include "math40/MATH40_EXPORT.h"

#include <boost/random/mersenne_twister.hpp> 
#include <boost/random/variate_generator.hpp> 
#include <boost/random/poisson_distribution.hpp> 
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <time.h>



/*** Poisson Distribution ***/
class /*MATH40_EXPORT*/ RandPoisson
{

private:
	int m_iMean;
	int m_iVariance;
	boost::mt19937 m_rnd_gen; //Mersenne Twister generator 
	     

	typedef boost::variate_generator< boost::mt19937&, 
		boost::poisson_distribution<> > rnd_poisson_gen; 
	rnd_poisson_gen m_poisson_gen;

public:
	RandPoisson(int mean);
	int GetMean()		{return m_iMean;}
	int GetVariance()	{return m_iVariance;}
	int operator() ()	{return m_poisson_gen();}

};


/*** Binomial Distribution ***/
class /*MATH40_EXPORT*/ RandBinomial
{

private:
	int		m_iN;
	double	m_dP;
	double	m_dMean;
	double	m_dVariance;
	boost::mt19937 m_rnd_gen; //Mersenne Twister generator 
    //static boost::mt19937 m_rnd_gen(static_cast<unsigned int>(std::time(0)));
	typedef boost::variate_generator< boost::mt19937&, 
		boost::binomial_distribution<> > rnd_binomial_gen; 
	rnd_binomial_gen m_binomial_gen;

public:
	RandBinomial(int n, double p);
	int		GetN()			{return m_iN;}
	double	GetP()			{return m_dP;}
	double	GetMean()		{return m_dMean;}
	double	GeVariance()	{return m_dVariance;}
	double operator() () 	{return m_binomial_gen();}

};

/*** Normal Distribution ***/
class /*MATH40_EXPORT*/ RandNormal
{

private:

	double	m_dMean;
	double	m_dStdDev;
	boost::mt19937 m_rnd_gen; //Mersenne Twister generator 

	typedef boost::variate_generator< boost::mt19937&, 
		boost::normal_distribution<> > rnd_normal_gen; 
	rnd_normal_gen m_normal_gen;

public:
	RandNormal(double mean, double stddev);
	double	GetMean()		{return m_dMean;}
	double	GetStdDev()	{return m_dStdDev;}
	double	GetVariance()	{return pow(m_dStdDev,2);}
	double operator() () 	{return m_normal_gen();}

};

#endif /*_RANDOM_H__ */


