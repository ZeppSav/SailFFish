/****************************************************************************
    SailFFish Library
    Copyright (C) 2022 Joseph Saverin j.saverin@tu-berlin.de

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Information on file:

    -> A range of important auxiliary functions required for definition of the Green's functions are defined here.
    -> The functions have been taken from:
        -> Mads Hejlesen's Fast Poisson solver: https://github.com/mmhej/poissonsolver  (bessel_int_J0 & sine_int)
        -> Denis-Gabriel Caprace & Thomas Gillis' FLUPS solver: https://github.com/vortexlab-uclouvain/flups (exp_int)

*****************************************************************************/

#include "../SailFFish_Math_Types.h"

//----------------------------------------------------------------------------//
// Approximation by Chebyshev polynomials to the integral Bessel function 
// of first kind and order 0: Ji0 = int( q^(-1) (1 - J0) )
// Luke, Y. L: Mathematical functions and their approximations (1975) Table 9.3
//----------------------------------------------------------------------------//
inline double bessel_int_J0( double x )
{
    const double pi    = 3.1415926535897932;
    const double gamma = 0.5772156649015329;

    double a[18] = {  1.35105091918187636388,
	                  0.83791030734868376979,
	                 -0.35047963978529462711,
	                  0.12777415867753198659,
	                 -0.02981035698255560990,
	                  0.00455219841169387328,
	                 -0.00048408621967185359,
	                  0.00003780202859916883,
	                 -0.00000225886908506771,
	                  0.00000010664609068423,
	                 -0.00000000408005443149,
	                  0.00000000012909996251,
	                 -0.00000000000343577839,
	                  0.00000000000007799552,
	                 -0.00000000000000152842,
	                  0.00000000000000002612,
	                 -0.00000000000000000039,
	                  0.00000000000000000001 };

    std::complex<double> c[39] = { { 0.95360150809738558095,-0.13917925930200001236},
	                               {-0.05860838853872331670,-0.12902065726135067062},
	                               {-0.01020283575659856676, 0.01103004348109535741},
	                               { 0.00196012704043622581, 0.00051817180856880364},
	                               {-0.00009574977697756219,-0.00030928210173975681},
	                               {-0.00003570479477043714, 0.00004647098443047525},
	                               { 0.00001169677960430223,-0.00000008198845340928},
	                               {-0.00000164386246452682,-0.00000191888381006925},
	                               {-0.00000007415845751760, 0.00000057813667761104},
	                               { 0.00000011434387527717,-0.00000008448997773317},
	                               {-0.00000003600903214141,-0.00000000525612161520},
	                               { 0.00000000601257386446, 0.00000000763257790924},
	                               { 0.00000000019124656215,-0.00000000268643963177},
	                               {-0.00000000054892028385, 0.00000000054279949860},
	                               { 0.00000000022740445656,-0.00000000001744365343},
	                               {-0.00000000005671490865,-0.00000000003975692920},
	                               { 0.00000000000607510983, 0.00000000002069683990},
	                               { 0.00000000000252060520,-0.00000000000639623674},
	                               {-0.00000000000191255246, 0.00000000000116359235},
	                               { 0.00000000000074056501, 0.00000000000006759603},
	                               {-0.00000000000018950214,-0.00000000000016557337},
	                               { 0.00000000000002021389, 0.00000000000008425597},
	                               { 0.00000000000001103617,-0.00000000000002824474},
	                               {-0.00000000000000889993, 0.00000000000000607698},
	                               { 0.00000000000000388558,-0.00000000000000003171},
	                               {-0.00000000000000119200,-0.00000000000000077237},
	                               { 0.00000000000000021456, 0.00000000000000048022},
	                               { 0.00000000000000002915,-0.00000000000000019502},
	                               {-0.00000000000000004877, 0.00000000000000005671},
	                               { 0.00000000000000002737,-0.00000000000000000862},
	                               {-0.00000000000000001080,-0.00000000000000000269},
	                               { 0.00000000000000000308, 0.00000000000000000309},
	                               {-0.00000000000000000042,-0.00000000000000000167},
	                               {-0.00000000000000000020, 0.00000000000000000066},
	                               { 0.00000000000000000020,-0.00000000000000000019},
	                               {-0.00000000000000000011, 0.00000000000000000003},
	                               { 0.00000000000000000004, 0.00000000000000000001},
	                               {-0.00000000000000000001,-0.00000000000000000001},
	                               { 0.00000000000000000000, 0.00000000000000000001} };


	int i;
    double ans;
    double T[3];
    double x8 = 0.125*x;
    double x5sh = 2.0*5.0/x - 1.0;
    double fac;
    std::complex<double> sum;

	if( x < 8.0 )
	{
		T[0] = 1.0;
		T[1] = x8;

		ans = T[0] * a[0];
		for( i = 2; i < 36; ++i )
		{
			T[2] = 2*x8*T[1] - T[0];

			if( i % 2 == 0 )
			{
				ans += a[i/2] * T[2];
			}
			T[0] = T[1];
			T[1] = T[2];
		}
	}
	else
	{
		T[0] = 1.0;
		T[1] = x5sh;

		sum  = c[0]*T[0];
		sum += c[1]*T[1];
		for( i = 2; i < 39; ++i )
		{
			T[2] = 2.0*x5sh*T[1] - T[0];
			T[0] = T[1];
			T[1] = T[2];

			sum += c[i] * T[2];
		}
        fac = cos(x + 0.25 * pi) * std::real( sum ) - sin(x + 0.25 * pi) * std::imag( sum );
		ans = sqrt( 2.0/(pi*x) )/x * fac + gamma + log(0.5*x);
	}

//----------------------------------------------------------------------------//
// Return
//----------------------------------------------------------------------------//
	return ans;
}

//----------------------------------------------------------------------------//
// Sine integral function
// This function is taken from: 
// W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery:
//'Numerical recipes in Fortran: The art of scientific computing' 2.ed, 1986
//----------------------------------------------------------------------------//

inline double sine_int( double x)
{
//----------------------------------------------------------------------------//
// Local variables
//----------------------------------------------------------------------------//
    const double pi    = 3.1415926535897932;
    const double gamma = 0.5772156649015329;

	int maxit = 1000;
    double eps = 1.0e-10;
    double fpmin = 1.0e-30;
    double fpmax = 1.0e30;
    double tmin = 2.0;

	int i,k;

    double a,err,fact,sign,sum,sumc,sums,t,term;
    std::complex<double> h,g,b,c,d,del;
	bool odd;

    double ans = 0.0;

	t = std::abs(x);
	if(t == 0.0)
	{
		ans = 0.0;
		return ans;
	}

// Evaluate continued fraction by modified Lentz’s method (§5.2).
	if(t > tmin)
	{
		b = {1.0,t};
		c = fpmax;
		d = 1.0/b;
		h = d;

		i = 2;
		err = 1.0;
        while( i < maxit && err > eps )
		{
			a = -pow( i-1 ,2);
			b = b + 2.0;
			d = 1.0/(a*d+b); // Denominators cannot be zero.
			c = b + a/c;
			del = c*d;
			h = h*del;

            err = std::abs( std::real( del-1.0 ) ) + std::abs( std::imag( del-1.0 ) );
			++i;

		}

		if(i >= maxit){ std::cerr << " [sine_int]: Continued fraction failed" << std::endl; }

		g = {cos(t),-sin(t)};
		h = g * h;

		ans = 0.5*pi + std::imag(h);
	}
// Evaluate both series simultaneously.
	else
	{

// Special case: avoid failure of convergence test because of underflow.
		if(t < sqrt(fpmin))
		{
			sumc = 0.0;
			sums = t;
		}
		else
		{
			sum  = 0.0;
			sums = 0.0;
			sumc = 0.0;
			sign = 1.0;
			fact = 1.0;
			odd  = true;

			k = 1;
			err = 1.0;
			while( k <= maxit && err > eps )
			{
				fact = fact*t/k;
				term = fact/k;
				sum = sum + sign*term;
				err = term/std::abs(sum);
				if(odd)
				{
					sign = -sign;
					sums = sum;
					sum  = sumc;
				}
				else
				{
					sumc = sum;
					sum  = sums;
				}

				odd = !odd;
				++k;
			}
			if(k >= maxit){ std::cerr << " [sine_int]: MAXIT exceeded" << std::endl; }
		}

		ans = sums;
	}

	if(x < 0.0){ ans = -ans; }

//----------------------------------------------------------------------------//
// Return
//----------------------------------------------------------------------------//
	return ans;
}

/**********************************************************************/
/*                                                                    */
/*                      double expint1()                              */
/*                                                                    */
/**********************************************************************/
/*                                                                    */
/*  DESCRIPTION:                                                      */
/*  Calculation of the exponential integral  for -4<= x <= 4 using an */
/*  expansion in terms of Chebyshev polynomials.                      */
/*                                                                    */
/**********************************************************************/

inline double expint1(double x)
{
    static int MAX = 23; /* The number of coefficients in a[].   */

    static double a[23] = {7.8737715392882774,
                           -8.0314874286705335,
                           3.8797325768522250,
                           -1.6042971072992259,
                           0.5630905453891458,
                           -0.1704423017433357,
                           0.0452099390015415,
                           -0.0106538986439085,
                           0.0022562638123478,
                           -0.0004335700473221,
                           0.0000762166811878,
                           -0.0000123417443064,
                           0.0000018519745698,
                           -0.0000002588698662,
                           0.0000000338604319,
                           -0.0000000041611418,
                           0.0000000004821606,
                           -0.0000000000528465,
                           0.0000000000054945,
                           -0.0000000000005433,
                           0.0000000000000512,
                           -0.0000000000000046,
                           0.0000000000000004};

    int    k;
    double arg, t, value, b0, b1, b2;

    arg = .25 * x; /* Argument in Chebyshev expansion is x/4. */
    t   = 2. * arg;

    b2 = 0.;
    b1 = 0.;
    b0 = a[MAX - 1];

    for (k = MAX - 2; k >= 0; k--) {
        b2 = b1;
        b1 = b0;
        b0 = t * b1 - b2 + a[k];
    }

    value = .5 * (b0 - b2);

    value += log(fabs(x));

    return (-value);

}

/**********************************************************************/
/*                                                                    */
/*                      double expint2()                              */
/*                                                                    */
/**********************************************************************/
/*                                                                    */
/*  DESCRIPTION:                                                      */
/*  Calculation of the exponential integral for x >= 4 using an expan-*/
/*  sionin terms of Chebyshev polynomials.                            */
/*                                                                    */
/**********************************************************************/

inline double expint2(double x)
{
    static int MAX = 23; /* The number of coefficients in a[].   */

    static double a[23] = {0.2155283776715125,
                           0.1028106215227030,
                           -0.0045526707131788,
                           0.0003571613122851,
                           -0.0000379341616932,
                           0.0000049143944914,
                           -0.0000007355024922,
                           0.0000001230603606,
                           -0.0000000225236907,
                           0.0000000044412375,
                           -0.0000000009328509,
                           0.0000000002069297,
                           -0.0000000000481502,
                           0.0000000000116891,
                           -0.0000000000029474,
                           0.0000000000007691,
                           -0.0000000000002070,
                           0.0000000000000573,
                           -0.0000000000000163,
                           0.0000000000000047,
                           -0.0000000000000014,
                           0.0000000000000004,
                           -0.0000000000000001};

    int    k;
    double arg, t, value, b0, b1, b2;

    arg = 4. / x; /* Argument in the Chebyshev expansion.       */
    t   = 2. * (2. * arg - 1.);

    b2 = 0.;
    b1 = 0.;
    b0 = a[MAX - 1];

    for (k = MAX - 2; k >= 0; k--) {
        b2 = b1;
        b1 = b0;
        b0 = t * b1 - b2 + a[k];
    }

    value = .5 * (b0 - b2);

    value *= exp(-x);

    return (value);

}

inline double expint_ei(double x)
{
    double value;

    if (x >= -4. && x <= 4.)
        value = expint1(x);
    else if (x > 4.)
        value = expint2(x);
    else {
        value = 0.;
    }

    return (value);
}
