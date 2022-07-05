// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include "misc/cpu_math.h"

namespace jittor {

#define CENTRAL_RANGE 0.7

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_erfinv(T y) {
/* Function to calculate inverse error function.  Rational approximation
is used to generate an initial approximation, which is then improved to
full accuracy by two steps of Newton's method.  Code is a direct
translation of the erfinv m file in matlab version 2.0.
Author:  Gary L. Pavlis, Indiana University
Date:  February 1996
*/
  T x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */
  T a[4]={ 0.886226899, -1.645349621,  0.914624893, -0.140543331};
  T b[4]={-2.118377725,  1.442710462, -0.329097515,  0.012229801};
  T c[4]={-1.970840454, -1.624906493,  3.429567803,  1.641345311};
  T d[2]={ 3.543889200,  1.637067800};
  T y_abs = std::abs(y);
  if(y_abs > 1.0) return std::numeric_limits<T>::quiet_NaN();
  if(y_abs == 1.0) return std::copysign(std::numeric_limits<T>::infinity(), y);
  if(y_abs <= static_cast<T>(CENTRAL_RANGE)) {
    z = y * y;
    num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
    dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0]) * z + static_cast<T>(1.0));
    x = y * num / dem;
  }
  else{
    z = std::sqrt(-std::log((static_cast<T>(1.0)-y_abs)/static_cast<T>(2.0)));
    num = ((c[3]*z + c[2])*z + c[1]) * z + c[0];
    dem = (d[1]*z + d[0])*z + static_cast<T>(1.0);
    x = std::copysign(num, y) / dem;
  }
  /* Two steps of Newton-Raphson correction */
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(3.14159265358979323846)))*std::exp(-x*x));
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(3.14159265358979323846)))*std::exp(-x*x));

  return x;
}

float _erfinv(float y) { return calc_erfinv(y); };
double _erfinv(double y) { return calc_erfinv(y); };

}

