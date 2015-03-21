/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _UTILS_
#define _UTILS_
#include <assert.h>
#include <math.h>

#include "common.h"
#include "Vec.h"

namespace rnn {

  // utils:
  void hardclipping( Vec& v, my_int b = -1, my_int e = -1){
    if(b == -1 ) b = 0;
    if(e == -1 ) e = v.size();
    for(my_int i = b; i < e; i++){
      if( v[i] < -15) v[i] = -15;
      if( v[i] > 15) v[i] = 15;
    }
  }

  /* uniform distribution, (0..1] */
  my_real drand()
  {
    return (rand()+1.0)/(RAND_MAX+1.0);
  }

  /* normal distribution, centered on 0, std dev 1 */
  my_real random_normal()
  {
    return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
  }

  my_real random(my_real min, my_real max){
    return rand()/(my_real)RAND_MAX*(max-min)+min;
  }

  void matrixXvector(Vec& dest, const Vec& srcvec, const Vec2D& srcmatrix,
      const my_int& obegin, const my_int& oend,
      const my_int& ibegin, const my_int& iend, const my_int& type)
  {
    // type = 0 -> srcmatrix * srcvec
    // type = 1 -> srcmatrix^T * srcvec

    assert(srcmatrix.nrow() >= oend);
    assert(srcmatrix.ncol() >= iend);

    my_int a, b;
    my_real val1, val2, val3, val4;
    my_real val5, val6, val7, val8;

    my_int matrix_width = srcmatrix.ncol();


    if (type==0) {		//ac mod
      assert(dest.size() >= oend);
      assert(srcvec.size() >= iend);
      for (b=0; b<(oend-obegin)/8; b++) {
        val1=0;
        val2=0;
        val3=0;
        val4=0;

        val5=0;
        val6=0;
        val7=0;
        val8=0;

        for (a=ibegin; a<iend; a++) {
          val1 += srcvec[a] * srcmatrix[a+(b*8+obegin+0)*matrix_width];
          val2 += srcvec[a] * srcmatrix[a+(b*8+obegin+1)*matrix_width];
          val3 += srcvec[a] * srcmatrix[a+(b*8+obegin+2)*matrix_width];
          val4 += srcvec[a] * srcmatrix[a+(b*8+obegin+3)*matrix_width];

          val5 += srcvec[a] * srcmatrix[a+(b*8+obegin+4)*matrix_width];
          val6 += srcvec[a] * srcmatrix[a+(b*8+obegin+5)*matrix_width];
          val7 += srcvec[a] * srcmatrix[a+(b*8+obegin+6)*matrix_width];
          val8 += srcvec[a] * srcmatrix[a+(b*8+obegin+7)*matrix_width];
        }
        dest[b*8+obegin+0] += val1;
        dest[b*8+obegin+1] += val2;
        dest[b*8+obegin+2] += val3;
        dest[b*8+obegin+3] += val4;

        dest[b*8+obegin+4] += val5;
        dest[b*8+obegin+5] += val6;
        dest[b*8+obegin+6] += val7;
        dest[b*8+obegin+7] += val8;
      }

      for (b=b*8; b<oend-obegin; b++) {
        for (a=ibegin; a<iend; a++) {
          dest[b+obegin] += srcvec[a] * srcmatrix[a+(b+obegin)*matrix_width];
        }
      }
    }
    else {		//er mod
      assert(dest.size() >= iend);
      assert(srcvec.size() >= oend);
      for (a=0; a<(iend-ibegin)/8; a++) {
        val1=0;
        val2=0;
        val3=0;
        val4=0;

        val5=0;
        val6=0;
        val7=0;
        val8=0;

        for (b=obegin; b<oend; b++) {
          val1 += srcvec[b] * srcmatrix[a*8+ibegin+0+b*matrix_width];
          val2 += srcvec[b] * srcmatrix[a*8+ibegin+1+b*matrix_width];
          val3 += srcvec[b] * srcmatrix[a*8+ibegin+2+b*matrix_width];
          val4 += srcvec[b] * srcmatrix[a*8+ibegin+3+b*matrix_width];

          val5 += srcvec[b] * srcmatrix[a*8+ibegin+4+b*matrix_width];
          val6 += srcvec[b] * srcmatrix[a*8+ibegin+5+b*matrix_width];
          val7 += srcvec[b] * srcmatrix[a*8+ibegin+6+b*matrix_width];
          val8 += srcvec[b] * srcmatrix[a*8+ibegin+7+b*matrix_width];
        }
        dest[a*8+ibegin+0] += val1;
        dest[a*8+ibegin+1] += val2;
        dest[a*8+ibegin+2] += val3;
        dest[a*8+ibegin+3] += val4;

        dest[a*8+ibegin+4] += val5;
        dest[a*8+ibegin+5] += val6;
        dest[a*8+ibegin+6] += val7;
        dest[a*8+ibegin+7] += val8;
      }

      for (a=a*8; a<iend-ibegin; a++) {
        for (b=obegin; b<oend; b++) {
          dest[a+ibegin] += srcvec[b] * srcmatrix[a+ibegin+b*matrix_width];
        }
      }
    }
  }


} // end of namespace


#endif
