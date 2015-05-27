/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _NONLINEARITY_
#define _NONLINEARITY_
#include <algorithm>
#include <vector>

#include "common.h"
#include "Vec.h"

namespace rnn{
  struct Softmax{

    void static forward(Vec& v, my_int b = 0, my_int e = -1){
      if(e == -1) e = v.size();
      my_real max=v[b], denom = 0;
      for(my_int i = b; i < e; i++)
        if(v[i] > max) max = v[i];

      for(my_int i = b; i < e; i++){
        v[i] = exp(v[i]-max);
        denom += v[i];
      }
      for(my_int i = b; i < e; i++)
        v[i] = v[i] / denom;
    }

    void static backward(Vec& err, const Vec& v, my_int b = 0, my_int e = -1){
      if(e == -1) e = v.size();

      Vec grad = err;
      for(my_int i = b; i < e; i++){
        grad[i] = err[i] * v[i];
        for(my_int j = b; j < e; j++)
          grad[i] -= err[j] * v[j] * v[i];
      }
      err =  grad;
    }


  };

  struct Sigmoid{

    void static forward(my_real& v){
      if(v > 50 ) v = 50;
      if(v < -50 ) v = -50;
      v = 1 / (1 + exp(-v));
    }

    void static forward(Vec& v, my_int b = -1, my_int e = -1){
      if(b == - 1) b =0;
      if(e == -1 ) e = v.size();
      for (my_int i = b; i < e; i++)
      {
        if(v[i] > 50 ) v[i] = 50;
        if(v[i] < -50 ) v[i] = -50;
        v[i] = 1 / (1 + exp(-v[i]));
      }
    }

    void static backward(my_real& err, const my_real& v){
      err = err * v * (1 - v);
    }


    void static backward(Vec& err, const Vec& v, my_int b = 0, my_int e = -1){
      if(e == -1) e = err.size();
      for(my_int i = b; i < e; i++)
        err[i] = err[i] * (v[i] * (1 - v[i]));
    }

  };


}

#endif
