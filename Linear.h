/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _LINEAR_
#define _LINEAR_
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include "common.h"
#include "Vec.h"
#include "utils.h"

namespace rnn {

  // Linear struct:
  struct Linear{
    public:
      /*** Constructors ***/

      Linear(){};

      explicit Linear(const my_int& si, const my_int& so) :
        _data(so, si, 0),
        _gradient(so, si, 0) {};

      Linear(const Linear& rhs) {
        this->_data = rhs._data;
        this->_gradient = rhs._gradient;
        this->_gradient.zeros();
      }


      /*** methods ***/

      my_int ncol() const {
        return this->_data.ncol();
      }

      my_int nrow() const {
        return this->_data.nrow();
      }

      void initialize(){
        for(my_int i = 0; i < _data.size(); i++)
          _data[i] = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
      }

      my_int size() { return this->_data.size();}
      my_int sizeIn() { return this->_data.ncol();}
      my_int sizeOut() { return this->_data.nrow();}

      void zeros(){ this->_data.zeros();};

      /*** forward methods ***/

      void forward(const my_int& idx, Vec& out){
        assert(out.size() == _data.nrow());
        for(my_int x = 0; x < out.size(); x++)
          out[x] += _data(x, idx);
      };

      void forward_transpose(const my_int& idx, Vec& out){
        assert(out.size() == _data.ncol());
        for(my_int x = 0; x < out.size(); x++)
          out[x] += _data(idx, x);
      };

      void forward_transpose(const my_int& idx, Vec& out,
          const my_int& obegin, const my_int& oend){
        assert(out.size() == _data.ncol());
        for(my_int x = obegin; x < oend; x++)
          out[x] += _data(idx, x);
      };

      void forward(const Vec& in, Vec& out,
          const my_int& ibegin, const my_int& iend,
          const my_int& obegin, const my_int& oend){
        assert(obegin >= 0);
        assert(oend <= _data.nrow());
        assert(oend <= out.size());
        assert(ibegin >= 0);
        assert(iend <= _data.ncol());
        assert(iend <=  in.size());
        matrixXvector(out, in, this->_data, obegin, oend, ibegin, iend, 0);
      }

      void forward(const Vec& in, Vec& out){
        matrixXvector(out, in, this->_data, 0, out.size(), 0, in.size(), 0);
      };

      /*** backward methods ***/

      void backward(Vec& in, const Vec& out){
        matrixXvector(in, out, this->_data, 0, out.size(), 0, in.size(), 1);
      };

      void backward(Vec& in, const Vec& out,
          const my_int& ibegin, const my_int& iend,
          const my_int& obegin, const my_int& oend){
        assert(obegin >= 0);
        assert(oend <= _data.nrow());
        assert(oend <= out.size());
        assert(ibegin >= 0);
        assert(iend <= _data.ncol());
        assert(iend <=  in.size());
        matrixXvector(in, out, this->_data, obegin, oend, ibegin, iend, 1);
      }

      /*** gradient methods ***/

      void resetGradient(){
        this->_gradient.zeros();
      };

      void computeGradient(const my_int& idx ,const Vec& out){
        for(my_int i = 0; i < _gradient.nrow(); i++)
          _gradient(i, idx) += out[i]; // gradient += out * in';
      };

      void computeGradient_transpose(const my_int& idx ,const Vec& out){
        for(my_int i = 0; i < _gradient.ncol(); i++)
          _gradient(idx,i) += out[i]; // gradient += out * in';
      };

      void computeGradient_transpose(const my_int& idx ,const Vec& out,
          const my_int& obegin, const my_int& oend){
        for(my_int i = obegin; i < oend; i++)
          _gradient(idx,i) += out[i]; // gradient += out * in';
      };

      void computeGradient(const Vec& in ,const Vec& out){
        computeGradient(in, out, 0, in.size(), 0, out.size());
      };

      void computeGradient(const Vec& in, const Vec& out,
          const my_int& ibegin, const my_int& iend,
          const my_int& obegin, const my_int& oend){
        assert(obegin >= 0);
        assert(oend <= _gradient.nrow());
        assert(oend <= out.size());
        assert(ibegin >= 0);
        assert(iend <= _gradient.ncol());
        assert(iend <=  in.size());
        for(my_int o = obegin; o < oend; o++){
          for(my_int i = ibegin; i < iend; i++){
            _gradient(o, i) += out[o] * in[i];
          }
        }
      }

      void update(const my_real& lr){
        for(my_int i =0; i < this->size(); i++)
          this->_data[i] += lr * this->_gradient[i];
      }

      // TODO make that private

      Vec2D    _data;
      Vec2D    _gradient;

  };


}// end namespace rnn

#endif
