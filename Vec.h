/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _VEC_
#define _VEC_
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <assert.h>

#include "common.h"


/************************************************

  Vec is a vector class use for matrix computation
  along with Vec2D.

 ********************************************/

namespace rnn {

  class Vec{
    public:
      typedef my_int          size_type;
      typedef my_real               value_type;
      typedef my_real*              iterator;
      typedef const my_real* 	const_iterator;
      typedef my_real& 		reference;
      typedef const my_real& 	const_reference;


      /*** Constructors ***/

      Vec()  {create();}

      explicit Vec(size_type s, const_reference v = my_real())  {create(s,v);};

      Vec( const Vec& v){ create(v.begin(), v.end()); }


      /*** Destructors ***/

      ~Vec() { this->uncreate();}

      /*** Iterators ***/

      iterator begin() { return this->_begin;}
      const_iterator begin() const { return this->_begin;}
      iterator end() { return this->_end;}
      const_iterator end() const { return this->_end;}

      /*** methods ***/

      void zeros(){
        for(iterator it = this->begin(); it != this->end(); it++)
          *it = 0;
      }

      size_type size() const { return this->_end - this->_begin;}


      /*** operators ***/

      reference operator[] (size_type i){ return this->_begin[i];}
      const_reference operator[] (size_type i) const { return this->_begin[i];}

      Vec& operator = (const Vec& rhs) {
        if( this != &rhs ){
          this->uncreate();
          this->create( rhs.begin(), rhs.end() );
        }
        return *this;
      }


    protected:
      iterator 	_begin;
      iterator 	_end;
      /*** private ***/

      void create(){ _begin = _end = NULL;}
      void create(size_type s, const_reference v = my_real());
      void create(const_iterator begin, const_iterator end);

      void uncreate();

  };


  /********************************* Method definition ***************************************/

  void Vec::create( Vec::size_type n,
      Vec::const_reference val){
    this->_begin = (my_real*) calloc(n , sizeof(my_real));
    this->_end = this->_begin + n;
  };

  void Vec::create( Vec::const_iterator b,
      Vec::const_iterator e){
    Vec::size_type n = e - b;
    this->_begin = (my_real*) calloc(n , sizeof(my_real));
    this->_end = this->_begin + n;
    memcpy(this->_begin, b, sizeof(my_real) * n);
  }

  void Vec::uncreate(){
    if( this->_begin != NULL ){
      free(this->_begin);
    }
    this->_begin = this->_end = NULL;
  }

  class  Vec2D : public Vec{

    private:
      my_int _ncol;
      my_int _nrow;

    public:
      explicit Vec2D() {};

      explicit Vec2D(my_int nr,
          my_int nc,
          Vec::const_reference v = my_real()) :
        Vec(nr*nc, v), _ncol(nc), _nrow(nr) {};

      ~Vec2D() { this->uncreate();}


      my_int nrow() const { return this->_nrow;}
      my_int ncol() const { return this->_ncol;}

      Vec2D& operator= (const Vec2D& rhs) {
        Vec::operator=(rhs);
        this->_ncol = rhs._ncol;
        this->_nrow = rhs._nrow;
        return *this;
      }


      Vec::reference operator() (my_int i,
          my_int j){
        return this->_begin[i * _ncol + j];
      }
      Vec::const_reference operator() (my_int i, my_int j) const {
        return this->_begin[i * _ncol + j];
      }

  };


} // end namespace rnn

#endif
