/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _TASK_
#define _TASK_
#include <string>
#include <cstdlib>

#include "common.h"

namespace rnn {

  // a^nb^n, a^nb^nc^n, a^nb^nc^nd^n...
  std::string task1(const my_int nmax, const my_int nmin, my_int nchar){
    my_int n = (rand() % (nmax-nmin)) + nmin ;
    std::string p( (nchar-1)* n + n, 'a');
    for(my_int c = 1; c < nchar; c++){
      for(my_int i = c * n; i < (c+1) * n; i++)
        p[i] = 'a'  + c;
    }
    return p;
  }

  // a^nb^kn n>=1
  std::string task2(const my_int nmax, const my_int nmin, my_int nchar, my_int nrep = 2){
    my_int n = (rand() % (nmax-nmin)) + nmin ;
    my_int c2 = rand() % (nchar-1) + 1;
    std::string p( n + nrep * n, 'a');
    for(my_int i = n; i <  n +  nrep * n; i++)
      p[i] = 'a'  + c2;
    return p;
  }


  // addition: a^nb^mc^{n+m}
  std::string task3(const my_int nmax, const my_int nmin, my_int nchar){
    my_int n = (rand() % (nmax-nmin)) + nmin ;
    my_int m = (rand() % (n-1)) + 1 ;
    n = n - m;
    std::string p( (nchar-2)* (n+m) + m  + n, 'a');
    for(my_int i = n  ; i < n + m; i++)
      p[i] = 'b';
    for(my_int i = n + m   ; i < p.size(); i++)
      p[i] = 'c';
    return p;
  }

  // memorization string (see paper)
  std::string task4(const my_int nmax, const my_int nmin, my_int nchar){
    my_int n = (rand() % (nmax-nmin)) + nmin ;
    std::string p( 2 * n + 1, 'a');
    for(my_int i = 0  ; i < n; i++)
      p[i] = 'a' + (rand() % (nchar-1) + 1);
    for(my_int i = 0  ; i <  n; i++)
      p[p.size()-1  - i] = p[i];
    return p;
  }

  // multiplication a^nb^nc^{nm}
  std::string task5(const my_int nmax, const my_int nmin, my_int nchar){
    my_int n = (rand() % (nmax-nmin)) + nmin ;
    my_int k = (rand() % (n-1)) + 1 ;
    n = n - k;
    my_int c1 = (rand() % (nchar -2)) + 2;
    std::string p( k + n +  k * n, 'a');
    for(my_int i = n; i < n + k; i++)
      p[i] = 'b';
    for(my_int i = n + k; i < p.size(); i++)
      p[i] = 'a' + c1;
    return p;
  }

  // a^nb^mc^nd^m
  std::string task6(const my_int nmax, const my_int nmin, my_int nchar){
    if(nchar != 4) exit(-1);
    my_int n = (rand() % (nmax-nmin)) + nmin ;
    my_int m = (rand() % (n-1)) + 1 ;
    n = n - m;
    std::string p( 2 * n + 2 * m, 'a');
    for(my_int i = n  ; i < n + m; i++)
      p[i] = 'b';
    for(my_int i = n + m   ; i < 2 * n + m ; i++)
      p[i] = 'c';
    for(my_int i = 2 * n + m   ; i < p.size(); i++)
      p[i] = 'd';
    return p;
  }

  std::string generate_addition(const my_int nmax, const my_int nmin, my_int base){
    if(base < 2) exit(-1);

    my_int i0 = 1;
    my_int tln;
    if(nmax > nmin+1)
      tln =(rand() % (nmax-nmin)) + nmin;
    else
      tln = nmin;
    my_int ln = rand() % (tln+1);
    my_int lm = tln - ln;

    std::string n = std::string(ln, '1' + (rand() % (base-1)));
    if(ln == 0) {n = "0"; ln = 1;}
    for(my_int i = i0; i < ln; i++)
      n[i] = '0' + (rand() % base);
    std::string m = std::string(lm, '1' + (rand() % (base-1)));
    if(lm == 0) {m = "0"; lm = 1;}
    for(my_int i = i0; i < lm; i++)
      m[i] = '0' + (rand() % base);

    std::string p = n;p += "+";p += m;p += "=";
    my_int carry = 0;
    my_int in = n.size() -1, im = m.size() -1;
    while ( in >= 0 || im >= 0 || carry > 0){
      my_int num = carry;
      if( in >= 0) num += n[in] - '0';
      if( im >= 0) num += m[im] - '0';
      p += '0' + (num % base);
      carry = num /base;
      in--; im--;
    }
    p+=".";
    return p;
  }

  std::string generate_next_sequence(const my_int nmax, const my_int nmin, my_int nchar, my_int nrep, my_int ntask){

    if(ntask == 2)
      return task2(nmax, nmin, nchar, nrep);
    if(ntask == 3){
      return task3(nmax, nmin, 3);
    }
    if(ntask == 6){
      return task6(nmax, nmin, nchar);
    }
    if(ntask == 5){
      return task5(nmax, nmin, nchar);
    }
    if(ntask == 4){
      return task4(nmax, nmin, nchar);
    }
    return task1(nmax, nmin, nchar);
  }


}

#endif
