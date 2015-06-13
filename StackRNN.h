/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _STACK_RNN_
#define _STACK_RNN_
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>

#include "common.h"
#include "Vec.h"
#include "Linear.h"
#include "Nonlinearity.h"


#define EMPTY_STACK_VALUE -1

namespace rnn
{

  enum {push,  pop, noop};

  struct StackRNN
  {
    public:

      StackRNN(const std::string& filename)
      {
        load(filename);
        emptyStacks();
      }

      StackRNN(my_int si,
          my_int sh,
          my_int nstack,
          my_int stack_capacity,
          my_int so,
          my_int sm,
          my_int bptt_step,
          my_int mod = 1,
          bool isnoop = false,
          my_int depth = 1,
          my_real reg = 0) :
        _reg(reg), // regularization by entropy -- NOT USED
        _count(0),
        _HIDDEN(sh), // size of the hidden layer
        _NB_STACK(nstack), // number of stacks
        _STACK_SIZE(stack_capacity), // stacks capacity - this is currently fix TODO make it flexible
        _ACTION(2 + ((isnoop)?1:0)), // size of the action layer
        _TOP_OF_STACK(0), // index of the top of the stack
        _BPTT(sm), // length of the bptt
        _BPTT_STEP(bptt_step), // step of bptt (how often backprop is perform)
        _IN(si), // size of the input layer
        _OUT(so), // size of the output layer
        _it_mem(_BPTT - 1),// iterator for the circular buffer
        _mod(mod), // mod=0 -> no-rec, mod=1 -> rec with stack, mod=2 -> rec through stack+full
        _DEPTH(depth), // depth used to predict next hidden units from stacks
        _in2hidTranspose(_HIDDEN, _IN),
        _hid2act(_NB_STACK, Linear(_HIDDEN, _ACTION)),
        _hid2hid(_HIDDEN, _HIDDEN),
        _hid2stack(_NB_STACK, Linear(_HIDDEN, _STACK_SIZE)),
        _stack2hid(_NB_STACK, Linear(_STACK_SIZE, _HIDDEN)),
        _hid2out(_HIDDEN, _OUT),
        _in(_BPTT,0),
        _hid(_BPTT, Vec (_HIDDEN, 0)),
        _act(_NB_STACK, std::vector<Vec>(_BPTT, Vec (_ACTION, 0))),
        _stack(_NB_STACK, std::vector<Vec>(_BPTT, Vec (_STACK_SIZE, 0))),
        _out(_BPTT, Vec(_OUT, 0)),
        _targets(_BPTT, 0),
        _err_out(_OUT, 0),
        _err_hid (_HIDDEN, 0),
        _err_stack(_NB_STACK, Vec(_STACK_SIZE, 0)),
        _err_act(_NB_STACK, Vec(_ACTION,0)),
        _pred_err_stack(_NB_STACK, Vec(_STACK_SIZE,0)),
        _pred_err_hid(_HIDDEN,0),
        _isemptied(_BPTT, false)
        {
          this->initialize();
        };


      void initialize()
      {

        // initialize input to output linear layer:
        _in2hidTranspose.initialize();

        if(_mod != 2) _hid2hid.zeros();
        else _hid2hid.initialize();

        //initialize transition between action, hidden and top of stack:
        for(my_int i = 0; i <_NB_STACK;i++)
        {
          _hid2act[i].initialize();
          _hid2stack[i].initialize();
          _stack2hid[i].initialize();
        }

        for(my_int s = 0; s <_NB_STACK;s++)
        {
          for(my_int j = 0; j < _HIDDEN; j++)
            for(my_int i = _TOP_OF_STACK +_DEPTH; i < _TOP_OF_STACK + _STACK_SIZE; i++)
              _stack2hid[s]._data(j,i) = 0;
          for(my_int i = _TOP_OF_STACK +1; i < _TOP_OF_STACK + _STACK_SIZE; i++)
            for(my_int j = 0; j < _HIDDEN; j++)
              _hid2stack[s]._data(i,j) = 0;
        }

        // initialize hidden to output linear layer:
        _hid2out.initialize();

        // initialize the stack with empty value:
        emptyStacks();
      };


      void emptyStacks()
      {
        if(_NB_STACK == 0) return;
        _count = 0;
        my_int m = _it_mem;
        _isemptied[m] = true;
        for(my_int s = 0; s <_NB_STACK;s++)
          for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE; i++)
            _stack[s][m][i] = EMPTY_STACK_VALUE;
      }

      void forward(const my_int& cur, const my_int& target, bool ishard = false)
      {

        // increment iterator on memory
        my_int old_it = _it_mem;
        _it_mem = ( _it_mem + 1) % _in.size();

        _isemptied[_it_mem] = false;

        // zeros the current hidden states:
        _out[_it_mem].zeros();
        _hid[_it_mem].zeros();
        for(my_int s = 0; s <_NB_STACK; s++)
        {
          _act[s][_it_mem].zeros();
          _stack[s][_it_mem].zeros();
        }

        //copy current word and target word in in memory
        _targets[ _it_mem ] = target;
        _in[ _it_mem ] = cur;

        // forward propagation from input to hidden:
        _in2hidTranspose.forward_transpose(cur, _hid[_it_mem]);

        // forward from hidden to hidden:
        //  (hidden + top of stack) (t-1) -> hidden (t):
        // mod = 1 -> recurrent only through stack
        // mod = 2 -> full hidden
        // mod 0 -> no recurrent
        if( _mod != 0)
        {
          // previous top of stack -> current hidden
          for(my_int s = 0; s <_NB_STACK;s++)
          {
            _stack2hid[s].forward(_stack[s][old_it], _hid[_it_mem],
                _TOP_OF_STACK, _TOP_OF_STACK + _DEPTH, 0, _HIDDEN);
          }
        }

        if(_mod == 2)
        {
          // previous hidden (t-1) -> current hidden (t)
          _hid2hid.forward(_hid[old_it], _hid[_it_mem]);
        }

        // nonlinearity on the hidden:
        Sigmoid::forward(_hid[_it_mem]);

        for(my_int s = 0; s <_NB_STACK;s++)
        {
          // current hidden  -> current action:
          _hid2act[s].forward(_hid[_it_mem], _act[s][_it_mem]);

          // non linearity
          // action
          Softmax::forward(_act[s][_it_mem]);
          if(ishard)
          {
            //if it s discretize, i.e. take the most probable action:
            my_int im =0; my_real pm = _act[s][_it_mem][0];
            _act[s][_it_mem][0] = 0;
            for(my_int i = 1; i < _ACTION; i++)
            {
              if( pm < _act[s][_it_mem][i])
              {
                im = i;
                pm = _act[s][_it_mem][i];
              }
              _act[s][_it_mem][i] = 0;
            }
            _act[s][_it_mem][im]  = 1;
          }

          my_real pop_weight = _act[s][_it_mem][pop];
          my_real push_weight = _act[s][_it_mem][push];

          // (action + hidden) -> (stack):

          // in case of push:
          // push from the top to the bottom:
          for(my_int i = _TOP_OF_STACK + 1; i < _STACK_SIZE; i++)
            _stack[s][_it_mem][i] += _stack[s][old_it][i-1] * push_weight;

          // the push on the top of the stack is weighted by push action:
          _stack[s][_it_mem][_TOP_OF_STACK] = 0;
          for(my_int i = 0; i < _HIDDEN; i++)
            _stack[s][_it_mem][_TOP_OF_STACK]  += _hid2stack[s]._data(_TOP_OF_STACK, i) * _hid[_it_mem][i];
          // add a non-linearity on the top of the stack:
          if(_stack[s][_it_mem][_TOP_OF_STACK]  > 50)
            _stack[s][_it_mem][_TOP_OF_STACK]  = 50;
          if(_stack[s][_it_mem][_TOP_OF_STACK]  < -50)
            _stack[s][_it_mem][_TOP_OF_STACK]  = -50;
          _stack[s][_it_mem][_TOP_OF_STACK] = 1 / ( 1 + exp( - _stack[s][_it_mem][_TOP_OF_STACK] ) );

          _stack[s][_it_mem][_TOP_OF_STACK] *=  push_weight;

          // in case of pop:
          for(my_int i = _TOP_OF_STACK; i < _STACK_SIZE - 1; i++)
            _stack[s][_it_mem][i] += _stack[s][old_it][i+1] * pop_weight;

          // last element of the stack get an empty value:
          _stack[s][_it_mem][_STACK_SIZE - 1] += EMPTY_STACK_VALUE * pop_weight;

          // in case of no-op:
          if(_ACTION == 3)
          {
            my_real noop_weight = _act[s][_it_mem][noop];
            for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE; i++)
              _stack[s][_it_mem][i] += _stack[s][old_it][i] * noop_weight;
          }
        }

        // propagation from hidden to out:
        _hid2out.forward(_hid[_it_mem], _out[_it_mem]);
        Softmax::forward(_out[_it_mem]);
      }


      void backward()
      {

        // put gradient to zeros:
        _in2hidTranspose.resetGradient();
        _hid2hid.resetGradient();
        _hid2out.resetGradient();

        for(my_int s = 0; s <_NB_STACK;s++)
        {
          _hid2stack[s].resetGradient();
          _hid2act[s].resetGradient();
          _stack2hid[s].resetGradient();
        }

        _err_hid.zeros();

        for(my_int s = 0; s <_NB_STACK;s++)
        {
          _err_stack[s].zeros();
          _err_act[s].zeros();
        }

        my_int itm = _it_mem, count = 0;
        _count++;

        //back prog through time
        while(count < std::min(_BPTT,_count))
        {

          if(_mod != 2) _err_hid.zeros();

          //out -> hidden
          if( count < _BPTT_STEP)
          {
            // backprop through softmax:
            for(my_int i = 0; i < _OUT; i++)  _err_out[i] = -_out[itm][i];
            _err_out[_targets[itm]] +=1;

            // Compute gradient from hidden -> out
            _hid2out.computeGradient(_hid[itm], _err_out);

            //propagate error from out -> hidden
            _hid2out.backward(_err_hid, _err_out);

            // clip the error:
            hardclipping(_err_hid);
          }

          if(_isemptied[itm]) break;

          _pred_err_hid.zeros();

          my_int old_it = itm - 1;
          if(old_it < 0) old_it = _in.size() - 1;

          for(my_int s = 0; s <_NB_STACK;s++)
          {

            _err_act[s].zeros();
            _pred_err_stack[s].zeros();

            if(itm == _it_mem)
            {
              for(my_int a = 0; a < _ACTION; a++)
              {
                _err_act[s][a] = _reg * ( log(_act[s][itm][a] + 1e-16) + 1);
              }
            }

            // gradient of hidden -> top of stack (due to push):
            // this is ugly but required: the gradient of hid->stack apply to the value before the sigmoid,
            // I don t store that value, so I need to recompute it (it would be better to simply store it...)
            my_real tmp_top_stack_in = 0;
            for(my_int i = 0; i < _HIDDEN; i++)
            {
              tmp_top_stack_in += _hid2stack[s]._data(_TOP_OF_STACK, i) * _hid[itm][i];
            }
            if(tmp_top_stack_in  > 50) tmp_top_stack_in  = 50;
            if(tmp_top_stack_in  < -50)  tmp_top_stack_in  = -50;
            tmp_top_stack_in = 1 / (1 + exp( - tmp_top_stack_in));

            my_real tmp_top_stack_err = _err_stack[s][_TOP_OF_STACK];
            tmp_top_stack_err *= _act[s][itm][push];
            tmp_top_stack_err *= tmp_top_stack_in * ( 1 - tmp_top_stack_in);

            if(tmp_top_stack_err > 15) tmp_top_stack_err = 15;
            if(tmp_top_stack_err < -15) tmp_top_stack_err = -15;

            // gradient if hid -> stack
            for(my_int i = 0; i < _HIDDEN; i++)
            {
              _hid2stack[s]._gradient(_TOP_OF_STACK, i) += _hid[itm][i] * tmp_top_stack_err;
            }
            // propagate error from stack(t) -> stack(t-1)
            for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE - 1; i++)
            {
              _pred_err_stack[s][i+1] += _err_stack[s][i] * _act[s][itm][pop];
            }
            // propagate error from stack(t) -> action[pop]
            for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE - 1; i++)
            {
              _err_act[s][pop] += _err_stack[s][i] * _stack[s][old_it][i+1];
            }
            _err_act[s][pop] += _err_stack[s][_TOP_OF_STACK + _STACK_SIZE - 1] * EMPTY_STACK_VALUE;

            // in case of push:
            // push from the top to the bottom:
            for(my_int i = _TOP_OF_STACK + 1; i < _TOP_OF_STACK + _STACK_SIZE; i++)
            {
              _pred_err_stack[s][i-1] += _err_stack[s][i] * _act[s][itm][push];
            }
            for(my_int i = _TOP_OF_STACK + 1; i < _TOP_OF_STACK + _STACK_SIZE; i++)
            {
              _err_act[s][push] += _err_stack[s][i] * _stack[s][old_it][i-1];
            }
            // propagate error from stack to action + hidden
            for(my_int i = 0; i < _HIDDEN; i++)
            {
              _err_hid[i] += _hid2stack[s]._data(_TOP_OF_STACK, i) * tmp_top_stack_err;
            }
            _err_act[s][push] += _err_stack[s][_TOP_OF_STACK] * tmp_top_stack_in;

            // in case of no-op action:
            if(_ACTION == 3)
            {
              for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE; i++)
              {
                _pred_err_stack[s][i] += _err_stack[s][i] * _act[s][itm][noop];
              }
              for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE; i++)
              {
                _err_act[s][noop] += _err_stack[s][i] * _stack[s][old_it][i];
              }
            }
            hardclipping(_err_act[s]);
            hardclipping(_pred_err_stack[s]);

            Softmax::backward(_err_act[s], _act[s][itm]);

            hardclipping(_err_act[s]);

            // gradient of hidden -> action:
            _hid2act[s].computeGradient( _hid[itm], _err_act[s]);

            // propagate error from action -> hidden:
            _hid2act[s].backward(_err_hid, _err_act[s]);
          }

          // at that point: err_hid = err_from_out + err_from_top_stack + err_from_action
          //propagate error on hidden layer through non-linearity:
          Sigmoid::backward(_err_hid, _hid[itm]);

          // clip the error:
          hardclipping(_err_hid);

          // compute  contribution of the hidden to the gradient of in2hid:
          _in2hidTranspose.computeGradient_transpose(_in[itm], _err_hid);

          //propagate error in the past:

          itm = old_it;

          // stop before doing last propagaton from hidden to hidden
          if(count == _BPTT - 1) break;

          if(_mod != 0)
          {
            // compute gradient of (hidden + top of stack) -> hidden
            for(my_int s = 0; s <_NB_STACK;s++)
            {
              _stack2hid[s].computeGradient( _stack[s][itm], _err_hid,
                  _TOP_OF_STACK, _TOP_OF_STACK + _DEPTH,
                  0, _HIDDEN);
              // Propagate error from hidden -> top of stack
              _stack2hid[s].backward(_pred_err_stack[s], _err_hid,
                  _TOP_OF_STACK, _TOP_OF_STACK + _DEPTH,
                  0, _HIDDEN);
              hardclipping(_pred_err_stack[s]);
            }
          }
          if(_mod == 2)
          {
            // compute gradient of (hidden ) -> hidden
            _hid2hid.computeGradient( _hid[itm], _err_hid);
            // Propagate error from hidden -> (hidden + top of stack)
            _hid2hid.backward(_pred_err_hid, _err_hid);
          }

          for(my_int i = 0; i < _HIDDEN; i++)
          {
            _err_hid[i] = _pred_err_hid[i];
          }
          hardclipping(_err_hid);

          for(my_int s = 0; s <_NB_STACK;s++)
          {
            for(my_int i = 0; i < _STACK_SIZE; i++)
              _err_stack[s][i] = _pred_err_stack[s][i];
            hardclipping(_err_stack[s]);
          }
          count++;
        }
      }

      void update(const my_real& lr)
      {
        _hid2out.update(lr);
        if(_mod == 2) _hid2hid.update(lr);
        for(my_int s = 0; s <_NB_STACK;s++)
        {
          _hid2act[s].update(lr);
          _stack2hid[s].update(lr);
          _hid2stack[s].update(lr);
        }
        _in2hidTranspose.update(lr);
      }

      my_real eval(const my_int& target) const {
        return _out[_it_mem][target];
      }

      my_int pred() const {
        my_int pred = 0;
        my_real pv = _out[_it_mem][0];
        for(my_int i = 1; i <_OUT; i++)
        {
          if(pv < _out[_it_mem][i])
          {
            pred = i; pv =_out[_it_mem][i];
          }
        }
        return pred;
      }

      /*************************************************************************************/


      void copy(StackRNN shrnn)
      {
        assert(_IN == shrnn._IN);
        assert(_HIDDEN == shrnn._HIDDEN);
        assert(_OUT == shrnn._OUT);
        assert(_BPTT == shrnn._BPTT);
        assert(_ACTION == shrnn._ACTION);
        assert(_STACK_SIZE == shrnn._STACK_SIZE);
        assert(_NB_STACK == shrnn._NB_STACK);
        assert(_DEPTH == shrnn._DEPTH);

        _it_mem = shrnn._it_mem;
        _mod = shrnn._mod;

        _in2hidTranspose._data = shrnn._in2hidTranspose._data;
        _hid2hid._data = shrnn._hid2hid._data;
        _hid2out._data = shrnn._hid2out._data;

        for(my_int s = 0; s < _NB_STACK; s++)
        {
          _hid2stack[s]._data = shrnn._hid2stack[s]._data;
          _hid2act[s]._data = shrnn._hid2act[s]._data;
          _stack2hid[s]._data = shrnn._stack2hid[s]._data;
          for(my_int m = 0; m < _BPTT; m++)
          {
            _act[s][m] = shrnn._act[s][m];
            _stack[s][m] = shrnn._stack[s][m];
          }
        }

        for(my_int m = 0; m < _BPTT; m++)
        {
          _out[m] = shrnn._out[m];
          _in[m] = shrnn._in[m];
          _hid[m] = shrnn._hid[m];
          _targets[m] = shrnn._targets[m];
          _isemptied[m] = shrnn._isemptied[m];
        }

      }


      void save(std::string filename)
      {
        FILE* f;
        f= fopen(filename.c_str(),"w");
        fprintf(f, "%d %d %d %d %d %d %d %d %d %d\n", _IN, _ACTION, _HIDDEN, _NB_STACK, _STACK_SIZE, _OUT,
            _BPTT, _BPTT_STEP, _mod, _DEPTH);
        for(my_int i = 0; i < _in2hidTranspose.size(); i++)  fprintf(f, "%f,", _in2hidTranspose._data[i]);
        for(my_int i = 0; i < _hid2hid.size(); i++) fprintf(f, "%f,", _hid2hid._data[i]);
        for(my_int s = 0; s <_NB_STACK;s++)
        {
          for(my_int i = 0; i < _hid2act[s].size(); i++) fprintf(f, "%f,", _hid2act[s]._data[i]);
          for(my_int i = 0; i < _hid2stack[s].size(); i++) fprintf(f, "%f,", _hid2stack[s]._data[i]);
          for(my_int i = 0; i < _stack2hid[s].size(); i++) fprintf(f, "%f,", _stack2hid[s]._data[i]);
        }
        for(my_int i = 0; i < _hid2out.size(); i++) fprintf(f, "%f,", _hid2out._data[i]);
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f;
        f= fopen(filename.c_str(),"r");
        fscanf(f, "%d %d %d %d %d %d %d %d %d %d\n", &_IN, &_ACTION, &_HIDDEN, &_NB_STACK, &_STACK_SIZE, &_OUT,
            &_BPTT, &_BPTT_STEP, &_mod, &_DEPTH);
        _TOP_OF_STACK = 0;
        _in2hidTranspose = Linear(_HIDDEN, _IN);
        _hid2hid = Linear(_HIDDEN, _HIDDEN);
        _hid2stack = std::vector<Linear>(_NB_STACK, Linear(_HIDDEN, _STACK_SIZE));
        _stack2hid = std::vector<Linear>(_NB_STACK, Linear(_STACK_SIZE, _HIDDEN));
        _hid2act = std::vector<Linear>(_NB_STACK, Linear(_HIDDEN, _ACTION));
        _hid2out = Linear(_HIDDEN, _OUT);
        for(my_int i = 0; i < _in2hidTranspose.size(); i++) fscanf(f, "%lf,", &_in2hidTranspose._data[i]);
        for(my_int i = 0; i < _hid2hid.size(); i++) fscanf(f, "%lf,", &_hid2hid._data[i]);
        for(my_int s = 0; s <_NB_STACK;s++)
        {
          for(my_int i = 0; i < _hid2act[s].size(); i++) fscanf(f, "%lf,", &_hid2act[s]._data[i]);
          for(my_int i = 0; i < _hid2stack[s].size(); i++) fscanf(f, "%lf,", &_hid2stack[s]._data[i]);
          for(my_int i = 0; i < _stack2hid[s].size(); i++) fscanf(f, "%lf,", &_stack2hid[s]._data[i]);
        }
        for(my_int i = 0; i < _hid2out.size(); i++) fscanf(f, "%lf,", &_hid2out._data[i]);
        fclose(f);
        _isemptied = std::vector< bool >(_BPTT, false);
        _it_mem = _BPTT - 1;
        _in = std::vector<my_int>(_BPTT, 0);
        _hid = std::vector<Vec >(_BPTT, Vec (_HIDDEN,0));
        _act = std::vector< std::vector<Vec > >(
            _NB_STACK, std::vector<Vec >(_BPTT, Vec (_ACTION, 0)));
        _stack = std::vector< std::vector<Vec > >(
            _NB_STACK, std::vector<Vec >(_BPTT, Vec (_STACK_SIZE, 0)));
        _out = std::vector<Vec >(_BPTT, Vec(_OUT,0));
        _targets = std::vector<my_int>( _BPTT,0);
        _err_out = Vec(_OUT, 0);
        _err_hid = Vec(_HIDDEN, 0);
        _err_act = std::vector<Vec > (
            _NB_STACK, Vec(_ACTION,0));
        _err_stack = std::vector<Vec > (
            _NB_STACK, Vec(_STACK_SIZE, 0));
        _pred_err_hid = Vec(_HIDDEN,0);
        _pred_err_stack = std::vector<Vec > (
            _NB_STACK, Vec(_STACK_SIZE,0));
        _reg = 0;
        _count = 0;
      }

      // TODO: make this private:

      my_real _reg;

      my_int _count;

      my_int _HIDDEN;
      my_int _NB_STACK;
      my_int _STACK_SIZE;
      my_int _ACTION;
      my_int _TOP_OF_STACK;
      my_int _BPTT;
      my_int _BPTT_STEP;
      my_int _IN;
      my_int _OUT;
      my_int _it_mem;
      my_int _mod;
      my_int _DEPTH;

      Linear                  _in2hidTranspose;
      std::vector< Linear >   _hid2act;
      Linear                  _hid2hid;
      std::vector< Linear >   _hid2stack;
      std::vector< Linear >   _stack2hid;
      Linear                  _hid2out;

      std::vector<my_int>               _in;
      std::vector< Vec >                _hid;
      std::vector< std::vector< Vec > > _act;
      std::vector< std::vector< Vec > > _stack;
      std::vector< Vec >                _out;
      std::vector<my_int>               _targets;

      Vec                   _err_out;
      Vec                   _err_hid;
      std::vector< Vec >    _err_stack;
      std::vector< Vec >    _err_act;
      std::vector< Vec >    _pred_err_stack;
      Vec                   _pred_err_hid;

      std::vector< bool > _isemptied;

  };



} // end namespace
#endif





