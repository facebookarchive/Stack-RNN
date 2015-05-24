/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unordered_map>

#include "common.h"
#include "task.h"
#include "StackRNN.h"

using namespace std;
using namespace rnn;

int main(int argc, char **argv){

  int nhid = 100;
  int nstack = 10;
  int stack_size = 200;
  int bptt = 50;
  float lr = 0.1;
  int mod = 1;
  int nmaxmax = 20;
  int nmin = 2;
  bool isnoop = true;
  bool ishard = false;
  int nreset = 10;
  int base = 2;
  int depth = 2;
  int nseq = 10000;
  int seed = 22;
  bool save = false;
  int nvalidmax = 20;
  float lrmin = 1e-5;

  int ai = 1;
  while(ai < argc){
    if( strcmp( argv[ai], "-nhid") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nhid = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nseq") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nseq = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nstack") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nstack = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-stack_size") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      stack_size = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-bptt") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      bptt = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-mod") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      mod = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-lr") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      lr = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nreset") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nreset = atoi(argv[ai+1]);
      if(nreset < 0) {printf("error nchar should be >= 0\n");return -1;}
    }
    else if( strcmp( argv[ai], "-base") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      base = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nmin") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nmin = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-seed") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      seed = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nmax") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nmaxmax = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nvalidmax") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nvalidmax = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-noop") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      isnoop = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-save") == 0){
      save = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-hard") == 0){
      ishard = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-lrmin") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      lrmin = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-depth") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      depth = atoi(argv[ai+1]);
      if(depth < 1) {printf("error in depth...\n"); return -1;}
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }
  srand(seed);

  cout<<"seed: "<<seed<<endl<<"nhid: "<<nhid<<endl<<"nstack: "<<nstack<<endl<<"bptt: "<<bptt<<
    endl<<"mod: "<<mod<<endl<<"depth: "<<depth<<endl<<"noop: "<<isnoop<<endl<<"lr: "<<lr<<endl;

  printf("INFO ABOUT PRINTING:\nlr = learning rate\nnmax = max n used to generate sequences\nentropy = entropy of the predictable part \ngoal = symbols to predict \npred = prediction (we only show prediction on the predictable part, else we print the symbol _)\n");
  printf("WARNING: The model takes quite some times to converge.\
 You can tune some parameters to make it faster:\
 the number of sequence (-nseq) is at 10K per epoch you can decrease it \
 to 5K per epoch (-nseq 5000), you can also play with the n max for train \
 (currently nmax = 20) by doing (-nmax 15) for example. Currently the model\
 is considered to have converge when the learning rate is below 1e-5 \
 (you can change to -lrmin 0.001 for example). \
 Finally you should try multiple seeds (-seed 1 for example) and pick the\
 ones which obtain low entropy on the validation set... \n");

  char buff[1000];
  sprintf(buff,"addition_base%d_nhid%d_nstack%d_bptt%d_mod%d_depth%d_noop%d_hard%d_seed%d",
      base, nhid, nstack, bptt, mod, depth, (int)((isnoop)? 1 : 0), ishard, seed);
  string modelname = "data/model_";
  modelname.append(buff);
  string  logfilename ("data/log_");
  logfilename.append(buff);
  string  testfilename = "data/test_";
  testfilename.append(buff);
  string  logtestfilename = "data/log_test_" ;
  logtestfilename.append(buff);
  if(save){
    cerr<< modelname << endl;
    cerr<< logfilename << endl;
    cerr<< testfilename << endl;
    cerr<< logtestfilename << endl;
  }

  int nchar = 3 + base;
  unordered_map<char, int> dic;
  vector<char> rdic(nchar,0);
  dic['+'] = 0; rdic[0] = '+';
  dic['='] = 1; rdic[1] = '=';
  dic['.'] = 2; rdic[2] = '.';
  for(int i = 0; i < nchar -3; i++)
  {  dic['0'+i] = 3 + i; rdic[3 + i] = '0' + i;}

  cout<<"create rnn...";
  StackRNN rnn(nchar, nhid, nstack, stack_size,
      nchar, bptt, 1, mod, isnoop, depth, 0);
  StackRNN back_up_model(nchar, nhid, nstack, stack_size,
      nchar, bptt, 1, mod, isnoop, depth, 0);
  cout<<"done"<<endl;

  int cur = nchar - 1, next = 0;
  int nmax = 3;
  if(nmin >= nmax) nmax = nmin + 1;
  int nseqv = 1000;

  string p = generate_addition(nmax, nmin, base);

  int count = 0, neval = 0;
  int ne = 0; double lo = 0;
  int nepoch = 100;
  bool iseval = true;
  float last_ent = 0;
  double loss;
  string spred, sgoal;

  FILE*    f;
  for(int e = 0; e < nepoch; e++){
    nmax = max(min(e+3,nmaxmax),3);
    nmin = 0;
    neval = 1; loss = 0;
    ne = 1; lo = 0;

    rnn.emptyStacks();

    /************* TRAIN *************/

    for(int iseq = 0; iseq < nseq; iseq++) {

      p = generate_addition(nmax, nmin, base);

      if(nreset > 0 && iseq % nreset == 0 ) rnn.emptyStacks();
      //spred += '_';  sgoal += '_';

      iseval = false;
      for(int ip = 0; ip < p.size(); ip++){
        next = dic[p[ip]];
        if(rdic[cur] == '=') iseval = true;

        rnn.forward(cur, next);

        spred += (iseval)? rdic[rnn.pred()] : '_'; sgoal += rdic[next];
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);
        if(ip == 0 && iseq == 0) rnn.emptyStacks();
        if(iseval) {
          rnn.backward();
          rnn.update(lr);
          lo -= log(rnn.eval(next)) / log(10); ne++;
          fprintf(stdout, "\r[train] lr: %.5f\tnmax: %02d\tentropy: %.3f\tgoal: %s pred: %s prog=%.1f%%",
              lr, nmax, lo / ne, sgoal.c_str(), spred.c_str(), 100.0 * iseq / nseq);
        }
        cur = next;
      }

    }
    fprintf(stdout, "\r[train] lr: %.5f\tnmax: %02d\tentropy: %.3f\tgoal: %s pred: %s\n",
        lr, nmax, lo / ne, sgoal.c_str(), spred.c_str());

    /************* VALID *************/

    nmax = max(nmaxmax, nvalidmax);
    nmin = min(nmaxmax, nvalidmax);
    ne = 1; lo = 0;

    rnn.emptyStacks();

    for(int iseq = 0; iseq < nseqv; iseq++){
      //spred += '_';  sgoal += '_';
      p = generate_addition(nmax, nmin, base);
      iseval = false;
      for(int ip = 0; ip < p.size(); ip++){
        next = dic[p[ip]];
        if(rdic[cur] == '=') iseval = true;
        rnn.forward(cur, next, ishard);
        spred += (iseval)? rdic[rnn.pred()] : '_'; sgoal += rdic[next];
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);
        if(ip == 0 && iseq == 0) rnn.emptyStacks();
        if(iseval){
          lo -= log(rnn.eval(next)) / log(10);
          ne++;
          fprintf(stdout, "\r[valid] lr: %.5f\tnmax: %d\tentropy: %.3f\tgoal: %s pred: %s prog=%.1f%%",
              lr, nmax, lo / ne, sgoal.c_str(), spred.c_str(), 100.0 * iseq / nseqv);
        }
        cur = next;
      }
    }

    fprintf(stdout, "\r[valid] lr: %.5f\tnmax: %02d\tentropy: %.3f \tgoal: %s pred: %s\n",
        lr, nmax, lo / ne, sgoal.c_str(), spred.c_str());

    if( e == 0 || lo / ne < last_ent){
      last_ent = lo / ne;
      back_up_model.copy(rnn);
      back_up_model.save(modelname);
    }
    else if( e > 0 ){
      if(e > nmaxmax/2){
        lr /= 2;
        rnn.copy(back_up_model);
      }
      if(last_ent < .1) //supervised | < .1 means it works
        rnn.copy(back_up_model);
    }
    if(lr < lrmin) break;
  }

  FILE* fseq;
  FILE* fres;
  fprintf(stdout,"Test set: \n");
  if(save){
    sprintf(buff,"data/test_seqence");
    cout << " Sequence used at test time saved at: "<< buff << endl;
    fseq = fopen(buff,"w");
    fres = fopen(testfilename.c_str(),"w");
    fprintf(fres,"validation:\t %f\n", lo / ne);
  }
  int ntest = 200;
  bool begin_seq = true;
  cur = nchar - 1;

  rnn.emptyStacks();

  for(int nm = 2; nm < 60; nm++){
    nmin = nm; nmax = nm + 1;
    float corr = 0, ecorr = 0;
    int sseq = 0; nseq = 0;
    neval = 0;
    ne = 0;lo = 0;
    if(save) f = fopen(logtestfilename.c_str(),"w");

    for(int iseq = 0; iseq < ntest; iseq++){
      p = generate_addition(nmax, nmin, base);
      iseval = false;
      if(nreset > 0 && iseq % nreset == 0 ) rnn.emptyStacks();

      for(int ip = 0; ip < p.size(); ip++){
        next = dic[p[ip]] ;
        if(save) fprintf(fseq, "%c", p[ip]);

        rnn.forward(cur, next, ishard);

        // begin of a sequence / end of evaluation:
        if (ip == 0) {
          if(iseq != 0){
            neval++;
            if( corr == sseq ) ecorr++;
            if(save)fprintf(f, "end eval - accuracy: %f \n", ecorr / neval);
          }
          sseq=0; corr = 0;
          iseval = false;
        }

        if(iseval && next == rnn.pred()) corr++;
        if(iseval) sseq++;

        lo -= log(rnn.eval(next)) / log(10);
        ne++;

        // begin of evaluation:
        if(rdic[next] == '=') {
          iseval = true;
          if(save) fprintf(f, "begin eval\n");
        }
        cur = next;
        count++;
      }
    }
    if(save){
      fprintf(fres,"%d \t %f\n", nm, ecorr / neval);
      fclose(f);
    }
    fprintf(stdout,"n: %d \t accuracy: %f \n", nm, ecorr / neval);
  }
  fprintf(stdout, "\n");
  if(save){
    fclose(fres);
    fclose(fseq);
  }
  return 0;
}
