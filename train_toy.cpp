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

#include "common.h"
#include "task.h"
#include "StackRNN.h"

using namespace std;
using namespace rnn;

/****************
  This files is used to learn a model for a given
  toy task (e.g., a^nb^n)

  See script_toy.sh to see examples of how to use this file to
  reproduce the experiments in our paper

 **************/
void print_help(){
  printf("train_toy is used ot train a model on simple toy tasks (see Joulin and Mikolov, 2015)\n");
  printf("We print every sequence seperated by a underscore [_]. The model does not see this character.\n");
  printf("usage: train_toy [options]\n");
  printf("options:\n");
  printf("-nhid [integer]\t\t number of units in the hidden layer. Default value: 40\n");
  printf("-nstack [integer]\t number of stacks. Default value: 10\n");
  printf("-depth [integer]\t depth used of the stack to predict the hidden units. Default value: 1\n");
  printf("-stack_size [integer]\t size of the stack container. Default value: 200\n");
  printf("-bptt [integer]\t\t number of step of the back-propagation through tie (BPTT). Default value: 50\n");
  printf("-nseq [integer]\t\t number of sequences used at each training epoch. Default value: 2000\n");
  printf("-mod [integer]\t\t switch between feedforward (-mod 0), recurrence only through stacks (-mod 1) and recurrence through hidden layer and stacks (-mod 2). Default value: 21\n");
  printf("-lr [float]\t\t Learning rate. Default value: 0.1\n");
  printf("-nreset [integer]\t how often the stacks are emptied. Default value: 1000\n");
  printf("-ntask [integer]\t choice the task (see readme or script_toy.sh). Default value: 1 \n");
  printf("-nchar [integer]\t number of characters for a task (works with ntaks - see readme). Default value: 2\n");
  printf("-nrep [integer]\t\t number of repetition in characters for a task (only use for ntask=2 - see readme). Default value: 1\n");
  printf("-seed [integer]\t\t seed for the random number generator. Default value: 1\n");
  printf("-nmax [integer]\t\t the maximum value for n for the tasks (e.g. n in a^nb^n). Default value: 10\n");
  printf("-save \t\t\t use to save a bunch of things, like the model, logs... Default: false\n");
  printf("-noop \t\t\t use a no-op action on the stack. Default: false\n");
  printf("-hard \t\t\t use adiscrete actions at validation and test time. Default: false\n");
  printf("Examples: \n ./train_toy -nhid 40 -nstack 10 -depth 2 -ntask 1 -nchar 2 -lr .1 -seed 1\n");
}

void print(StackRNN& rnn, FILE* f, int cur, int next){
  int nstack = rnn._NB_STACK;
  bool isnoop = (rnn._ACTION == 3);
  int naction = rnn._ACTION;
  fprintf(f, "cur: %c next: %c pred: %c ", 'a' + cur, 'a' + next, 'a' + rnn.pred());
  fprintf(f, "prob[%c]: %f ", 'a'+next, rnn.eval(next));
  for(int s = 0; s < nstack; s++) {
    if(rnn._act[s][rnn._it_mem][push] * naction > 1. )
      fprintf(f, " push[%f] ", rnn._act[s][rnn._it_mem][push]);
    if(rnn._act[s][rnn._it_mem][pop] * naction  > 1. )
      fprintf(f, " pop[%f] ",rnn._act[s][rnn._it_mem][pop]);
    if(isnoop && rnn._act[s][rnn._it_mem][noop] * naction > 1. )
      fprintf(f, " noop[%f] ", rnn._act[s][rnn._it_mem][noop]);
  }
  for(int s = 0; s < nstack; s++) {
    fprintf(f,"stack[%d]: ",s);
    for(int d = 0; d < 3; d++)
      fprintf(f," [%d]:%.3f" , d, rnn._stack[s][rnn._it_mem][d]);
  }
  fprintf(f,"\n");
}

int main(int argc, char **argv){

  int nhid = 40;
  int nstack = 10;
  int stack_size = 200;
  int bptt = 50;
  float lr = 0.1;
  int  max_count_train = 10000000;
  string modelname = "model";
  int mod = 1;
  int nmaxmax = 5;
  int nmin = 2;
  bool isnoop = false;
  bool ishard = false;
  int nchar = 2;
  int nrep = 1;
  int nreset = 1000;
  int ntask = 1;
  int depth = 2;
  int nseq = 2000;
  int seed = 1;
  double reg = 0;
  bool save = false;

  printf("For help: train_toy --help\n");

  int ai = 1;
  while(ai < argc){
    if( strcmp( argv[ai], "--help") == 0){
      print_help();
      return 1;
    }
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
    else if( strcmp( argv[ai], "-reg") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      reg = atof(argv[ai+1]);
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
    else if( strcmp( argv[ai], "-nrep") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nrep = atoi(argv[ai+1]);
      if(nrep < 1) {printf("error nchar should be >= 1\n");return -1;}
    }
    else if( strcmp( argv[ai], "-ntask") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      ntask = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nchar") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nchar = atoi(argv[ai+1]);
      if(nchar < 2) {printf("error nchar should be >= 2\n");return -1;}
    }
    else if( strcmp( argv[ai], "-ntrain") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      max_count_train = atoi(argv[ai+1]);
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
    else if( strcmp( argv[ai], "-noop") == 0){
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
    else if( strcmp( argv[ai], "-depth") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      depth = atoi(argv[ai+1]);
      if(depth < 1) {printf("error blabla depth...\n"); return -1;}
    }
    else if( strcmp( argv[ai], "-name") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      modelname = argv[ai+1];
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }

  cout<<"seed: "<<seed<<endl<<"nhid: "<<nhid<<endl<<"nstack: "<<nstack<<endl<<"bptt: "<<bptt<<
    endl<<"mod: "<<mod<<endl<<"depth: "<<depth<<endl<<"noop: "<<isnoop<<endl<<"lr: "<<lr<<endl<<"task: "<<ntask<<" nchar:"<<nchar<<" nrep: "<<nrep<<endl;

  srand(seed);

  char buff[1000];
  sprintf(buff,"ntask%d_nchar%d_nhid%d_nstack%d_bptt%d_mod%d_depth%d_noop%d_nrep%d_hard%d_seed%d",
      ntask, nchar, nhid, nstack, bptt, mod, depth, (int)((isnoop)? 1 : 0), nrep, ishard, seed);
  modelname = "data/model_";
  modelname.append(buff);
  string  logfilename ("data/log_");
  logfilename.append(buff);
  string  testfilename = "data/test_";
  testfilename.append(buff);
  string  logtestfilename = "data/log_test_" ;
  logtestfilename.append(buff);

  sprintf(buff,"_nseq%d_nmax%d", nseq, nmaxmax);
  testfilename.append(buff);

  if(save){
    cout<< "Model saved in: "<< modelname << endl;
    cout<< "Log file for training (for current epoch): "<<logfilename << endl;
    cout<< "Test results:" << testfilename << endl;
    cout<< "Log file for the test data: "<< logtestfilename << endl;
  }


  double loss = 0;

  int nback = 1;


  cout<<"create rnn...";
  StackRNN rnn(nchar, nhid, nstack, stack_size,
      nchar, bptt, nback, mod, isnoop, depth, reg);
  StackRNN back_up_model(nchar, nhid, nstack, stack_size,
      nchar, bptt, nback, mod, isnoop, depth, reg);
  cout<<"done"<<endl;


  int cur = nchar - 1, next = 0;

  int nmax = 3;

  if(nmin >= nmax) nmax = nmin + 1;

  string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);

  // string to be print:
  string spred(50,'#');
  string sgoal(50,'#');

  vector<string> sstacks(nstack);
  for(int s = 0; s < nstack; s++)
    sstacks[s] = string(50,'#');


  int count = 0, neval = 0;
  int ne = 0;
  double lo = 0;


  int nepoch = 100;

  float last_ent = 0;
  FILE*    f;
  for(int e = 0; e < nepoch; e++){

    if(save) f = fopen(logfilename.c_str(), "w");
    nmax = max(min(e+3,nmaxmax),3);
    neval = 1; loss = 0;
    ne = 1; lo = 0;
    count = 0;

    rnn.emptyStacks();

    // train on increasingly more challenging tasks:
    for(int iseq = 0; iseq < nseq; iseq++){
      p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);

      if(save) fprintf(f,"begin sequence\n");
      spred += '_';  sgoal += '_';
      for(int s = 0; s < nstack; s++) sstacks[s] += '_';
      if(nreset == 1 || (nreset > 0 && iseq % nreset == 0 )) rnn.emptyStacks();

      for(int ip = 0; ip < p.size(); ip++){
        next = p[ip] - 'a';

        rnn.forward(cur, next);

        if(ip == 0 && iseq == 0) rnn.emptyStacks();
        else{
          rnn.backward();
          rnn.update(lr);
        }
        if (ip == 0) {
          loss -= log(rnn.eval(next)) / log(10);
          neval++;
        }
        lo -= log(rnn.eval(next)) / log(10);
        ne++;

        // print stuff:
        if(save) print(rnn, f, cur, next);

        spred += 'a' + rnn.pred(); sgoal += 'a' + next;
        for(int s = 0; s < nstack; s++) {
          if(rnn._act[s][rnn._it_mem][pop] > 0.7) sstacks[s] += '-';
          else if(rnn._act[s][rnn._it_mem][push] > 0.7) sstacks[s] += '+';
          else if(isnoop && rnn._act[s][rnn._it_mem][noop] > 0.7) sstacks[s] += '|';
          else  sstacks[s] += 'X';
        }
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);
        for(int s = 0; s < nstack; s++) if (sstacks[s].size() > 30){
          sstacks[s].erase(sstacks[s].begin(), sstacks[s].end() - 30);
        }
        if(ip == 0){
          fprintf(stdout, "\r [train] lr: %.5f it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s ",
              lr, count, nmax,  lo / ne, sgoal.c_str(), spred.c_str());
        }

        cur = next;
        count++;


      }
    }


    fprintf(stdout, "\r [train] lr: %.5f it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s ",
        lr, count, nmax,  lo / ne, sgoal.c_str(), spred.c_str());
    for(int s = 0; s < min(nstack,5); s++)
      fprintf(stdout, "| actions on stack[%d] = %s", s, sstacks[s].c_str());
    fprintf(stdout," [ - = pop, + = push, | = no-op,  X = not determined yet  ]");
    fprintf(stdout, "\n");

    // evaluation on every sequences:
    nmax = max(nmaxmax, 20), nmin = 2;
    if(nstack==0) nmax = nmaxmax; // else it does not work for standard rnn...
    neval = 1; loss = 0;
    ne = 1; lo = 0;
    count = 0;

    rnn.emptyStacks();
    cur = nchar - 1;

    if(save) fprintf(f, "[VALID]\n");

    for(int iseq = 0; iseq < 1000; iseq++){
      p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      spred += '_';  sgoal += '_';
      for(int s = 0; s < nstack; s++) sstacks[s] += '_';

      if( nreset == 1) rnn.emptyStacks();

      if(save) fprintf(f,"begin sequence\n");
      for(int ip = 0; ip < p.size(); ip++){
        next = p[ip] - 'a';

        rnn.forward(cur, next, ishard);

        //if(ip == 0 && iseq == 0) rnn.emptyStacks();

        if (ip == 0) {
          loss -= log(rnn.eval(next)) / log(10);
          neval++;
        }
        lo -= log(rnn.eval(next)) / log(10);
        ne++;


        // printing stuff
        if(save) print(rnn, f, cur, next);

        spred += 'a' + rnn.pred(); sgoal += 'a' + next;
        for(int s = 0; s < nstack; s++) {
          if(rnn._act[s][rnn._it_mem][pop] > 0.7) sstacks[s] += '-';
          else if(rnn._act[s][rnn._it_mem][push] > 0.7) sstacks[s] += '+';
          else if(isnoop && rnn._act[s][rnn._it_mem][noop] > 0.7) sstacks[s] += '|';
          else  sstacks[s] += 'X';
        }
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);
        for(int s = 0; s < nstack; s++) if (sstacks[s].size() > 30){
          sstacks[s].erase(sstacks[s].begin(), sstacks[s].end() - 30);
        }

        fprintf(stdout, "\r [valid] lr: %.5f it=%7d nmax:%d  entropy:  %.3f  goal: %s pred: %s ",
            lr, count, nmax,  lo / ne, sgoal.c_str(), spred.c_str());

        cur = next;
        count++;
      }
    }
    if(save)fprintf(f, "\n [valid] lr: %.5f it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s \n",
        lr, count, nmax, lo / ne, sgoal.c_str(), spred.c_str());


    fprintf(stdout, "\r [valid] lr: %.5f it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s ",
        lr, count, nmax,  lo / ne, sgoal.c_str(), spred.c_str());
    for(int s = 0; s < min(nstack,5); s++)
      fprintf(stdout, "| actions on stack[%d] = %s", s, sstacks[s].c_str());
    fprintf(stdout," [ - = pop, + = push, | = no-op,  X = not determined yet ]");
    fprintf(stdout, "\n");

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
    }
    if(lr < 1e-5) break;

    rnn._reg *= 2;
    if(save) fclose(f);
  }

  srand(10);

  rnn.copy(back_up_model);

  cout<< testfilename << endl;
  cout<< logtestfilename << endl;
  count = 0;

  FILE* fseq;
  FILE* fres;
  fprintf(stdout,"Test set: \n");
  if(save){
    sprintf(buff,"data/test_seqence_ntask%d_nchar%d", ntask, nchar);
    cout << " Sequence used at test time saved at: "<< buff << endl;
    fseq = fopen(buff,"w");
    fres = fopen(testfilename.c_str(),"w");
    fprintf(fres,"validation:\t %f\n", lo / ne);
  }

  int ntest = 200;
  bool iseval = false;
  rnn.emptyStacks();
  cur = nchar - 1;

  // task =4: 1st element is not part of the evaluation
  bool iscountfirstelement = (ntask != 4);

  for(int nm = 2; nm < 60; nm++){
    nmin = nm; nmax = nm + 1;
    float corr = 0, ecorr = 0;
    int sseq = 0; nseq = 0;
    neval = 0;
    ne = 0;lo = 0;
    if(save)f = fopen(logtestfilename.c_str(),"w");

    for(int iseq = 0; iseq < ntest; iseq++){

      if(ntask >= 7) rnn.emptyStacks();
      p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      iseval = false;

      for(int ip = 0; ip < p.size(); ip++){
        next = p[ip] - 'a';
        if(save)fprintf(fseq, "%c", p[ip]);

        rnn.forward(cur, next, ishard);

        //if(ip == 0 && iseq == 0) rnn.emptyStacks();

        // begin of a sequence / end of evaluation:
        if (ip == 0) {
          if(iseq != 0){
            neval++;
            if( corr == sseq && (!iscountfirstelement || next == rnn.pred()))
              ecorr++;
            if(save) fprintf(f, "end eval - accuracy: %f \n", ecorr / neval);
          }
          sseq=0; corr = 0;
          iseval = false;
        }

        if(iseval && next == rnn.pred()) corr++;
        if(iseval) sseq++;

        lo -= log(rnn.eval(next)) / log(10);
        ne++;

        // printing stuff
        if(save)print(rnn, f, cur, next);

        // begin of evaluation:
        if( (ntask == 1 && cur == 0 && next != 0)
            || (ntask == 2 && cur == 0 && next!= 0)
            || (ntask == 3 && cur == nchar -2 && next == nchar - 1)
            || (ntask == 4 && next == 0)
            || (ntask == 6 && cur == 1 && next == 2)
            || (ntask == 5 && cur == nchar -2 && next == nchar - 1) ){
          iseval = true;
          if(save)fprintf(f, "begin eval\n");
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
  if(save) fclose(fres);
  if(save) fclose(fseq);

  return 0;
}
