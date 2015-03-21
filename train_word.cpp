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
#include <string>
#include <math.h>

#include "Dic.h"
#include "StackHRNN.h"

using namespace std;
using namespace rnn;

void ReadWord(string& word, FILE *fin) {
  word = "";
  int a = 0;
  char ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        word = "</s>";
        return;
      } else continue;
    }
    word += ch;
    a++;
  }
}

bool my_compare(std::pair<std::string, int> a, std::pair<std::string, int> b){
  return a.second > b.second;
}

vector<int> setHierachicalSoftmax(Dic& dic, const int& nhier){
  vector< pair<string, int> > scounts = dic.vectorize();
  sort( scounts.begin(), scounts.end(), my_compare);

  for(int i = 0; i < scounts.size(); i++)
    dic.setIdx(scounts[i].first, i);
  int dgc =  dic.getCount();
  int nc = dgc / nhier;

  int nh = nhier;
  int i = 0, j;
  vector<int> pos;
  while(i < scounts.size()){
    int c = 0; j = i;
    while( c <= nc && j < scounts.size() ){
      c += scounts[j].second;
      j++;
    }
    nh--;
    dgc -= c;
    nc = dgc / nh;
    pos.push_back(i);
    i = j;
  }
  pos.push_back(dic.size());
  return pos;
}


int main(int argc, char **argv){
  srand(1);

  int nhid = 40;
  int nstack = 10;
  int bptt = 50;
  float alpha = 0.1;
  int  nhier = 100;
  int  max_count_train = -1;
  int stack_size = 200;
  int mod = 1;
  bool isnoop = false;
  bool ishard = false;
  int depth = 1;
  bool isreset = false;

  string filename_train = "data/ptb.train.txt";
  string filename_valid = "data/ptb.valid.txt";
  string filename_test = "data/ptb.test.txt";

  int ai = 1;
  while(ai < argc){

    if ( strcmp( argv[ai], "-db") == 0 ){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      if( strcmp(argv[ai + 1], "text8") == 0 ){
        filename_train = "data/text8/train.txt";
        filename_valid = "data/text8/valid.txt";
        filename_test = "data/text8/valid.txt";
      }
      else if(strcmp(argv[ai + 1], "ptb") == 0 ){
        filename_train = "data/ptb.train.txt";
        filename_valid = "data/ptb.valid.txt";
        filename_test = "data/ptb.test.txt";
      }
      else{
        printf("unknown database %s\n", argv[ai+1]);
        return -1;
      }
    }
    else if( strcmp( argv[ai], "-nhid") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nhid = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nhier") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nhier = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nstack") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      nstack = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-bptt") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      bptt = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-lr") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      alpha = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-stack_size") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      stack_size = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-mod") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      mod = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-noop") == 0){
      isnoop = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-hard") == 0){
      ishard = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-reset") == 0){
      isreset = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-depth") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return - 1;}
      depth = atoi(argv[ai+1]);
      if(depth < 1) {printf("error blabla depth...\n"); return -1;}
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }

  char buff[1000];
  sprintf(buff,"word_nhid%d_nstack%d_bptt%d_mod%d_depth%d_noop%d_hard%d",
      nhid, nstack, bptt, mod, depth, (int)((isnoop)? 1 : 0),  ishard);
  string modelname = "data/model_";
  modelname.append(buff);
  string  logfilename ("data/log_");
  logfilename.append(buff);
  string  testfilename = "data/res/test_";
  testfilename.append(buff);
  string  logtestfilename = "data/log_test_" ;
  logtestfilename.append(buff);
  cerr<< modelname << endl;
  cerr<< logfilename << endl;
  cerr<< testfilename << endl;
  cerr<< logtestfilename << endl;

  bool change_alpha = false;
  float loss = 0;
  float lastent = -999999999;
  FILE *f;

  f = fopen(filename_train.c_str(), "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  Dic dic(true);
  dic.setUnknownWord();
  int cur=0, next = 0,count = 0;
  string scur, snext;
  while( 1 ){
    ReadWord(scur, f);
    if (feof(f)) break;
    dic.update(scur);
    count++;
  }
  int total_count = count;

  cout<<"dic size: "<<dic.size()<<endl;
  fclose(f);

  vector<int> pos = setHierachicalSoftmax(dic, nhier);

  int nback = (bptt > 20)? 10 : (bptt + (bptt%2)) / 2;

  cout<<"create rnn...";
  StackHRNN rnn(dic.size(), nhid, nstack, stack_size,
      dic.size(), bptt, nback, pos, mod, isnoop, false, depth);
  StackHRNN best_model(dic.size(), nhid, nstack, stack_size,
      dic.size(), bptt, nback, pos, mod, isnoop, false, depth);
  cout<<"done"<< rnn._OUT<<endl;

  count = 0;
  float trainentr;
  float trainperp;

  for(int e = 0;  e < 100; e++){
    cur=0;
    loss = 0;
    count = 0;
    int it = 0;
    f = fopen(filename_train.c_str(), "rb");
    rnn.emptyStacks();

    while( 1 ){
      ReadWord(snext, f);
      if(feof(f)) break;
      count++;
      next = dic.getWordIdx(snext);

      if(isreset && dic.getWord(cur) == "</s>") rnn.emptyStacks();

      rnn.forward(cur, next);

      loss -= log10(rnn.eval(next))/log10(2);

      if(count%1000 == 0){
        printf("\rIter: %3d\t Alpha: %f\t TRAIN entropy:%.4f\t perplexity: %.4f\t Progress: %.1f%% ", e+1, alpha, loss/count, pow(2, loss/count), (float)(count)/(total_count-1) * 100);
      }

      if(it == nback && count >= bptt){
        rnn.backward();
        rnn.update(alpha);
        it = 0;
      }
      else if(it == nback && count < bptt)  it = 0;
      cur = next;
      it++;
    }
    trainentr = loss/count;
    trainperp = pow(2,loss/count);
    printf("\rIter: %3d\t Alpha: %f\t TRAIN entropy:%.4f\t perplexity: %.4f\t\t\t\t", e+1, alpha, loss/count, pow(2, loss/count));


    fclose(f);

    f = fopen(filename_valid.c_str(), "rb");
    rnn.emptyStacks();

    it = 0; loss = 0;
    cur=0;
    count = 0;

    while( 1 ){
      ReadWord(snext, f);
      if(feof(f)) break;
      count++;

      next = dic.getWordIdx(snext);

      if(feof(f)) break;

      if(isreset && dic.getWord(cur) == "</s>") rnn.emptyStacks();

      rnn.forward(cur, next, ishard);
      loss -= log10(rnn.eval(next))/log10(2);
      if(count%1000==0)
        printf("\rIter: %3d\t Alpha: %f\t TRAIN entropy:%.4f\t perplexity: %.4f\t VALID entropy:%.4f\t perplexity: %.4f   ", e+1, alpha, trainentr, trainperp,loss/count, pow(2, loss/count));
      cur = next;
    }

    fclose(f);
    printf("\rIter: %3d\t Alpha: %f\t TRAIN entropy:%.4f\t perplexity: %.4f\t VALID entropy:%.4f\t perplexity: %.4f\n", e+1, alpha, trainentr, trainperp,loss/count, pow(2, loss/count));

    if ( e!=0 && loss * 1.001 > lastent){
      change_alpha = true;
      //rnn.copy(best_model);
    }
    else{
      change_alpha = false;
        best_model.copy(rnn);
    }

    if(change_alpha)
      alpha*=0.5;

    if(alpha < 1e-5)
      break;

    lastent = loss;
  }

  //rnn.copy(best_model);

  printf("Results on the test sets\n");
  f = fopen(filename_test.c_str(), "rb");
  rnn.emptyStacks();
  loss = 0;
  cur=0;
  count = 0;


  while( 1 ){
    ReadWord(snext, f);
    if(feof(f)) break;
    count++;

    next = dic.getWordIdx(snext);

    if(feof(f)) break;

    rnn.forward(cur, next, ishard);
    loss -= log10(rnn.eval(next))/log10(2);
    if(count%1000==0)
      printf("\r Alpha: %f\t TRAIN entropy:%.4f\t perplexity: %.4f\t TEST entropy:%.4f\t perplexity: %.4f   ", alpha, trainentr, trainperp,loss/count, pow(2, loss/count));
    cur = next;
  }

  printf("\r Alpha: %f\t TRAIN entropy:%.4f\t perplexity: %.4f\t TEST entropy:%.4f\t perplexity: %.4f   \n", alpha, trainentr, trainperp,loss/count, pow(2, loss/count));
  fclose(f);
}






