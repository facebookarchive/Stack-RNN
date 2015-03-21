/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _TOKENIZER_
#define _TOKENIZER_
#include <string>
#include <assert.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <ctype.h>
#include <fstream>
#include <vector>
#include <utility>

namespace rnn {

  struct Dic {

    private:

      struct _Node{
        int _idx;
        int _count;
        _Node() : _idx(-1), _count(0) {};
        _Node(int idx) : _idx(idx), _count(1) {};
      };


    public:
      typedef std::unordered_map<std::string, _Node> ht_type;
      typedef std::unordered_map<std::string, _Node>::iterator iterator;
      typedef std::unordered_map<std::string, _Node>::const_iterator const_iterator;

      static const std::string unk;

      Dic() : _count(0) { this->setUnknownWord();};

      Dic(bool ip) : _count(0)  {if(ip)this->setUnknownWord();};

      int size() const{
        return this->_ht.size();
      }


      std::string getUnkToken() const{
        return unk;
      }

      void setUnknownWord(){
        addWord(unk);
      }


      std::vector< std::pair<std::string, int> > vectorize() const {
        std::vector< std::pair<std::string, int> > v(_ht.size());
        for(const_iterator it = _ht.begin(); it != _ht.end(); it++)
          v[it->second._idx] = std::pair<std::string, int>(it->first, it->second._count);

        return v;
      }


      void setIdx(const std::string& str, const int& idx){
        if(this->isWordIn(str))
          this->_ht[str]._idx = idx;
      }

      bool isValidWord(const std::string& str) const{
        return true;
      }

      int getCount() const {
        return this->_count;
      }


      void addWord(const std::string& str){
        if(!this->isWordIn(str)){
          int s = this->_ht.size();
          this->_ht[str] = _Node(s);
          _count++;
        }
      }

      void update(const std::string& str){
        if(!this->isWordIn(str)) this->addWord(str);
        else {this->_ht[str]._count +=1; _count++;}
      }

      bool  isWordIn(const std::string& str) const {
        return this->_ht.find(str) != this->_ht.end();
      }

      int getWordIdx(const std::string& str) const {
        ht_type::const_iterator it;
        if(!this->isWordIn(str)){
          if(!this->isWordIn(unk)) return -1;
          it = _ht.find(unk);
        }
        else
          it = _ht.find(str);
        return it->second._idx;
      }

      std::string getWord(int idx) const {
        for(const_iterator it = _ht.begin(); it != _ht.end(); it++)
          if(idx == it->second._idx) return it->first;
        return unk;
      }

      int getCount(const std::string& str) const {
        std::string nstr = str;
        ht_type::const_iterator it;
        if(!this->isWordIn(nstr)){
          if(!this->isWordIn(unk)) return -1;
          it = _ht.find(unk);
        }
        else
          it = _ht.find(nstr);
        return it->second._count;
      }

      void print() const{
        for(const_iterator it = _ht.begin(); it != _ht.end(); it++)
          std::cout<<"[ "<<it->first<<" ] idx = "<<
            it->second._idx<<" count = "<< it->second._count<<std::endl;
      }
    private:
      ht_type _ht;

      int _count;
  };
  const std::string Dic::unk= "<unk>";

}// end namespace

#endif
