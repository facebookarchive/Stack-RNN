# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


CC = g++
CFLAGS = -std=c++0x  -lm -O3 -march=native -Wall -funroll-loops -ffast-math

all: toy add

toy : train_toy.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_toy.cpp -o train_toy

add : train_add.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_add.cpp -o train_add

clean:
	rm train_toy train_add
