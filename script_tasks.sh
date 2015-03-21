#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#


# compile:
make toy

# experiments made with our model with 40 hidden units and 10 stacks:

# a^nb^n
./train_toy  -ntask 1 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1
# a^nb^nc^n
./train_toy  -ntask 1 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1
# a^nb^nc^nd^n
./train_toy  -ntask 1 -nchar 4 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1

# a^nb^2n
./train_toy  -ntask 2 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nrep 2
# a^nb^3n
./train_toy  -ntask 2 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 3

# memorization (with smaller epochs i.e. nreset = 100 instead of 1000)
./train_toy  -ntask 4 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -nreset 100

# example where discretization helps on a^nb^mc^{n+m}:
./train_toy  -ntask 3 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 
./train_toy  -ntask 3 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -hard 

# multiplication a^nb^nc^{nm} (with and without noop):
./train_toy  -ntask 5 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -noop
./train_toy  -ntask 5 -nchar 3 -nhid 40 -nstack 20 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1

# experiments made with our model with a small number of hidden units and stacks:

./train_toy  -ntask 1 -nchar 2 -nhid 10 -nstack 1 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 1 -nchar 3 -nhid 10 -nstack 2 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 1 -nchar 4 -nhid 10 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 2 
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 2 -hard
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 3 
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 3 -hard #used to generate table of actions on a^nb^3n
./train_toy  -ntask 3 -nchar 3 -nhid 10 -nstack 1 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 3 -nchar 3 -nhid 10 -nstack 1 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -hard 

# RNN exepriments:

./train_toy  -ntask 1 -nchar 2 -nhid 40 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2
./train_toy  -ntask 1 -nchar 2 -nhid 100 -nstack 0 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 2

./train_toy  -ntask 1 -nchar 3 -nhid 40 -nstack 0 -lr .1 -nmax 5 -depth 2 -bptt 50 -mod 2
./train_toy  -ntask 1 -nchar 3 -nhid 100 -nstack 0 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 2

./train_toy  -ntask 1 -nchar 4 -nhid 40 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2
./train_toy  -ntask 1 -nchar 4 -nhid 100 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2

./train_toy  -ntask 2 -nchar 2 -nhid 40 -nstack 0 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 2 -nrep 2
./train_toy  -ntask 2 -nchar 2 -nhid 100 -nstack 0 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 2 -nrep 2

./train_toy  -ntask 2 -nchar 2 -nhid 40 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2 -nrep 3
./train_toy  -ntask 2 -nchar 2 -nhid 100 -nstack 0 -lr .1 -nmax 5 -depth 2 -bptt 50 -mod 2 -nrep 3

./train_toy  -ntask 3 -nchar 3 -nhid 40 -nstack 0 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 2
./train_toy  -ntask 3 -nchar 3 -nhid 100 -nstack 0 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 2

./train_toy  -ntask 5 -nchar 3 -nhid 40 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2
./train_toy  -ntask 5 -nchar 3 -nhid 100 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2

./train_toy  -ntask 4 -nchar 3 -nhid 40 -nstack 0 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 2
./train_toy  -ntask 4 -nchar 3 -nhid 100 -nstack 0 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 2
