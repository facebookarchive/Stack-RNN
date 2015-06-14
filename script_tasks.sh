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
# example where discretization helps on a^nb^mc^{n+m}:
./train_toy  -ntask 3 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 
./train_toy  -ntask 3 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -hard 

# memorization (with smaller epochs i.e. nreset = 100 instead of 1000)
./train_toy  -ntask 4 -nchar 3 -nhid 100 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nreset 100 
# note that to reproduce the results in the paper, one needs to cycle over the seed and restart the ones which give
# entropy above average.


# experiments with noop:

./train_toy  -ntask 1 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nseq 10000  -noop
# data/test_ntask1_nchar2_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep1_hard0_seed1_nseq10000_nmax20
./train_toy  -ntask 1 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nseq 10000  -noop -hard
# data/test_ntask1_nchar2_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep1_hard1_seed1_nseq10000_nmax20
./train_toy  -ntask 1 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -nseq 10000  -noop
./train_toy  -ntask 1 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -nseq 10000  -noop -hard
# data/test_ntask1_nchar3_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep1_hard0_seed1_nseq10000_nmax10
./train_toy  -ntask 1 -nchar 4 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nseq 10000  -noop
# data/test_ntask1_nchar4_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep1_hard0_seed1_nseq10000_nmax20
./train_toy  -ntask 1 -nchar 4 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nseq 5000  -noop -hard
# data/test_ntask1_nchar4_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep1_hard1_seed1_nseq5000_nmax20

./train_toy  -ntask 2 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nseq 10000  -noop -nrep 2
# data/test_ntask2_nchar2_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep2_hard0_seed1_nseq10000_nmax15
./train_toy  -ntask 2 -nchar 2 -nhid 40 -nstack 10 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nseq 1000  -noop -hard -nrep 2
# data/test_ntask2_nchar2_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep2_hard1_seed1_nseq1000_nmax20

./train_toy  -ntask 3 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -nseq 1000  -noop
#data/test_ntask3_nchar3_nhid40_nstack10_bptt50_mod1_depth2_noop1_nrep1_hard0_seed1_nseq1000_nmax10

# experiments made with our model with a small number of hidden units and stacks:

./train_toy  -ntask 1 -nchar 2 -nhid 10 -nstack 1 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 1 -nchar 3 -nhid 10 -nstack 2 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 1 -nchar 3 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 1 -nchar 4 -nhid 10 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 
./train_toy  -ntask 1 -nchar 4 -nhid 20 -nstack 2 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1 -nseq 5000
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 2 
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 2 -hard
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 2 -bptt 50 -mod 1 -nrep 3 
./train_toy  -ntask 3 -nchar 3 -nhid 10 -nstack 1 -lr .1 -nmax 20 -depth 2 -bptt 50 -mod 1
./train_toy  -ntask 3 -nchar 3 -nhid 10 -nstack 1 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -hard 

./train_toy  -ntask 4 -nchar 3 -nhid 40 -nstack 10 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1 -nreset 100
#example with depth 1:
./train_toy  -ntask 1 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 20 -depth 1 -bptt 50 -mod 1 
./train_toy  -ntask 1 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 20 -depth 1 -bptt 50 -mod 1 -hard
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 1 -bptt 50 -mod 1 -nrep 2
./train_toy  -ntask 2 -nchar 2 -nhid 20 -nstack 2 -lr .1 -nmax 15 -depth 1 -bptt 50 -mod 1 -nrep 2 -hard

