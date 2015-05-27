# Stack RNN
Stack RNN is a project gathering the code from the paper 
*Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets* by Armand Joulin and Tomas Mikolov ([pdf](http://arxiv.org/abs/1503.01007)).
In this research project, we focus on extending Recurrent Neural Networks (RNN) with a stack to allow them to learn sequences which require
some form of persistent memory. 

Examples are given in the script `script_tasks.sh`. The code is still under construction. 
We are working on releasing the code for the list RNN. If you have any suggestion, please let us know (contacts below).


## Examples
To run the code on a task:
```
> make toy
> ./train_toy  -ntask 1 -nchar 2 -nhid 10 -nstack 1 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1
```
To run the code on binary addition:
```
> make add
> ./train_add 
```

## Requirements
Stack RNN works on:
* Mac OS X
* Linux

It was not tested on Windows. To compile the code a relatively recent version of g++ is required.

## Building Stack RNN
Run `make` to compile everything. 


## Options
For more help about the options:
```
> make toy
> ./train_toy --help
```
Note that `train_add` can take the same options as `train_toy`.


## Join the Stack RNN community
* Paper: http://arxiv.org/abs/1503.01007
* Facebook page: https://www.facebook.com/fair
* Contact: ajoulin@fb.com

See the CONTRIBUTING file for how to help out.

## License
Stack RNN is BSD-licensed. We also provide an additional patent grant





