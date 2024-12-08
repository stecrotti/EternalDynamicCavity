# InfiniteMatrixProductBP

Code for the article "Nonequilibrium steady-state dynamics of Markov processes on graphs" ([arXiv:2411.19100](https://arxiv.org/abs/2411.19100)). 

## Installation
On Julia >= 1.9, run
```julia
import Pkg; Pkg.add(url="https://github.com/stecrotti/InfiniteMatrixProductBP.jl")
```

## Usage

To play around with the method on your own stochastic dynamics, check out the syntax of [MatrixProductBP.jl](https://github.com/stecrotti/MatrixProductBP.jl) and [TensorTrains.jl](https://github.com/stecrotti/TensorTrains.jl) where most of the source code lies. Code in this repo only adds functionalities to truncate infinite tensor trains and can be used to reproduce the results in the article.

To reproduce results in the article, clone the repo, begin in the main directory and run
```julia
import Pkg; Pkg.activate(".")
```
then run scripts from `/scripts/`, which will save data to `/data/`. Generate plots via [/scripts/plots.ipynb](/scripts/plots.ipynb). 