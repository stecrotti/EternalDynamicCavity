# InfiniteMatrixProductBP

Code for the article [Nonequilibrium steady-state dynamics of Markov processes on graphs](https://arxiv.org/abs/2411.19100).

To use the package on your own stochastic dynamics, check out the syntax of [MatrixProductBP.jl](https://github.com/stecrotti/MatrixProductBP.jl) and [TensorTrains.jl](https://github.com/stecrotti/TensorTrains.jl). Code in this repo only adds functionalities to truncate infinite tensor trains and can be used to reproduce the results in the article. 

## Installation
On Julia >= 1.9, run
```julia
import Pkg; Pkg.add(url="https://github.com/stecrotti/InfiniteMatrixProductBP.jl")
```

## Usage
Begin in the home directory of the package and run
```julia
include("setup.jl")
```

To reproduce results in the article, run scripts from `/scripts/`, which will save data to `/data/`. Generate plots via [/scripts/plots.ipynb](/scripts/plots.ipynb). 
