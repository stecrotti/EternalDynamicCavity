# InfiniteMatrixProductBP

Code for the paper [Nonequilibrium steady-state dynamics of Markov processes on graphs](https://arxiv.org/abs/2411.19100).

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
