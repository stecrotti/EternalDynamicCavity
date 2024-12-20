# EternalDynamicCavity
[![CI](https://github.com/stecrotti/EternalDynamicCavity/actions/workflows/CI.yml/badge.svg)](https://github.com/stecrotti/EternalDynamicCavity/actions/workflows/CI.yml)

Code for the article [Nonequilibrium steady-state dynamics of Markov processes on graphs](https://arxiv.org/abs/2411.19100).

## Usage
Clone this repo, move to the main directory, then run
```
julia -E 'import Pkg; Pkg.activate("."); Pkg.instantiate()'
```
to start Julia and add the necessary dependencies.

To reproduce results in the article, run (`include(<path/to/file.jl>)`) scripts from [`scripts/`](scripts/), which will save data to [`data/`](data/). Generate plots via [`scripts/plots.ipynb`](scripts/plots.ipynb). 


To try the algorithm on your own stochastic dynamics, use [MatrixProductBP.jl](https://github.com/stecrotti/MatrixProductBP.jl). 
The best way to get acquainted with the syntax is to check out the examples in [`scripts/`](scripts/).
