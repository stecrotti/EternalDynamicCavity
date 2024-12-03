import Pkg
import Logging

# activate necessary source code
Pkg.activate(@__DIR__)
Pkg.add(url="https://github.com/stecrotti/MPSKit.jl")
# Pkg.add(url="https://github.com/stecrotti/MatrixProductBP.jl", rev="uniform_tt")

# disable long VUMPS outputs
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)