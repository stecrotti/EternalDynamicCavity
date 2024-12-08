import Pkg
import Logging

# activate necessary source code
Pkg.activate(@__DIR__)

# disable long VUMPS outputs
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)