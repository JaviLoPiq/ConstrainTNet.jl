# ConstrainTNet.jl

ConstrainTNet.jl requires a forked version of ITensors.jl from the `constraints` branch.

## Installation

To install ConstrainTNet.jl along with the forked version of ITensors.jl as a submodule, please follow these steps:

1. Clone the repository with its submodules:
   ```bash
   git clone --recurse-submodules https://github.com/JaviLoPiq/ConstrainTNet.jl.git

2. Open the Julia REPL and set up the environment:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

3. Load ConstrainTNet.jl
    ```julia 
    using ConstrainTNet