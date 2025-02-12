# ConstrainTNet.jl

Repository for the paper: ["Cons-training tensor networks"](https://arxiv.org/abs/2405.09005). 

ConstrainTNet.jl requires a forked version of ITensors.jl containing relevant functionality related to constraint handling. 

Currently the maximum number of constraints that can be handled is fixed to 20. This can be increased by modifying the following line in qn/qn.jl within ITensors.jl.

```julia
const maxQNs = 20 # increase for > 20 constraints
```

## Installation

To install ConstrainTNet.jl along with the forked version of ITensors.jl as a submodule, please follow these steps:

1. Clone the repository with its submodules:
   ```bash
   git clone --recurse-submodules https://github.com/JaviLoPiq/ConstrainTNet.jl.git

2. Open the Julia REPL within the repo and set up the environment:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

3. Load ConstrainTNet.jl
    ```julia 
    using ConstrainTNet

In case of questions or suggestions, please feel free to reach out at jlopezpiquer@umass.edu.  