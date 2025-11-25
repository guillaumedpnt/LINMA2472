# include("test.jl")
include(joinpath(@__DIR__, "test.jl"))

# Reference implementation we test against
# include("forward.jl")
include(joinpath(@__DIR__, "forward.jl"))


# include("flatten.jl")
include(joinpath(@__DIR__, "flatten.jl"))

## First order
# include("reverse_vectorized.jl")
include(joinpath(@__DIR__, "reverse_vectorized.jl"))


run_gradient_tests(Forward.gradient, VectReverse.gradient)