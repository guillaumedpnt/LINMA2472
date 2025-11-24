include("test.jl")

# Reference implementation we test against
include("forward.jl")
using .Forward

include("flatten.jl")

## First order
include("reverse_vectorized.jl")
import .VectReverse: gradient


run_gradient_tests(Forward.gradient, VectReverse.gradient)