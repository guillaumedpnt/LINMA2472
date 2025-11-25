module VectReverse

mutable struct VectNode
    op::Union{Nothing,Symbol}
    args::Vector{VectNode}
    value::Union{AbstractArray,Number}
    derivative::Union{AbstractArray,Number}
end

VectNode(value::Union{AbstractArray,Number}) = VectNode(nothing, VectNode[], value, zero(value))
VectNode(op::Symbol, args::Vector{VectNode}, value::Union{AbstractArray,Number}) = VectNode(op, args, value, zero(value))

Base.zero(x::VectNode) = VectNode(zero(x.value))

# For `tanh.(X)`
function Base.broadcasted(op::Function, x::VectNode)
    if op === tanh
        return VectNode(:tanh, [x], tanh.(x.value))
    elseif op === exp
        return VectNode(:exp, [x], exp.(x.value))
    elseif op === log
        return VectNode(:log, [x], log.(x.value))
    elseif op === relu
        return VectNode(:relu, [x], relu.(x.value))
    else
        error("Unary operation $op not supported")
    end
end

# For `X .* Y`
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    if op === *
        return VectNode(:mul, [x, y], x.value .* y.value)
    elseif op === +
        return VectNode(:add, [x, y], x.value .+ y.value)
    elseif op === -
        return VectNode(:sub, [x, y], x.value .- y.value)
    elseif op === /
        return VectNode(:div, [x, y], x.value ./ y.value)
    elseif op === ^
        return VectNode(:pow, [x, y], x.value .^ y.value)
    else
        error("Binary operation $op not supported")
    end
end

# For `X .* Y` where `Y` is a constant
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    y_node = VectNode(y)
    return Base.broadcasted(op, x, y_node)
end

# For `X .* Y` where `X` is a constant
function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    x_node = VectNode(x)
    return Base.broadcasted(op, x_node, y)
end

# For `x .^ 2`
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
	Base.broadcasted(^, x, y)
end

# size and size(x, d)
Base.size(x::VectNode) = size(x.value)
Base.size(x::VectNode, d::Integer) = size(x.value, d)

# length, ndims, axes
Base.length(x::VectNode) = length(x.value)
Base.ndims(x::VectNode) = ndims(x.value)
Base.axes(x::VectNode) = axes(x.value)

# element type (useful for many generic routines)
Base.eltype(x::VectNode) = eltype(x.value)

# make VectNode iterable like its underlying array
Base.iterate(x::VectNode) = iterate(x.value)
Base.iterate(x::VectNode, s) = iterate(x.value, s)

# getindex / setindex! (handy for some algorithms/tests)
Base.getindex(x::VectNode, inds...) = x.value[inds...]
Base.setindex!(x::VectNode, v, inds...) = (x.value[inds...] = v; x)

# Matrix multiplication
Base.:*(x::VectNode, y::VectNode) = VectNode(:matmul, [x, y], x.value * y.value)
Base.:*(x::VectNode, y::Union{AbstractArray,Number}) = x * VectNode(y)
Base.:*(x::Union{AbstractArray,Number}, y::VectNode) = VectNode(x) * y

# Addition
Base.:+(x::VectNode, y::VectNode) = VectNode(:add_scalar, [x, y], x.value .+ y.value)
Base.:+(x::VectNode, y::Union{AbstractArray,Number}) = x + VectNode(y)
Base.:+(x::Union{AbstractArray,Number}, y::VectNode) = VectNode(x) + y

# Subtraction
Base.:-(x::VectNode, y::VectNode) = VectNode(:sub_scalar, [x, y], x.value .- y.value)
Base.:-(x::VectNode, y::Union{AbstractArray,Number}) = x - VectNode(y)
Base.:-(x::Union{AbstractArray,Number}, y::VectNode) = VectNode(x) - y
Base.:-(x::VectNode) = VectNode(:neg, [x], -x.value)

# Division
# Base.:/(x::VectNode, y::Union{AbstractArray,Number}) = x * VectNode(inv(y))
Base.:/(x::VectNode, y::Union{AbstractArray,Number}) = VectNode(:div, [x, VectNode(y)], x.value ./ y)

# Sum reduction
# Base.sum(x::VectNode) = VectNode(:sum, [x], sum(x.value))
# Base.sum(x::VectNode; dims) = VectNode(:sum_dims, [x], sum(x.value; dims=dims))
Base.sum(x::VectNode; kwargs...) =
    VectNode(
        isempty(kwargs) ? :sum : :sum_dims,
        [x],
        sum(x.value; kwargs...)
    )

# Maximum reduction
Base.maximum(x::VectNode; dims) = VectNode(:maximum_dims, [x], maximum(x.value; dims=dims))

# Comparison operators for control flow
Base.isless(x::VectNode, y::Number) = x.value < y
Base.isless(x::Number, y::VectNode) = x < y.value
Base.isless(x::VectNode, y::VectNode) = x.value < y.value

# Max operation (needed for ReLU)
Base.max(x::VectNode, y::Union{Number,AbstractArray}) = VectNode(:max, [x, VectNode(y)], max.(x.value, y))
Base.max(x::Union{Number,AbstractArray}, y::VectNode) = VectNode(:max, [VectNode(x), y], max.(x, y.value))

# We assume `Flatten` has been defined in the parent module.
# If this fails, run `include("/path/to/Flatten.jl")` before
# including this file.
# import ..Flatten
import Main: Flatten, relu

function topo_sort!(visited, topo, f::VectNode)
    if !(f in visited)
        push!(visited, f)
        for arg in f.args
            topo_sort!(visited, topo, arg)
        end
        push!(topo, f)
    end
end

# Helper function to add to derivative, handling both scalars and arrays
function add_derivative!(node::VectNode, grad)
    if node.derivative isa Number
        node.derivative += grad
    else
        node.derivative .+= grad
    end
end

function sub_derivative!(node::VectNode, grad)
    if node.derivative isa Number
        node.derivative -= grad
    else
        node.derivative .-= grad
    end
end

function _backward!(f::VectNode)
    if isnothing(f.op)
        return
    elseif f.op == :add || f.op == :add_scalar
        for arg in f.args
            add_derivative!(arg, f.derivative)
        end
    elseif f.op == :sub || f.op == :sub_scalar
        add_derivative!(f.args[1], f.derivative)
        sub_derivative!(f.args[2], f.derivative)
    elseif f.op == :neg
        sub_derivative!(f.args[1], f.derivative)
    elseif f.op == :mul
        # Element-wise multiplication: d/dx(x.*y) = y, d/dy(x.*y) = x
        add_derivative!(f.args[1], f.derivative .* f.args[2].value)
        add_derivative!(f.args[2], f.derivative .* f.args[1].value)
    elseif f.op == :div
        # Element-wise division: d/dx(x./y) = 1/y, d/dy(x./y) = -x/y^2
        add_derivative!(f.args[1], f.derivative ./ f.args[2].value)
        sub_derivative!(f.args[2], f.derivative .* f.args[1].value ./ (f.args[2].value .^ 2))
    elseif f.op == :pow
        # Element-wise power: d/dx(x.^n) = n * x.^(n-1)
        add_derivative!(f.args[1], f.derivative .* f.args[2].value .* (f.args[1].value .^ (f.args[2].value .- 1)))
        # If args[2] is not constant, we'd need: d/dn(x.^n) = x.^n * log(x)
        # But for typical use cases, the exponent is constant
        if !isnothing(f.args[2].op)
            add_derivative!(f.args[2], f.derivative .* (f.args[1].value .^ f.args[2].value) .* log.(f.args[1].value))
        end
    elseif f.op == :matmul
        # Matrix multiplication: d/dX(X*Y) = dL/dZ * Y^T, d/dY(X*Y) = X^T * dL/dZ
        add_derivative!(f.args[1], f.derivative * f.args[2].value')
        add_derivative!(f.args[2], f.args[1].value' * f.derivative)
    elseif f.op == :tanh
        # d/dx(tanh(x)) = 1 - tanh(x)^2
        add_derivative!(f.args[1], f.derivative .* (1 .- tanh.(f.args[1].value).^2))
    elseif f.op == :exp
        # d/dx(exp(x)) = exp(x)
        add_derivative!(f.args[1], f.derivative .* exp.(f.args[1].value))
    elseif f.op == :log
        # d/dx(log(x)) = 1/x
        add_derivative!(f.args[1], f.derivative ./ f.args[1].value)
    elseif  f.op == :relu
        # d/dx(relu(x)) = 1 if x > 0, 0 otherwise (with subgradient at x == 0)
        mask = f.args[1].value .> 0
        add_derivative!(f.args[1], f.derivative .* mask)

    elseif f.op == :sum
        # d/dx(sum(x)) = ones(size(x))
        # The derivative flows back to all elements
        # add_derivative!(f.args[1], f.derivative)
        add_derivative!(f.args[1], f.derivative .* ones(size(f.args[1].value)))
    elseif f.op == :sum_dims
        add_derivative!(f.args[1], f.derivative .*ones(size(f.args[1].value)))
    elseif f.op == :maximum_dims
        max_vals = f.value
        mask = f.args[1].value .== max_vals
        add_derivative!(f.args[1], f.derivative .* mask)

    elseif f.op == :max
        # d/dx(max(x, c)) = 1 if x > c, 0 otherwise (with subgradient at x == c)
        # For max between two nodes or node and constant
        if isnothing(f.args[1].op) && !isnothing(f.args[2].op)
            # First arg is constant, second is variable
            mask = f.args[2].value .>= f.args[1].value
            add_derivative!(f.args[2], f.derivative .* mask)
        elseif !isnothing(f.args[1].op) && isnothing(f.args[2].op)
            # First arg is variable, second is constant (typical for ReLU)
            # mask = f.args[1].value .>= f.args[2].value
            mask = f.args[1].value .>= f.args[2].value .* ones(size(f.args[1].value))

            add_derivative!(f.args[1], f.derivative .* mask)
        else
            # Both are variables
            mask1 = f.args[1].value .>= f.args[2].value
            mask2 = f.args[2].value .> f.args[1].value
            add_derivative!(f.args[1], f.derivative .* mask1)
            add_derivative!(f.args[2], f.derivative .* mask2)
        end
    else
        error("Operator $(f.op) not supported yet")
    end
end

function backward!(f::VectNode)
    topo = VectNode[]
    topo_sort!(Set{VectNode}(), topo, f)
    reverse!(topo)
    
    # Initialize all derivatives to zero
    for node in topo
        node.derivative = zero(node.value)
    end
    
    # Seed the output gradient
    # f.derivative = one(f.value)
    f.derivative = ones(size(f.value))
    
    # Backpropagate
    for node in topo
        _backward!(node)
    end
    
    return f
end

function gradient!(f, g::Flatten, x::Flatten)
	x_nodes = Flatten(VectNode.(x.components))
	expr = f(x_nodes)
	backward!(expr)
	for i in eachindex(x.components)
		g.components[i] .= x_nodes.components[i].derivative
	end
	return g
end

gradient(f, x) = gradient!(f, zero(x), x)

end