#=
Container for activation functions.
Largely copied from neat-python. (Copyright 2015-2017, CodeReclaimers, LLC.)
=#


function softmax_activation( x::Array{T,1} ) where{T<:Real}
    e_x = exp.( x .- maximum(x) )
    return e_x ./ sum(e_x)
end


function sigmoid_activation( x::Array{T,1} ) where{T<:Real}
    return (exp.(-x) .+ 1 ).^-1
end


function tanh_activation( x::Array{T,1} ) where{T<:Real}
    return tanh.(x)
end


function sin_activation( x::Array{T,1} ) where{T<:Real}
    return sin.(x)
end


function tan_activation( x::Array{T,1} ) where{T<:Real}
    return tan.(x)
end


function cos_activation( x::Array{T,1} ) where{T<:Real}
    return cos.(x)
end


function gauss_activation( x::Array{T,1} ) where{T<:Real}
    return exp.(-5*(max.(-3.4, min.(3.4,x) )).^2 )
end


function sharp_gauss_activation( x::Array{T,1} ) where{T<:Real}
    return exp.(-100*(x).^2 )
end


function sharp_gauss_mu_2_activation( x::Array{T,1} ) where{T<:Real}
    return exp.(-100*(x .- 2).^2 )
end


function relu_activation( x::Array{T,1} ) where{T<:Real}
    return [y>0 ? y : 0 for y in x]
end


function log_activation( x::Array{T,1} ) where{T<:Real}
    return log.(max.(1e-7, x))
end


function exp_activation( x::Array{T,1} ) where{T<:Real}
    return exp.( max.(-60, min.(60,x) ) )
end


function linear_activation( x::Array{T,1} ) where{T<:Real}
    return x
end


struct ActivationFunctionSet
    functions::Dict{Symbol,Function}
    ActivationFunctionSet() =  new( Dict(:softmax=>softmax_activation,
                                         :sigmoid=>sigmoid_activation,
                                         :tanh=>tanh_activation,
                                         :sin=>sin_activation,
                                         :tan=>tan_activation,
                                         :cos=>cos_activation,
                                         :gaus=>gauss_activation,
                                         :sharp_gauss=>sharp_gauss_activation,
                                         :sharp_gauss2=>sharp_gauss_mu_2_activation,
                                         :relu=>relu_activation,
                                         :log=>log_activation,
                                         :exp=>exp_activation,
                                         :linear=>linear_activation) )
end


function add(a::ActivationFunctionSet, key::Symbol, f::Function)
    a.functions[key] = f
    return nothing
end    


function get(a::ActivationFunctionSet, key::Symbol)
    return Base.get(a.functions, key, nothing)
end    