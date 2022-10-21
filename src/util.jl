#=
Common functions used throughout DeepHyperNEAT.
Contains iterator tools and basic statistics function. The iterator tools are
copied directly from six (Copyright (c) 2010-2018 Benjamin Peterson).
=#


using Statistics


function iterkeys(d::Dict)
    return keys(d)
end


function iteritems(d::Dict)
    return d
end


function itervalues(d::Dict)
    return values(d)
end


variance( x::Array{T,1} ) where{T<:Real} = var(x)


stdev( x::Array{T,1} ) where{T<:Real} = std(x)


function softmax( x::Array{T,1} ) where{T<:Real}
    e_x = exp.( x .- maximum(x) )
    return e_x ./ sum(e_x)
end