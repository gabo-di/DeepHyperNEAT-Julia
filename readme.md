#  Deep HyperNEAT: Extending HyperNEAT to Evolve the Architecture and Depth of Deep Networks

This implementation is a julia translation of Felix Sosa's python code in https://github.com/flxsosa/DeepHyperNEAT,
which itself takes some things from PurePLES python code in https://github.com/ukuleleplayer/pureples

The code was tested with julia 1.7, it migth work with julia 1.6 but not less. 

## Example

An example is provided in the file xor.jl. You can run it the following

julia> using Revise
julia> includet("xor.jl")

i.e., it assumes that you have instald Revise 

## Notes

The LICENSE file is a copy of Felix's LICENSE.

If you have any observation please let me know.
