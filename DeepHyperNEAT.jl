module DeepHyperNEAT

import Base.run
using Printf

include("activations.jl")
include("util.jl")
include("genome.jl")
include("phenomes.jl")
include("decode.jl")
include("species.jl")
include("stagnation.jl")
include("reproduction.jl")
include("population.jl")
include("reporters.jl")

end