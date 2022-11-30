#=
Maintains stagnation behavior for speciation in Deep HyperNEAT.
Largely copied from neat-python (Copyright 2015-2017, CodeReclaimers, LLC.)
=#


mutable struct Stagnation
    # Stagnation struct
    species_fitness_func::Function
    reporters::Union{Nothing}
    species_elitism::Int
    max_stagnation::Int
end


#Stagnation(species_elitism::Int) = Stagnation(mean, nothing, 0, 15) #OOD in the original code 15
Stagnation(max_stagnation::Int) = Stagnation(mean, nothing, 0, max_stagnation)

function update(self::Stagnation, species_set::SpeciesSet, generation::Int)
    """
    Updates species fitness history, checks for stagnated species,
    and returns a list with species to remove
    
    species_set  --set containing the species and their ids
    generation   --the current generation number
    """
    species_data = Tuple{Int,Species}[]
    for (sid, species) in iteritems(species_set.species)
        if !isempty(species.fitness_history)
            prev_fitness = maximum( species.fitness_history )
        else
            prev_fitness = -prevfloat(typemax(Float64))
        end
        species.fitness = self.species_fitness_func( get_fitnesses(species) )
        append!( species.fitness_history, species.fitness )
        species.adjusted_fitness = nothing
        if isnothing( prev_fitness ) || (species.fitness > prev_fitness)
            species.last_improved = generation
        end
        push!(species_data, (sid, species) )
    end
    # Sort in ascending fitness order
    sort!(species_data, by=x->x[2].fitness)
    result = Tuple{Int,Species,Bool}[]
    species_fitness = Float64[]
    num_non_stagnant_species = length(species_data)
    for (idx, (sid, species)) in enumerate(species_data)
        # Override stagnant state if making this species as stagnant would
        # result in the total number of species dropping below the limit
        stagnant_time = generation - species.last_improved
        is_stagnant = false
        if num_non_stagnant_species > self.species_elitism
            is_stagnant = stagnant_time >= self.max_stagnation
        end
        if (length(species_data) - idx + 1) <= self.species_elitism
            is_stagnant = false
        end
        if is_stagnant
            num_non_stagnant_species -= 1
        end
        push!(result, (sid, species, is_stagnant) )
        append!(species_fitness, species.fitness)
    end
    return result
end