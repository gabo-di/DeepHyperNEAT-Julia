#=
Class for maintaining and implementing reproductive behavior in Deep HyperNEAT.
Largely copied from neat-python. (Copyright 2015-2017, CodeReclaimers, LLC.)
=#


mutable struct Reproduction
    genome_indexer::Int
    reporters::Union{Nothing} #FALTA 
    # Numbers of elites allowed to be cloned into species each gen
    species_elitism::Int
    stagnation::Stagnation
    # Fraction of members of a species allowed to reproduce each gen
    species_reproduction_threshold::Float64
end


Reproduction(max_stagnation::Int) = Reproduction(1, nothing, 1, Stagnation(max_stagnation), 0.2)


function create_new_population(self::Reproduction, num_genomes::Int, sheet_dimensions::Union{Nothing, Array{Int,1}})
    """
    Creates a fresh population
    num_genomes   --number of genomes to create for the population
    """
    new_genomes = Dict{Int,Genome}()
    # Create num_genomes new, minimal genomes
    for ii in range(0,length=num_genomes)
        gid = self.genome_indexer
        self.genome_indexer += 1
        # Create genome
        new_genome = Genome(gid, sheet_dimensions)

        new_genomes[gid] = new_genome
    end
    return new_genomes
end


function compute_species_sizes(adjusted_fitness::Array{Float64,1}, previous_sizes::Array{Int,1}, pop_size::Int, min_species_size::Int)
    """
    Compute the proper number of offspring per species (prportional to fitness)
    adjusted_fitness   --normalized fitness of members in the population
    previous_sizes      --previous sizes of the species
    pop_size           --population size
    min_species_size   --minimum species size
    """
    
    adjust_fitness_sum = sum(adjusted_fitness)
    species_sizes = Int[]
    for (adjusted_fit, prev_size) in zip(adjusted_fitness, previous_sizes)
        if adjust_fitness_sum > 0
            # Species size should be proportional to fitness if positive
            species_size = max(min_species_size, adjusted_fit/adjust_fitness_sum*pop_size)
        else
            species_size = min_species_size
        end
        # This is basically determining if the species improved in fitness or
        # decreased
        difference = (species_size - prev_size)*0.5
        count = Int(round(difference))
        curr_size = prev_size
        # If species sees large increase in fitness, increase accordingly
        if abs(count) > 0
            curr_size += count
        elseif difference > 0
            curr_size += 1
        elseif difference < 0
            curr_size -= 1
        end
        
        append!(species_sizes, curr_size)
    end
    # Normalize the amounts so that the next generation is roughly
    # the population size requested by the user
    total_spawn = sum(species_sizes)
    norm = pop_size/total_spawn
    species_sizes = [ max(min_species_size, Int(round(n*norm))) for n in species_sizes ]
    return species_sizes
end


function reproduce_with_species(self::Reproduction, species_set::SpeciesSet, pop_size::Int, generation::Int)
    """
    Creates and speciates genomes
    
    species_set  --set of current species
    pop_size     --population size
    generation   --current generation
    """
    all_fitnesses = Float64[]
    remaining_species = Species[]
    # Traverse species and grab fitnesses from non-stagnated species
    for (sid, species, species_is_stagnant) in update(self.stagnation, species_set, generation)
        if species_is_stagnant
            println("!!! Species $sid Stagnated!!!")
            continue
        else
            # Add fitnesses of members of current species
            append!(all_fitnesses, member.fitness for member in itervalues(species.members) )
            push!(remaining_species, species)
        end
    end
    # No species
    if isempty(remaining_species)
        species_set.species = Dict{Int,Species}()
        return Dict{Int,Genome}()
    end
    # Find min/max fitness across entire population
    min_population_fitness = minimum(all_fitnesses)
    max_population_fitness = maximum(all_fitnesses)
    # Do not allow the fitness range to be zero, as we divide by it below
    population_fitness_range = max(1.0, max_population_fitness - min_population_fitness )
    # Compute adjuted fitness and record minimum species size
    adjusted_fitness = Float64[] 
    previous_sizes = Int[]
    for species in remaining_species
        # Determine current species average fitness
        mean_species_fitness = mean( [member.fitness for member in itervalues(species.members) ] )
        max_species_fitness = maximum( [member.fitness for member in itervalues(species.members) ] )
        # Determine current species adjusted fitness and update it
        species.adjusted_fitness = (mean_species_fitness - min_population_fitness)/population_fitness_range
        species.max_fitness = max_species_fitness
        
        append!(adjusted_fitness, species.adjusted_fitness )
        append!(previous_sizes, length(species.members) )
    end
    avg_adjusted_fitness = mean( adjusted_fitness )
    # Compute the number of new members for each species in the new generation
    min_species_size = max(2, self.species_elitism)
    spawn_amounts = compute_species_sizes(adjusted_fitness, previous_sizes, pop_size, min_species_size)
        
    new_population = Dict{Int,Genome}()
    species_set.species = Dict{Int,Species}()
    for (spawn, species) in zip(spawn_amounts, remaining_species)
        # If eltism is enabled, each species always at least gets to retain its elites
        spawn = max(spawn, self.species_elitism)
        @assert spawn > 0 "assert spawn > 0 in reproduction.jl"
        # The species has at least one memberfor the next generation, so retain it
        old_species_members = [ item for item in iteritems(species.members) ]
        # Update species with blank slate
        species.members = Dict{Int,Genome}()
        # Update species in species set accordingly
        species_set.species[ species.key ] = species
        # Sort old speces members in order of descending fitness
        sort!(old_species_members, rev=true, by=x->x[2].fitness) 
        # Clone elites to new generation
        if self.species_elitism > 0
            for (member_key, member) in old_species_members[1:self.species_elitism]
                new_population[member_key] = member
                spawn -= 1
            end
        end
        # If the species only has room for the elites, move onto next species
        if spawn <= 0
            continue
        end
        # Only allow fractions of species members to reproduce
        reproduction_cutoff = Int(ceil( self.species_reproduction_threshold*length(old_species_members) ))
        # Use at least two parents no matter what the threshold fraction result is
        reproduction_cutoff = min(max(reproduction_cutoff, 2),length(old_species_members))

        old_species_members = old_species_members[1:reproduction_cutoff]

        
        # Randomly choose parents and produce the number of offspring allotted to the species
        # NOTE: Asexual reproduction for now
        while spawn > 0
            spawn -= 1
            parent1_key, parent1 = old_species_members[ rand(1:length(old_species_members)) ]
            child_key = self.genome_indexer
            self.genome_indexer += 1
            child = Genome(child_key)
            copy(child, parent1,generation)
            mutate(child, generation)
            new_population[child_key] = child
        end
    end
    return new_population
end