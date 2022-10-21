#=
Population class for Deep HyperNEAT
Felix Sosa
=#

mutable struct Population
    """
    Struct for populations
    key     --population key
    size    --population size
    elitism --number of members that must be passed from previus gen to next gen
    """
    key::Int
    size::Int
    best_genome::Union{Nothing,Genome}
    max_complex_genome::Union{Nothing,Genome}
    min_complex_genome::Union{Nothing,Genome}
    avg_complexity::Float64
    max_dict::Dict{Int,Genome}
    last_best::Int
    current_gen::Int
    elitism::Int
    reproduction::Reproduction
    species::SpeciesSet
    population::Dict{Int,Genome}
end


function Population(key::Int, size::Int, elitism::Int=1, state::Union{Nothing, Tuple{Dict{Int,Genome}, Reproduction}}=nothing)
    #Create new population  
    species = SpeciesSet(3.5)
    if isnothing(state)    
        reproduction = Reproduction()    
        population = create_new_population(reproduction, size )
        speciate( species, population, 0)
    else
        population, reproduction = state
    end
    Population(key, size, nothing, nothing, nothing, 0.0, Dict{Int,Genome}(), 0, 0, elitism, reproduction, species, population )
end    


function run(self::Population, task::Function, goal::Real, generations::Int, log_report::Bool=false)
    """
    Run evolution on a given task for a number of generations or until
    a goal is reached.
    task -- the task to be solved
    goal -- the goal to reach for the given task that defines a solution
    generations -- the max number of generations to run evolution for
    """
    
    self.current_gen = 0
    reached_goal = false
    # Plot data
    #best_fitnesses = Array{Float64,1}[]
    
    while (self.current_gen < generations) & ( !reached_goal )
        # Assess fitness of current population
        task(iteritems(self.population))
        # Find best genome in current generation and update avg fitness        
        curr_best = nothing
        curr_max_complex = nothing
        curr_min_complex = nothing
        avg_complexities = 0
        for genome in itervalues(self.population)
            avg_complexities += complexity(genome)
            # Update generation's most fit
            if isnothing( curr_best ) || (genome.fitness > curr_best.fitness)
                curr_best = genome
            end
            # Update generation's most complex
            if isnothing( curr_max_complex ) || (complexity(genome) > complexity(curr_max_complex))
                curr_max_complex = genome
            end
            # Update generation's least complex
            if isnothing( curr_min_complex ) || (complexity(genome) < complexity(curr_min_complex))
                curr_min_complex = genome
            end 
        end
        
        # Update global best genome if possible
        if isnothing( self.best_genome ) || (curr_best.fitness > self.best_genome.fitness) 
            self.best_genome = curr_best
        end
        # Update global most and least complex genomes
        if isnothing( self.max_complex_genome ) || (complexity(curr_max_complex) > complexity(self.max_complex_genome))
            self.max_complex_genome = curr_max_complex
        end    
        if isnothing( self.min_complex_genome ) || (complexity(curr_min_complex) < complexity(self.min_complex_genome))
            self.min_complex_genome = curr_min_complex
        end    
        
        self.max_dict[self.current_gen] = self.max_complex_genome
        
        # Reporters
        if log_report
            report_fitness(self)
            report_species(self.species, self.current_gen)
        end
        #report_output(self)
        #append!(best_fitnesses, self.best_genome.fitness)
        
        self.avg_complexity  = (avg_complexities+0.0)/length(self.population)
        
        # Reached fitness goal, we can stop
        if self.best_genome.fitness >= goal
            reached_goal = true
        end

        # Create new unspeciated popuation based on current population's fitness
        self.population = reproduce_with_species(self.reproduction, self.species, self.size, self.current_gen)
        
        # Check for species extinction (species did not perform well)
        if isempty(self.species.species)
            println("!!! Species went extinct !!!")
            self.population = create_new_population(self.reproduction, self.size)
        end

        # Speciate new population
        speciate(self.species, self.population, self.current_gen)
        self.current_gen += 1
    end
    
    #plot_fitness( range(0,self.current_gen-1), best_fitnesses)
    return self.best_genome
end