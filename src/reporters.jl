#=
Set of functions for reporting status of an evolutionary run.
NOTE: Only meant for XOR at the moment. Working on generalizing to any task.
=#

function report_fitness(pop::Population)
	"""
	Report average, min, and max fitness of a population
	pop -- population to be reported
	"""
	avg_fitness = 0.0
	# Find best genome in current generation and update avg fitness
	for genome in itervalues(pop.population)
		avg_fitness += genome.fitness
    end
	println("\n=================================================")
	@printf "\t\tGeneration: %6i" pop.current_gen
	println("\n=================================================")
	println("Best Fitness \t Avg Fitness \t Champion")
	println("============ \t =========== \t ========")
	@printf "%.2f \t\t %.2f \t\t %6i" pop.best_genome.fitness avg_fitness/pop.size pop.best_genome.key
	println("\n=================================================")
	println("Max Complexity \t Avg Complexity")
	println("============ \t =========== \t ========")
	@printf "%6i \t\t %6i" complexity(pop.max_complex_genome) pop.avg_complexity
    return nothing
end


function report_species(species_set::SpeciesSet, generation::Int)
	"""
	Reports species statistics
	species_set -- set contained the species
	generation  -- current generation
	"""
	println("\nSpecies Key \t Fitness Mean/Max \t Sp. Size")
	println("=========== \t ================ \t ========")
	for species in keys(species_set.species)
		# print("{} \t\t {:.2} / {:.2} \t\t {}".format(species,
		# 	species_set.species[species].fitness,
		# 	species_set.species[species].max_fitness,
		# 	len(species_set.species[species].members)))
		println(species,"   ",
			species_set.species[species].fitness,"   ",
			species_set.species[species].max_fitness,"   ",
			length(species_set.species[species].members))
    end
end