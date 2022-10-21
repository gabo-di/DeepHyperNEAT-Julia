using Revise, Infiltrator


import Base.run
using Printf

includet("activations.jl")
includet("util.jl")
includet("genome.jl")
includet("phenomes.jl")
includet("decode.jl")
includet("species.jl")
includet("stagnation.jl")
includet("reproduction.jl")
includet("population.jl")
includet("reporters.jl")




# Substrate parameters
sub_in_dims = [1,2]
sub_sh_dims = [1,3]
sub_o_dims = 1

# Evolutionary parameters
goal_fitness=0.98
pop_key = 0
pop_size = 150
pop_elitism = 2
num_generations = 500


function f_xor(genomes)
	# Task parameters
	xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
	expected_outputs = [0.0, 1.0, 1.0, 0.0]
	# Iterate through potential solutions
	for (genome_key, genome) in genomes
		cppn = CPPN_create(genome)
		substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims)
		sum_square_error = 0.0
		for (inputs, expected) in zip(xor_inputs, expected_outputs)
			inputs_v = [in for in in inputs]
            push!(inputs_v, 1.0)
			actual_output = activate(substrate, inputs_v)[1]
			sum_square_error += ((actual_output - expected)^2)/4
        end
		genome.fitness = 1.0 - sum_square_error
    end
    return nothing
end


function report_output(pop::Population)
    genome = pop.best_genome
    cppn = CPPN_create(genome)
    substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims)
    sum_square_error = 0.0
    println("\n=================================================")
	@printf "\tChampion Output at Generation: %6i" pop.current_gen
	println("\n=================================================")
	for (inputs, expected) in zip(xor_inputs, expected_outputs)
		@printf "\nInput: (%.2f   %.2f)\nExpected Output: %.2f" inputs[1]  inputs[2] expected
		inputs = [ii for ii in inputs]
        push!(inputs, 1.0)
		actual_output = activate(substrate, inputs)[1]
		sum_square_error += ((actual_output - expected)^2.0)/4.0
		@printf "\nActual Output: %g\nLoss: %g\n" actual_output sum_square_error
    end
	@printf "\nTotal Loss: %g" sum_square_error
end


# Inititalize population
pop = Population(pop_key, pop_size, pop_elitism)

# Run population on the defined task for the specified number of generations
#	and collect the winner
winner_genome = run(pop, f_xor, goal_fitness, num_generations)

# Decode winner genome into CPPN representation
cppn = CPPN_create(winner_genome)

# Decode Substrate from CPPN
substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims)


# Run winning genome on the task again
@printf "\nChampion Genome: %6i with Fitness %g\n" winner_genome.key winner_genome.fitness

xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
expected_outputs = [0.0, 1.0, 1.0, 0.0]

report_output(pop)	  					