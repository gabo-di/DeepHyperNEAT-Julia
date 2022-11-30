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
sub_in_dims = [1,2];
sub_sh_dims = [1,1];   #  [breadth , depth] of initial genome. Note that depth >=1 makes changes in genome. and breadth >=2 makes changes in genome 
sub_o_dims = 1;

substrate_type = 2;
act_func = :relu;
last_act_func = :linear;


xor_inputs = [(ii,jj) for ii in 0.0:1.0:3.0 for jj in 0.0:1.0:3.0];
expected_outputs = [ ii+ii*jj+jj^2 for (ii,jj) in xor_inputs ];


# Evolutionary parameters
goal_fitness=2.0;
pop_key = 0;
pop_size = 25; #100
pop_elitism = 2;
num_generations = 200;


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


function f_xor_2(genomes)
	# Iterate through potential solutions
	for (genome_key, genome) in genomes
		cppn = CPPN_create(genome)
		substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims; substrate_type=substrate_type, act_func=act_func, last_act_func=last_act_func )
		sum_square_error = 0.0
		n = length(expected_outputs)
		for (inputs, expected) in zip(xor_inputs, expected_outputs)
			inputs_v = [in for in in inputs]
            push!(inputs_v, 1.0)
			actual_output = activate(substrate, inputs_v)[1]
			sum_square_error += ((actual_output - expected)^2)/n
        end
		genome.fitness = - log10(sum_square_error) #1.0 - sum_square_error
    end
    return nothing
end


function f_xor_3(genomes, info::Bool=false)
	# Iterate through potential solutions
	for (genome_key, genome) in genomes
		cppn = CPPN_create(genome)
		substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims; substrate_type=1, act_func = :sigmoid, last_act_func=:sigmoid)
		substrate_2 = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims; substrate_type=substrate_type, act_func=act_func, last_act_func=last_act_func )
		sum_square_error = 0.0
		n = length(expected_outputs)
		nn = 0
		if info
			println(genome_key)
		end
		for (inputs, expected) in zip(xor_inputs, expected_outputs)
			inputs_v = [in for in in inputs]
            push!(inputs_v, 1.0)
			partial_output = activate(substrate, inputs_v)[1]
			if partial_output > 0.9
				actual_output = activate(substrate_2, inputs_v)[1]
				sum_square_error += ((actual_output - expected)^2)
				nn += 1
				if info 
					println(inputs_v)
				end
			end
        end
		if nn > 0 
			genome.fitness = - log10(sum_square_error/(nn) + (n-nn)/nn) #1.0 - sum_square_error
			if info
				println(genome.fitness, "   "  ,sum_square_error, "    ", nn, "   ", genome_key )
				println("####################################\n")
			end
		else
			genome.fitness = -Inf
		end
    end
    return nothing
end


function report_output(pop::Population)
    genome = pop.best_genome
    cppn = CPPN_create(genome)
    substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims; substrate_type=substrate_type, act_func=act_func, last_act_func=last_act_func)
    sum_square_error = 0.0
	n = length(expected_outputs)
    println("\n=================================================")
	@printf "\tChampion Output at Generation: %6i" pop.current_gen
	println("\n=================================================")
	for (inputs, expected) in zip(xor_inputs, expected_outputs)
		@printf "\nInput: (%.2f   %.2f)\nExpected Output: %.2f" inputs[1]  inputs[2] expected
		inputs = [ii for ii in inputs]
        push!(inputs, 1.0)
		actual_output = activate(substrate, inputs)[1]
		sum_square_error += ((actual_output - expected)^2.0)/n
		@printf "\nActual Output: %g\nLoss: %g\n" actual_output sum_square_error
    end
	@printf "\nTotal Loss: %g" sum_square_error
end


# Inititalize population
pop = Population(pop_key, pop_size, Int(div(num_generations,1)), pop_elitism, sub_sh_dims);

# Run population on the defined task for the specified number of generations
#	and collect the winner
winner_genome = run(pop, f_xor_2, goal_fitness, num_generations, sub_sh_dims)

# Decode winner genome into CPPN representation
cppn = CPPN_create(winner_genome)

# Decode Substrate from CPPN
substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims; substrate_type=substrate_type, act_func=act_func, last_act_func=last_act_func)


# Run winning genome on the task again
@printf "\nChampion Genome: %6i with Fitness %g\n" winner_genome.key winner_genome.fitness
substrate.node_evals

#report_output(pop)