#=
Contains classes for CPPN and Substrate phenomes.
Largely copied from neat-python. (Copyright 2015-2017, CodeReclaimers, LLC.)
=#

function creates_cycle(connections::Dict{Tuple{Int,Int},ConnectionGene}, test::Tuple{Int,Int})
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o
        return true
    end
    
    visited = Set(o)
    while true
        num_added = 0
        for (a, b) in keys(connections)
            if (a in visited) & !(b in visited)
                if b == i
                    return true
                end
                
                push!(visited, b)
                num_added += 1
            end
        end
        if num_added == 0
            return false
        end
    end
end


function required_for_output(inputs::Array{Int, 1}, outputs::Array{Int, 1}, connections::Array{Tuple{Int,Int},1})
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.
    Returns a set of identifiers of required nodes.
    """
    required = Set(outputs)
    s = Set(outputs)
    while true
        # Find nodes not in S whose output is consumed by a node in s.
        t = Set(a for (a,b) in connections if (b in s) & !(a in s))
        if isempty(t)
            break
        end
        layer_nodes = Set(x for x in t if !(x in inputs))
        if isempty(layer_nodes)
            break
        end
        union!(required, layer_nodes)
        union!(s, t)
    end
    return required
end


function feed_forward_layers(inputs::Array{Int, 1}, outputs::Array{Int, 1}, connections::Array{Tuple{Int,Int},1})
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """
    
    required = required_for_output(inputs, outputs, connections)
    
    layers = Set{Int}[]
    s = Set(inputs)
    while true
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = Set(b for (a, b) in connections if ((a in s) & !(b in s)))
        # Keep only the used nodes whose entire input set is contained in s.
        t = Set{Int}()
        for n in c
            if (n in required) & (all(a in s for (a, b) in connections if b == n))
                push!(t, n)
            end
        end
        
        if isempty(t)
            break
        end
        
        push!(layers,t)
        union!(s, t)
    end
    
    return layers
end


mutable struct FeedForwardCPPN
    """
    Feed forward representation of a CPPN.
    inputs     -- input nodes of CPPN
    outpusts   -- output nodes of CPPN
    node_evals -- objects containing information for each node
    nodes      -- all nodes of CPPN
    mapping_tuples -- mapping tuples associated with each output node
    """
    input_nodes::Array{Int,1} 
    output_nodes::Dict{Int,Tuple{Tuple{Int,Int},Tuple{Int,Int}}}
    node_evals::Array{ Tuple{ Int, Function, Function, Array{Tuple{Int,Float64},1} } ,1}
    values::Dict{Int,Float64}
    nodes::Union{Nothing,Dict{Int,NodeGene}}
end


function FeedForwardCPPN(inputs::Array{Int,1}, outputs::Dict{Int,Tuple{Tuple{Int,Int},Tuple{Int,Int}}},
    node_evals::Array{ Tuple{ Int, Function, Function, Array{Tuple{Int,Float64},1}},1},
    nodes::Union{Nothing,Dict{Int,NodeGene}}=nothing, mapping_tuples::Union{Nothing,Dict{Int,Tuple{Tuple{Int,Int},Tuple{Int,Int}}}}=nothing )
    input_nodes = inputs
    if isnothing(mapping_tuples)
        output_nodes = outputs
    else
        output_nodes = Dict(key=>mapping_tuples[key] for key in keys(mapping_tuples))
    end
    values = Dict(key=>0.0 for key in union(Set(inputs), keys(output_nodes))) 
    return FeedForwardCPPN(input_nodes, output_nodes, node_evals, values, nodes)    
end
   

function activate(self::FeedForwardCPPN, inputs::Array{Float64, 1})
    if length(self.input_nodes) != length(inputs)
        return nothing
    end
    
    for (k, v) in zip(self.input_nodes, inputs)
        self.values[k] = v
    end
    

    for (node, act_func, agg_func, incoming_connections) in self.node_evals
        node_inputs = Float64[]
        for (node_id, conn_weight) in incoming_connections
            append!(node_inputs, self.values[node_id]*conn_weight)
        end
        s = agg_func(node_inputs)
        self.values[node] = act_func([s])[1]
    end
    return self.values
end


function CPPN_create(genome::Genome)
    connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
    layers = feed_forward_layers(genome.input_keys, genome.output_keys, connections)
    node_evals = Tuple{Int, Function, Function, Array{Tuple{Int, Float64}, 1}}[] 
    mapping_tuples = Dict{Int,Tuple{Tuple{Int,Int},Tuple{Int,Int}}}()
    # Traverse layers
    for layer in layers
        # For each node in each layer, collect all incoming connections to the node
        for node in layer
            incoming_connections = Tuple{Int, Float64}[] 
            for conn_key in connections
                input_node, output_node = conn_key
                if output_node == node
                    cg = genome.connections[conn_key]
                    push!(incoming_connections, (input_node, cg.weight))
                end
            end
            # Gather node gene information
            node_gene = genome.nodes[node]
            activation_function = node_gene.activation
            push!(node_evals, (node, activation_function, sum, incoming_connections))
        end
    end
    # Gather mapping tuples
    for key in genome.output_keys
        mapping_tuples[key] = genome.nodes[key].cppn_tuple        
    end    
    for key in genome.bias_keys
        mapping_tuples[key] = genome.nodes[key].cppn_tuple
    end
    return FeedForwardCPPN(genome.input_keys, mapping_tuples, node_evals, genome.nodes)
end    


mutable struct FeedForwardSubstrate
    input_nodes::Array{Int,1} 
    bias_node::Array{Int,1}
    output_nodes::Array{Int,1}
    node_evals::Array{ Tuple{ Int, Function, Function, Array{Tuple{Int,Float64},1} } ,1}
    values::Dict{Int,Float64}   
end


function FeedForwardSubstrate(inputs::Array{Int,1}, bias::Array{Int,1}, outputs::Array{Int,1}, node_evals::Array{ Tuple{ Int, Function, Function, Array{Tuple{Int,Float64},1} } ,1})
    input_nodes = inputs
    bias_nodes = bias
    output_nodes = outputs
    values = Dict(key=>0.0 for key in union(Set(inputs), Set(outputs))) 
    return FeedForwardSubstrate(input_nodes, bias_nodes, output_nodes, node_evals, values)
end


function activate(self::FeedForwardSubstrate, inputs::Array{Float64, 1})
    if (length(self.input_nodes) + length(self.bias_node)) != length(inputs)
        return nothing
    end
    
    for (k, v) in zip(self.input_nodes, inputs)
        self.values[k] = v
    end
    
    self.values[self.bias_node[1]] = inputs[end]
    evaluations = reverse(self.node_evals)
    for (node, act_func, agg_func, links) in evaluations
        node_inputs = Float64[]
        for (i, w) in links
            append!(node_inputs,self.values[i]*w)
        end
        s = agg_func(node_inputs)
        self.values[node] = act_func([s])[1]
    end
    return [self.values[i] for i in self.output_nodes]
end


function Substrate_create(genome::Genome)
    connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
    layers = feed_forward_layers(genome.input_keys, genome.output_keys, connections)
    node_evals = Tuple{ Int, Function, Function, Array{Tuple{Int,Float64},1} }[]
    # Traverse layers
    for layer in layers
        # For each node in each layer, collect all incoming connections to the node
        for node in layer
            inputs = Tuple{Int,Float64}[]
            for conn_key in connections
                input_node, output_node = conn_key
                if output_node == node
                    cg = genome.connections[conn_key]
                    push!(inputs, (input_node, cg.weight))
                end
            end
            # Gather node gene information
            node_gene = genome.nodes[node]
            activation_function = node_gene.activation
            push!(node_evals, (node, activation_function, sum, node_gene.bias, inputs))
        end
    end
    return FeedForwardSubstrate(genome.input_keys, genome.bias_keys, genome.output_keys, node_evals)
end    
