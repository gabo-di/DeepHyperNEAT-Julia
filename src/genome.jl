#=
Class for the DeepHyperNEAT genome and genes.
Largely copied from neat-python. (Copyright 2015-2017, CodeReclaimers, LLC.),
though heavily modified for DeepHyperNEAT.
=#


# Mutation probabilities
node_add_prob = 0.2
node_delete_prob = 0.2
conn_add_prob = 0.5
conn_delete_prob = 0.5
weight_mutation_rate = 0.9
weight_mutation_power = 0.1
inc_depth_prob = 0.1
inc_breadth_prob = 0.1


mutable struct ConnectionGene
    """
    Base class for CPPN connection genes
    
    key     --node key
    weight  --connection gene weight
    """
    key::Tuple{Int,Int}
    weight::Float64
    enabled::Bool
end


ConnectionGene(key::Tuple{Int,Int}, weight::Float64) = ConnectionGene(key, weight, true)


mutable struct NodeGene
    """
    Base class for CPPN node genes
    
    key             --node key 
    node_type       --node type
    activation      --activation function of node
    mapping_tuple   --mapping tuple (if output node)
    """
    
    type::String
    key::Int
    bias::Float64
    activation::Function
    response::Float64
    cppn_tuple::Union{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Nothing}
end    


NodeGene(key::Int, node_type::String, activation::Function, mapping_tuple::Union{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Nothing}) = NodeGene(node_type, key, 2*rand()-1, activation, 1.0, mapping_tuple)


mutable struct Genome
    """
    Base class for the CPPN genome.
    key -- genome key
    """
    key::Int
    node_indexer::Union{Nothing,Int}
    # Nodes and connections
    connections::Dict{Tuple{Int,Int},ConnectionGene}
    nodes::Dict{Int,NodeGene}
    fitness::Union{Nothing,Float64}
    # I/O and substrate values
    num_inputs::Int
    num_outputs::Int
    num_layers::Int
    input_keys::Array{Int,1} 
    output_keys::Array{Int,1}
    bias_keys::Array{Int,1}
    # (0,0) is designated as the output layer. (1,1) is designated
    #   as the bias sheet. Input sheet is designated as (1,0). Hidden
    #   layers range from (2,k) to (n,k). Where n is the layer number
    #   and k is the sheet number. Note again that 1 and 0 are reserved
    #   for input and output layers, respectively.
    cppn_tuples::Array{Tuple{Tuple{Int,Int},Tuple{Int,Int}},1}
    activations::ActivationFunctionSet 
    _complexity::Int
    substrate::Dict{Int,Array{Int,1}}
end


function Genome(key::Int)
    num_inputs = 4
    num_outputs = 2
    num_layers = 2
    bias_keys = [1]
    cppn_tuples = [((1,0), (0,0)),((1,1),(0,0))]
    genome = Genome(key,
                    nothing, 
                    Dict{Tuple{Int,Int},ConnectionGene}(), 
                    Dict{Int,NodeGene}(), 
                    nothing, 
                    num_inputs, 
                    num_outputs, 
                    num_layers, 
                    [-i - 1 for i in range(0,length=num_inputs)], 
                    [i for i in range(0,length=num_outputs)],  
                    bias_keys,
                    cppn_tuples,
                    ActivationFunctionSet(),
                    0,
                    Dict(1=>[0,1],0=>[0]))
    configure(genome)
    complexity(genome)
    return genome
end


function complexity(self::Genome)
    """
    Genome complexity
    """
    self._complexity = length(self.nodes) + length(self.connections)
    return self._complexity
end


function configure(self::Genome)
    """
    Configure a new fully connected genome
    """
    for input_id in self.input_keys
        for output_id in self.output_keys
            create_connection(self, input_id, output_id)
        end    
    end
    for (key, cppn_tuple) in zip(self.output_keys, self.cppn_tuples)
        create_node(self, "out", cppn_tuple, key)
    end
    return nothing
end


function copy(self::Genome, genome::Genome, gen::Int)
    """
    Copies the genes of another genome
    
    genome  --genome to be copied
    gen     --the current generation the copy is taking place
    """
    self.node_indexer = deepcopy(genome.node_indexer)
    self.num_inputs = deepcopy(genome.num_inputs)
    self.num_outputs = deepcopy(genome.num_outputs)
    self.input_keys = [x for x in genome.input_keys]
    self.output_keys = [x for x in genome.output_keys]
    self.cppn_tuples = [x for x in genome.cppn_tuples]
    self.num_layers = deepcopy(genome.num_layers)
    self.substrate = deepcopy(genome.substrate)
    self.bias_keys = [x for x in genome.bias_keys]
    self.nodes = Dict{Int,NodeGene}() 
    self.connections = Dict{Int,ConnectionGene}() 
    # Nodes
    for node_copy in itervalues(genome.nodes)
        node_to_add = NodeGene(node_copy.key, node_copy.type, node_copy.activation, node_copy.cppn_tuple)
        node_to_add.bias = node_copy.bias
        self.nodes[node_to_add.key] = node_to_add
    end
    # Connections
    for conn_copy in itervalues(genome.connections)
        conn_to_add = ConnectionGene(conn_copy.key, conn_copy.weight)
        self.connections[conn_to_add.key] = conn_to_add
    end
    return nothing
end


function create_connection(self::Genome, source_key::Int, target_key::Int, weight::Float64)
    """
    Creates a new connection gene in the genome
    
    source_key  --key of the source node of the connection
    target_key  --key of the target node of the connection
    weight      --optional weight value for connection
    """
    new_conn = ConnectionGene( (source_key, target_key), weight)
    self.connections[new_conn.key] = new_conn
    return new_conn
end


function create_connection(self::Genome, source_key::Int, target_key::Int)
    return create_connection(self, source_key, target_key, 2*rand()-1)
end    


function create_node(self::Genome, node_type::String="hidden", mapping_tuple::Union{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Nothing}=nothing, key::Union{Int,Nothing}=nothing)
    """
    Create a new node gene in the genome
    
    node_type      --node type
    mapping_tuple  --mapping tuple for output nodes
    """
    if node_type == "hidden"
        activation_key = rand(self.activations.functions).first
    else
        activation_key = :linear
    end
    if isnothing(key)
        new_node_key = get_new_node_key(self)
    else
        new_node_key = key
    end    
    activation = get(self.activations, activation_key)
    new_node = NodeGene(new_node_key, node_type, activation, mapping_tuple)
    self.nodes[new_node.key] = new_node
    return new_node
end


function mutate(self::Genome, gen::Union{Int,Nothing}=nothing, single_struct::Bool=true)
    """
    Randomly choose a mutation to execute on the genome
    
    gen             --optional argument for generation mutation occurs
    single_struct   --optional flag for only allowing one topoplogical mutation to occur per generation
    """
    if single_struct
        d = max(1, (node_add_prob + node_delete_prob + conn_add_prob + conn_delete_prob + inc_depth_prob + inc_breadth_prob))
        r = rand()
        if r < node_add_prob/d
            mutate_add_node(self, gen)
        elseif r < (node_add_prob + node_delete_prob)/d
            mutate_delete_node(self, gen)
        elseif r < (node_add_prob + node_delete_prob + conn_add_prob)/d
            mutate_add_connection(self, gen)
        elseif r < (node_add_prob + node_delete_prob + conn_add_prob + conn_delete_prob)/d    
            mutate_delete_connection(self, gen)
        elseif r < (node_add_prob + node_delete_prob + conn_add_prob + conn_delete_prob + inc_depth_prob)/d
            mutate_increment_depth(self, gen)
        elseif r < (node_add_prob + node_delete_prob + conn_add_prob + conn_delete_prob + inc_depth_prob + inc_breadth_prob)/d
            mutate_increment_bredth(self, gen)
        end    
    else
        if rand() < node_add_prob
            mutate_add_node(self, gen)
        end
        if rand() < node_delete_prob
            mutate_delete_node(self, gen)
        end
        if rand() < conn_add_prob
            mutate_add_connection(self, gen)
        end
        if  rand() < conn_delete_prob
            mutate_delete_connection(self, gen)
        end
        if rand() < inc_depth_prob
            mutate_increment_depth(self, gen)
        end
        if rand() < inc_breadth_prob
            mutate_increment_bredth(self, gen)
        end    
    end
    
    # Mutate connection genes
    for conn_gene in itervalues(self.connections)
        mutate(conn_gene, self, gen)
    end
    return nothing
end    


function mutate_add_node(self::Genome, gen::Union{Int,Nothing}=nothing)
    """
    Mutation for adding a node gene to the genome
    
    gen --optional argument for current generation mutation occurs
    """
    if !isempty(self.connections)
        conn_to_split = rand(self.connections).first
    else
        return nothing
    end
    # Create new hidden node and add to genome
    new_node = create_node(self)
    # Get weight from old connection
    old_weight = self.connections[conn_to_split].weight
    # Delete connection from genome
    delete!(self.connections, conn_to_split)
    # Create i/o connection from genome
    i, o = conn_to_split
    create_connection(self, i, new_node.key, 1.0)
    create_connection(self, new_node.key, o, old_weight)
    return nothing
end


function mutate_add_connection(self::Genome, gen::Union{Int,Nothing}=nothing) 
    """
    Mutation for adding a connection gene to the genome
    
    gen --optional argument for current generation mutation occurs
    """
    # Gather possible target nodes and source nodes
    if isempty(self.nodes)
        return nothing
    end
    possible_targets = [ii for ii in iterkeys(self.nodes)]
    target_key = possible_targets[rand(1:length(possible_targets))]
    possible_sources = vcat(possible_targets, self.input_keys)
    source_key = possible_sources[rand(1:length(possible_sources))]
    # Determine if new connection creates cycles. Currently, only
    # supports feed forward networks
    if creates_cycle(self.connections, (source_key,target_key) )
        return nothing
    end
    # Ensure connection isn't duplicate
    if (source_key, target_key) in keys(self.connections)
        self.connections[(source_key,target_key)].enabled = true
        return nothing
    end
    # Don't allow connections between two output nodes
    if (source_key in self.output_keys) & (target_key in self.output_keys)
        return nothing
    end
    new_conn = create_connection(self, source_key, target_key)
    return nothing
end


function mutate_delete_node(self::Genome, gen::Union{Int,Nothing}=nothing) 
    """
    Mutation for deleting a node gene to the genome
    
    gen --optional argument for current generation mutation occurs
    """
    available_nodes = [k for k in iterkeys(self.nodes) if !(k in self.output_keys) ]
    if isempty(available_nodes)
        return nothing
    end
    # Choose random node to delete
    del_key = available_nodes[rand(1:length(available_nodes))]
    # Iterate through all connections and find connections to node
    conn_to_delete = Set{Tuple{Int,Int}}()
    for (k, v) in iteritems(self.connections)
        if del_key in v.key
            if !(v.key in conn_to_delete)
                push!(conn_to_delete, v.key)
            end
        end
    end
    for i in conn_to_delete
        delete!(self.connections, i)
    end
    # Delete node key
    delete!(self.nodes, del_key)
    return nothing
end


function mutate_delete_connection(self::Genome, gen::Union{Int,Nothing}=nothing) 
    """
    Mutation for deleting a connection gene to the genome
    
    gen --optional argument for current generation mutation occurs
    """
    if !isempty(self.connections)
        key = rand(self.connections).first
        delete!(self.connections, key)
    end
    return nothing
end


function mutate_increment_depth(self::Genome, gen::Union{Int,Nothing}=nothing)
    """
    Mutation for adding an output node gene to the genome allowing
    it to represent a new layer in the encoded Substrate.
    gen     -- optional argument for current generation mutation occurs    
    """
    source_layer, source_sheet, = self.num_layers, 0
    target_layer, target_sheet = 0, 0
    cppn_tuple = (( source_layer, source_sheet), (target_layer, target_sheet))
    self.substrate[source_layer] = [0]
    b_key = nothing
    # Create bias nodes
    for bias_key in self.bias_keys
        if self.nodes[bias_key].cppn_tuple == ((1,1), (0,0))
            # Create new bias output node in CPPN
            new_bias_output_node = create_node(self, "out", ((1,1), (0,0)))
            # Copy over activation
            new_bias_output_node.activation = get( self.activations, :linear)
            new_bias_output_node.bias = 0
            # Sve key
            b_key = new_bias_output_node.key
            # Add connections
            for conn in iterkeys(self.connections)
                if conn[2] == bias_key
                    n = create_connection(self, conn[1], new_bias_output_node.key, 0.0)
                end    
            end
            append!(self.output_keys, new_bias_output_node.key)   
        end    
    end
    append!(self.bias_keys,  b_key)
    
    # Adjust tuples for previous CPPNONs
    for key in self.output_keys
        tup = self.nodes[key].cppn_tuple
        if (tup[2] == (0,0)) & (key != b_key)
            self.nodes[key].cppn_tuple = (tup[1], (source_layer, source_sheet))
        end
    end
    
    # Create two new gaussian nodes
    gauss_1_node = create_node(self)
    gauss_1_node.activation = get(self.activations, :sharp_gauss)
    gauss_1_node.bias = 0.0
    gauss_2_node = create_node(self)
    gauss_2_node.activation = get(self.activations, :sharp_gauss)
    gauss_2_node.bias = 0.0
    gauss_3_node = create_node(self)
    gauss_3_node.activation = get(self.activations, :sharp_gauss2)
    gauss_3_node.bias = 0.0
    # Create new CPPN Output Node (CPPNON)
    output_node = create_node(self, "out", cppn_tuple)
    output_node.activation = get(self.activations, :linear)
    output_node.bias = 0.0
    # Add new CPPNON key to list of output keys in genome
    self.num_outputs += 1
    self.num_layers += 1
    append!(self.output_keys,output_node.key)
    # Add connections 
    # x1 to gauss 1
    create_connection(self, self.input_keys[1], gauss_1_node.key, -1.0)
    # x2 to gauss 1
    create_connection(self, self.input_keys[3], gauss_1_node.key, 1.0)
    # y1 to gauss 2
    create_connection(self, self.input_keys[2], gauss_2_node.key, -1.0)
    # y2 to gauss 2
    create_connection(self, self.input_keys[4], gauss_2_node.key, 1.0)
    # Gauss 1 to gauss 3
    create_connection(self, gauss_1_node.key, gauss_3_node.key, 1.0)
    # Gauss 2 to gauss 3
    create_connection(self, gauss_2_node.key, gauss_3_node.key, 1.0)
    # Gauss 3 to CPPNON
    create_connection(self, gauss_3_node.key, output_node.key, 1.0)
    
    return nothing
end


function mutate_increment_bredth(self::Genome, gen::Union{Int,Nothing}=nothing)
    """
    Mutation for adding an output node gene to the genome allowing
    it to represent a new sheet to a preexisting layer in the encoded
    Substrate.
    gen      -- optional argument for current generation mutation occurs
    """
    # Can only expand a layer with more sheets if there is a hidden layer
    if self.num_layers <=2
        mutate_increment_depth(self)
    else
        layer = rand(range(2,stop=self.num_layers-1))
        # Find out how many sheets are represented by current CPPNONs
        num_sheets = length(self.substrate[layer])
        sheet = rand(range(0,stop=num_sheets))
        append!(self.substrate[layer], sheet)
        copied_sheet = (layer, sheet)
        keys_to_append = Int[]
        # Create bias
        b_key = nothing
        # Create bias nodes
        for bias_key in self.bias_keys
            if self.nodes[bias_key].cppn_tuple == ((1,1), copied_sheet)
                # Create new bias output node in CPPN
                new_bias_output_node = create_node(self, "out", ((1,1), (layer,num_sheets)))
                # Copy over activation
                new_bias_output_node.activation = deepcopy(self.nodes[bias_key].activation)
                new_bias_output_node.bias = deepcopy(self.nodes[bias_key].bias)
                # Save key
                b_key = new_bias_output_node.key
                append!(self.bias_keys, b_key)
                append!(self.output_keys, b_key)
                # Add connections
                for conn in iterkeys(self.connections)
                    if conn[2] == bias_key
                        create_connection(self, conn[1], new_bias_output_node.key, self.connections[conn].weight/2)
                        self.connections[conn].weight /= 2
                     end
                end
            end
        end
        
        # Search for CPPNONs that contain the copied sheet
        for key in self.output_keys
            # Create CPPNONs to represent outgoing connections
            if (self.nodes[key].cppn_tuple[1] == copied_sheet) & !(key in self.bias_keys)
                # create new cppn node for newly copied sheet
                cppn_tuple = ((layer, num_sheets), self.nodes[key].cppn_tuple[2])
                output_node = create_node(self, "out", cppn_tuple)
                output_node.activation = self.nodes[key].activation
                output_node.bias = self.nodes[key].bias
                append!(keys_to_append, output_node.key)
                # Create connections in CPPN and halve existing connections
                for conn in iterkeys(self.connections)
                    if conn[2] == key
                        self.connections[conn].weight /= 2
                        create_connection(self, conn[1], output_node.key, self.connections[conn].weight)
                    end                
                end
            end
            
            # Create CPPNONs to represent the incoming connections
            if (self.nodes[key].cppn_tuple[2] == copied_sheet) & !(key in self.bias_keys)
                # create new cppn node for newly copied sheet
                cppn_tuple = (self.nodes[key].cppn_tuple[1], (layer, num_sheets))
                output_node = create_node(self, "out", cppn_tuple)
                output_node.activation = self.nodes[key].activation
                output_node.bias = self.nodes[key].bias
                append!(keys_to_append, output_node.key)
                # Create connections in CPPN
                for conn in iterkeys(self.connections)
                    if conn[2] == key
                        create_connection(self, conn[1], output_node.key, self.connections[conn]. weight) 
                    end
                end
            end
        end
        
        # Add new CPPNONs to genome
        self.num_outputs += length(keys_to_append)
        append!(self.output_keys, keys_to_append)
    end    
    return nothing
end


function get_new_node_key(self::Genome)
    """
    Returns new node key
    """
    if isnothing( self.node_indexer )
        self.node_indexer = max(self.output_keys...) + 1
    end
    new_id = self.node_indexer
    self.node_indexer += 1
    @assert !(new_id in keys(self.nodes)) "assertion new_id not in self.nodes in genome.jl"
    return new_id
end    


function mutate(self::ConnectionGene, g::Genome, gen::Union{Int,Nothing}=nothing)
    # Mutate attributes of connection gene
    if rand() < weight_mutation_rate
        delta = (2*rand() - 1)*weight_mutation_power
        self.weight += delta
    end
    return nothing
end    









