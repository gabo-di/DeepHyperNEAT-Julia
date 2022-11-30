#=
Contains functions for decoding a given CPPN into a Substrate.
create_substrate() and query_cppn() are based on corresponding
functions from PurePLES (but are heavily modified for DeepHyperNEAT).
Felix Sosa
=#

function decode(cppn::FeedForwardCPPN, input_dimensions::Array{Int,1}, output_dimensions::Int, sheet_dimensions::Union{Nothing, Array{Int,1}}=nothing; substrate_type::Int=1, act_func::Symbol=:relu, last_act_func::Symbol=:linear)
    """
    Decodes a CPPN into a substrate.
    cppn             -- CPPN
    input_dimensions -- dimensions of substrate input layer
    output_dimension -- dimensions of substrate output layer
    sheet_dimensions -- optional substrate sheet dimensions
    """
    # Create input layer coordinate map from specified input dimensions
    if (input_dimensions[2] > 1)
        x = range(-1.0, 1.0, length=input_dimensions[2])
    else
        x = [0.0]
    end
    if (input_dimensions[1] > 1)
        y = range(-1.0, 1.0, length=input_dimensions[1])
    else
        y = [-1.0]
    end    
    input_layer = [(x,y) for x in x for y in y]

    # Create output layer coordinate map from specified output dimensions
    if (output_dimensions > 1)
        x = range(-1.0,1.0, length=output_dimensions) 
    else
        x = [0.0]
    end
    y = [1.0]
    output_layer = [(x,y) for x in x for y in y] 

    # Create sheet coordinate map from given sheet dimensions (if any)
    
    if !isnothing(sheet_dimensions)
        if (sheet_dimensions[2] > 1)
            x = range(-1.0, 1.0, length=sheet_dimensions[2])  
        else
            x = [0.0]
        end
        if (sheet_dimensions[1] > 1)
            y = range(-1.0, 1.0, length=sheet_dimensions[1])  
        else
            y = [0.0]
        end
        sheet = [(x,y) for x in x for y in y]
    else
        sheet = input_layer
    end
         
    # Create list of mappings to be created between substrate sheets
    connection_mappings = [cppn.nodes[x].cppn_tuple for x in keys(cppn.output_nodes) if cppn.nodes[x].cppn_tuple[1] != (1,1)]

    # Create substrate representation (dictionary of sheets and their respective coordinate maps)
    hidden_sheets = Set(cppn.nodes[node].cppn_tuple[1] for node in keys(cppn.output_nodes))
    substrate = Dict(s=>sheet for s in hidden_sheets)
    substrate[(1,0)] = input_layer
    substrate[(0,0)] = output_layer
    substrate[(1,1)] = [(0.0, 0.0)]
    layers = gather_layers(substrate)
    a = length(layers)
    if a>2
        for (key, yi ) in zip(keys(layers), range(-1.0, 1.0, length=a))
            if key != 1 && key != 0
                layer_length = length(layers[key])
                if layer_length>1 
                    x = range(-1.0, 1.0, length=layer_length) 
                else
                    x = [0.0]
                end
                for (keyi, xi) in zip(layers[key], x)
                    substrate[keyi] = [(xi, yi)]
                end                     
            end
        end
    end

    # Create dictionary of output node IDs to their respective mapping tuples
    cppn_idx_dict = Dict(cppn.nodes[idx].cppn_tuple=>idx for idx in keys(cppn.output_nodes))
    # Create the substrate
    return create_substrate(cppn, substrate, connection_mappings, cppn_idx_dict, layers; act_func = act_func, last_act_func = last_act_func, substrate_type = substrate_type)    

end


function create_substrate(cppn::FeedForwardCPPN, substrate::Dict{Tuple{Int,Int}, Array{Tuple{Float64, Float64}, 1}}, mapping_tuples::Array{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}, 1}, id_dict::Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Int}, layers::Dict{Int,Vector{Tuple{Int,Int}}};
     act_func::Symbol=:relu, last_act_func::Symbol=:linear, substrate_type::Int=1)
    """
    Creates a neural network from a CPPN and substrate representation.
    Based on PurePLES. Copyright (c) 2017 Adrian Westh & Simon Krabbe Munck.
    cppn      -- CPPN
    substrate -- substrate representation (a dictionary of sheets and their respective coordinate maps)
    mapping_tuples -- list of mappings to be created between substrate sheets
    id_dict   -- dictionary of output node IDs and their respective mapping tuples
    act_func  -- optional argument for the activation function of the substrate
    """
    node_evals = Tuple{Int, Function, Function, Array{Tuple{Int, Float64}, 1}}[]

    # Assign coordinates to input, output, and bias layers
    input_coordinates, output_coordinates, bias_coordinates = (substrate[(1,0)],(1,0)), (substrate[(0,0)],(0,0)), (substrate[(1,1)],(1,1))

    # Assign ids to nodes in the substrate
    input_node_ids = collect(range(1,length(input_coordinates[1])))
    bias_node_ids = collect(range(length(input_node_ids) + 1, length(input_node_ids) + length(bias_coordinates[1])))
    output_node_ids = collect(range(length(input_node_ids) + length(bias_node_ids) + 1, length(input_node_ids) + length(bias_node_ids) + length(output_coordinates[1])))

    # Remove the input and output layers from the substrate dictionary
    delete!(substrate, (1,0))
    delete!(substrate, (0,0))
    delete!(substrate, (1,1))

    # Create hidden layer coordinate maps
    hidden_coordinates = [(substrate[k], k) for k in keys(substrate)]

    # Assign ids to nodes in all hidden layers
    number_of_hidden_nodes = sum([length(layer[1]) for layer in hidden_coordinates])
    start_index = length(input_node_ids) + length(output_node_ids) + length(bias_node_ids)
    hidden_node_ids = collect(range(start_index + 1 , start_index + number_of_hidden_nodes))

    # Get activation function for substrate
    act_func_set = ActivationFunctionSet()
    hidden_activation = get( act_func_set, act_func)
    output_activation = get( act_func_set, last_act_func)
    
    # Decode depending on whether there are hidden layers or not
    if !isempty(hidden_node_ids)
        # Query CPPN for mapping between output layer and topmost hidden layer
        out_hid_mapping_tuples = [mapping for mapping in mapping_tuples if mapping[2] == (0,0)]
        out_node_counter, idx, hidden_idx = 0, 0, 0
        # For each coordinate in output sheet
        for oc in output_coordinates[1]
            # Adding Biases from Output to Hidden
            node_connections = query_cppn(cppn, oc, output_coordinates, bias_coordinates, bias_node_ids[1], id_dict)
            # For each connection mapping
            for mapping in out_hid_mapping_tuples
                source_sheet_id = mapping[1]
                append!(node_connections, query_cppn(cppn, oc, output_coordinates, (substrate[source_sheet_id],source_sheet_id), hidden_node_ids[idx+1], id_dict))
                idx += length(substrate[source_sheet_id])
            end    
            if !isempty(node_connections)
                push!(node_evals, (output_node_ids[out_node_counter+1], output_activation, sum, node_connections))
            end    
            hidden_idx = idx
            idx = 0
            out_node_counter += 1
        end    

        # Query CPPN for mapping between hidden layers (from top to bottom)
        hid_node_counter = 0
        next_idx = hidden_idx 
        idx = hidden_idx
        # For each hidden layer in the substrate, going from top to bottom
        for layer_idx in range(length(layers)-1, 2+1, step=-1)
            # For each sheet in the current layer, i
            for sheet_idx in range(0, length(layers[layer_idx])-1)
                # Assign target sheet id
                target_sheet_id = layers[layer_idx][sheet_idx+1]
                hid_hid_mapping_tuple = [mapping for mapping in mapping_tuples if (mapping[2] == target_sheet_id)]
                # For each coordinate in target sheet
                for hc in substrate[target_sheet_id]
                    # Adding Biases from Hidden to Hidden
                    node_connections = query_cppn(cppn, hc, (substrate[target_sheet_id], target_sheet_id), bias_coordinates, bias_node_ids[1], id_dict)
                    for mapping in hid_hid_mapping_tuple
                        source_sheet_id = mapping[1]
                        append!(node_connections, query_cppn(cppn, hc, (substrate[target_sheet_id], target_sheet_id), (substrate[source_sheet_id], source_sheet_id), hidden_node_ids[idx+1], id_dict))
                        idx += length(substrate[source_sheet_id])
                    end    
                    if !isempty(node_connections)
                        push!(node_evals, (hidden_node_ids[hid_node_counter+1], hidden_activation, sum, node_connections))
                    end    
                    hid_node_counter += 1
                    next_idx = idx
                    idx = hidden_idx
                end  
            end  
            idx = next_idx
            hidden_idx = next_idx
        end    

        # Query CPPN for mapping between bottom hidden layer to input layer
        idx = 0
        for i in range(0, length(layers[2])-1)
            # Assign target
            target_sheet_id = layers[2][i+1]
            # For each coordinate in target sheet
            for hc in substrate[target_sheet_id]
                node_connections = query_cppn(cppn, hc, (substrate[target_sheet_id],target_sheet_id), input_coordinates, input_node_ids[idx+1], id_dict)
                # Adding Biases from Hidden to Input
                append!(node_connections, query_cppn(cppn, hc, (substrate[target_sheet_id],target_sheet_id), bias_coordinates, bias_node_ids[1], id_dict))
                if !isempty(node_connections)
                    push!(node_evals, (hidden_node_ids[hid_node_counter+1], hidden_activation, sum, node_connections))
                end    
                hid_node_counter += 1
            end
        end    

    # No hidden layers
    else
        # Output Input Layer
        idx, counter = 0, 0
        for i in range(0, length(layers[0])-1)
            # Assign target
            target_sheet_id = layers[0][i+1]
            # For each coordinate in target sheet
            for oc in output_coordinates[1]
                node_connections = query_cppn(cppn, oc, output_coordinates, input_coordinates, input_node_ids[idx+1], id_dict)
                append!(node_connections, query_cppn(cppn, oc, output_coordinates, bias_coordinates, bias_node_ids[idx+1], id_dict))
                if !isempty(node_connections)
                    push!(node_evals, (output_node_ids[counter+1], output_activation, sum, node_connections))
                end
                counter += 1
            end
        end    
    end

    if substrate_type == 1
        return FeedForwardSubstrate(input_node_ids, bias_node_ids, output_node_ids, node_evals)
    elseif substrate_type == 2
        return FeedForwardSubstrate_2(input_node_ids, bias_node_ids, output_node_ids, node_evals)
    end
end

    
function query_cppn(cppn::FeedForwardCPPN, source_coordinate::Tuple{Float64, Float64}, source_layer::Tuple{Array{Tuple{Float64, Float64}, 1}, Tuple{Int,Int}}, target_layer::Tuple{Array{Tuple{Float64, Float64}, 1}, Tuple{Int,Int}}, node_idx::Int, id_dict::Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Int}, max_weight::Float64=5.0) #OJO max_weight::Float64=5.0
    """
    Given a single node's coordinates and a layer of nodes, query the CPPN for potential weights
    for all possible connections between the layer and the single node.
    Based on PurePLES. Copyright (c) 2017 Adrian Westh & Simon Krabbe Munck.
    cppn         -- CPPN
    source_coordinate -- coordinate of single node to be connected to a set of nodes
    source_layer -- layer of nodes in which source_coordinate resides
    target_layer -- layer of nodes to which source_coordinate will be connected
    node_idx     -- node index to begin on when traversing target_layer
    id_dict      -- dictionary of CPPN output node ids and their respective mapping tuples
    """
    node_connections = Tuple{Int, Float64}[]
    target_coordinates = target_layer[1]
    target_layer_id = target_layer[2]
    source_layer_id = source_layer[2]
    mapping_tuple = (target_layer_id,source_layer_id)
    cppnon_id = id_dict[mapping_tuple]
    for target_coordinate in target_coordinates
        i = [target_coordinate[1], target_coordinate[2], source_coordinate[1], source_coordinate[2]]
        #println("")
        #println(" cppnon_id ", cppnon_id, " mapping tuple ", mapping_tuple,  " coordinates ", i)
        #println(" result ", activate(cppn, i))
        #println("")
        w = activate(cppn, i)[cppnon_id]

        if abs(w) < max_weight
            push!(node_connections, (node_idx, w*max_weight))
        elseif abs(w) > max_weight
            push!(node_connections, (node_idx, max_weight))
        else
            push!(node_connections, (node_idx, 0.0))
        end    
        node_idx += 1
    end
    return node_connections
end

    
function gather_layers(substrate::Dict{Tuple{Int,Int}, Array{Tuple{Float64, Float64},1}})
    """
    Takes a dictionary representation of a substrate and returns
    a list of the layers and the sheets within those layers.
    substrate -- dictionary representation of a substrate
    """
    layers = Dict{Int, Array{Tuple{Int,Int},1}}()
    for i in range(0, length(substrate)-1)
        layers[i] = Tuple{Int,Int}[]
        for key in keys(substrate)
            if (key[1] == i) & !(key in layers[i])
                push!(layers[i], key)
            end
        end    
        if length(layers[i]) == 0
            delete!(layers ,i)
        end
    end    
    return layers
end