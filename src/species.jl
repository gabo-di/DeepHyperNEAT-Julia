#=
Classes that handle speciation in Deep HyperNEAT.
Largely copied from neat-python. Copyright 2015-2017, CodeReclaimers, LLC.
=#


mutable struct GenomeDistanceCache
    # Cache for genome distances
    distances::Dict{Tuple{Int,Int},Float64} #FALTA
    hits::Int
    misses::Int
    compatibility_disjoint_coefficient::Float64
    compatibility_weight_coefficient::Float64
end


GenomeDistanceCache() = GenomeDistanceCache(Dict{Tuple{Int,Int},Float64}(), 0, 0, 1.0, 0.5) 


function (self::GenomeDistanceCache)(genome0::Genome, genome1::Genome)
    genome_key_0 = min(genome0.key, genome1.key)
    genome_key_1 = max(genome0.key, genome1.key)
    distance = Base.get( self.distances, (genome_key_0, genome_key_1), nothing)
    if isnothing( distance )
        # Distance is not already computed
        distance = genome_distance(self, genome0, genome1)
        self.distances[ (genome_key_0, genome_key_1) ] = distance
        self.misses += 1
    else
        self.hits +=1
    end
    return distance
end


function genome_distance(self::GenomeDistanceCache, genome0::Genome, genome1::Genome )
    """
    Computes genome distance between two genomes
    """
    node_distance = 0.0
    # Determine node distance
    if !isempty(genome0.nodes) || !isempty(genome1.nodes)
        # Number of disjoint nodes between genomes
        disjoint_nodes = 0
        for genome_1_node_key in iterkeys(genome1.nodes)
            if !(genome_1_node_key in keys(genome0.nodes))
                disjoint_nodes += 1
            end    
        end
        # Determine genetic distance between individual node genes
        for (genome_0_node_key, genome_0_node) in iteritems(genome0.nodes)
            genome_1_node = Base.get(genome1.nodes, genome_0_node_key, nothing)
            if isnothing( genome_1_node )
                disjoint_nodes += 1 
            else
                # Homologous genes compute their own distance value
                node_distance += node_gene_distance(self, genome_0_node, genome_1_node)
            end
        end
        # Find most number of nodes in either genome
        max_nodes = max(length(genome0.nodes), length(genome1.nodes))
        # Determine final node genetic distance
        node_distance = (node_distance + (self.compatibility_disjoint_coefficient*disjoint_nodes))/max_nodes
    end
    
    # Determine connection gene distance
    connection_distance = 0.0
    if !isempty(genome0.connections) || !isempty(genome1.connections)
        disjoint_connections = 0
        for genome_1_conn_key in iterkeys(genome1.connections)
            if !(genome_1_conn_key in keys(genome0.connections))
                disjoint_connections += 1
            end
        end
        
        for (genome_0_conn_key, genome_0_conn) in iteritems(genome0.connections)
            genome_1_conn = Base.get(genome1.connections, genome_0_conn_key, nothing)
            if isnothing( genome_1_conn )
                disjoint_connections += 1
            else
                # Homologous genes compute their own distance value
                connection_distance += connection_gene_distance(self, genome_0_conn, genome_1_conn)
            end
        end
        
        max_conn = max(length(genome0.connections), length(genome1.connections) )
        connection_distance = (connection_distance + (self.compatibility_disjoint_coefficient*disjoint_connections))/max_conn
    end
    
    return node_distance + connection_distance
end


function node_gene_distance(self::GenomeDistanceCache, node_gene_0::NodeGene, node_gene_1::NodeGene)
    """
    Computes genetic distance between node genes
    """
    distance = abs(node_gene_0.bias - node_gene_1.bias)
    if node_gene_0.activation != node_gene_1.activation
        distance += 1
    end
    return distance*self.compatibility_weight_coefficient
end


function connection_gene_distance(self::GenomeDistanceCache, conn_gene_0::ConnectionGene, conn_gene_1::ConnectionGene)
    return abs(conn_gene_0.weight - conn_gene_1.weight)*self.compatibility_weight_coefficient
end


mutable struct Species
    # Struct for inidividual species
    key::Int
    created::Int
    last_improved::Int
    representative::Union{Nothing,Genome}
    members::Dict{Int,Genome}
    fitness::Union{Nothing,Float64}
    max_fitness::Union{Nothing,Float64}
    adjusted_fitness::Union{Nothing,Float64}
    fitness_history::Array{Float64,1}
end


Species(key::Int, created::Int) = Species(key, created, created, nothing, Dict{Int,Genome}(), nothing, nothing, nothing, Float64[])


function update(self::Species, representative::Genome, members::Dict{Int,Genome})
    self.representative = representative
    self.members = members
    return nothing
end


function get_fitnesses(self::Species)
    return [m.fitness for m in itervalues(self.members)]
end


mutable struct SpeciesSet
    # Struct for handling sets of species within a population
    threshold::Float64
    species::Dict{Int,Species}
    species_indexer::Int
    genome_to_species::Dict{Int,Int}
end


SpeciesSet(threshold::Real) = SpeciesSet(threshold, Dict{Int,Species}(), 1, Dict{Int,Int}() )


function speciate(self::SpeciesSet, population::Dict{Int,Genome}, generation::Int)
    """
    Speciates a population
    """
    # Compatibility threshold
    compatibility_threshold = self.threshold
    # Set of unspeciated members of the population
    unspeciated = Set(iterkeys(population))
    # Means of determining distances
    distances = GenomeDistanceCache()
    # New representatives and members of species
    new_representatives = Dict{Int,Int}()
    new_members = Dict{Int,Array{Int,1}}()
    # Traverse through set of species from last generation
    for (sid, species) in iteritems(self.species)
        # Candidates for current species representatives
        candidate_representatives = Tuple{Float64,Genome}[]
        # Traverse genomes in the unspeciated and check their distance
        # from the current species representative
        for gid in unspeciated
            genome = population[gid]
            genome_distance = distances(species.representative, genome)
            push!(candidate_representatives, (genome_distance, genome))
        end
        # The new representative for the current species is the
        # closest to the current representative
        _, new_rep = sort(candidate_representatives, by= x->x[1] )[1]
        new_rid = new_rep.key
        new_representatives[sid] = new_rid
        new_members[sid] = [new_rid]
        delete!(unspeciated, new_rid)
    end
    # Partition the population in species pased on genetic similarity
    while !isempty(unspeciated)
        gid = pop!(unspeciated)
        genome = population[gid]
        # Find the species with the most similar representative to the 
        # current genome from the unspeciated set
        candidate_species = Tuple{Float64, Int}[] #!FALTA
        # Traverse species and their representatives
        for (sid, rid) in iteritems(new_representatives)
            representative = population[rid]
            # Determine current genome's distance from representative
            genome_distance = distances(representative, genome)
            # If it's below threshold, add it to list for adding to the species
            if genome_distance < compatibility_threshold
                push!(candidate_species, (genome_distance, sid))
            end            
        end
        # Add current genome to the species its most genetically similar to
        if !isempty(candidate_species)
            _, sid = sort(candidate_species, by=x->x[1])[1]
            append!(new_members[sid],gid)
        else
            # No species is similar enough so we create a new species with
            # the current genome as its representative
            sid = self.species_indexer
            self.species_indexer += 1
            new_representatives[sid] = gid
            new_members[sid] = [gid]            
        end
        # Update species collection based on new speciation
        self.genome_to_species = Dict{Int,Int}()
        for (sid, rid) in iteritems(new_representatives)
            # Add species if not existing in current species set
            s = Base.get(self.species, sid, nothing)
            if isnothing( s )
                s = Species(sid, generation)
                self.species[sid] = s
            end
            # Collect and add members to current species
            members = new_members[sid]
            for gid in members
                self.genome_to_species[gid] = sid
            end
            # Update current species members and representative
            member_dict = Dict(gid=>population[gid] for gid in members)
            update(s, population[rid], member_dict)
        end    
    end
end    
    
function get_species_key(self::SpeciesSet, key::Int)
    return self.genome_to_species[key]
end
    
    
function get_species(self::SpeciesSet, key::Int)
    return self.species[get_species_key(self,key)]
end