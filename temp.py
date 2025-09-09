from helpers import *




if __name__ == "__main__":
    k_neighbors = 3
    layer= 5
    #pickle_file = f"pkls/story_L{layer}_gpt2-xl_NO_CN.pkl"
    #pickle_file = f"pkls/L{layer}_gptneo_FK_GA.pkl"
    pickle_file = "pkls/EU_L5_gptneo_EU.pkl"
    #pickle_file = "pkls/story_L23_gpt2xl.pkl"
    
    output_gexf_path = f"country_graphs/L{layer}_gptneo_20sums_EU_k{k_neighbors}.gexf"
    
    
    
    embeddings_dict = load_centroids(pickle_file=pickle_file) # shape (1024, 1600) if not centroid
    #G = create_full_knn_graph(embeddings_dict, k=k_neighbors) # Full embedding space(all tokens per country)
    G = create_centroid_knn_graph(embeddings_dict, k=k_neighbors)
    
    """ format_positions_for_gephi(G)
    nx.write_gexf(G, output_gexf_path, version="1.2draft")
    print(f"Saved batch k-NN graph to {output_gexf_path}") 
    """
    
    centroids_to_gephi_with_labels(pickle_file, output_gexf_path, k=k_neighbors, centroid=False)