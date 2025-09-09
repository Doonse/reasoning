import umap.umap_ as umap
import tqdm
import seaborn as sns 

from helpers import *



def main_computation(model_id, 
                     all_hidden_states, 
                     group_ids, 
                     base_colors_rgb, 
                     layers_path,
                     latent_graphs_path, 
                     countries=None):
    """Sir Computealot

    Args:
        model_id (_type_): _description_
        all_hidden_states (_type_): _description_
        group_ids (_type_): _description_
        base_colors_rgb (_type_): _description_
        layers_path (_type_): _description_
        countries (_type_, optional): _description_. Defaults to None.
    
    Return:
        avg_dists_full, avg_dists_umap, layer_modularities, layer_graphs
    """

    avg_dists_full      = []
    avg_dists_umap      = []
    layer_modularities  = []
    layer_graphs        = {}
    all_pairs = None
    for layer_nr, layer_state in tqdm.tqdm(enumerate(all_hidden_states)):
        layer = layer_state.cpu().squeeze().numpy()  # [seq_len, hidden_size]
        orig_centroids = compute_cluster_centroids(layer, group_ids)
        groups, orig_dist_matrix, avg_dist_full = compute_distance_matrix(orig_centroids)
        avg_dists_full.append(avg_dist_full)
        
        reducer = umap.UMAP(
            n_neighbors=5,
            n_components=2,
            metric='euclidean',
            repulsion_strength=2,
            random_state=42
        )
        umap_result = reducer.fit_transform(layer)
        
        #  graphs for storing, later visualized in (Gephi) software
        G_latent = create_graph(layer, group_ids, k=5) 
        layer_graphs[layer_nr] = G_latent
        layer_modularities.append(compute_modularity(G_latent))
        
        umap_centroids = compute_umap_centroids(umap_result, group_ids)
        umap_groups, umap_dist_matrix, avg_dist_umap = compute_distance_matrix(umap_centroids)
        avg_dists_umap.append(avg_dist_umap)
        
        final_colors = create_final_colors(
            group_ids=group_ids, 
            base_colors_rgb=base_colors_rgb,
            start_alpha=0.2,  
            end_alpha=1.0,     
            light_factor=0.5   
        )
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        ax1.scatter(umap_result[:, 0], umap_result[:, 1],
                    c=final_colors, s=20)
        
        for g, centroid in umap_centroids.items():
            label_color = base_colors_rgb[g % len(base_colors_rgb)]
            ax1.text(centroid[0], centroid[1], f"{g}",
                    fontsize=14, fontweight='bold',
                    color="black")
        
        legend_handles = []
        unique_groups = sorted(np.unique(group_ids))
        for g in unique_groups:
            label_color = base_colors_rgb[g % len(base_colors_rgb)]
            if countries:
                patch = mpatches.Patch(color=label_color, label=f"{g}; {countries[g]}")
            else:
                patch = mpatches.Patch(color=label_color, label=f"Group {g}")
            legend_handles.append(patch)
        ax1.legend(handles=legend_handles, title="Groups", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.set_title(f"{model_id} UMAP – Layer: {layer_nr}")
        ax1.axis("off")
        
        sns.heatmap(orig_dist_matrix, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=groups, yticklabels=groups, ax=ax2)
        ax2.set_title(f"Embedded Space Centroid Distances – Layer: {layer_nr}")
        
        sns.heatmap(umap_dist_matrix, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=umap_groups, yticklabels=umap_groups, ax=ax3)
        ax3.set_title(f"UMAP (2D) Centroid Distances – Layer: {layer_nr}")
        
        plt.tight_layout()
        save_path = f"{layers_path}/layer{layer_nr}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return avg_dists_full, avg_dists_umap, layer_modularities, layer_graphs


