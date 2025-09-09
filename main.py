from helpers import *
from compute import main_computation
from text import text_grouping, input_texts, countries

class EmbeddingVisualizer:
    def __init__(self, 
                 model_name: str, 
                 text_input_desc: str, 
                 base_folder: str="images"):
        
        self.model_name = model_name
        self.text_input_desc = text_input_desc
        self.base_folder = base_folder
        
        self.model, self.device, self.config, self.tokenizer, \
        self.model_name, self.model_id = load_model(self.model_name)
        
        self.input_texts_token_lengths = text_grouping(
            tokenizer=self.tokenizer, paragraphs=input_texts
        )

        self.group_ids, self.num_groups, self.point_colors, \
        self.cmap, self.base_colors_rgb, self.all_hidden_states = grouping_and_coloring(
            self.tokenizer,
            self.model,
            self.input_texts_token_lengths
        )

        self.folder_path, self.latent_graphs_path, \
        self.layers_path, self.full_folder_path = create_folders(
            self.model_name, self.text_input_desc
        )
        
    def run(self):
        avg_dists_full, avg_dists_umap, \
        layer_modularities, layer_graphs = main_computation(
            self.model_id,
            self.all_hidden_states,
            self.group_ids,
            self.base_colors_rgb,
            self.layers_path,
            self.latent_graphs_path,
            countries=countries
        )

        measurement_plots(
            avg_dists_full,
            avg_dists_umap,
            layer_modularities,
            self.full_folder_path
        )

        create_gephi_graphs(
            layer_graphs,
            self.latent_graphs_path,
            self.model_id
        )
""" 

if __name__ == "__main__":
    EmbeddingVisualizer("EleutherAI/gpt-neo-1.3B", "") """