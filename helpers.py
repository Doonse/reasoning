import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from sklearn.neighbors import kneighbors_graph
import gc, torch
import os
from text import *
import pandas as pd
import pickle
from pandas_datareader import wb
import wbgapi as wb
import pypopulation
import pycountry



# 1=AF, 2=EU, 3=AS, 4=NA, 5=SA, 6=OC, 7=AN
continents = {
    1: "AF", 
    2: "EU", 
    3: "AS", 
    4: "NA", 
    5: "SA", 
    6: "OC", 
    7: "AN"
}

    
continent_map = {
    # Africa (1)
    "DZ": 1, "AO": 1, "BJ": 1, "BW": 1, "BF": 1, "BI": 1, "CM": 1, "CV": 1, "CF": 1,
    "TD": 1, "KM": 1, "CG": 1, "CD": 1, "CI": 1, "DJ": 1, "EG": 1, "GQ": 1, "ER": 1,
    "SZ": 1, "ET": 1, "GA": 1, "GM": 1, "GH": 1, "GN": 1, "GW": 1, "KE": 1, "LS": 1,
    "LR": 1, "LY": 1, "MG": 1, "MW": 1, "ML": 1, "MR": 1, "MU": 1, "YT": 1, "MA": 1,
    "MZ": 1, "NA": 1, "NE": 1, "NG": 1, "RW": 1, "ST": 1, "SN": 1, "SC": 1, "SL": 1,
    "SO": 1, "ZA": 1, "SS": 1, "SD": 1, "TZ": 1, "TG": 1, "TN": 1, "UG": 1, "ZM": 1,
    "ZW": 1,

    # Europe (2)
    "AX": 2, "AL": 2, "AD": 2, "AT": 2, "BY": 2, "BE": 2, "BA": 2, "BG": 2, "HR": 2,
    "CY": 2, "CZ": 2, "DK": 2, "EE": 2, "FI": 2, "FR": 2, "GE": 2, "DE": 2, "GI": 2,
    "GR": 2, "HU": 2, "IS": 2, "IE": 2, "IT": 2, "LV": 2, "LI": 2, "LT": 2, "LU": 2,
    "MT": 2, "MD": 2, "MC": 2, "ME": 2, "MK": 2, "NL": 2, "NO": 2, "PL": 2, "PT": 2,
    "RO": 2, "RU": 2, "SM": 2, "RS": 2, "SK": 2, "SI": 2, "ES": 2, "SE": 2, "CH": 2,
    "UA": 2, "GB": 2,

    # Asia (3)
    "AF": 3, "AM": 3, "AZ": 3, "BD": 3, "BT": 3, "BN": 3, "KH": 3, "CN": 3, "CX": 3,
    "CC": 3, "TL": 3, "IN": 3, "ID": 3, "IR": 3, "IQ": 3, "IL": 3, "JP": 3, "JO": 3,
    "KZ": 3, "KW": 3, "KG": 3, "LA": 3, "LB": 3, "MO": 3, "MY": 3, "MV": 3, "MN": 3,
    "MM": 3, "NP": 3, "OM": 3, "PK": 3, "PS": 3, "PH": 3, "QA": 3, "SA": 3, "SG": 3,
    "KR": 3, "LK": 3, "SY": 3, "TW": 3, "TJ": 3, "TH": 3, "TM": 3, "TR": 3, "AE": 3,
    "UZ": 3, "VN": 3, "YE": 3, "HK": 3, "KP": 3,

    # North America (4)
    "AG": 4, "AI": 4, "AW": 4, "BS": 4, "BB": 4, "BZ": 4, "BM": 4, "CA": 4, "KY": 4,
    "CR": 4, "CU": 4, "CW": 4, "DM": 4, "DO": 4, "SV": 4, "GL": 4, "GD": 4, "GP": 4,
    "GT": 4, "HT": 4, "HN": 4, "JM": 4, "MQ": 4, "MX": 4, "MS": 4, "NI": 4, "PA": 4,
    "PR": 4, "BL": 4, "KN": 4, "LC": 4, "PM": 4, "VC": 4, "TT": 4,
    "TC": 4, "US": 4, "VI": 4,

    # South America (5)
    "AR": 5, "BO": 5, "BR": 5, "CL": 5, "CO": 5, "EC": 5, "FK": 5, "GF": 5, "GY": 5,
    "PY": 5, "PE": 5, "SR": 5, "UY": 5, "VE": 5,

    # Oceania (6)
    "AS": 6, "AU": 6, "FJ": 6, "FM": 6, "GU": 6, "KI": 6, "MH": 6, "NR": 6, "NC": 6,
    "NZ": 6, "NU": 6, "NF": 6, "MP": 6, "PW": 6, "PG": 6, "PN": 6, "WS": 6, "SB": 6,
    "TK": 6, "TO": 6, "TV": 6, "VU": 6, "WF": 6,

    # Antarctica (7)
    "AQ": 7
}

""" 

continent_map = {
    "FK": 5,
    "GA": 1
    } 
 """


df_countries = pd.read_csv("country_data.csv")

def get_country_info(cc_cont: str) -> str:
    pure_cc = cc_cont[:2] # since the cc passed in is of form "NO.2", Norway, 2 = Europe
    match = df_countries.loc[df_countries['alpha-2'] == pure_cc]
    
    if not match.empty:
        return match.iloc[0]['country_name'], match.iloc[0]['population']
    return "CC. Not Found", "Pop. Not Found"


def load_country_summaries(data_folder, countries):
    """Fetch the summary of a country story

    Args:
        data_folder (directory, csv): data folder ..
        countries (dict): Country mapping

    Returns:
        List[Tuple]: list of tuples where first index is country code and second is the summary
    """
    summaries = []
    for cc in countries:
        path = f"data/{cc}/{cc}_summaries.csv"
        df = pd.read_csv(path)
        summary_text = df['Summaries'].iloc[:20] # holds many summaries
        summaries.append((cc, summary_text))
    return summaries

def load_country_stories(data_folder, countries, tokenizer, max_tokens=1024):
    stories = []
    for cc in countries:
        path = f"data/{cc}/{cc}_stories.csv"
        df = pd.read_csv(path)
        story_text = df['Story'].iloc[18]  # holds the full story 
        # Tokenize and truncate to context window
        tokens = tokenizer.encode(story_text, truncation=True, max_length=max_tokens)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

        stories.append((cc, truncated_text))
    return stories

def batch_summaries(summaries, batch_size=10):
    batches = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]
    return batches

def embed_batch(tokenizer, model, texts, layer_idx, do_centroid=True):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Get embeddings from the specified layer 
    hidden_states = outputs.hidden_states[layer_idx]  # [batch_size, seq_len, hidden_dim]
    # Take mean over the sequence length to get one embedding per summary (find cluster)
    if do_centroid:
        return hidden_states.mean(dim=1).cpu().numpy()  # [batch_size, hidden_dim]
    return hidden_states.cpu().numpy()

def embedding_std(embedding):
    """
    Compute the standard deviation of a single embedding vector.
    Accepts a NumPy array or a 1‑D PyTorch tensor.
    Returns a Python float.
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    return float(np.std(embedding))

def plot_country_std(country_std, top_k=25):
    sorted_std = sorted(country_std.items(),
                        key=lambda x: x[1],
                        reverse=False)
    """
    Bar‑plot the top‑k countries by embedding standard deviation.
    Call this *after* extract_centroids, e.g.:

        _, sorted_std = extract_centroids(...)
        plot_country_std(sorted_std, top_k=20)
    """
    # keep only the first k elements
    shown = sorted_std[:top_k]
    for cc, std in shown:
        print(cc, round(std,3))
    labels, values = zip(*shown)
""" 
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values, width=0.5)
    plt.xticks(range(len(values)), labels, rotation=45)
    plt.ylabel("σ of embedding")
    plt.ylim(top=2.3,bottom=2)
    plt.title(f"Top {top_k} countries by embedding spread")
    plt.tight_layout()
    plt.show()
 """

# reset CUDA mem
def unload_model(*objs, full: bool = False):
    """
    *objs: any tensors / nn.Modules / optimizers you want gone.
    full=True also runs IPC + stat resets.
    """
    # remove Python references
    objs = list(objs)
    for o in objs:
        del o

    #  garbage‑collect dangling tensors
    gc.collect()

    # free pytorch  CUDA caching allocator
    torch.cuda.empty_cache()

    if full:
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    
    _MODEL_CACHE.clear() # empty the model cache

# Load model
_MODEL_CACHE: dict[tuple] = {} 
def load_model(model_name: str):
    """Load device with model

    Args:
        model_name (str): Name of the original model
        tokenizer (any): Tokenizer

    Returns:
        tuple: Model, Device, Config(hidden states), Tokenizer
    """
    cached = _MODEL_CACHE.get(model_name)
    if cached:
        model, device, config, tokenizer = cached
        # double‑check the model really sits on cuda
        if next(model.parameters()).is_cuda:
            print(f"[cache‑hit] {model_name} already on {device}")
            model.eval()
            model_id = {
                "EleutherAI/gpt-neo-1.3B": "gptneo",
                "gpt2-xl":                "gpt2xl",
                "facebook/opt-1.3b":      "opt",
            }[model_name]
            return model, device, config, tokenizer, model_name, model_id
    
    cache_dir = "Z:\\LLMCache"
    try:
        if model_name == "EleutherAI/gpt-neo-1.3B":
            from transformers import (
            GPTNeoForCausalLM,
            GPTNeoConfig,
            GPT2Tokenizer
            )
            cache_dir = "Z:\\LLMCache"
            config = GPTNeoConfig.from_pretrained(model_name, 
                                                output_hidden_states=True)
            model = GPTNeoForCausalLM.from_pretrained(model_name, 
                                                    config=config, 
                                                    cache_dir=cache_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        if model_name == "gpt2-xl":
            from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
            config = GPT2Config.from_pretrained(model_name, 
                                                output_hidden_states=True)
            model = GPT2LMHeadModel.from_pretrained(model_name, 
                                                    config=config)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if model_name == "facebook/opt-1.3b":
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            config = AutoConfig.from_pretrained(model_name, 
                                                output_hidden_states=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                         config=config)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
    except:
        raise KeyError("Model name does not exist in function load_model()")

    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model device: ", device)
    
    model_map = {
        "EleutherAI/gpt-neo-1.3B": "gptneo",
        "gpt2-xl": "gpt2xl",
        "facebook/opt-1.3b": "opt",
    }
    model_id = model_map.get(model_name)
        
    return model, device, config, tokenizer, model_name, model_id

# Populate folders
def create_folders(model_name: str, text_input_desc: str):
    """Populates the folder for injection of data/images later

    Args:
        model_name (str): name of the model used
        text_input_desc (str): Description of the text used (ex.: 10 stories: meaning 10 paragraphs)

    Returns:
        tuple: folder_path, latent_graphs_path, layers_path
    """
    base_folder = "images"
    model_map = {
        "EleutherAI/gpt-neo-1.3B": "gptneo",
        "gpt2-xl": "gpt2xl",
        "facebook/opt-1.3b": "opt",
    }
    model_path = model_map.get(model_name, "temp_stash")
    folder_path        = f"{model_path}/{text_input_desc}"
    latent_graphs_path = f"{folder_path}/latent_graphs"
    layers_path        = f"{folder_path}/layers"
    full_folder_path    = os.path.join(base_folder, model_path, text_input_desc)
    latent_graphs_path  = os.path.join(full_folder_path, "latent_graphs")
    layers_path         = os.path.join(full_folder_path, "layers")
    os.makedirs(latent_graphs_path, exist_ok=True)
    os.makedirs(layers_path,        exist_ok=True)
    
    return folder_path, latent_graphs_path, layers_path, full_folder_path

# Group/Tokens
def grouping_and_coloring(tokenizer: any, model: any, input_texts_token_lengths: list):
    """Grouping and Coloring of input text paragraphs

    Args:
        tokenizer (any): Loaded Tokenizer
        model (any): Loaded Model
        input_texts_token_lengths (str): ..

    Returns:
        tuple: (group_ids, num_groups, point_colors, cmap, base_colors_rgb )
    """


    input_text = " ".join(input_texts)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
    all_hidden_states = outputs.hidden_states 

    seq_len = all_hidden_states[1].size()[1]
    token_indices = np.arange(seq_len)


    cumulative_lengths = np.cumsum(input_texts_token_lengths)
    seq_len = all_hidden_states[0].size(1)   
    token_indices = np.arange(seq_len) 
    group_ids = np.searchsorted(cumulative_lengths, token_indices, side="right")
    num_groups = group_ids.max() + 1 

    base_colors_hex = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#bcbd22",  # olive

        "#17becf",  # cyan
        "#7f7f7f",  # gray
        "#ffbb78",  # light orange
        "#98df8a",  # light green
    ]

    point_colors = [base_colors_hex[g] for g in group_ids]
    cmap = plt.cm.get_cmap("Set1", num_groups)
    base_colors_rgb = [mcolors.to_rgb(h) for h in base_colors_hex]

    return group_ids, num_groups, point_colors, cmap, base_colors_rgb, all_hidden_states


def create_gephi_graphs(layer_graphs, latent_graphs_path, model_path):
    number_of_layers = len(layer_graphs)
    for layer_number in range(number_of_layers):
        graph = layer_graphs[layer_number]
        for node in graph.nodes():
            pos = graph.nodes[node].get("pos")
            if isinstance(pos, (list, tuple)):
                graph.nodes[node]["pos"] = ",".join(str(x) for x in pos)

        nx.write_gexf(graph, f"{latent_graphs_path}/{model_path}_layer{layer_number}.gexf")

def load_centroids(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

### 

# Helpers (Coloring)
def get_group_fractions(group_ids):
    fractions = np.zeros_like(group_ids, dtype=float)
    unique_groups = np.unique(group_ids)
    
    for g in unique_groups:
        g_indices = np.where(group_ids == g)[0]
        if len(g_indices) <= 1:
            fractions[g_indices] = 0.0
            continue
        local_positions = np.arange(len(g_indices))
        local_fractions = local_positions / (len(g_indices) - 1)
        fractions[g_indices] = local_fractions
    return fractions

def blend_color_and_alpha(base_color, frac, 
                          start_alpha=0.2, end_alpha=1.0, 
                          light_factor=0.5):
    alpha = start_alpha + frac * (end_alpha - start_alpha)
    
    r0, g0, b0 = base_color
    white_blend = light_factor * (1.0 - frac)
    r = (1.0 * white_blend) + r0 * (1.0 - white_blend)
    g = (1.0 * white_blend) + g0 * (1.0 - white_blend)
    b = (1.0 * white_blend) + b0 * (1.0 - white_blend)
    
    return (r, g, b, alpha)

def create_final_colors(group_ids, base_colors_rgb,
                        start_alpha=0.2, end_alpha=1.0, 
                        light_factor=0.5):
    fractions = get_group_fractions(group_ids)
    final_colors = np.zeros((len(group_ids), 4), dtype=float)
    
    for i, g in enumerate(group_ids):
        base_color = base_colors_rgb[g % len(base_colors_rgb)]
        frac       = fractions[i]
        rgba       = blend_color_and_alpha(base_color, frac,
                                           start_alpha=start_alpha,
                                           end_alpha=end_alpha,
                                           light_factor=light_factor)
        final_colors[i] = rgba
    return final_colors


# Helpers (Clusters)
def compute_cluster_centroids(layer, group_ids):
    """
    Compute the mean hidden state (centroid) for each group.
    layer: 2D numpy array of shape [seq_len, hidden_size]
    group_ids: 1D array-like of length seq_len, each token's group id
    Returns a dictionary: {group_id: centroid vector}
    """
    unique_groups = np.unique(group_ids)
    centroids = {}
    for g in unique_groups:
        indices = np.where(group_ids == g)[0]
        centroids[g] = layer[indices].mean(axis=0)
    return centroids

def compute_umap_centroids(umap_result, group_ids):
    """
    Compute centroids in UMAP space by averaging the UMAP coordinates per group.
    """
    unique_groups = np.unique(group_ids)
    umap_centroids = {}
    for g in unique_groups:
        indices = np.where(group_ids == g)[0]
        umap_centroids[g] = umap_result[indices].mean(axis=0)
    return umap_centroids

def compute_distance_matrix(centroids):
    """
    Compute a pairwise Euclidean distance matrix between centroids.
    centroids: dictionary {group_id: centroid vector}
    Returns:
      groups: sorted list of group IDs
      dist_matrix: 2D numpy array of shape [num_groups, num_groups]
    """
    groups = sorted(centroids.keys())
    num_groups = len(groups)
    dist_matrix = np.zeros((num_groups, num_groups))
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist_matrix[i, j] = np.linalg.norm(centroids[g1] - centroids[g2])
    
    mean_rows = dist_matrix.mean(0) # mean of each rows -> [1,num_groups]
    avg_distance = mean_rows.mean(0) # mean of all columns -> single value
    #avg_distance = dist_matrix.mean()
    return groups, dist_matrix, avg_distance

def create_graph(matrix, group_ids, k=5):
    """
    Creates a graph using networkx kneighbors_graph function  
    
    Parameters:
      matrix: numpy array of shape [n_points, dim]
      group_ids: 1D array-like of length n_points, each node's group ID
      k: (currently unused) integer, originally desired number of edges per node
      n_candidates: number of nearest neighbor candidates to consider for each node

    Returns:
      G: a networkx Graph with:
         - Node label = integer index
         - 'pos' attribute for each node's (x, y) coordinates
         - 'group' attribute for each node's group ID
    """
    n_points = matrix.shape[0]

    adjacency = kneighbors_graph(
        matrix,
        n_neighbors=k,
        metric='euclidean',
        mode='distance',
        include_self=False
    ).tolil()

    G = nx.Graph()
    for i in range(n_points):
        G.add_node(i,
                   pos=(matrix[i, 0], matrix[i, 1]),
                   group=group_ids[i])

    for i in range(n_points):
        for j in adjacency.rows[i]:
            if not G.has_edge(i, j):
                G.add_edge(i, j)

    return G

def compute_modularity(G):
    """
    Compute the modularity of the graph based on the 'group' attribute assigned to each node.
    
    Parameters:
      G: a networkx Graph where each node has a 'group' attribute.
      
    Returns:
      mod: modularity value (float)
    """
    # dictionary grouping nodes by their group id
    communities_dict = {}
    for node in G.nodes():
        group = G.nodes[node]['group']
        if group not in communities_dict:
            communities_dict[group] = set()
        communities_dict[group].add(node)
    
    # list of communities
    communities = list(communities_dict.values())
    
    # modularity using networkx built-in function
    mod = nx.algorithms.community.quality.modularity(G, communities)
    return mod

def visualize_graph(G, folder_path, layer_nr, base_colors_rgb):
    """
    Visualize the graph using the node positions stored in the 'pos' attribute,
    and color the nodes by their group IDs using the provided base color mapping.
    
    Parameters:
      G: networkx Graph with node attributes:
         - 'pos': (x, y) position for each node
         - 'group': group ID for each node
      base_colors_rgb: list of RGB tuples (or hex strings) to map group IDs to colors.
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    node_groups = [G.nodes[node]['group'] for node in G.nodes()]
    node_colors = [base_colors_rgb[group % len(base_colors_rgb)] for group in node_groups]
    
    plt.figure(figsize=(8, 8))
    
    nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.9, width=1)
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=10,
        node_color=node_colors
    )
    
    unique_groups = sorted(set(node_groups))
    legend_handles = []
    for group in unique_groups:
        color = base_colors_rgb[group % len(base_colors_rgb)]
        patch = mpatches.Patch(color=color, label=f"Group {group}")
        legend_handles.append(patch)
    plt.legend(handles=legend_handles, title="Group ID", loc="best")
    
    plt.title("Graph from 2D UMAP")
    plt.axis("off")
    plt.tight_layout()
    
    save_path = f"{folder_path}/graph_layer{layer_nr}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_consecutive_centroid_distances(layer_features, group_ids, centroids):
    """
    """
    group_list = sorted(centroids.keys())
    
    distances = []
    pairs = []
    for i in range(len(group_list) - 1):
        g1 = group_list[i]
        g2 = group_list[i+1]
        d = np.linalg.norm(centroids[g1] - centroids[g2])
        distances.append(d)
        pairs.append((g1, g2))
    
    return pairs, distances

def measurement_plots(avg_dists_full, avg_dists_umap, layer_modularities, full_folder_path):
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

    x_values_full = list(range(len(avg_dists_full)))
    ax1.plot(x_values_full, avg_dists_full, marker='o', linestyle='-', linewidth=2, color='blue')
    ax1.set_title("Average Distances (Full Embedding Space)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Average Distance")
    ax1.set_xticks(x_values_full)
    ax1.set_xticklabels([str(x) for x in x_values_full])
    ax1.grid(True)

    x_values_umap = list(range(len(avg_dists_umap)))
    ax2.plot(x_values_umap, avg_dists_umap, marker='o', linestyle='-', linewidth=2, color='green')
    ax2.set_title("Average Distances (UMAP Space)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Average Distance")
    ax2.set_xticks(x_values_umap)
    ax2.set_xticklabels([str(x) for x in x_values_umap])
    ax2.grid(True)

    x_values_mod = list(range(len(layer_modularities)))
    ax3.plot(x_values_mod, layer_modularities, marker='o', linestyle='-', linewidth=2, color='purple')
    ax3.set_title("Layer Modularities")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Modularity")
    ax3.set_xticks(x_values_mod)
    ax3.set_xticklabels([str(x) for x in x_values_mod])
    ax3.grid(True)

    plt.tight_layout()
    save_path = f"{full_folder_path}/measurements.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()






# Load centroids
def load_centroids(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

# Create k-NN graph with positions and proper labels for Gephi
def create_centroid_knn_graph(centroids, k=5):
    country_codes = list(centroids.keys())
    
    
    
    embeddings = [centroids[cc] for cc in country_codes]

    adjacency = kneighbors_graph(
        embeddings,
        n_neighbors=k,
        mode='distance',
        metric='euclidean',
        include_self=False
    ).tolil()

    G = nx.Graph()
    for idx, cc in enumerate(country_codes):
        country_name, population_count = get_country_info(cc)
        G.add_node(
            idx,
            label=country_name,                  
            pos=embeddings[idx],
            country_code=cc,
            continents=continents[int(cc[-1])],
            population=population_count
        )

    rows, cols = adjacency.nonzero()
    for i, j in zip(rows, cols):
        if not G.has_edge(i, j):
            distance = adjacency[i, j]
            G.add_edge(i, j, weight=float(distance))

    return G

def format_positions_for_gephi(G):
    for node in G.nodes():
        pos = G.nodes[node]["pos"]
        if isinstance(pos, (list, tuple, np.ndarray)):
            G.nodes[node]["pos"] = ",".join(str(x) for x in pos)

def save_gephi_graph(G, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    format_positions_for_gephi(G)
    nx.write_gexf(G, save_path, version="1.2draft")

def centroids_to_gephi_with_labels(pickle_file, output_gexf_path, k=5, centroid=True):
    pkl_obj = load_centroids(pickle_file)
    if not centroid:
        graph = create_full_knn_graph(pkl_obj, k=k)
    else:
        graph = create_centroid_knn_graph(pkl_obj, k=k)
    save_gephi_graph(graph, output_gexf_path)
    print(f"Graph with visible labels saved at {output_gexf_path}")
    
    
    
    
    
    


def fetch_population(iso_list, year=2020, indicator="SP.POP.TOTL", chunk_size=30):
    """
    Fetch population data from the WDI database in small chunks.
    Returns a dict {iso2: population}.
    """
    pop_map = {}
    for i in range(0, len(iso_list), chunk_size):
        chunk = iso_list[i:i+chunk_size]
        try:
            # db="WDI" ensures we use the World Development Indicators endpoint
            df = wb.data.DataFrame(
                indicator,
                economy=chunk,
                time=year,
                db="WDI"
            )
            # df.loc[iso, year] → population
            for iso in chunk:
                try:
                    pop_map[iso] = df.loc[iso, year]
                except KeyError:
                    # country missing data for that year
                    continue
        except Exception as e:
            print(f"Warning: failed to fetch chunk {chunk[:5]}...: {e}")
    return pop_map


def plot_degree_vs_population(G):
    """
    For each node in G (with label 'XX.C' or 'XX'), fetch population via pypopulation,
    then plot node degree vs. population.

    G: networkx.Graph with node attribute 'label' (e.g. "FR.2" or "US.4")
    """
    labels  = nx.get_node_attributes(G, 'label')
    iso_map = {node: lbl.split('.')[0] for node, lbl in labels.items()}

    pop_map = {}
    for iso in set(iso_map.values()):
        try:
            pop_map[iso] = pypopulation.get_population(iso)
        except Exception as e:
            # skip any codes pypopulation doesn't know
            continue

    pops, degrees = [], []
    for node, iso in iso_map.items():
        pop = pop_map.get(iso)
        if not pop or pop <= 0:
            continue
        pops.append(pop)
        degrees.append(G.degree(node))

    plt.figure(figsize=(10, 6))
    plt.scatter(pops, degrees, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Population (log scale)")
    plt.ylabel("Node Degree (k-NN graph)")
    plt.title("Country Node Degree vs. Population")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
    
    

def rank_countries_by_degree_population(G):
    """
    Given a NetworkX graph G whose nodes have a 'label' attribute like "US.4",
    returns:
      - df_by_degree: DataFrame sorted by node degree desc
      - df_by_population: DataFrame sorted by population desc
      - corr: Pearson correlation coefficient between degree and population

    Columns of both DataFrames: ['country', 'degree', 'population']
    """
    labels = nx.get_node_attributes(G, 'label')
    iso_map = {node: lbl.split('.')[0] for node, lbl in labels.items()}

    pop_map = {}
    for iso in set(iso_map.values()):
        try:
            pop_map[iso] = pypopulation.get_population(iso)
        except Exception:
            pop_map[iso] = None

    records = []
    for node, iso in iso_map.items():
        pop = pop_map.get(iso)
        deg = G.degree(node)
        if pop is None or pop <= 0:
            continue
        records.append({
            'country': iso,
            'degree': deg,
            'population': pop
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid country-population pairs found.")

    corr = df['degree'].corr(df['population'], method='pearson')

    df_by_degree     = df.sort_values('degree', ascending=False).reset_index(drop=True)
    df_by_population = df.sort_values('population', ascending=False).reset_index(drop=True)

    return df_by_degree, df_by_population, corr


def add_country_names(df: pd.DataFrame, iso_column: str = "country") -> pd.DataFrame:
    """
    Given a DataFrame with an ISO-2 code column, returns a copy with an extra
    'country_name' column filled in via pycountry lookup. Unknown codes become None.
    """
    def lookup_name(iso):
        try:
            return pycountry.countries.get(alpha_2=iso).name
        except Exception:
            return None

    df = df.copy()
    df["country_name"] = df[iso_column].apply(lookup_name)
    return df


def export_rankings_to_csv(df_by_degree: pd.DataFrame,
                           df_by_population: pd.DataFrame,
                           degree_csv_path: str,
                           population_csv_path: str):
    """
    Save the country‐degree and country‐population rankings to CSV files,
    adding a 'country_name' column.

    Parameters
    ----------
    df_by_degree : pd.DataFrame
        DataFrame with columns ['country','degree','population'], sorted by 'degree' desc.
    df_by_population : pd.DataFrame
        Same DataFrame, but sorted by 'population' desc.
    degree_csv_path : str
        Path where df_by_degree is written (e.g. "ranking_by_degree.csv").
    population_csv_path : str
        Path where df_by_population is written (e.g. "ranking_by_population.csv").
    """
    # Enrich with full country names
    df_deg_named = add_country_names(df_by_degree, iso_column="country")
    df_pop_named = add_country_names(df_by_population, iso_column="country")

    # Write to CSV
    df_deg_named.to_csv(degree_csv_path, index=False)
    df_pop_named.to_csv(population_csv_path, index=False)
    print(f"– Wrote degree ranking to `{degree_csv_path}` with country names")
    print(f"– Wrote population ranking to `{population_csv_path}` with country names")
    
    
    


def plot_degree_vs_population_from_csv(csv_path):
    """
    Load a CSV with columns ['country','country_name','degree','population']
    and plot node degree vs. population with:
      - left: log-scaled x-axis with trend line and Pearson r
      - right: linear x-axis with trend line and Pearson r
    """
    # Load data
    df = pd.read_csv(csv_path)
    pops = df['population']
    degrees = df['degree']
    
    # pearson corr
    corr = pops.corr(degrees)
    
    # Prepare trend line for log-scale plot (degree vs log10(pop))
    log_pops = np.log10(pops)
    m1, b1 = np.polyfit(log_pops, degrees, 1)
    log_x = np.linspace(log_pops.min(), log_pops.max(), 100)
    log_y = m1 * log_x + b1
    
    # Prepare trend line for linear plot (degree vs pop)
    m2, b2 = np.polyfit(pops, degrees, 1)
    lin_x = np.linspace(pops.min(), pops.max(), 100)
    lin_y = m2 * lin_x + b2

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log-scale plot
    ax1.scatter(pops, degrees, alpha=0.7)
    #ax1.plot(10**log_x, log_y)
    ax1.set_xscale('log')
    ax1.set_xlabel('Population (log scale)')
    ax1.set_ylabel('Node Degree')
    ax1.set_title('Degree vs Population (log scale)')
    # Linear-scale plot
    ax2.scatter(pops, degrees, alpha=0.7)
    #ax2.plot(lin_x, lin_y)
    ax2.set_xlabel('Population')
    ax2.set_ylabel('Node Degree')
    ax2.set_title('Degree vs Population (linear scale)')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
def create_full_knn_graph(embeddings_dict, k=5):
    """
    Build a k-NN graph from per-country embedding batches.
    
    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        Mapping country_code → array of shape [n_points, dim].
        { "US": np.array([[...],...]), "FR": np.array([[...],...]), ... }
    k : int
        Number of nearest neighbors for each node.
    
    Returns
    -------
    networkx.Graph
        - Nodes are numbered 0..(total_points−1)
        - Node attribute 'label' = country_code
        - Node attribute 'pos'   = embedding vector (tuple)
        - Edge attribute 'weight' = distance between nodes
    """
    country_codes = []
    all_embs = []
    for cc, arr in embeddings_dict.items():
        # ensure 2D array
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for {cc}, got shape {arr.shape}")
        for vec in arr:
            country_codes.append(cc)
            all_embs.append(vec)
    all_embs = np.vstack(all_embs)  # shape [N_total, dim]

    adjacency = kneighbors_graph(
        all_embs,
        n_neighbors=k,
        metric='euclidean',
        mode='distance',
        include_self=False
    ).tolil()

    G = nx.Graph()
    for idx, (cc, vec) in enumerate(zip(country_codes, all_embs)):        
        country_name, population_count = get_country_info(cc)
        G.add_node(
            idx,
            label=country_name,                  
            pos=tuple(vec),
            country_code=cc,
            continents=continents[int(cc[-1])],
            population=population_count
        )

    # 4) Add edges with distance weight
    rows, cols = adjacency.nonzero()
    for i, j in zip(rows, cols):
        if not G.has_edge(i, j):
            G.add_edge(i, j, weight=float(adjacency[i, j]))

    return G


def country_distances(G):
    
    return nx.resistance_distance(G)

def get_furthest_countries(G, source, top_n=None, weight="weight"):
    """
    Compute resistance-distance from `source` to every other node in G.
    Returns a list of (node, distance) sorted descending by distance.
    
    Parameters
    ----------
    G : networkx.Graph
      Must contain `source` as a node, edges may have a 'weight' attribute.
    source : node‐label
      The node from which distances are measured.
    top_n : int or None
      If int, only return the top_n furthest nodes; otherwise return all.
    weight : str or None
      The edge-data key corresponding to distance (None for unweighted).
    """
    dists = {}
    for node in G:
        if node == source:
            continue
        try:
            d = nx.resistance_distance(G, source, node, weight=weight)
        except Exception:
            d = float('inf')
        dists[node] = d

    sorted_d = sorted(dists.items(), key=lambda x: x[1], reverse=True)
    return sorted_d[:top_n] if top_n is not None else sorted_d



if __name__ == "__main__":
    unload_model(_MODEL_CACHE)