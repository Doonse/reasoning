from main import *
from helpers import *
import pickle
import tqdm


""" model_names = ["EleutherAI/gpt-neo-1.3B",  "gpt2-xl", "facebook/opt-1.3b"]
embs = EmbeddingVisualizer(model_names[0], text_input_desc="test")
embs.run()
 """



class CountryEmbeddingExtractor:
    def __init__(self, model_name, data_folder, layer_idx, do_centroid):
        self.model, self.device, _, self.tokenizer, _, _ = load_model(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.do_centroid = do_centroid

        self.data_folder = data_folder
        self.layer_idx = layer_idx

        
    def extract_centroids(self, countries):
        #summaries = load_country_stories(self.data_folder, countries, self.tokenizer)
        summaries = load_country_summaries(self.data_folder, countries)
        batches = batch_summaries(summaries, batch_size=1) 

        country_centroids = {}
        country_std = {}
        for batch in tqdm.tqdm(batches):
            # batch = (cc, pandas data.frame of paragraphs)
            country_code, texts = zip(*batch)
            texts = " ".join(texts[0])
            
            
            # text_embds can be one centroid vector or all embeding vectors 
            # text_embds if not do_centroid shape is (1, 1024, 1600)
            text_embds = embed_batch(self.tokenizer, 
                                        self.model, 
                                        texts, 
                                        self.layer_idx, 
                                        self.do_centroid)
            
            print(text_embds.shape)
            
            for cc, embedding in zip(country_code, text_embds):
                std_val = embedding_std(embedding)             
                continent_id = continent_map.get(cc, 0)  # fallback to 0 if not found
                
                # Make continent_id to population mauybe??
                key = f"{cc}.{continent_id}"
                country_centroids[key] = embedding
                country_std[key] = std_val 

        return country_centroids, country_std
    
"""     def extract_centroids(self, countries):
        #summaries = load_country_summaries(self.data_folder, countries)
        summaries = load_country_stories(self.data_folder, countries, self.tokenizer)
        batches = batch_summaries(summaries, batch_size=1)
        
        country_centroids = {}
        for batch in batches:
            country_codes, texts = zip(*batch)
            embeddings = embed_batch(self.tokenizer, self.model, texts, self.layer_idx)
            for cc, embedding in zip(country_codes, embeddings):
                country_centroids[cc] = embedding

        return country_centroids 
"""


#model_name = "EleutherAI/gpt-neo-1.3B"
model_name = "gpt2-xl"
data_folder = "data" 
target_layer = 5
local, continent_nr = True, 2
do_centroid = True


if __name__ == "__main__":
    extractor = CountryEmbeddingExtractor(
        model_name=model_name,
        data_folder=data_folder, 
        layer_idx=target_layer,
        do_centroid=do_centroid
    )
    
    if local:
        df_pop = pd.read_csv("countries_by_population.csv").set_index("country")
        local_continent_map = {}
        for iso, cont in continent_map.items():
            if cont == continent_nr and iso in df_pop.index:
                local_continent_map[iso] = int(df_pop.loc[iso, "population"])
    
        
        centroids, country_std = extractor.extract_centroids(local_continent_map)
        
        with open(f"pkls/{continents[continent_nr]}_L{target_layer}_gptneo_EU.pkl", "wb") as f:
            pickle.dump(centroids, f)
            
            
    else:
        centroids, country_std = extractor.extract_centroids(continent_map)
        #with open(f"pkls/L{target_layer}_{model_name}_FK_GA.pkl", "wb") as f:
        with open(f"pkls/L{target_layer}_gptneo_EU.pkl", "wb") as f:
            pickle.dump(centroids, f)

