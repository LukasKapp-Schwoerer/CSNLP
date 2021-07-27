import numpy as np
from wiktionary.wiktionary import Wiktionary
from transformers import RobertaTokenizer, RobertaModel
import torch
from tqdm import tqdm

def cpnet_to_wiktionary_defs(vocab_path, output_path):
    """
        loads a text file of concepts at location vocab_path
        stores the corresponding wiktionary definitions at ouput_path
        if a concept is not found in wiktionary, we just keep the concept name instead
    """
    with open(vocab_path, "r") as entities:
        ent_arr = np.array(entities.read().splitlines())
        wiktionary = Wiktionary()

        for i in tqdm(range(ent_arr.size), desc='cpnet_to_wiktionary_defs'):         
            try:
                # print("retrieving concept net entity", ent_arr[i], "in wiktionary", flush=True)
                ent_arr[i] = wiktionary[ent_arr[i].replace("_", " ")]
            except KeyError:
                # print("concept net entity not found", ent_arr[i], "not found", flush=True)
                pass
        
        np.save(output_path, ent_arr)



def embed_wiktionary_defs(definition_path, output_path):
    """
        Embeds the wiktionary defininitions through roberta-base (for now)
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    lang_model = RobertaModel.from_pretrained('roberta-base')
    emb_dim = 768

    if torch.cuda.is_available():
        lang_model.cuda()
    
    lang_model.eval()
    with torch.no_grad():
        all_defs = np.load(definition_path)
    
        cp_net_embeddings = torch.zeros((all_defs.size, emb_dim)) # size (num_concepts, 768)
        
        for i in tqdm(range(all_defs.size), desc='embed_wiktinoary_defs'):
            tokens = tokenizer.encode(all_defs[i], add_special_tokens=True, return_tensors="pt")
            attention_mask = torch.ones((1,tokens.size(1))) # TODO @team: is this reasonable to encode like this?
            
            if torch.cuda.is_available():
                tokens = tokens.cuda()
                attention_mask = attention_mask.cuda()

            output = lang_model(tokens, attention_mask) 

            cp_net_embeddings[i] = output[1][0] # this should be the pooled representation over the sentence

        cp_net_embeddings = cp_net_embeddings.cpu()
        cp_net_embeddings = cp_net_embeddings.numpy()
        np.save(output_path, cp_net_embeddings)

        print(output_path, "now stores embeddings of shape", cp_net_embeddings.shape)


# Testing code, TODO remove
#cpnet_to_wiktionary_defs("data/cpnet/concept.txt", "data/cpnet/concept_defs.npy")
#embed_wiktionary_defs("data/cpnet/concept_defs.npy", "data/cpnet/concept_emb.npy")
