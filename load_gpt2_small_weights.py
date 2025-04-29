import pickle
import urllib.request
import torch
from gpt_download import download_and_load_gpt2
import numpy as np
from model import GPTModel
from config import get_config
from utilities import generate, text_to_ids, ids_to_text
import tiktoken



def retrive_downloader_code(url = 
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"):

    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)



def download_and_save():
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    with open("settings.pkl", "wb") as f:
        pickle.dump(settings, f)
    with open("params.pkl", "wb") as f:
        pickle.dump(params, f)
    return settings, params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))



def load_weights_into_gpt(gpt, params):
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.blocks[b].attention.w_q.weight = assign(gpt.blocks[b].attention.w_q.weight, q_w.T)
        gpt.blocks[b].attention.w_k.weight = assign(gpt.blocks[b].attention.w_k.weight, k_w.T)
        gpt.blocks[b].attention.w_v.weight = assign(gpt.blocks[b].attention.w_v.weight, v_w.T)
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.blocks[b].attention.w_q.bias = assign(gpt.blocks[b].attention.w_q.bias, q_b)
        gpt.blocks[b].attention.w_k.bias = assign(gpt.blocks[b].attention.w_k.bias, k_b)
        gpt.blocks[b].attention.w_v.bias = assign(gpt.blocks[b].attention.w_v.bias, v_b)
        gpt.blocks[b].attention.w_o.weight = assign(gpt.blocks[b].attention.w_o.weight,params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.blocks[b].attention.w_o.bias = assign(gpt.blocks[b].attention.w_o.bias,params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.blocks[b].ffn.layers[0].weight = assign(gpt.blocks[b].ffn.layers[0].weight,params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.blocks[b].ffn.layers[0].bias = assign(gpt.blocks[b].ffn.layers[0].bias,params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.blocks[b].ffn.layers[2].weight = assign(gpt.blocks[b].ffn.layers[2].weight,params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.blocks[b].ffn.layers[2].bias = assign(gpt.blocks[b].ffn.layers[2].bias,params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.blocks[b].norm1.scale = assign(gpt.blocks[b].norm1.scale,params["blocks"][b]["ln_1"]["g"])
        gpt.blocks[b].norm1.shift = assign(gpt.blocks[b].norm1.shift,params["blocks"][b]["ln_1"]["b"])
        gpt.blocks[b].norm2.scale = assign(gpt.blocks[b].norm2.scale,params["blocks"][b]["ln_2"]["g"])
        gpt.blocks[b].norm2.shift = assign(gpt.blocks[b].norm2.shift,params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def get_new_config(model_name = "gpt2-small (124M)"):
    GPT_CONFIG_124M = get_config()
    model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True})
    return NEW_CONFIG

if __name__ == "__main__":
    #retrive_downloader_code()
    settings, params = download_and_save()
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NEW_CONFIG = get_new_config()
    gpt = GPTModel(NEW_CONFIG)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    token_ids = generate(
    model=gpt,
    idx=text_to_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
    )
    print("Output text:\n", ids_to_text(token_ids, tokenizer))

