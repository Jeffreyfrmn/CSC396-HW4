'''
CSc 396 - Homework 4
Jeffrey Freeman
'''
import os
from tqdm import tqdm
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaModel
import torch

# --------------------------------------------------------------------------- #
# - PROBLEM 1 --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Read dataset
dataset_path = os.path.join(os.getcwd(), "dataset.txt")
try:
    with open(dataset_path, "r", errors='ignore', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"File not found: {dataset_path}")
    exit()

# Split text into manageable chunks
max_token_length = 512  # Transformer token limit
text_chunks = [text[i:i+max_token_length] for i in range(0, len(text), max_token_length)]

# Dictionary to store embeddings for each token
token_embeddings = defaultdict(list)

# Process each chunk
for chunk in tqdm(text_chunks, desc="Processing Chunks"):
    encoded_input = tokenizer(chunk, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)

    # Get embeddings and tokens
    embeddings = output.last_hidden_state  # [batch_size, seq_length, hidden_size]
    token_ids = encoded_input["input_ids"]  # [batch_size, seq_length]

    # Map embeddings to tokens
    for i, token_id in enumerate(token_ids[0]):  # Batch size is 1 here
        token_embeddings[token_id.item()].append(embeddings[0, i].cpu())

# Compute average embeddings for each token
avg_token_embeddings = {
    token_id: torch.mean(torch.stack(embeds), dim=0)
    for token_id, embeds in token_embeddings.items()
}

# Map token IDs back to strings (optional)
vocab_string_map = {token_id: tokenizer.decode([token_id]) for token_id in avg_token_embeddings.keys()}

print(f"Computed average embeddings for {len(avg_token_embeddings)} tokens.")
torch.save(avg_token_embeddings, "avg_token_embeddings.pt")