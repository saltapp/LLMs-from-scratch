# %%
"""
Complete LLM Classification Pipeline from Chapter 5 of 'Build a Large Language Model From Scratch'
Extended for numerical sequence classification

This script implements the full pipeline for training a classification LLM from scratch including:
- GPT-based model with classification head
- Numerical data loading and preprocessing
- Binary classification training
- Model evaluation and prediction
- Model saving/loading functionality
"""

import os
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time


# GPT Model and related components from previous chapters
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # For classification, we add a classifier head that takes the pooled representation
        self.classifier_head = nn.Linear(cfg["emb_dim"], cfg["classifier_num"])  # Number of classes based on classifier_num config

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Use the representation of the last token for classification
        # or could use mean pooling across all tokens
        last_token_repr = x[:, -1, :]  # Take the last token representation
        logits = self.classifier_head(last_token_repr)  # Shape: (batch_size, num_classes)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# Text processing utilities from chapter 5
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


# Loss calculation utilities for classification
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # For classification, target_batch should be class indices (not next token prediction)
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# Enhanced text generation with temperature and top-k sampling from chapter 5
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

        # Calculate accuracy
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss, val_acc


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, total_predictions = 0, 0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)  # Shape: (batch_size, num_classes)
            predicted_labels = torch.argmax(logits, dim=1)  # Shape: (batch_size,)
            correct_predictions += (predicted_labels == target_batch).sum().item()
            total_predictions += target_batch.size(0)
        else:
            break
    model.train()
    return correct_predictions / total_predictions


def print_model_predictions(model, val_loader, device, num_examples=3):
    """Print model predictions on validation examples for classification"""
    model.eval()
    print("Model predictions on validation examples:")
    count = 0

    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            if count >= num_examples:
                break

            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)  # Shape: (batch_size,)

            batch_size = input_batch.size(0)
            for i in range(min(batch_size, num_examples - count)):
                true_label = target_batch[i].item()
                pred_label = predicted_labels[i].item()
                confidence = probabilities[i][pred_label].item()

                print(f"  Example {count+1}: True label={true_label}, Predicted={pred_label}, Confidence={confidence:.3f}")
                count += 1

                if count >= num_examples:
                    break

    model.train()


def generate_numerical_sequence(model, start_sequence, max_new_tokens, device):
    """Generate a numerical sequence using the trained model"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]

    # Convert start sequence to tensor
    encoded = torch.tensor([start_sequence], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_tokens = generate(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            top_k=50,
            temperature=0.7
        )

    # Convert back to coordinates if needed
    generated_sequence = generated_tokens[0].tolist()
    model.train()
    return generated_sequence


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses, tokens seen, and accuracies
    train_losses, val_losses, track_tokens_seen = [], [], []
    val_accuracies = []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss, val_acc = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Val acc {val_acc:.3f}")

        # Print model predictions after each epoch
        print_model_predictions(model, val_loader, device, num_examples=3)

    return train_losses, val_losses, track_tokens_seen, val_accuracies


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


def get_device():
    """Get the appropriate device (GPU, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")
    return device


def load_numerical_dataset():
    """Load the numerical dataset from dataset.txt file"""
    file_path = "dataset/dataset.txt"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    print(f"Loading numerical dataset from {file_path}")

    sequences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                # Replace 'nan' with None first, then replace None with a valid number
                safe_line = line.replace('nan', 'float("nan")')

                # Parse each line as a Python list structure
                try:
                    # Use eval instead of literal_eval since literal_eval doesn't support float("nan")
                    parsed_line = eval(safe_line)
                    label = parsed_line[0]
                    coords = parsed_line[1]
                    # Check if any coordinates are NaN and skip if so
                    has_nan = any(x != x or y != y for x, y in coords)  # x != x is True only if x is NaN
                    if not has_nan:
                        # Convert to a format that includes both the label and the sequence
                        labels.append(label)

                        # Convert coordinates to tokens
                        token_seq = []
                        for x, y in coords:
                            # Quantize x and y coordinates to discrete bins separately
                            # Using smaller number of bins since data is in limited range [-100, 100]
                            x_bin = min(int((x + 100) / 200 * 200), 199)
                            y_bin = min(int((y + 100) / 200 * 200), 199)

                            # Add both x and y as separate tokens in sequence
                            # This prevents the vocab size from exploding
                            token_seq.extend([x_bin, y_bin])
                        sequences.append(token_seq)
                except Exception as e:
                    print(f"Error parsing line: {line}, Error: {e}")

    print(f"Loaded {len(sequences)} sequences from dataset (NaN entries filtered out)")
    print(f"Label distribution: {dict(zip(*torch.unique(torch.tensor(labels), return_counts=True)))}")
    return sequences, labels


def normalize_coordinates(sequences, min_val=-100, max_val=100):
    """Normalize coordinate values to a range suitable for tokenization"""
    # Find min and max across all coordinates
    all_coords = []
    for label, seq in sequences:
        for x, y in seq:
            all_coords.extend([x, y])

    data_min = min(all_coords)
    data_max = max(all_coords)

    normalized_sequences = []
    for label, seq in sequences:
        normalized_seq = []
        for x, y in seq:
            # Normalize to [0, 1] then scale to desired range
            norm_x = (x - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
            norm_y = (y - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
            normalized_seq.append((norm_x, norm_y))
        normalized_sequences.append((label, normalized_seq))

    return normalized_sequences


def numerical_sequence_to_tokens(sequences, num_bins=200):  # Reduced number of bins
    """Convert normalized numerical sequences to token IDs"""
    tokenized_sequences = []

    for label, seq in sequences:
        token_seq = []
        for x, y in seq:
            # Quantize x and y coordinates to discrete bins separately
            # Using smaller number of bins since data is in limited range
            x_bin = min(int((x + 100) / 200 * num_bins), num_bins - 1)
            y_bin = min(int((y + 100) / 200 * num_bins), num_bins - 1)

            # Add both x and y as separate tokens in sequence
            # This prevents the vocab size from exploding
            token_seq.extend([x_bin, y_bin])
        tokenized_sequences.append(token_seq)

    return tokenized_sequences


class NumericalSequenceClassificationDataset(Dataset):
    """Dataset class for numerical sequence classification data"""
    def __init__(self, sequences, labels, context_length):
        self.sequences = sequences
        self.labels = labels
        self.context_length = context_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Ensure sequence is the right length
        if len(seq) > self.context_length:
            # Truncate if too long
            seq = seq[:self.context_length]
        elif len(seq) < self.context_length:
            # Pad if too short
            padding_length = self.context_length - len(seq)
            seq = seq + [0] * padding_length  # Pad with zeros

        input_ids = torch.tensor(seq, dtype=torch.long)
        target = torch.tensor(label, dtype=torch.long)  # The class label (0 or 1)

        return input_ids, target


def create_classification_dataloader(sequences, labels, batch_size=4, context_length=64, shuffle=True, drop_last=False, num_workers=0):
    """Create dataloader for numerical sequence classification data"""
    dataset = NumericalSequenceClassificationDataset(sequences, labels, context_length)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader

def predict_class(model, input_sequence, device):
    """Predict the class for a numerical sequence in multi-class classification"""
    model.eval()
    with torch.no_grad():
        # Convert input to tensor and add batch dimension
        input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(device)
        logits = model(input_tensor)  # Shape: (1, num_classes)
        probabilities = torch.softmax(logits, dim=1)  # Shape: (1, num_classes)
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Predicted class index
        confidence = probabilities[0][predicted_class].item()  # Confidence in prediction

    return predicted_class, confidence, probabilities[0].tolist()


def plot_accuracy(epochs_tensor, val_accuracies):
    plt.figure(figsize=(5, 3))
    plt.plot(epochs_tensor, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Validation Accuracy Over Time")
    plt.grid(True, alpha=0.3)
    plt.savefig("classification-accuracy-plot.pdf")
    plt.show()


def load_model(model_path, config):
    """Utility function to load a saved model"""
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# %%
print("Starting LLM classification training pipeline with numerical dataset...")

# Configuration for the GPT model adapted for classification
# With separate tokens for x and y, vocab size is just the number of bins + small buffer
num_bins = 200  # Number of bins for quantization
vocab_size = num_bins + 50  # Add buffer to vocab size

GPT_CONFIG_CLASSIFICATION = {
    "vocab_size": vocab_size,        # More reasonable vocab size
    "context_length": 64,            # Increased for longer coordinate sequences (up to 20 coordinate pairs = 40 tokens)
    "emb_dim": 128,                  # Reduced embedding dimension for numerical data
    "classifier_num": 5,             # Number of classes to classify
    "n_heads": 8,                    # Reduced number of attention heads
    "n_layers": 4,                   # Reduced number of layers for faster training
    "drop_rate": 0.1,                # Dropout rate
    "qkv_bias": False,               # Query-key-value bias
    "balance_classes": False          # Whether to balance classes in dataset
}

print(f"Model config: {GPT_CONFIG_CLASSIFICATION}")

# Get device
device = get_device()


# %%
# Load and prepare the numerical training data
print("Loading numerical training data...")
sequences, labels = load_numerical_dataset()

print(f"Total sequences: {len(sequences)}")
print(f"Sample sequence length: {len(sequences[0]) if sequences else 0}")
print(f"Label distribution: {dict(zip(*torch.unique(torch.tensor(labels), return_counts=True))) if labels else 'N/A'}")

# %%
# Split the dataset into train and validation sets based on classifier_num parameter
import random

# Combine sequences and labels to maintain alignment during processing
data_pairs = list(zip(sequences, labels))

# Filter out sequences that have NaN values or length < 20
filtered_pairs = []
for seq, label in data_pairs:
    # Check if sequence length is less than 20
    if len(seq) < 20:
        continue  # Skip this sequence

    # Check for NaN values in the sequence
    has_nan = False
    for token in seq:
        if token != token:  # This checks for NaN (nan != nan is True)
            has_nan = True
            break

    if not has_nan:
        filtered_pairs.append((seq, label))

print(f"Total sequences after filtering: {len(filtered_pairs)}")

# Identify all unique labels in the dataset
unique_labels = sorted(list(set(label for _, label in filtered_pairs)))
print(f"Unique labels in dataset: {unique_labels}")

# Shuffle the filtered pairs
random.shuffle(filtered_pairs)

# Separate back into sequences and labels
all_sequences = [pair[0] for pair in filtered_pairs]
all_labels = [pair[1] for pair in filtered_pairs]

# Group data by label for potential balancing
label_data_groups = {}
for label in unique_labels:
    label_data_groups[label] = [(seq, label_val) for seq, label_val in filtered_pairs if label_val == label]

# Print original counts
for label in unique_labels:
    print(f"Label {label} count: {len(label_data_groups[label])}")

# Conditionally balance the labels based on config parameter
if GPT_CONFIG_CLASSIFICATION.get("balance_classes", True):
    # Balance the labels by taking the minimum count across all available labels
    min_count = min([len(label_data_groups[label]) for label in unique_labels]) if unique_labels else 0

    # Trim each label group to the minimum count to ensure balanced dataset
    balanced_data_groups = {}
    for label in unique_labels:
        balanced_data_groups[label] = label_data_groups[label][:min_count]

    # Print balanced counts
    for label in unique_labels:
        print(f"After balancing - Label {label} count: {len(balanced_data_groups[label])}")

    # Shuffle each balanced label group separately
    for label in unique_labels:
        random.shuffle(balanced_data_groups[label])

    # Calculate split points for balanced distribution in train/validation
    train_ratio = 0.90
    num_classes = len(unique_labels)
    total_balanced = min_count * num_classes  # Total samples after balancing
    train_count_per_label = int(train_ratio * min_count)

    # Split each balanced label group
    train_data = []
    val_data = []
    for label in unique_labels:
        label_train_data = balanced_data_groups[label][:train_count_per_label]
        label_val_data = balanced_data_groups[label][train_count_per_label:]
        train_data.extend(label_train_data)
        val_data.extend(label_val_data)

    print("Dataset balancing enabled: All classes have equal representation")
else:
    # Use original data distribution without balancing
    print("Dataset balancing disabled: Using original class distribution")

    # Calculate split points based on original distribution
    train_ratio = 0.90
    train_data = []
    val_data = []
    for label in unique_labels:
        label_data = label_data_groups[label]
        random.shuffle(label_data)  # Shuffle the original data for this label
        label_train_count = int(train_ratio * len(label_data))
        label_train_data = label_data[:label_train_count]
        label_val_data = label_data[label_train_count:]
        train_data.extend(label_train_data)
        val_data.extend(label_val_data)

    print("Using original dataset distribution without balancing")

# Shuffle the train and validation sets to mix the labels
random.shuffle(train_data)
random.shuffle(val_data)

# Separate sequences and labels for train and validation
train_sequences = [pair[0] for pair in train_data]
train_labels = [pair[1] for pair in train_data]
val_sequences = [pair[0] for pair in val_data]
val_labels = [pair[1] for pair in val_data]

print(f"Training set: {len(train_sequences)} sequences, Label distribution: {dict(zip(*torch.unique(torch.tensor(train_labels), return_counts=True)))}")
print(f"Validation set: {len(val_sequences)} sequences, Label distribution: {dict(zip(*torch.unique(torch.tensor(val_labels), return_counts=True)))}")

# Verify that the number of classes matches the model configuration
actual_num_classes = len(torch.unique(torch.tensor(train_labels + val_labels)))
expected_num_classes = GPT_CONFIG_CLASSIFICATION["classifier_num"]
if actual_num_classes != expected_num_classes:
    print(f"WARNING: Dataset has {actual_num_classes} classes, but model is configured for {expected_num_classes} classes.")
    print(f"Consider updating GPT_CONFIG_CLASSIFICATION['classifier_num'] to {actual_num_classes}")
    # Update the config to match the actual number of classes in the dataset
    GPT_CONFIG_CLASSIFICATION["classifier_num"] = actual_num_classes
    print(f"Updated classifier_num to {actual_num_classes}")

# %%
# Create data loaders for numerical sequences with labels
torch.manual_seed(123)

train_loader = create_classification_dataloader(
    train_sequences,
    train_labels,
    batch_size=4,
    context_length=GPT_CONFIG_CLASSIFICATION["context_length"],
    drop_last=False,  # Don't drop last batch as it may be small but still valuable for classification
    shuffle=True,
    num_workers=0
)

val_loader = create_classification_dataloader(
    val_sequences,
    val_labels,
    batch_size=4,
    context_length=GPT_CONFIG_CLASSIFICATION["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# %%
# Initialize the model with classification config
print("Initializing model...")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_CLASSIFICATION)
model.to(device)

# %%
# Print initial loss before training
print("Calculating initial loss...")
with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"Initial Training loss: {train_loss}")
print(f"Initial Validation loss: {val_loss}")


# %%
# Calculate initial accuracy
initial_acc = calc_accuracy_loader(val_loader, model, device)
print(f"Initial Validation accuracy: {initial_acc}")

# %%
# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# Start training
print("Starting training...")
start_time = time.time()

num_epochs = 5  # Reduced epochs for initial testing
train_losses, val_losses, tokens_seen, val_accuracies = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=30, eval_iter=3
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# %%
# Plot losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Plot accuracy
plot_accuracy(epochs_tensor, val_accuracies)

print("\nLLM classification training pipeline with numerical data completed!")

# %%
# Save the model
print("\nSaving trained model...")
torch.save(model.state_dict(), "trained_numerical_gpt_classification_model.pth")
print("Model saved as 'trained_numerical_gpt_classification_model.pth'")

# %%
# Demonstrate classification predictions
print("\nTesting classification predictions:")
model.eval()
for i in range(min(5, len(val_sequences))):
    seq = val_sequences[i]
    true_label = val_labels[i]
    pred_class, confidence, all_probs = predict_class(model, seq, device)
    print(f"Sequence {i+1}: True label={true_label}, Predicted={pred_class}, Confidence={confidence:.3f}")

    # Print all class probabilities dynamically
    prob_str = ", ".join([f"Class {j}: {prob:.3f}" for j, prob in enumerate(all_probs)])
    print(f"  All probabilities: {prob_str}")



# %%
