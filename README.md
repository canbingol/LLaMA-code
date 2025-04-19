# 🦙 LLaMA-like Transformer Implementation

This repository contains a minimalist PyTorch implementation of a LLaMA-style transformer model from scratch. It is structured for educational purposes and supports training with HuggingFace tokenizers.

---

## Model Architecture

The transformer is built with the following core components:

- **Token Embedding Layer**
- **Rotary Positional Embeddings (RoPE)**
- **Multi-Head Self-Attention (`SelfAttention`)**
  - Includes **KV-Cache** for inference-time efficiency
  - Custom `repeat_kv()` for `n_kv_heads < n_heads` support
- **Feed-Forward Network (FFN)**
- **RMS Layer Normalization**
- **Output Projection Layer**

Each Transformer block follows this structure:

`x → RMSNorm → SelfAttention → residual → RMSNorm → FFN → residual`

---

##  Dataset & Dataloaders

Data is loaded from a raw `.txt` file and tokenized using a HuggingFace tokenizer. The dataset pipeline includes:

- Sentence tokenization
- Chunking into fixed `max_seq_len`
- Input/target pair generation with stride
- Padding + truncation
- `train_loader` and `val_loader` built using `torch.utils.data.DataLoader`

---

##  Training Loop

Training logic includes:

- `cross_entropy` loss
- `Adam` optimizer
- Epoch-wise tracking of training & validation loss
- `model.train()` / `model.eval()` to toggle KV-cache usage
- Visualization of loss using `matplotlib`
- Model checkpointing:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    ...
}, "checkpoint.pt")
