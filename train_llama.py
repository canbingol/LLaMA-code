import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import ModelArgs, LLaMA
from lm_dataset import train_loader, val_loader
from warnings import filterwarnings
filterwarnings('ignore')

args = ModelArgs()
EPOCH = 2
LR = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')


model = LLaMA(ModelArgs)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"{'Total parameters:':<20} {total_params:,}")
print(f"{'Trainable parameters:':<20} {trainable_params:,}")
print(f"{'Non-trainable parameters:':<20} {total_params - trainable_params:,}")

print(f'len train_loader: {len(train_loader)}')
print(f'len val_loader: {len(val_loader)}')


optimizer = torch.optim.Adam(model.parameters(),lr=LR)

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

train_losses = []
val_losses = []

for epoch in range(EPOCH):
    args.train=True

    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")

    for input_batch, target_batch in progress_bar:

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        logits = model(input_batch, start_pos=0)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), target_batch.view(-1))
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        avg_loss = total_train_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(train_loss=avg_loss)

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    args.train=False
    model.eval()
    total_val_loss = 0
    progress_bar = tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validating")  

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(progress_bar): 
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch, start_pos=0)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), target_batch.view(-1))
            total_val_loss += loss.item()

            avg_val_loss_so_far = total_val_loss / (i + 1)  
            progress_bar.set_postfix(val_loss=avg_val_loss_so_far)  

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses,
    'val_loss': val_losses,
}, f"checkpoint_epoch_{epoch+1}.pt")


plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
