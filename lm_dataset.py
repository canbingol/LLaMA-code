import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import ModelArgs
import os
from dotenv import load_dotenv


from warnings import filterwarnings
filterwarnings('ignore')

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', token=hf_token)
args = ModelArgs()
max_len = args.max_seq_len
pad_token = tokenizer.pad_token_id

# for kaggle
if os.path.exists("/kaggle/input/tr-news/data.txt"):
    data_path = "/kaggle/input/tr-news/data.txt"
else:
    data_path = "sml_data.txt"  

with open(data_path, "r", encoding="utf-8") as f:
    data = f.readlines()
    
train_ratio = .9
train_len = int(len(raw_data) * train_ratio)

train_data = raw_data[:train_len]
val_data = raw_data[train_len:]


class language_model_dataset(Dataset):

    def __init__(self, texts, tokenizer, max_len, stride):
        super().__init__()
        self.inputs = []
        self.targets = []
        input_tokens = []

        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            input_tokens.extend(tokens)  

            
        for i in range(0, len(input_tokens) - max_len, stride):
            input_chunk = input_tokens[i: i+max_len]
            target_chunk = input_tokens[1+ i: 1+ i+max_len]
            
            # truncating
            input_chunk = input_chunk[:max_len]
            target_chunk = target_chunk[:max_len]

            # padding
            input_chunk += [pad_token] * (max_len - len(input_chunk))
            target_chunk += [pad_token] * (max_len - len(target_chunk)) 

            input_chunk = input_chunk[:max_len]
            target_chunk = target_chunk[:max_len]

            self.inputs.append(torch.tensor(input_chunk, dtype=torch.long))
            self.targets.append(torch.tensor(target_chunk, dtype=torch.long))
            

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]



def create_dataloader(text: list, batch_size: int, drop_last: bool,
                      max_len: int, stride: int, shuffle: bool, tokenizer):
    
    dataset = language_model_dataset(text, tokenizer,max_len, stride)
    dataloader = DataLoader(
        dataset  =dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last=drop_last
    )
    
    return dataloader

train_loader = create_dataloader(train_data,args.batch_size,args.drop_last,
                                 args.max_seq_len,args.max_seq_len,args.shuffle ,tokenizer)


val_loader = create_dataloader(val_data,args.batch_size,args.drop_last,
                                 args.max_seq_len,args.max_seq_len,args.shuffle, tokenizer )


def save_sample_batch(dataloader, tokenizer, filename: str, num_samples: int = 2):
    with open(filename, "w", encoding="utf-8") as f:
        for i, (inputs, targets) in enumerate(dataloader):
            for j in range(min(num_samples, inputs.size(0))):
                input_ids = inputs[j].tolist()
                target_ids = targets[j].tolist()

                input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

                f.write(f"Sample {i * num_samples + j + 1}\n")
                f.write(f"Input IDs  : {input_ids}\n")
                f.write(f"Input Text : {input_text}\n")
                f.write(f"Target IDs : {target_ids}\n")
                f.write(f"Target Text: {target_text}\n")
                f.write("\n" + "=" * 80 + "\n\n")
            break  # Sadece ilk batch'ten örnek al


save_sample_batch(train_loader, tokenizer, "train_sample.txt")
save_sample_batch(val_loader, tokenizer, "val_sample.txt")
