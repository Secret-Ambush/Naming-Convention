import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

# Load data
df2 = pd.read_csv('Datasets/dataset_final.csv')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# Define your dataset class
class NameCorrectionDataset(Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings.input_ids)


incorrect = [str(r) for r in df2['Incorrect']]
correct = [str(r) for r in df2['Correct']]

# Preprocess the data
encodings = tokenizer(incorrect,
                      correct,
                      truncation=True,
                      padding='max_length',
                      max_length=50,
                      return_tensors='pt')

unique_correct = list(set(correct))
labels = torch.tensor([unique_correct.index(name) for name in correct])
encodings['labels'] = labels
print(f"Label Distribution: {torch.unique(labels, return_counts=True)}")

# Create the dataset
dataset = NameCorrectionDataset(encodings)

# Create data loaders
loader = DataLoader(dataset, batch_size=16)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-cased')

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(3):
    model.train()
    batch_num = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Batch {batch_num} complete!')

    print(f'Epoch {epoch+1} complete!')

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'name_correction_model.pth')
