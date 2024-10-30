import json
from tqdm import tqdm
from pathlib import Path
from models.extractor.dataset import WLASLDataset, video_transform
from models.extractor.model import ModifiedI3D
from utils import EarlyStopping, save_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.output_path = Path(self.config['output_path'])
        self.output_path.mkdir(exist_ok=True)
        self.transform = video_transform()
        print(f"Using Device: {self.device}")

        with open(self.config['train_data']) as f:
            self.train_data = json.load(f)

        with open(self.config['val_data']) as f:
            self.val_data = json.load(f)

        print(f"Num Classes: {self.config['num_classes']}")

        self.model = ModifiedI3D(self.config['num_classes']).to(self.device)
        self.train_loader = self.get_dataloader(self.train_data)
        self.val_loader = self.get_dataloader(self.val_data)

    def get_dataloader(self, data):
        dataset = WLASLDataset(data, self.config['video_root'], transform=self.transform)
        dataloader = DataLoader(dataset, 
                                batch_size=self.config['batch_size'], 
                                shuffle=True, 
                                num_workers=self.config['num_workers'])
        return dataloader

    def train_epoch(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            _, outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(self.train_loader)
        train_acc = 100. * correct_train / total_train
        return train_acc, train_loss

    def validate(self):
        val_loss = 0.
        correct_val = 0
        total_val = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc=f"Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct_val / total_val
        return val_acc, val_loss

    def train(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        early_stopping = EarlyStopping(patience=self.config['patience'], verbose=True)

        # Training
        for epoch in range(self.config['epochs']):
            train_acc, train_loss = self.train_epoch(epoch)
            val_acc, val_loss = self.validate()
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")

            # Check early stopping condition
            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                print('Best Model. Saving ...')
                save_model(self.model, self.optimizer, self.config, val_loss, self.output_path/'checkpoint_best.pt')

            if early_stopping.early_stop and self.config['enable_earlystop']:
                print("Early stopping triggered. Stopping training.")
                save_model(self.model, self.optimizer, self.config, val_loss, self.output_path/'checkpoint_final.pt')
                break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and validate video classification model')
    parser.add_argument('--config', type=str, default='models/extractor/config.json', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.train()