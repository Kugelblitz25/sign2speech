import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import EarlyStopping, save_model

from models.transformer.dataset import SpectrogramDataset
from models.transformer.model import SpectrogramGenerator


class Trainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.output_path = Path(self.config['output_path'])
        self.output_path.mkdir(exist_ok=True)

        self.model = SpectrogramGenerator().to(self.device)
        self.train_loader = self.get_dataloader(self.config['features_csv_train'], 
                                                self.config['specs_csv'])
        self.val_loader = self.get_dataloader(self.config['features_csv_val'], 
                                                self.config['specs_csv'])

    def get_dataloader(self, features_csv, spec_csv):
        dataset = SpectrogramDataset(features_csv, spec_csv)
        dataloader = DataLoader(dataset, 
                                batch_size=self.config['batch_size'], 
                                shuffle=True, 
                                num_workers=self.config['num_workers'])
        return dataloader
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for features, spectrograms in tqdm(self.train_loader, f'Epoch {epoch}/{self.config["epochs"]}'):
            features = features.to(self.device)
            spectrograms = spectrograms.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, spectrograms)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
  
        with torch.no_grad():
            for features, spectrograms in tqdm(self.val_loader, 'Validation'):
                features = features.to(self.device)
                spectrograms = spectrograms.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, spectrograms)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)
        
    def train(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        early_stopping = EarlyStopping(patience=self.config['patience'], verbose=True)

        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            print(f'Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
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
    
    parser = argparse.ArgumentParser(description='Transforming video features into spectrogram featurres')
    parser.add_argument('--config', type=str, default='models/transformer/config.json', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.train()
