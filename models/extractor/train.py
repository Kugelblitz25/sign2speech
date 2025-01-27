import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.extractor.dataset import WLASLDataset
from utils import EarlyStopping, save_model


class Trainer:
    def __init__(self, config, model_name, freeze):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.output_path = Path(self.config["output_path"])
        self.output_path.mkdir(exist_ok=True)
        print(f"Using Device: {self.device}")

        with open(self.config["train_data"]) as f:
            self.train_data = json.load(f)

        with open(self.config["val_data"]) as f:
            self.val_data = json.load(f)

        print(f"Num Classes: {self.config['num_classes']}")

        self.model = self.load_model(model_name, self.config['num_classes'], freeze).to(self.device)
        self.train_loader = self.get_dataloader(self.train_data)
        self.val_loader = self.get_dataloader(self.val_data)
        self.freeze = freeze
    
    def load_model(self, model_name, num_classes, freeze):
        if model_name == "i3d":
            from models.extractor.model import ModifiedI3D
            model = ModifiedI3D(num_classes)
            for i in range(freeze): 
                for param in model.i3d.blocks[i].parameters():
                    param.requires_grad = False
            return model
        elif model_name == "x3d":
            from models.extractor.model import ModifiedX3D
            model = ModifiedX3D(num_classes)
            for i in range(freeze): 
                for param in model.x3d.blocks[i].parameters():
                    param.requires_grad = False
            return model
        elif model_name == "r2p1d":
            from models.extractor.model import ModifiedR2P1D
            model = ModifiedR2P1D(num_classes)
            for i in range(freeze): 
                for param in model.r2p1d.blocks[i].parameters():
                    param.requires_grad = False
            return model

            
    def get_dataloader(self, data):
        dataset = WLASLDataset(
            data, self.config["video_root"]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        return dataloader

    def train_epoch(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}"
        ):
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
        train_acc = 100.0 * correct_train / total_train
        return train_acc, train_loss

    def validate(self):
        val_loss = 0.0
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
        val_acc = 100.0 * correct_val / total_val
        return val_acc, val_loss


    def train(self):
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(
            #self.model.parameters(),
            #lr=self.config["lr"],
            #weight_decay=self.config["weight_decay"],
        #)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["lr"], momentum=0.9, weight_decay=self.config["weight_decay"])
        # Learnng rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)

        # Training
        best_train_acc = 0.0
        best_test_acc = 0.0

        for epoch in range(self.config["epochs"]):
            train_acc, train_loss = self.train_epoch(epoch)
            val_acc, val_loss = self.validate()
            print(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%"
            )

            # Check early stopping condition
            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                print("Best Model. Saving ...")
                save_model(
                    self.model,
                    self.optimizer,
                    self.config,
                    val_loss,
                    self.output_path / f"checkpoint_best_{self.model.name}.pt",
                )
                best_train_acc, best_test_acc = round(train_acc, 2), round(val_acc, 2)
                

            if early_stopping.early_stop and self.config["enable_earlystop"]:
                print("Early stopping triggered. Stopping training.")
                save_model(
                    self.model,
                    self.optimizer,
                    self.config,
                    val_loss,
                    self.output_path / f"checkpoint_final_{self.model.name}.pt",
                )
                break
        
        save_model(
            self.model,
            self.optimizer,
            self.config,
            val_loss,
            self.output_path / f"checkpoint_final_{self.model.name}.pt",
        )
        return best_train_acc, best_test_acc, round(train_acc, 2), round(val_acc, 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and validate video classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/extractor/config.json",
        help="Path to the config file",
    )

    parser.add_argument(
        "--model",
        type = str,
        default = "i3d",
        choices=["x3d", "i3d", "r2p1d"],
        help = "Model to be used for training (i3d, x3d, r2p1d)"
    )
    
    parser.add_argument(
        "--freeze",
        type=int,
        choices=range(0, 7),
        default=0,
        help="Number of layers to freeze"
    )
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    trainer = Trainer(config, args.model, args.freeze)
    trainer.train()
