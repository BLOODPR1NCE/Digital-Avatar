#model_training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class LipSyncDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        audio = torch.FloatTensor(np.array(sample['audio_features'])).T
        visual = torch.FloatTensor(np.array(sample['visual_features']))
        
        visual_mean = visual.mean(dim=0)
        visual_flat = visual_mean.view(-1)
        
        return audio, visual_flat

def collate_fn(batch):
    audio_list = []
    visual_list = []
    
    for audio, visual in batch:
        audio_list.append(audio)
        visual_list.append(visual)
    
    max_len = max(a.shape[0] for a in audio_list)
    padded_audio = []
    
    for audio in audio_list:
        if audio.shape[0] < max_len:
            pad_size = max_len - audio.shape[0]
            padding = torch.zeros(pad_size, audio.shape[1])
            padded = torch.cat([audio, padding], dim=0)
        else:
            padded = audio
        padded_audio.append(padded)
    
    visual_tensor = torch.stack(visual_list)
    
    return torch.stack(padded_audio), visual_tensor

class LipSyncModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=136):  # 68*2=136
        super(LipSyncModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.input_projection(x)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        lstm_mean = lstm_out.mean(dim=1)
        
        x = self.fc1(lstm_mean)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModelTrainer:
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
    def load_data(self):
        print("Загрузка подготовленных данных")
        data_path = self.data_dir / 'processed_data.json'
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data['samples']
    
    def prepare_dataloaders(self, samples, batch_size=8, train_split=0.8):
        print("Подготовка даталоадеров")
        if not samples:
            return None, None
        
        train_size = int(len(samples) * train_split)
        val_size = len(samples) - train_size
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:]
        
        train_dataset = LipSyncDataset(train_samples)
        val_dataset = LipSyncDataset(val_samples)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        print(f"   Обучающая выборка: {len(train_samples)} образцов")
        print(f"   Валидационная выборка: {len(val_samples)} образцов")
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader, epochs=30):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        train_losses = []
        val_losses = []
        
        print("\n🚀 Начало обучения модели...")
        print("="*50)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (audio, visual) in enumerate(train_loader):
                audio = audio.to(self.device)
                visual = visual.to(self.device)
                
                optimizer.zero_grad()
                output = model(audio)
                loss = criterion(output, visual)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 10 == 0 and batch_idx > 0:
                    print(f"   Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / max(train_batches, 1)
            train_losses.append(avg_train_loss)
            
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for audio, visual in val_loader:
                    audio = audio.to(self.device)
                    visual = visual.to(self.device)
                    output = model(audio)
                    loss = criterion(output, visual)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(model, 'best_model.pth')
                print(f"   ✨ Новая лучшая модель! Val Loss: {avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)
        
        print("\n✅ Обучение завершено!")
        print(f"   Лучшая валидационная loss: {best_val_loss:.4f}")
        
        self.plot_losses(train_losses, val_losses)
        self.plot_training_curves(train_losses, val_losses)

        return model
    
    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
        plt.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        plt.title('Кривые обучения модели', fontsize=14, fontweight='bold')
        plt.xlabel('Эпоха', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(train_losses, label='Train Loss', linewidth=2, color='blue')
        plt.semilogy(val_losses, label='Validation Loss', linewidth=2, color='red')
        plt.title('Кривые обучения (логарифмическая шкала)', fontsize=14, fontweight='bold')
        plt.xlabel('Эпоха', fontsize=12)
        plt.ylabel('Loss (log scale)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'training_curves.png', dpi=100)
        plt.show()
        
        print(f"\n📊 Графики сохранены: {self.data_dir / 'training_curves.png'}")
        
    def plot_training_curves(self, train_losses, val_losses):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='blue')
        axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Кривые обучения модели', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].semilogy(train_losses, label='Train Loss', linewidth=2, color='blue')
        axes[1].semilogy(val_losses, label='Validation Loss', linewidth=2, color='red')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss (log scale)', fontsize=12)
        axes[1].set_title('Кривые обучения (логарифмическая шкала)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = min(val_losses)
        axes[0].axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.5)
        axes[0].text(best_val_epoch, best_val_loss * 1.1, 
                    f'Best: {best_val_loss:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n📊 График обучения сохранен: {self.models_dir / 'training_curves.png'}")
        
        losses_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_val_epoch': best_val_epoch
        }
        
        with open(self.models_dir / 'losses.json', 'w') as f:
            json.dump(losses_data, f)
        print(f"📊 Значения потерь сохранены: {self.models_dir / 'losses.json'}")

    def save_model(self, model, path='model.pth'):
        model_path = self.data_dir / path
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': 13,
                'hidden_dim': 128,
                'output_dim': 136
            }
        }, model_path)
        print(f"💾 Модель сохранена: {model_path}")
        
        return model_path
    
    def load_model(self, model, path='best_model.pth'):
        model_path = self.data_dir / path
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📦 Модель загружена из {model_path}")
            return True
        else:
            print(f"⚠️ Модель не найдена: {model_path}")
            return False

if __name__ == "__main__":
    print("🚀 Запуск обучения модели...")
    
    trainer = ModelTrainer()
    samples = trainer.load_data()
    
    if samples:
        train_loader, val_loader = trainer.prepare_dataloaders(samples, batch_size=8)
        
        if train_loader and val_loader and len(train_loader) > 0 and len(val_loader) > 0:
            model = LipSyncModel().to(trainer.device)
            trained_model = trainer.train_model(model, train_loader, val_loader, epochs=30)
            trainer.save_model(trained_model, 'final_model.pth')
            print("\n🎉 Обучение успешно завершено!")
        else:
            print("❌ Не удалось подготовить даталоадеры")
    else:
        print("❌ Нет данных для обучения. Сначала запустите data_preparation.py")