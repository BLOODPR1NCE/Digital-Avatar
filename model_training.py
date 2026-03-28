# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Tuple, List, Dict
warnings.filterwarnings('ignore')

class LipSyncDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        audio = torch.FloatTensor(np.array(sample['audio_features'])).T
        visual = torch.FloatTensor(np.array(sample['visual_features']))
        visual_flat = visual.mean(dim=0).view(-1)
        return audio, visual_flat

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    audio_list, visual_list = zip(*batch)
    
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
    
    return torch.stack(padded_audio), torch.stack(visual_list)

class LipSyncModel(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 136):
        super().__init__()
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out.mean(dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class ModelTrainer:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
    def load_data(self) -> Optional[List[Dict]]:
        data_path = self.data_dir / 'processed_data.json'
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Загружено {len(data['samples'])} образцов")
        return data['samples']
    
    def prepare_dataloaders(self, samples: List[Dict], batch_size: int = 8, train_split: float = 0.8) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if not samples:
            return None, None
        
        train_size = int(len(samples) * train_split)
        train_dataset = LipSyncDataset(samples[:train_size])
        val_dataset = LipSyncDataset(samples[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
        
        print(f"   Обучающая выборка: {len(train_dataset)} образцов")
        print(f"   Валидационная выборка: {len(val_dataset)} образцов")
        
        return train_loader, val_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 30) -> nn.Module:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        
        print("\n🚀 Начало обучения модели...")
        print("="*50)
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._validate_epoch(model, val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model, 'best_model.pth')
                print(f"   ✨ Новая лучшая модель! Val Loss: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print(f"\n✅ Обучение завершено! Лучшая валидационная loss: {best_val_loss:.4f}")
        self._plot_training_curves(train_losses, val_losses)
        
        return model
    
    def _train_epoch(self, model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        
        for audio, visual in loader:
            audio, visual = audio.to(self.device), visual.to(self.device)
            
            optimizer.zero_grad()
            output = model(audio)
            loss = criterion(output, visual)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for audio, visual in loader:
                audio, visual = audio.to(self.device), visual.to(self.device)
                output = model(audio)
                total_loss += criterion(output, visual).item()
        
        return total_loss / len(loader)
    
    def _plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='blue')
        axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Кривые обучения модели', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].semilogy(train_losses, label='Train Loss', linewidth=2, color='blue')
        axes[1].semilogy(val_losses, label='Validation Loss', linewidth=2, color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log scale)')
        axes[1].set_title('Кривые обучения (логарифмическая шкала)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        best_epoch = np.argmin(val_losses)
        best_loss = min(val_losses)
        axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        axes[0].text(best_epoch, best_loss * 1.1, f'Best: {best_loss:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.models_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 График обучения сохранен: {self.models_dir / 'training_curves.png'}")
        
        with open(self.models_dir / 'losses.json', 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_loss, 'best_val_epoch': int(best_epoch)}, f)

    def save_model(self, model: nn.Module, path: str = 'model.pth') -> Path:
        model_path = self.data_dir / path
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {'input_dim': 13, 'hidden_dim': 128, 'output_dim': 136}
        }, model_path)
        print(f"💾 Модель сохранена: {model_path}")
        return model_path
    
    def load_model(self, model: nn.Module, path: str = 'best_model.pth') -> bool:
        model_path = self.data_dir / path
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📦 Модель загружена из {model_path}")
            return True
        print(f"⚠️ Модель не найдена: {model_path}")
        return False


if __name__ == "__main__":
    print("🚀 Запуск обучения модели...")
    trainer = ModelTrainer()
    samples = trainer.load_data()
    
    if samples:
        train_loader, val_loader = trainer.prepare_dataloaders(samples, batch_size=8)
        if train_loader and val_loader and len(train_loader) > 0:
            model = LipSyncModel().to(trainer.device)
            trained_model = trainer.train_model(model, train_loader, val_loader, epochs=30)
            trainer.save_model(trained_model, 'final_model.pth')
            print("\n🎉 Обучение успешно завершено!")
        else:
            print("❌ Не удалось подготовить даталоадеры")
    else:
        print("❌ Нет данных для обучения. Сначала запустите data_preparation.py")