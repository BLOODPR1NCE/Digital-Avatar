# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

class AvatarDataset(Dataset):
    """Датасет для обучения модели аватара"""
    def __init__(self, data_dir, transform=None, img_size=128):
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Загрузка атрибутов
        attrs_path = f'{data_dir}/train_attributes.npy'
        if os.path.exists(attrs_path):
            self.attributes = np.load(attrs_path)
            print(f"Загружено атрибутов: {self.attributes.shape}")
        else:
            raise FileNotFoundError(f"Файл {attrs_path} не найден. Сначала запустите data_preparation.py")
        
        # Список изображений
        images_dir = f'{data_dir}/images'
        self.images = []
        if os.path.exists(images_dir):
            self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            self.images = sorted(self.images)[:len(self.attributes)]
        
        print(f"Загружено изображений: {len(self.images)}")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'images', self.images[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            image = torch.randn(3, self.img_size, self.img_size)
        
        attrs = torch.FloatTensor(self.attributes[idx])
        
        return image, attrs

class Generator(nn.Module):
    """Генератор изображений"""
    def __init__(self, latent_dim=100, n_attributes=40, img_size=128):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_attributes = n_attributes
        
        # Начальный размер: 4x4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + n_attributes, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z, attrs):
        # Конкатенация шума и атрибутов
        combined = torch.cat([z, attrs], dim=1)
        x = self.fc(combined)
        x = x.view(-1, 512, 4, 4)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    """Дискриминатор"""
    def __init__(self, n_attributes=40, img_size=128):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8 + n_attributes, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, attrs):
        features = self.conv(img)
        features = features.view(features.size(0), -1)
        combined = torch.cat([features, attrs], dim=1)
        validity = self.fc(combined)
        return validity

def train_model():
    """Обучение GAN модели"""
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ГЕНЕРАЦИИ АВАТАРА")
    print("=" * 60)
    
    # Параметры
    batch_size = 32
    epochs = 20
    lr = 0.0002
    latent_dim = 100
    img_size = 128
    n_attributes = 40
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Используемое устройство: {device}")
    
    # Загрузка данных
    dataset = AvatarDataset('processed_data', img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, drop_last=True)
    
    print(f"Количество батчей: {len(dataloader)}")
    
    # Инициализация моделей
    generator = Generator(latent_dim, n_attributes, img_size).to(device)
    discriminator = Discriminator(n_attributes, img_size).to(device)
    
    # Оптимизаторы
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Функции потерь
    adversarial_loss = nn.BCELoss()
    
    # Создание директории
    os.makedirs('models', exist_ok=True)
    os.makedirs('generated_samples', exist_ok=True)
    
    # Обучение
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, attrs in pbar:
            batch_size_curr = images.size(0)
            images = images.to(device)
            attrs = attrs.to(device)
            
            # Метки
            valid = torch.ones(batch_size_curr, 1).to(device)
            fake = torch.zeros(batch_size_curr, 1).to(device)
            
            # ---------------------
            # Обучение дискриминатора
            # ---------------------
            d_optimizer.zero_grad()
            
            # Потеря на реальных изображениях
            real_validity = discriminator(images, attrs)
            d_real_loss = adversarial_loss(real_validity, valid)
            
            # Генерация фейковых изображений
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake_images = generator(z, attrs)
            fake_validity = discriminator(fake_images.detach(), attrs)
            d_fake_loss = adversarial_loss(fake_validity, fake)
            
            # Полная потеря дискриминатора
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            # Обучение генератора
            # ---------------------
            g_optimizer.zero_grad()
            
            # Генерация изображений
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            generated_images = generator(z, attrs)
            
            # Adversarial loss для генератора
            gen_validity = discriminator(generated_images, attrs)
            g_loss = adversarial_loss(gen_validity, valid)
            
            g_loss.backward()
            g_optimizer.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
        
        # Сохранение модели каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
                'g_loss': avg_g_loss,
                'latent_dim': latent_dim,
                'n_attributes': n_attributes,
                'img_size': img_size
            }, f'models/generator_epoch_{epoch+1}.pt')
            
            # Генерация примеров
            generate_samples(generator, epoch + 1, device, n_attributes)
    
    # Сохранение финальной модели
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'latent_dim': latent_dim,
        'n_attributes': n_attributes,
        'img_size': img_size
    }, 'models/avatar_generator_final.pt')
    
    print("\nОбучение завершено! Модель сохранена.")
    
    return generator

def generate_samples(generator, epoch, device, n_attributes):
    """Генерация примеров для мониторинга"""
    generator.eval()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    with torch.no_grad():
        for i in range(4):
            for j in range(4):
                # Создаем случайные атрибуты
                attrs = torch.randint(0, 2, (1, n_attributes)).float().to(device)
                z = torch.randn(1, 100).to(device)
                generated = generator(z, attrs)
                
                img = generated.squeeze(0).cpu().numpy()
                img = (img.transpose(1, 2, 0) + 1) / 2
                img = np.clip(img, 0, 1)
                
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'generated_samples/epoch_{epoch}.png', dpi=100)
    plt.close()

def generate_avatar(attributes, model_path='models/avatar_generator_final.pt', device='cuda'):
    """Генерация аватара на основе атрибутов"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        print("Сначала обучите модель с помощью train_model()")
        return None
    
    # Загрузка модели
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Создание генератора
    latent_dim = checkpoint.get('latent_dim', 100)
    n_attributes = checkpoint.get('n_attributes', 40)
    img_size = checkpoint.get('img_size', 128)
    
    generator = Generator(latent_dim, n_attributes, img_size).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Подготовка атрибутов
    attrs_tensor = torch.FloatTensor(attributes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated = generator(z, attrs_tensor)
    
    # Преобразование в изображение
    generated = generated.squeeze(0).cpu().numpy()
    generated = (generated.transpose(1, 2, 0) + 1) / 2
    generated = np.clip(generated * 255, 0, 255).astype(np.uint8)
    
    return generated

if __name__ == "__main__":
    train_model()