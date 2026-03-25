# data_preparation.py
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CelebADataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.images_path = os.path.join(data_path, 'img_align_celeba', 'img_align_celeba')
        
    def load_attributes(self):
        """Загрузка атрибутов"""
        attr_path = os.path.join(self.data_path, 'list_attr_celeba.xlsx')
        df_attr = pd.read_excel(attr_path, index_col=0)
        # Преобразование -1/1 в 0/1
        df_attr = df_attr.replace({-1: 0, 1: 1})
        return df_attr
    
    def load_bbox(self):
        """Загрузка bounding boxes"""
        bbox_path = os.path.join(self.data_path, 'list_bbox_celeba.xlsx')
        df_bbox = pd.read_excel(bbox_path, index_col=0)
        return df_bbox
    
    def load_landmarks(self):
        """Загрузка landmarks"""
        landmarks_path = os.path.join(self.data_path, 'list_landmarks_align_celeba.xlsx')
        df_landmarks = pd.read_excel(landmarks_path, index_col=0)
        return df_landmarks
    
    def load_partition(self):
        """Загрузка разбиения на train/val/test"""
        partition_path = os.path.join(self.data_path, 'list_eval_partition.xlsx')
        df_partition = pd.read_excel(partition_path, index_col=0)
        return df_partition
    
    def analyze_data(self):
        """Анализ данных"""
        print("=" * 60)
        print("АНАЛИЗ ДАННЫХ CELEBA")
        print("=" * 60)
        
        # Загрузка всех данных
        df_attr = self.load_attributes()
        df_bbox = self.load_bbox()
        df_landmarks = self.load_landmarks()
        df_partition = self.load_partition()
        
        print("\n1. ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ")
        print("-" * 40)
        print(f"Количество изображений: {len(df_attr)}")
        print(f"Количество атрибутов: {len(df_attr.columns)}")
        print(f"Атрибуты: {list(df_attr.columns)[:5]}...")
        
        print("\n2. АНАЛИЗ АТРИБУТОВ")
        print("-" * 40)
        
        # Статистика по атрибутам
        attr_stats = pd.DataFrame({
            'count': df_attr.count(),
            'positive_ratio': df_attr.mean(),
            'std': df_attr.std()
        }).sort_values('positive_ratio', ascending=False)
        
        print("\nТоп-10 наиболее распространенных атрибутов:")
        print(attr_stats.head(10))
        
        print("\n3. АНАЛИЗ РАЗБИЕНИЯ ДАННЫХ")
        print("-" * 40)
        partition_stats = df_partition['partition'].value_counts().sort_index()
        partition_names = {0: 'Train', 1: 'Validation', 2: 'Test'}
        for part, count in partition_stats.items():
            print(f"{partition_names[part]}: {count} изображений ({count/len(df_partition)*100:.1f}%)")
        
        return df_attr, df_bbox, df_landmarks, df_partition
    
    def preprocess_data(self, img_size=128, max_images=50000):
        """Предобработка данных для обучения"""
        print("\n" + "=" * 60)
        print("ПРЕДОБРАБОТКА ДАННЫХ")
        print("=" * 60)
        
        df_attr = self.load_attributes()
        df_partition = self.load_partition()
        
        # Разделение данных
        train_images = df_partition[df_partition['partition'] == 0].index.tolist()
        val_images = df_partition[df_partition['partition'] == 1].index.tolist()
        test_images = df_partition[df_partition['partition'] == 2].index.tolist()
        
        print(f"Всего train: {len(train_images)} изображений")
        print(f"Всего validation: {len(val_images)} изображений")
        print(f"Всего test: {len(test_images)} изображений")
        
        # Ограничиваем количество для быстрого обучения
        train_images = train_images[:max_images]
        
        # Создание директорий для сохранения
        os.makedirs('processed_data', exist_ok=True)
        os.makedirs('processed_data/images', exist_ok=True)
        
        # Обработка изображений
        print("\nОбработка изображений...")
        
        all_attributes = []
        all_image_names = []
        
        for img_name in tqdm(train_images, desc="Processing training images"):
            img_path = os.path.join(self.images_path, img_name)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    img_path_save = f'processed_data/images/{img_name}'
                    cv2.imwrite(img_path_save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    attrs = df_attr.loc[img_name].values
                    all_attributes.append(attrs)
                    all_image_names.append(img_name)
        
        # Сохранение атрибутов
        np.save('processed_data/train_attributes.npy', np.array(all_attributes))
        np.save('processed_data/train_image_names.npy', np.array(all_image_names))
        
        print(f"\nОбработано {len(all_attributes)} изображений")
        
        # Сохранение метаданных
        metadata = {
            'img_size': img_size,
            'attributes': list(df_attr.columns),
            'n_attributes': len(df_attr.columns),
            'train_count': len(all_attributes)
        }
        
        import json
        with open('processed_data/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print("\nМетаданные сохранены")
        
        return metadata, np.array(all_attributes)

def main():
    data_path = r"C:\Users\prince\Downloads\archive"
    loader = CelebADataLoader(data_path)
    
    # Анализ данных
    df_attr, df_bbox, df_landmarks, df_partition = loader.analyze_data()
    
    # Предобработка
    metadata, attributes = loader.preprocess_data(img_size=128, max_images=10000)
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ И ПРЕДОБРАБОТКА ЗАВЕРШЕНЫ")
    print(f"Подготовлено {metadata['train_count']} изображений для обучения")
    print("=" * 60)

if __name__ == "__main__":
    main()