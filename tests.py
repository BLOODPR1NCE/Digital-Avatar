# test_avatar_system.py
import unittest
import numpy as np
import torch
import os
import json
import requests
import cv2
import tempfile
import base64

class TestAvatarSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Подготовка тестовых данных"""
        cls.test_attrs = np.random.randint(0, 2, 40).tolist()
        
        # Создание тестового изображения
        cls.test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cls.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(cls.temp_file.name, cls.test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Очистка"""
        os.unlink(cls.temp_file.name)
    
    def test_1_model_exists(self):
        """Тест 1: Проверка наличия модели"""
        print("\n" + "=" * 60)
        print("ТЕСТ 1: Проверка наличия модели")
        print("=" * 60)
        
        model_paths = [
            'models/avatar_generator_final.pt',
            'models/avatar_model_best.pt'
        ]
        
        for path in model_paths:
            self.assertTrue(os.path.exists(path), f"Модель не найдена: {path}")
            print(f"✓ Модель найдена: {path}")
    
    def test_2_data_exists(self):
        """Тест 2: Проверка наличия данных"""
        print("\n" + "=" * 60)
        print("ТЕСТ 2: Проверка наличия данных")
        print("=" * 60)
        
        required_files = [
            'processed_data/train_attributes.npy',
            'processed_data/val_attributes.npy',
            'processed_data/test_attributes.npy',
            'processed_data/metadata.json'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Файл не найден: {file_path}")
            print(f"✓ Файл найден: {file_path}")
        
        # Проверка метаданных
        with open('processed_data/metadata.json', 'r') as f:
            metadata = json.load(f)
            self.assertIn('attributes', metadata)
            self.assertIn('n_attributes', metadata)
            print(f"✓ Метаданные: {metadata['n_attributes']} атрибутов")
    
    def test_3_api_health(self):
        """Тест 3: Проверка API"""
        print("\n" + "=" * 60)
        print("ТЕСТ 3: Проверка API")
        print("=" * 60)
        
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('status', data)
            self.assertIn('model_loaded', data)
            self.assertIn('device', data)
            
            print(f"✓ Статус: {data['status']}")
            print(f"✓ Модель загружена: {data['model_loaded']}")
            print(f"✓ Устройство: {data['device']}")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")
    
    def test_4_model_prediction(self):
        """Тест 4: Предсказание модели"""
        print("\n" + "=" * 60)
        print("ТЕСТ 4: Предсказание модели")
        print("=" * 60)
        
        model_path = 'models/avatar_generator_final.pt'
        if not os.path.exists(model_path):
            self.skipTest("Модель не найдена")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        # Тест генерации
        attrs_tensor = torch.FloatTensor(self.test_attrs).unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                generated = model.generate_from_attrs(attrs_tensor, device)
            
            self.assertEqual(generated.shape, (1, 3, 128, 128))
            print(f"✓ Форма выхода: {generated.shape}")
            
            self.assertTrue(torch.all(generated >= -1) and torch.all(generated <= 1))
            print("✓ Диапазон значений корректный")
            
        except Exception as e:
            self.fail(f"Ошибка при генерации: {e}")
    
    def test_5_api_generate(self):
        """Тест 5: API генерации"""
        print("\n" + "=" * 60)
        print("ТЕСТ 5: API генерации")
        print("=" * 60)
        
        try:
            response = requests.post(
                "http://localhost:5000/generate",
                json={'attributes': self.test_attrs},
                timeout=10
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn('success', data)
            self.assertTrue(data['success'])
            self.assertIn('image', data)
            
            # Проверка декодирования изображения
            img_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.assertIsNotNone(img)
            
            print(f"✓ Изображение сгенерировано, размер: {img.shape}")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")
    
    def test_6_api_reconstruct(self):
        """Тест 6: API реконструкции"""
        print("\n" + "=" * 60)
        print("ТЕСТ 6: API реконструкции")
        print("=" * 60)
        
        try:
            with open(self.temp_file.name, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    "http://localhost:5000/reconstruct",
                    files=files,
                    timeout=10
                )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn('success', data)
            self.assertTrue(data['success'])
            self.assertIn('reconstructed', data)
            
            print("✓ Реконструкция выполнена")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")
    
    def test_7_api_attributes(self):
        """Тест 7: API списка атрибутов"""
        print("\n" + "=" * 60)
        print("ТЕСТ 7: API списка атрибутов")
        print("=" * 60)
        
        try:
            response = requests.get("http://localhost:5000/attributes", timeout=5)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('attributes', data)
            self.assertIn('count', data)
            
            print(f"✓ Загружено {data['count']} атрибутов")
            print(f"✓ Первые 5 атрибутов: {data['attributes'][:5]}")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")
    
    def test_8_model_structure(self):
        """Тест 8: Структура модели"""
        print("\n" + "=" * 60)
        print("ТЕСТ 8: Структура модели")
        print("=" * 60)
        
        model_path = 'models/avatar_generator_final.pt'
        if not os.path.exists(model_path):
            self.skipTest("Модель не найдена")
        
        model = torch.load(model_path)
        
        # Проверка наличия компонентов
        self.assertTrue(hasattr(model, 'encoder'))
        self.assertTrue(hasattr(model, 'decoder'))
        
        # Проверка параметров
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 1000000)
        
        print(f"✓ Структура модели корректна")
        print(f"✓ Количество параметров: {total_params:,}")

def run_tests():
    print("\n" + "=" * 70)
    print("ЗАПУСК МОДУЛЬНЫХ ТЕСТОВ СИСТЕМЫ ГЕНЕРАЦИИ АВАТАРОВ")
    print("=" * 70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAvatarSystem)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Пропущено: {len(result.skipped)}")
    
    if result.failures or result.errors:
        print("\nДЕТАЛИ ОШИБОК:")
        for test, traceback in result.failures + result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    return result

if __name__ == '__main__':
    run_tests()