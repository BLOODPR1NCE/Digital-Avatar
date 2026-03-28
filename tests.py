# tests.py - 4 простых юнит-теста
import unittest
import cv2
import numpy as np
import torch
import json
import tempfile
import os
from pathlib import Path

class TestDataPreparation(unittest.TestCase):
    """Тест 1: Проверка подготовки данных"""
    
    def test_audio_features_extraction(self):
        """Проверка извлечения аудио-признаков"""
        from data_preparation import DataPreparation
        
        # Создаем тестовое аудио
        sample_rate = 16000
        duration = 1
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Сохраняем временный файл
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, test_audio, sample_rate)
            audio_path = tmp.name
        
        try:
            # Тестируем извлечение признаков
            prep = DataPreparation()
            features = prep.extract_audio_features(audio_path)
            
            # Проверяем наличие всех признаков
            self.assertIsNotNone(features)
            self.assertIn('mfcc', features)
            self.assertIn('energy', features)
            self.assertIn('zcr', features)
            self.assertIn('duration', features)
            
            # Проверяем размерности
            self.assertEqual(features['mfcc'].shape[0], 13)  # 13 MFCC коэффициентов
            self.assertGreater(features['duration'], 0)
            
            print("✅ Аудио-признаки успешно извлечены")
            
        finally:
            # Очистка
            os.unlink(audio_path)
    
    def test_emotion_mapping(self):
        """Проверка маппинга эмоций"""
        from data_preparation import DataPreparation
        
        prep = DataPreparation()
        
        # Проверяем коды эмоций
        test_cases = [
            ('01', 'neutral'),
            ('02', 'calm'),
            ('03', 'happy'),
            ('04', 'sad'),
            ('05', 'angry'),
            ('06', 'fearful'),
            ('07', 'disgust'),
            ('08', 'surprised')
        ]
        
        for code, expected_emotion in test_cases:
            emotion = prep.get_emotion_name(code)
            self.assertEqual(emotion, expected_emotion)
        
        # Проверяем неизвестный код
        unknown_emotion = prep.get_emotion_name('99')
        self.assertEqual(unknown_emotion, 'unknown')
        
        print("✅ Маппинг эмоций работает корректно")


class TestModelTraining(unittest.TestCase):
    """Тест 2: Проверка обучения модели"""
    
    def test_model_forward_pass(self):
        """Проверка forward pass модели"""
        from model_training import LipSyncModel
        
        # Создаем модель
        model = LipSyncModel(input_dim=13, hidden_dim=128, output_dim=136)
        model.eval()
        
        # Тестовый вход
        batch_size = 2
        time_steps = 50
        test_input = torch.randn(batch_size, time_steps, 13)
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
        
        # Проверяем размерность выхода
        expected_shape = (batch_size, 136)
        self.assertEqual(output.shape, expected_shape)
        
        # Проверяем, что значения корректны
        self.assertTrue(torch.all(torch.isfinite(output)))
        
        print(f"✅ Forward pass успешен")
        print(f"   Вход: {test_input.shape}")
        print(f"   Выход: {output.shape}")
    
    def test_dataset_loading(self):
        """Проверка загрузки датасета"""
        from model_training import LipSyncDataset
        
        # Создаем тестовые данные
        test_samples = []
        for i in range(5):
            sample = {
                'audio_features': np.random.randn(30, 13).tolist(),
                'visual_features': np.random.randn(20, 68, 2).tolist()
            }
            test_samples.append(sample)
        
        # Создаем датасет
        dataset = LipSyncDataset(test_samples)
        
        # Проверяем размер
        self.assertEqual(len(dataset), 5)
        
        # Проверяем получение элемента
        audio, visual = dataset[0]
        self.assertIsInstance(audio, torch.Tensor)
        self.assertIsInstance(visual, torch.Tensor)
        self.assertEqual(visual.shape[0], 136)  # 68 точек * 2 координаты
        
        print("✅ Датасет работает корректно")


class TestFaceAnimation(unittest.TestCase):
    """Тест 3: Проверка анимации лица"""
    
    def test_audio_energy_extraction(self):
        """Проверка извлечения энергии из аудио"""
        from api import DigitalAvatarService
        
        # Создаем тестовое аудио
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Сохраняем временный файл
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, test_audio, sample_rate)
            audio_path = tmp.name
        
        try:
            # Извлекаем энергию
            service = DigitalAvatarService()
            energy = service.extract_audio_energy(audio_path)
            
            # Проверяем результат
            self.assertIsNotNone(energy)
            self.assertGreater(len(energy), 0)
            
            # Энергия должна быть в диапазоне [0, 1]
            for e in energy:
                self.assertGreaterEqual(e, 0)
                self.assertLessEqual(e, 1)
            
            print(f"✅ Энергия аудио извлечена: {len(energy)} кадров")
            
        finally:
            os.unlink(audio_path)
    
    def test_face_warping(self):
        """Проверка деформации лица"""
        from api import FaceWarper
        
        # Создаем тестовое изображение
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        
        # Создаем тестовые точки
        source_landmarks = np.array([[100, 100], [120, 100], [100, 120]])
        target_landmarks = np.array([[105, 100], [115, 100], [100, 125]])
        
        # Применяем деформацию
        warper = FaceWarper()
        warped = warper.warp_face(test_image, source_landmarks, target_landmarks)
        
        # Проверяем, что изображение не изменило размер
        self.assertEqual(warped.shape, test_image.shape)
        
        print("✅ Деформация лица работает")


class TestAPI(unittest.TestCase):
    """Тест 4: Проверка API"""
    
    def test_api_health_check(self):
        """Проверка health endpoint API"""
        import requests
        
        try:
            # Проверяем доступность API
            response = requests.get("http://localhost:8000/health", timeout=5)
            
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('status', data)
            
            print(f"✅ API доступен")
            print(f"   Статус: {data.get('status')}")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")
    
    def test_api_root_endpoint(self):
        """Проверка root endpoint API"""
        import requests
        
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('message', data)
            self.assertIn('endpoints', data)
            
            print(f"✅ Root endpoint работает")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")


def run_tests():
    """Запуск всех тестов"""
    print("\n" + "="*60)
    print("ЗАПУСК 4 ЮНИТ-ТЕСТОВ")
    print("="*60)
    
    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем все тесты
    suite.addTest(loader.loadTestsFromTestCase(TestDataPreparation))
    suite.addTest(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTest(loader.loadTestsFromTestCase(TestFaceAnimation))
    suite.addTest(loader.loadTestsFromTestCase(TestAPI))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"✅ Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Провалено: {len(result.failures)}")
    print(f"⚠️ Ошибок: {len(result.errors)}")
    print(f"⏭️ Пропущено: {len(result.skipped)}")
    
    return result

if __name__ == '__main__':
    run_tests()