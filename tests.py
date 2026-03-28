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
        from data_preparation import DataPreparation
        
        sample_rate = 16000 #тестовое аудио
        duration = 1
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, test_audio, sample_rate)
            audio_path = tmp.name
        
        try:
            prep = DataPreparation()
            features = prep.extract_audio_features(audio_path)
            
            self.assertIsNotNone(features)
            self.assertIn('mfcc', features)
            self.assertIn('energy', features)
            self.assertIn('zcr', features)
            self.assertIn('duration', features)
            self.assertEqual(features['mfcc'].shape[0], 13)  # 13 MFCC коэффициентов
            self.assertGreater(features['duration'], 0)
            
            print("✅ Аудио-признаки успешно извлечены")
            
        finally:
            os.unlink(audio_path)
    
    def test_emotion_mapping(self):
        from data_preparation import DataPreparation
        
        prep = DataPreparation()
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
        
        unknown_emotion = prep.get_emotion_name('99')
        self.assertEqual(unknown_emotion, 'unknown')
        
        print("✅ Маппинг эмоций работает корректно")


class TestModelTraining(unittest.TestCase):
    """Тест 2: Проверка обучения модели"""
    
    def test_dataset_loading(self):
        from model_training import LipSyncDataset
    
        test_samples = []
        for i in range(5):
            sample = {
                'audio_features': np.random.randn(30, 13).tolist(),
                'visual_features': np.random.randn(20, 68, 2).tolist()
            }
            test_samples.append(sample)
        
        dataset = LipSyncDataset(test_samples)
        
        self.assertEqual(len(dataset), 5)
        audio, visual = dataset[0]
        self.assertIsInstance(audio, torch.Tensor)
        self.assertIsInstance(visual, torch.Tensor)
        self.assertEqual(visual.shape[0], 136)  # 68 точек * 2 координаты
        
        print("✅ Датасет работает корректно")


class TestFaceAnimation(unittest.TestCase):
    """Тест 3: Проверка анимации лица"""
    
    def test_audio_energy_extraction(self):
        from api import DigitalAvatarService

        sample_rate = 16000 # тестовое аудио
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, test_audio, sample_rate)
            audio_path = tmp.name
        
        try:
            service = DigitalAvatarService()
            energy = service.extract_audio_energy(audio_path)
            
            self.assertIsNotNone(energy)
            self.assertGreater(len(energy), 0)
            for e in energy:
                self.assertGreaterEqual(e, 0)
                self.assertLessEqual(e, 1)
            
            print(f"✅ Энергия аудио извлечена: {len(energy)} кадров")
            
        finally:
            os.unlink(audio_path)
    
    def test_face_warping(self):
        from api import FaceWarper
        
        test_image = np.zeros((200, 200, 3), dtype=np.uint8) # тестовое фото
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        source_landmarks = np.array([[100, 100], [120, 100], [100, 120]])
        target_landmarks = np.array([[105, 100], [115, 100], [100, 125]])
        
        warper = FaceWarper()
        warped = warper.warp_face(test_image, source_landmarks, target_landmarks)
        
        self.assertEqual(warped.shape, test_image.shape)
        
        print("✅ Деформация лица работает")


class TestAPI(unittest.TestCase):
    """Тест 4: Проверка API"""
    
    def test_api_health_check(self):
        import requests
        
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('status', data)
            
            print(f"✅ API доступен")
            print(f"   Статус: {data.get('status')}")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен")
    
    def test_api_root_endpoint(self):
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
    print("\n" + "="*60)
    print("ЗАПУСК ЮНИТ-ТЕСТОВ")
    print("="*60)
    
    loader = unittest.TestLoader() #создание тестового набора
    suite = unittest.TestSuite()
    
    suite.addTest(loader.loadTestsFromTestCase(TestDataPreparation)) #добавление тестов
    suite.addTest(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTest(loader.loadTestsFromTestCase(TestFaceAnimation))
    suite.addTest(loader.loadTestsFromTestCase(TestAPI))
    
    runner = unittest.TextTestRunner(verbosity=2) #запуск
    result = runner.run(suite)
    
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