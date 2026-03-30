import unittest
import cv2
import numpy as np
import torch
import tempfile
import os

class TestDataPreparation(unittest.TestCase): 
    def test_audio_features_extraction(self):
        from data_preparation import DataPreparation
        sample_rate, duration = 16000, 1
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
            self.assertEqual(features['mfcc'].shape[0], 13)
            self.assertGreater(features['duration'], 0)
            print("✅ Аудио-признаки успешно извлечены")
        finally: os.unlink(audio_path)
    
    def test_emotion_mapping(self):
        from data_preparation import DataPreparation
        prep = DataPreparation()
        test_cases = [('01', 'neutral'), ('02', 'calm'), ('03', 'happy'), ('04', 'sad'), ('05', 'angry'), ('06', 'fearful'), ('07', 'disgust'), ('08', 'surprised'), ('99', 'unknown')]
        for code, expected in test_cases:
            self.assertEqual(prep.get_emotion_name(code), expected)
        print("✅ Маппинг эмоций работает корректно")

class TestModelTraining(unittest.TestCase):
    def test_dataset_loading(self):
        from model_training import LipSyncDataset
        test_samples = [{'audio_features': np.random.randn(30, 13).tolist(), 'visual_features': np.random.randn(20, 68, 2).tolist()} for _ in range(5)]
        dataset = LipSyncDataset(test_samples)
        self.assertEqual(len(dataset), 5)
        audio, visual = dataset[0]
        self.assertIsInstance(audio, torch.Tensor)
        self.assertIsInstance(visual, torch.Tensor)
        self.assertEqual(visual.shape[0], 136)
        print("✅ Датасет работает корректно")

class TestFaceAnimation(unittest.TestCase):
    def test_audio_energy_extraction(self):
        from api import DigitalAvatarService
        sample_rate, duration = 16000, 2
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
            for e in energy: self.assertGreaterEqual(e, 0); self.assertLessEqual(e, 1)
            print(f"✅ Энергия аудио извлечена: {len(energy)} кадров")
        finally: os.unlink(audio_path)
    
    def test_face_warping(self):
        from api import FaceWarper
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        src_landmarks = np.array([[100, 100], [120, 100], [100, 120]])
        dst_landmarks = np.array([[105, 100], [115, 100], [100, 125]])
        warper = FaceWarper()
        warped = warper.warp_face(test_image, src_landmarks, dst_landmarks)
        self.assertEqual(warped.shape, test_image.shape)
        print("✅ Деформация лица работает")

class TestAPI(unittest.TestCase):
    def _check_api_available(self) -> bool:
        import requests
        try:
            requests.get("http://localhost:8000/health", timeout=5)
            return True
        except: return False
    
    def test_api_health(self):
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        print(f"✅ API доступен: {data.get('status')}")
    
    def test_root_endpoint(self):
        import requests
        response = requests.get("http://localhost:8000/", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('message', data)
        self.assertIn('endpoints', data)
        print("✅ Root endpoint работает")

def run_tests():
    print("\n" + "="*60 + "\nЗАПУСК ЮНИТ-ТЕСТОВ\n" + "="*60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestDataPreparation))
    suite.addTest(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTest(loader.loadTestsFromTestCase(TestFaceAnimation))
    suite.addTest(loader.loadTestsFromTestCase(TestAPI))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n" + "="*60 + "\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n" + "="*60)
    print(f"✅ Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Провалено: {len(result.failures)}")
    print(f"⚠️ Ошибок: {len(result.errors)}")
    print(f"⏭️ Пропущено: {len(result.skipped)}")
    return result

if __name__ == '__main__':
    run_tests()