import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import librosa

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class DataPreparation:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.df = None
    
    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
        return {
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=400, hop_length=160),
            'energy': librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0],
            'zcr': librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)[0],
            'sr': sr,
            'duration': len(y) / sr,
            'audio': y
        }
    
    def generate_realistic_visual_features(self, audio_features):
        mfcc = audio_features['mfcc']
        energy = audio_features['energy']
        zcr = audio_features['zcr']
        n_frames = mfcc.shape[1]
        
        energy_norm = np.ones(n_frames) * 0.5
        if len(energy) > 0 and energy.max() > 0:
            energy_norm = energy / energy.max()
        zcr_norm = np.ones(n_frames) * 0.3
        if len(zcr) > 0 and zcr.max() > 0:
            zcr_norm = zcr / zcr.max()
        
        if len(energy) != n_frames and len(energy) > 0:
            energy_norm = np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(energy)), energy_norm)
        if len(zcr) != n_frames and len(zcr) > 0:
            zcr_norm = np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(zcr)), zcr_norm)
        
        visual_features = np.zeros((n_frames, 68, 2))
        
        for frame in range(n_frames):
            amplitude = min(0.8, 0.1 + energy_norm[frame] * 0.4 + zcr_norm[frame] * 0.2)
            
            for i in range(48, 60):
                angle = (i - 48) / 12 * 2 * np.pi
                visual_features[frame, i, 0] = 0.5 + np.cos(angle) * 0.12
                visual_features[frame, i, 1] = 0.6 + np.sin(angle) * (0.05 + amplitude * 0.15)
            
            for i in range(60, 68):
                angle = (i - 60) / 8 * 2 * np.pi
                visual_features[frame, i, 0] = 0.5 + np.cos(angle) * 0.08
                visual_features[frame, i, 1] = 0.6 + np.sin(angle) * (0.03 + amplitude * 0.12)
            
            eye_closure = 0.1 + energy_norm[frame] * 0.4
            for idx, i_val in enumerate(range(36, 42)):
                visual_features[frame, i_val, 0] = 0.4 + (idx - 3) * 0.03
                visual_features[frame, i_val, 1] = 0.4 + eye_closure * 0.02
            for idx, i_val in enumerate(range(42, 48)):
                visual_features[frame, i_val, 0] = 0.6 + (idx - 3) * 0.03
                visual_features[frame, i_val, 1] = 0.4 + eye_closure * 0.02
            
            brow_raise = 0.05 + energy_norm[frame] * 0.15
            for idx, i_val in enumerate(range(17, 22)):
                visual_features[frame, i_val, 0] = 0.35 + (idx - 2) * 0.05
                visual_features[frame, i_val, 1] = 0.25 - brow_raise * 0.05
            for idx, i_val in enumerate(range(22, 27)):
                visual_features[frame, i_val, 0] = 0.65 + (idx - 2) * 0.05
                visual_features[frame, i_val, 1] = 0.25 - brow_raise * 0.05
        
        return visual_features
    
    @staticmethod
    def get_emotion_name(code):
        emotions = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                   '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
        return emotions.get(code, 'unknown')
    
    def process_dataset(self):
        print("\n" + "="*50 + "\nОБРАБОТКА RAVDESS ДАТАСЕТА\n" + "="*50)

        audio_dir = self.data_dir
        audio_files = list(Path(audio_dir).rglob("*.wav"))
        max_samples = len(audio_files)
        print(f"\n🔄 Обрабатывается {max_samples} файлов...")
        
        data = []
        for audio_path in tqdm(audio_files[:max_samples], desc="Обработка аудио"):
            audio_features = self.extract_audio_features(audio_path)
            if not audio_features:
                continue
                
            visual_features = self.generate_realistic_visual_features(audio_features)
            metadata = self._extract_metadata(audio_path)
            metadata.update({'duration': audio_features['duration'], 'sample_rate': audio_features['sr']})
            
            data.append({
                'audio_path': str(audio_path),
                'audio_features': audio_features['mfcc'].tolist(),
                'audio_energy': audio_features['energy'].tolist(),
                'audio_zcr': audio_features['zcr'].tolist(),
                'visual_features': visual_features.tolist(),
                'metadata': metadata
            })
        
        if not data:
            print("❌ Не удалось обработать ни одного файла")
            return None
        
        processed_data = {'samples': data}
        
        output_path = self.data_dir / 'processed_data.json'
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"✅ Данные сохранены в {output_path}")
        return processed_data
    
    def _extract_metadata(self, audio_path):
        filename = audio_path.stem
        parts = filename.split('-')
        
        if len(parts) >= 6:
            emotion_code = parts[2]
            emotion = self.get_emotion_name(emotion_code)
            actor_id = parts[6]
            intensity = parts[3]
            intensity_name = 'strong' if intensity == '02' else 'normal'
        else:
            emotion, emotion_code, actor_id, intensity_name = 'unknown', '00', 0, 'normal'
        
        return {'emotion': emotion, 'emotion_code': emotion_code, 'actor_id': actor_id, 'intensity': intensity_name}
        
if __name__ == "__main__":
    print("🚀 Запуск обработки реального RAVDESS датасета...")
    prep = DataPreparation()
    processed_data = prep.process_dataset()