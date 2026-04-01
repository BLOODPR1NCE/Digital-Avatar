from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import librosa
import base64
import dlib
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="DigitalAvatar API", description="API для анимации лица")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class LipSyncModel(torch.nn.Module):
    def __init__(self, input_dim = 13, hidden_dim = 128, output_dim = 136):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out.mean(dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class FaceWarper:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = self._get_predictor_path()
        self.predictor = dlib.shape_predictor(self.predictor_path) if self.predictor_path else None
    
    @staticmethod
    def _get_predictor_path():
        possible_path = "./data/shape_predictor_68_face_landmarks.dat"
        return possible_path
    
    def detect_landmarks(self, image):
        if not self.predictor: return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces: return None
        landmarks = self.predictor(gray, faces[0])
        return np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
    
    def warp_face(self, image, src_landmarks, dst_landmarks):
        if src_landmarks is None or dst_landmarks is None: return image
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x, map_y = grid_x.astype(np.float32), grid_y.astype(np.float32)
        
        for src, dst in zip(src_landmarks, dst_landmarks):
            displacement = dst - src
            dist = np.sqrt((grid_x - src[0])**2 + (grid_y - src[1])**2)
            weight = np.clip(np.exp(-dist**2 / (2 * 50**2)), 0, 1)
            map_x += displacement[0] * weight
            map_y += displacement[1] * weight
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def interpolate_landmarks(self, landmarks1, landmarks2, t):
        return landmarks1 * (1 - t) + landmarks2 * t
    
    def create_lip_animation(self, image, audio_energy):
        src_landmarks = self.detect_landmarks(image)
        if src_landmarks is None: return [image] * len(audio_energy)
        
        frames = []
        for energy in audio_energy:
            frame = image.copy()
            current = src_landmarks.copy()
            mouth_open = min(0.8, 0.2 + energy * 1.2)
            
            for i in range(48, 60):
                offset = 15 if i < 54 else -15
                current[i][1] = src_landmarks[i][1] + offset * mouth_open
            for i in range(60, 68):
                offset = 12 if i < 64 else -12
                current[i][1] = src_landmarks[i][1] + offset * mouth_open
            if np.random.random() < 0.05:
                for i in range(36, 48):
                    current[i][1] = src_landmarks[i][1] + 8
            
            frames.append(self.warp_face(frame, src_landmarks, current))
        return frames

class DigitalAvatarService:
    def __init__(self, model_path = "./data/best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LipSyncModel().to(self.device)
        self.face_warper = FaceWarper()
        
        model_path = Path(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.eval()
    
    def extract_audio_energy(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
            frame_length, hop_length = int(sr * 0.04), int(sr * 0.02)
            energy = [np.sqrt(np.mean(y[i:i+frame_length]**2)) for i in range(0, len(y) - frame_length, hop_length)]
            
            if energy:
                energy = np.array(energy)
                max_energy = energy.max()
                if max_energy > 0:
                    energy = energy / max_energy
                energy = np.clip(energy, 0.1, 1.0)
                from scipy.ndimage import gaussian_filter1d
                energy = gaussian_filter1d(energy, sigma=2)
            
            num_frames = 30
            indices = np.linspace(0, len(energy) - 1, num_frames) if len(energy) > 0 else np.zeros(num_frames)
            return [float(energy[int(idx)]) if 0 <= int(idx) < len(energy) else 0.5 for idx in indices]
        except Exception as e:
            print(f"Ошибка извлечения энергии: {e}")
            return [0.5] * 30
    
    def generate_animation(self, image_path, audio_path, output_path= "output.gif"):
        image = cv2.imread(image_path)
        audio_energy = self.extract_audio_energy(audio_path)
        frames = self.face_warper.create_lip_animation(image, audio_energy) or [image] * len(audio_energy)
        
        try:
            import imageio
            resized_frames = [cv2.resize(frame, (int(frame.shape[1] * 800 / frame.shape[1]), int(frame.shape[0] * 800 / frame.shape[1]))) if frame.shape[1] > 800 else frame for frame in frames]
            imageio.mimsave(output_path, resized_frames, fps=25, loop=0)
            return output_path
        except Exception as e:
            print(f"Ошибка сохранения GIF: {e}")
            return None

service = DigitalAvatarService()

@app.get("/")
async def root():
    return {"message": "DigitalAvatar API", "status": "running", "version": "1.0.0", "endpoints": {"/generate": "POST", "/health": "GET"}}

@app.post("/generate")
async def generate_animation(photo = File(...), audio = File(...)):
    temp_files = []
    try:
        img_path, audio_path, output_path = [tempfile.NamedTemporaryFile(delete=False, suffix=s).name for s in [".jpg", ".wav", ".gif"]]
        temp_files.extend([img_path, audio_path, output_path])
        
        for path, file in [(img_path, photo), (audio_path, audio)]:
            with open(path, 'wb') as f:
                f.write(await file.read())
        
        result_path = service.generate_animation(img_path, audio_path, output_path)
        if not result_path: raise HTTPException(500, "Ошибка при создании анимации")
        
        with open(result_path, "rb") as f:
            gif_data = base64.b64encode(f.read()).decode()
        
        return JSONResponse({"success": True, "animation": gif_data, "format": "gif", "message": "Анимация успешно создана"})
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file): os.unlink(temp_file)
            except: pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(service.device), "model_status": "loaded" if service.model else "not_loaded", "face_detector_status": "loaded" if service.face_warper.predictor else "not_loaded"}

if __name__ == "__main__":
    print("🚀 Запуск DigitalAvatar API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")