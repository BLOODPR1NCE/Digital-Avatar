#api
from fastapi import FastAPI, File, UploadFile, HTTPException
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

app = FastAPI(title="DigitalAvatar API", description="API для анимации лица")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LipSyncModel(torch.nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=136):
        super(LipSyncModel, self).__init__()
        
        self.input_projection = torch.nn.Linear(input_dim, hidden_dim)
        
        self.lstm = torch.nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.fc1 = torch.nn.Linear(hidden_dim * 2, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_mean = lstm_out.mean(dim=1)
        x = self.fc1(lstm_mean)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FaceWarper:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = self._get_predictor_path()
        self.predictor = dlib.shape_predictor(self.predictor_path)
    
    def _get_predictor_path(self):
        possible_paths = [
            "./data/shape_predictor_68_face_landmarks.dat",
            "shape_predictor_68_face_landmarks.dat",
            os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def detect_landmarks(self, image):
        if self.predictor is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))
        
        return np.array(points)
    
    def warp_face(self, image, source_landmarks, target_landmarks):
        if source_landmarks is None or target_landmarks is None:
            return image
        
        h, w = image.shape[:2]
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h)) #сетка деформации
        
        map_x = grid_x.astype(np.float32) #смещение
        map_y = grid_y.astype(np.float32)
        
        for i in range(len(source_landmarks)):
            src = source_landmarks[i].astype(np.float32)
            dst = target_landmarks[i].astype(np.float32)
            
            displacement = dst - src
            
            radius = 50
            
            dist = np.sqrt((grid_x - src[0])**2 + (grid_y - src[1])**2)
            
            weight = np.exp(-dist**2 / (2 * radius**2))
            weight = np.clip(weight, 0, 1)
            
            map_x += displacement[0] * weight
            map_y += displacement[1] * weight
        
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped
    
    def interpolate_landmarks(self, landmarks1, landmarks2, t):
        return landmarks1 * (1 - t) + landmarks2 * t
    
    def create_lip_animation(self, image, audio_energy):
        source_landmarks = self.detect_landmarks(image)
        
        if source_landmarks is None:
            return [image] * len(audio_energy)
        
        target_landmarks = source_landmarks.copy()
        
        frames = []
        
        for energy in audio_energy:
            frame = image.copy()
            current_landmarks = source_landmarks.copy()
            
            mouth_open = 0.2 + energy * 1.2
            mouth_open = min(0.8, mouth_open)
            
            for i in range(48, 60):
                if i < 54:  # Верхняя губа
                    current_landmarks[i][1] = source_landmarks[i][1] - mouth_open * 15
                else:  # Нижняя губа
                    current_landmarks[i][1] = source_landmarks[i][1] + mouth_open * 15
            
            for i in range(60, 68):
                if i < 64:  # Верхняя часть внутренней губы
                    current_landmarks[i][1] = source_landmarks[i][1] - mouth_open * 12
                else:  # Нижняя часть
                    current_landmarks[i][1] = source_landmarks[i][1] + mouth_open * 12
            
            if np.random.random() < 0.05:  # 5% шанс мигания
                eye_closure = 8
                for i in range(36, 48):
                    if 36 <= i < 42:  # Левый глаз
                        current_landmarks[i][1] = source_landmarks[i][1] + eye_closure
                    elif 42 <= i < 48:  # Правый глаз
                        current_landmarks[i][1] = source_landmarks[i][1] + eye_closure
            
            warped = self.warp_face(frame, source_landmarks, current_landmarks)
            frames.append(warped)
        
        return frames

class DigitalAvatarService:
    def __init__(self, model_path="./data/best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        self.model = LipSyncModel().to(self.device)
        
        model_path = Path(model_path)
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Модель загружена из {model_path}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
        else:
            print(f"⚠️ Модель не найдена: {model_path}")
        
        self.model.eval()
        self.face_warper = FaceWarper()
        
    def extract_audio_energy(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
            
            frame_length = int(sr * 0.04)
            hop_length = int(sr * 0.02)
            
            energy = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                frame_energy = np.sqrt(np.mean(frame**2))
                energy.append(frame_energy)
            
            if len(energy) > 0:
                energy = np.array(energy)
                max_energy = energy.max()
                if max_energy > 0:
                    energy = energy / max_energy
                energy = np.clip(energy, 0.1, 1.0)
                
                from scipy.ndimage import gaussian_filter1d
                energy = gaussian_filter1d(energy, sigma=2)
            
            num_frames = 30
            animation_energy = []
            
            for i in range(num_frames):
                t = i / num_frames
                idx = int(t * len(energy)) if len(energy) > 0 else 0
                if idx < len(energy):
                    val = energy[idx]
                else:
                    val = 0.5
                animation_energy.append(float(val))
            
            return animation_energy
            
        except Exception as e:
            print(f"Ошибка при извлечении энергии: {e}")
    
    def generate_animation(self, image_path, audio_path, output_path="output.gif"):
        image = cv2.imread(image_path)
        audio_energy = self.extract_audio_energy(audio_path)
        frames = self.face_warper.create_lip_animation(image, audio_energy)
        
        if not frames:
            frames = [image] * len(audio_energy)
        
        try:
            import imageio
            resized_frames = []
            for frame in frames:
                if frame.shape[1] > 800:
                    scale = 800 / frame.shape[1]
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                resized_frames.append(frame)
            
            imageio.mimsave(output_path, resized_frames, fps=25, loop=0)
            print(f"✅ Анимация сохранена: {output_path}, кадров: {len(resized_frames)}")
            return output_path
            
        except Exception as e:
            print(f"Ошибка сохранения GIF: {e}")
            return None

service = DigitalAvatarService()

@app.get("/")
async def root():
    return {
        "message": "DigitalAvatar API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - Generate animation from photo and audio",
            "/health": "GET - Check service health"
        }
    }

@app.post("/generate")
async def generate_animation(
    photo: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    temp_files = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
            img_content = await photo.read()
            img_file.write(img_content)
            img_path = img_file.name
            temp_files.append(img_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_content = await audio.read()
            audio_file.write(audio_content)
            audio_path = audio_file.name
            temp_files.append(audio_path)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
        temp_files.append(output_path)
        
        result_path = service.generate_animation(img_path, audio_path, output_path)
        
        if result_path is None:
            raise HTTPException(500, "Ошибка при создании анимации")
        
        with open(result_path, "rb") as f:
            gif_data = base64.b64encode(f.read()).decode()
        
        return JSONResponse({
            "success": True,
            "animation": gif_data,
            "format": "gif",
            "message": "Анимация успешно создана"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

@app.get("/health")
async def health_check():
    model_status = "loaded" if service.model else "not_loaded"
    face_detector_status = "loaded" if service.face_warper.predictor else "not_loaded"
    
    return {
        "status": "healthy",
        "device": str(service.device),
        "model_status": model_status,
        "face_detector_status": face_detector_status,
        "available_endpoints": ["/", "/generate", "/health"]
    }

if __name__ == "__main__":
    print("🚀 Запуск DigitalAvatar API...")
    print("="*50)
    print("API доступен по адресу: http://localhost:8000")
    print("Документация: http://localhost:8000/docs")
    print("="*50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )