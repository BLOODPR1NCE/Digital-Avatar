# api.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
import cv2
import base64
import os
import json
from io import BytesIO
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем классы модели
from model_training import Generator

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для GUI

# Глобальные переменные
generator = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 100
n_attributes = 40
img_size = 128

# Список атрибутов
ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

def load_model():
    """Загрузка обученной модели"""
    global generator, latent_dim, n_attributes, img_size
    
    model_path = 'models/avatar_generator_final.pt'
    
    if os.path.exists(model_path):
        print(f"Загрузка модели из {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        latent_dim = checkpoint.get('latent_dim', 100)
        n_attributes = checkpoint.get('n_attributes', 40)
        img_size = checkpoint.get('img_size', 128)
        
        # Загрузка генератора
        generator = Generator(latent_dim, n_attributes, img_size).to(device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        print(f"Модель загружена на {device}")
        print(f"Latent dim: {latent_dim}, Attributes: {n_attributes}, Image size: {img_size}")
        return True
    else:
        print(f"Модель не найдена: {model_path}")
        print("Сначала обучите модель с помощью model_training.py")
        return False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': generator is not None,
        'device': str(device)
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Генерация аватара по атрибутам"""
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    
    if 'attributes' not in data:
        return jsonify({'error': 'Attributes required'}), 400
    
    attributes = data['attributes']
    
    if len(attributes) != n_attributes:
        return jsonify({'error': f'Expected {n_attributes} attributes, got {len(attributes)}'}), 400
    
    try:
        # Преобразование атрибутов
        attrs_tensor = torch.FloatTensor(attributes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Случайный шум
            z = torch.randn(1, latent_dim).to(device)
            generated = generator(z, attrs_tensor)
        
        # Преобразование в изображение
        generated_img = generated.squeeze(0).cpu().numpy()
        generated_img = (generated_img.transpose(1, 2, 0) + 1) / 2
        generated_img = np.clip(generated_img * 255, 0, 255).astype(np.uint8)
        
        # Кодирование в base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'attributes': attributes
        })
        
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Тестовый эндпоинт - генерация случайного аватара"""
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Случайные атрибуты
        random_attrs = np.random.randint(0, 2, n_attributes).tolist()
        attrs_tensor = torch.FloatTensor(random_attrs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z = torch.randn(1, latent_dim).to(device)
            generated = generator(z, attrs_tensor)
        
        generated_img = generated.squeeze(0).cpu().numpy()
        generated_img = (generated_img.transpose(1, 2, 0) + 1) / 2
        generated_img = np.clip(generated_img * 255, 0, 255).astype(np.uint8)
        
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'attributes': random_attrs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/attributes', methods=['GET'])
def get_attributes():
    """Получение списка атрибутов"""
    return jsonify({
        'attributes': ATTRIBUTES,
        'count': len(ATTRIBUTES)
    })

@app.route('/generate_from_face', methods=['POST'])
def generate_from_face():
    """Генерация аватара на основе исходного лица с изменением атрибутов"""
    if generator is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Получаем атрибуты для изменения
    data = request.get_json() or {}
    target_attributes = data.get('attributes', None)
    
    if target_attributes is None:
        return jsonify({'error': 'Attributes required'}), 400
    
    try:
        # Создаем аватар с указанными атрибутами
        attrs_tensor = torch.FloatTensor(target_attributes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z = torch.randn(1, latent_dim).to(device)
            generated = generator(z, attrs_tensor)
        
        # Преобразование в изображение
        generated_img = generated.squeeze(0).cpu().numpy()
        generated_img = (generated_img.transpose(1, 2, 0) + 1) / 2
        generated_img = np.clip(generated_img * 255, 0, 255).astype(np.uint8)
        
        # Сохраняем результат
        os.makedirs('output', exist_ok=True)
        cv2.imwrite('output/generated_avatar.jpg', cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR))
        
        # Кодируем в base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Формируем список измененных атрибутов
        changed_attrs = []
        for i, attr in enumerate(ATTRIBUTES):
            if target_attributes[i] == 1:
                changed_attrs.append({'attribute': attr, 'value': 1})
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'final_attributes': target_attributes,
            'changed_attributes': changed_attrs
        })
        
    except Exception as e:
        print(f"Ошибка генерации из лица: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("ЗАПУСК API ДЛЯ ГЕНЕРАЦИИ АВАТАРОВ")
    print("=" * 60)
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️ ВНИМАНИЕ: Модель не загружена!")
        print("Сначала обучите модель:")
        print("1. Запустите data_preparation.py для подготовки данных")
        print("2. Запустите model_training.py для обучения модели")
        print("3. Затем перезапустите этот API")
    
    print("\nAPI запущен на http://localhost:5000")
    print("Доступные эндпоинты:")
    print("  GET  /health - проверка состояния")
    print("  GET  /test - тестовая генерация")
    print("  POST /generate - генерация аватара по атрибутам")
    print("  GET  /attributes - список атрибутов")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)