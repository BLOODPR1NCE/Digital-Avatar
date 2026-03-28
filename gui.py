#gui,
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import base64
import tempfile
import os
from PIL import Image, ImageTk
import threading
import shutil

class DigitalAvatarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DigitalAvatar - Анимация лица")
        self.root.geometry("800x650")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(True, True)
        
        self.photo_path = None
        self.audio_path = None
        self.api_url = "http://localhost:8000"
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка интерфейса - простая и надежная версия"""
        
        # Заголовок
        title_label = tk.Label(
            self.root,
            text="🎭 DigitalAvatar",
            font=("Arial", 28, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=20)
        
        # Описание
        desc_label = tk.Label(
            self.root,
            text="Создайте анимированный аватар по фото и голосу",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666'
        )
        desc_label.pack(pady=5)
        
        # Разделитель
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', pady=10, padx=20)
        
        # Фрейм для кнопок загрузки
        buttons_frame = tk.Frame(self.root, bg='#f0f0f0')
        buttons_frame.pack(pady=20)
        
        # Кнопка загрузки фото
        self.photo_btn = tk.Button(
            buttons_frame,
            text="📸 1. Загрузить фото",
            command=self.load_photo,
            width=20,
            height=2,
            bg='#4CAF50',
            fg='white',
            font=("Arial", 11, "bold"),
            cursor='hand2'
        )
        self.photo_btn.pack(side=tk.LEFT, padx=10)
        
        # Кнопка загрузки аудио
        self.audio_btn = tk.Button(
            buttons_frame,
            text="🎵 2. Загрузить аудио",
            command=self.load_audio,
            width=20,
            height=2,
            bg='#2196F3',
            fg='white',
            font=("Arial", 11, "bold"),
            cursor='hand2'
        )
        self.audio_btn.pack(side=tk.LEFT, padx=10)
        
        # Информация о загруженных файлах
        info_frame = tk.Frame(self.root, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        info_frame.pack(pady=10, padx=20, fill='x')
        
        self.photo_label = tk.Label(
            info_frame,
            text="📷 Фото: не загружено",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='gray',
            pady=5
        )
        self.photo_label.pack(anchor='w', padx=10)
        
        self.audio_label = tk.Label(
            info_frame,
            text="🎵 Аудио: не загружено",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='gray',
            pady=5
        )
        self.audio_label.pack(anchor='w', padx=10)
        
        # Область предпросмотра
        preview_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=2)
        preview_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        preview_title = tk.Label(
            preview_frame,
            text="Предпросмотр",
            font=("Arial", 12, "bold"),
            bg='white'
        )
        preview_title.pack(pady=5)
        
        self.image_label = tk.Label(
            preview_frame,
            bg='#e0e0e0',
            text="Загрузите фото для предпросмотра",
            font=("Arial", 10),
            width=50,
            height=15,
            relief=tk.SUNKEN
        )
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Статус
        self.status_label = tk.Label(
            self.root,
            text="✅ Готов к работе",
            font=("Arial", 10),
            bg='#e0e0e0',
            fg='#333',
            relief=tk.SUNKEN,
            anchor='w',
            padx=10,
            pady=5
        )
        self.status_label.pack(side=tk.BOTTOM, fill='x')
        
        # Фрейм для кнопки генерации (отдельный, чтобы точно была видна)
        generate_frame = tk.Frame(self.root, bg='#f0f0f0')
        generate_frame.pack(pady=20)
        
        # КНОПКА ГЕНЕРАЦИИ - самая важная, делаем её большой и заметной
        self.generate_btn = tk.Button(
            generate_frame,
            text="🎬 3. СОЗДАТЬ АНИМАЦИЮ 🎬",
            command=self.generate_animation,
            width=35,
            height=3,
            bg='#FF9800',
            fg='white',
            font=("Arial", 14, "bold"),
            cursor='hand2',
            state=tk.DISABLED
        )
        self.generate_btn.pack()
        
        # Кнопка очистки
        self.clear_btn = tk.Button(
            generate_frame,
            text="🗑 Очистить всё",
            command=self.clear_all,
            width=20,
            height=1,
            bg='#f44336',
            fg='white',
            font=("Arial", 10),
            cursor='hand2'
        )
        self.clear_btn.pack(pady=5)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=400
        )
        
        # Инструкция
        instruction_frame = tk.Frame(self.root, bg='#fff8e7', relief=tk.GROOVE, bd=1)
        instruction_frame.pack(pady=10, padx=20, fill='x')
        
        instruction_text = tk.Label(
            instruction_frame,
            text="💡 Инструкция:\n"
                 "1. Нажмите 'Загрузить фото' и выберите изображение с лицом\n"
                 "2. Нажмите 'Загрузить аудио' и выберите аудиофайл (WAV или MP3)\n"
                 "3. После загрузки обоих файлов кнопка 'СОЗДАТЬ АНИМАЦИЮ' станет активной\n"
                 "4. Нажмите её и дождитесь результата (10-30 секунд)\n"
                 "5. Готовую анимацию можно будет сохранить",
            bg='#fff8e7',
            font=("Arial", 9),
            justify=tk.LEFT,
            pady=10,
            padx=10
        )
        instruction_text.pack()
        
        # Проверка API при запуске
        self.root.after(1000, self.check_api_status)
        
    def check_api_status(self):
        """Проверка доступности API"""
        def check():
            try:
                response = requests.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200:
                    self.update_status("✅ API сервер доступен")
                else:
                    self.update_status("⚠️ API сервер не отвечает. Запустите api.py")
                    messagebox.showwarning("API недоступен", 
                        "API сервер не отвечает.\nЗапустите api.py в отдельном терминале:\npython api.py")
            except:
                self.update_status("⚠️ API сервер не запущен. Запустите api.py")
                messagebox.showwarning("API не запущен", 
                    "API сервер не запущен.\nПожалуйста, запустите api.py в отдельном терминале:\npython api.py")
        
        threading.Thread(target=check, daemon=True).start()
        
    def load_photo(self):
        """Загрузка фото"""
        file_path = filedialog.askopenfilename(
            title="Выберите фото",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.photo_path = file_path
            self.photo_label.config(
                text=f"📷 Фото: {os.path.basename(file_path)}", 
                fg="green"
            )
            self.update_status("✅ Фото загружено")
            self.check_ready()
            
            # Показать миниатюру
            try:
                img = Image.open(file_path)
                # Сохраняем пропорции
                img.thumbnail((400, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
            except Exception as e:
                self.update_status(f"Ошибка загрузки фото: {e}")
            
    def load_audio(self):
        """Загрузка аудио"""
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Audio files", "*.wav *.mp3")]
        )
        
        if file_path:
            self.audio_path = file_path
            self.audio_label.config(
                text=f"🎵 Аудио: {os.path.basename(file_path)}", 
                fg="green"
            )
            self.update_status("✅ Аудио загружено")
            self.check_ready()
            
    def check_ready(self):
        """Проверка готовности к генерации"""
        if self.photo_path and self.audio_path:
            self.generate_btn.config(state=tk.NORMAL, bg='#FF9800')
            self.update_status("✅ Готов к генерации! Нажмите СОЗДАТЬ АНИМАЦИЮ")
        else:
            self.generate_btn.config(state=tk.DISABLED, bg='#FF9800')
            
    def clear_all(self):
        """Очистка всех данных"""
        self.photo_path = None
        self.audio_path = None
        self.photo_label.config(text="📷 Фото: не загружено", fg="gray")
        self.audio_label.config(text="🎵 Аудио: не загружено", fg="gray")
        self.image_label.config(image='', text="Загрузите фото для предпросмотра")
        self.image_label.image = None
        self.generate_btn.config(state=tk.DISABLED)
        self.update_status("🗑 Все данные очищены")
            
    def generate_animation(self):
        """Генерация анимации через API"""
        if not self.photo_path or not self.audio_path:
            messagebox.showwarning("Ошибка", "Загрузите фото и аудио")
            return
        
        # Запуск в отдельном потоке
        threading.Thread(target=self._generate_animation_thread, daemon=True).start()
        
    def _generate_animation_thread(self):
        """Поток для генерации анимации"""
        try:
            # Обновление UI
            self.generate_btn.config(state=tk.DISABLED)
            self.progress.pack(pady=10, padx=20, fill='x')
            self.progress.start()
            self.update_status("🎬 Генерация анимации... Пожалуйста, подождите 20-30 секунд")
            
            # Подготовка данных
            with open(self.photo_path, 'rb') as f:
                photo_data = f.read()
            
            with open(self.audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Отправка запроса к API
            files = {
                'photo': (os.path.basename(self.photo_path), photo_data, 'image/jpeg'),
                'audio': (os.path.basename(self.audio_path), audio_data, 'audio/wav')
            }
            
            response = requests.post(f'{self.api_url}/generate', files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    # Декодирование GIF
                    gif_data = base64.b64decode(result['animation'])
                    
                    # Сохранение временного файла
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as f:
                        f.write(gif_data)
                        temp_path = f.name
                    
                    # Отображение в GUI
                    self.show_gif(temp_path)
                    self.update_status("✅ Анимация создана успешно!")
                    
                    # Предложение сохранить
                    if messagebox.askyesno("Успех", "Анимация создана!\nСохранить файл?"):
                        save_path = filedialog.asksaveasfilename(
                            defaultextension=".gif",
                            filetypes=[("GIF files", "*.gif")],
                            initialfile="animation.gif"
                        )
                        if save_path:
                            shutil.copy(temp_path, save_path)
                            self.update_status(f"💾 Анимация сохранена: {os.path.basename(save_path)}")
                            messagebox.showinfo("Сохранено", f"Файл сохранен:\n{save_path}")
                    
                    # Очистка
                    os.unlink(temp_path)
                else:
                    raise Exception(result.get('message', 'Неизвестная ошибка'))
            else:
                raise Exception(f"Ошибка API: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.update_status("❌ Ошибка: API сервер не запущен")
            messagebox.showerror("Ошибка", 
                "API сервер не запущен!\n\n"
                "Пожалуйста, откройте новый терминал и запустите:\n"
                "python api.py\n\n"
                "После запуска API вернитесь в это окно и попробуйте снова.")
        except Exception as e:
            self.update_status(f"❌ Ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось создать анимацию:\n{str(e)}")
            
        finally:
            self.progress.stop()
            self.progress.pack_forget()
            if self.photo_path and self.audio_path:
                self.generate_btn.config(state=tk.NORMAL, bg='#FF9800')
            else:
                self.generate_btn.config(state=tk.DISABLED, bg='#FF9800')
            
    def show_gif(self, gif_path):
        """Отображение GIF в интерфейсе"""
        try:
            # Открываем GIF
            gif = Image.open(gif_path)
            
            # Показываем первый кадр
            frame = gif.copy()
            frame.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(frame)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            self.update_status("✅ Анимация отображается в предпросмотре")
            
        except Exception as e:
            print(f"Ошибка отображения GIF: {e}")
            self.image_label.config(text="Ошибка отображения анимации")
            
    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_label.config(text=message)
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Настройка стиля прогресс-бара
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TProgressbar", thickness=20)
    
    app = DigitalAvatarGUI(root)
    
    # Центрирование окна
    root.update_idletasks()
    width = 800
    height = 650
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()