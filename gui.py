import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import base64
import tempfile
import os
from PIL import Image, ImageTk
import threading
import shutil
from typing import Optional

class DigitalAvatarGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DigitalAvatar - Анимация лица")
        self.root.geometry("800x650")
        self.root.configure(bg='#f0f0f0')
        
        self.photo_path: Optional[str] = None
        self.audio_path: Optional[str] = None
        self.api_url = "http://localhost:8000"
        
        self.setup_ui()
        self.center_window()
    
    def setup_ui(self):
        # Заголовок
        tk.Label(self.root, text="🎭 DigitalAvatar", font=("Arial", 28, "bold"),
                bg='#f0f0f0', fg='#333').pack(pady=20)
        tk.Label(self.root, text="Создайте анимированный аватар по фото и голосу",
                font=("Arial", 12), bg='#f0f0f0', fg='#666').pack(pady=5)
        
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=10, padx=20)
        
        # Кнопки загрузки
        btn_frame = tk.Frame(self.root, bg='#f0f0f0')
        btn_frame.pack(pady=20)
        
        self.photo_btn = self._create_button(btn_frame, "📸 1. Загрузить фото", self.load_photo, '#4CAF50')
        self.audio_btn = self._create_button(btn_frame, "🎵 2. Загрузить аудио", self.load_audio, '#2196F3')
        
        # Информационная панель
        info_frame = tk.Frame(self.root, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        info_frame.pack(pady=10, padx=20, fill='x')
        
        self.photo_label = self._create_info_label(info_frame, "📷 Фото: не загружено")
        self.audio_label = self._create_info_label(info_frame, "🎵 Аудио: не загружено")
        
        # Предпросмотр
        preview_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=2)
        preview_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        tk.Label(preview_frame, text="Предпросмотр", font=("Arial", 12, "bold"), 
                bg='white').pack(pady=5)
        
        self.image_label = tk.Label(preview_frame, bg='#e0e0e0', 
                                    text="Загрузите фото для предпросмотра",
                                    font=("Arial", 10), width=50, height=15, relief=tk.SUNKEN)
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Статус
        self.status_label = tk.Label(self.root, text="✅ Готов к работе", font=("Arial", 10),
                                      bg='#e0e0e0', fg='#333', relief=tk.SUNKEN,
                                      anchor='w', padx=10, pady=5)
        self.status_label.pack(side=tk.BOTTOM, fill='x')
        
        # Кнопки управления
        gen_frame = tk.Frame(self.root, bg='#f0f0f0')
        gen_frame.pack(pady=20)
        
        self.generate_btn = self._create_button(gen_frame, "🎬 3. СОЗДАТЬ АНИМАЦИЮ 🎬", 
                                                 self.generate_animation, '#FF9800', 
                                                 width=35, height=3, state=tk.DISABLED)
        
        tk.Button(gen_frame, text="🗑 Очистить всё", command=self.clear_all,
                 width=20, height=1, bg='#f44336', fg='white', font=("Arial", 10),
                 cursor='hand2').pack(pady=5)
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate', length=400)
    
    def _create_button(self, parent, text, command, color, width=20, height=2, state=tk.NORMAL):
        btn = tk.Button(parent, text=text, command=command, width=width, height=height,
                       bg=color, fg='white', font=("Arial", 11, "bold"), cursor='hand2',
                       state=state)
        btn.pack(side=tk.LEFT, padx=10)
        return btn
    
    def _create_info_label(self, parent, text):
        label = tk.Label(parent, text=text, font=("Arial", 10), bg='#f0f0f0', fg='gray', pady=5)
        label.pack(anchor='w', padx=10)
        return label
    
    def center_window(self):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 800) // 2
        y = (self.root.winfo_screenheight() - 650) // 2
        self.root.geometry(f'800x650+{x}+{y}')
    
    def load_photo(self):
        file_path = filedialog.askopenfilename(title="Выберите фото", 
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.photo_path = file_path
            self.photo_label.config(text=f"📷 Фото: {os.path.basename(file_path)}", fg="green")
            self.update_status("✅ Фото загружено")
            self.check_ready()
            self._preview_image(file_path)
    
    def _preview_image(self, file_path):
        try:
            img = Image.open(file_path)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.update_status(f"Ошибка загрузки фото: {e}")
    
    def load_audio(self):
        file_path = filedialog.askopenfilename(title="Выберите аудиофайл",
                                               filetypes=[("Audio files", "*.wav *.mp3")])
        if file_path:
            self.audio_path = file_path
            self.audio_label.config(text=f"🎵 Аудио: {os.path.basename(file_path)}", fg="green")
            self.update_status("✅ Аудио загружено")
            self.check_ready()
    
    def check_ready(self):
        if self.photo_path and self.audio_path:
            self.generate_btn.config(state=tk.NORMAL)
            self.update_status("✅ Готов к генерации! Нажмите СОЗДАТЬ АНИМАЦИЮ")
        else:
            self.generate_btn.config(state=tk.DISABLED)
    
    def clear_all(self):
        self.photo_path = None
        self.audio_path = None
        self.photo_label.config(text="📷 Фото: не загружено", fg="gray")
        self.audio_label.config(text="🎵 Аудио: не загружено", fg="gray")
        self.image_label.config(image='', text="Загрузите фото для предпросмотра")
        self.image_label.image = None
        self.generate_btn.config(state=tk.DISABLED)
        self.update_status("🗑 Все данные очищены")
    
    def generate_animation(self):
        threading.Thread(target=self._generate_animation_thread, daemon=True).start()
    
    def _generate_animation_thread(self):
        try:
            self.generate_btn.config(state=tk.DISABLED)
            self.progress.pack(pady=10, padx=20, fill='x')
            self.progress.start()
            self.update_status("🎬 Генерация анимации... Пожалуйста, подождите 20-30 секунд")
            
            with open(self.photo_path, 'rb') as f:
                photo_data = f.read()
            with open(self.audio_path, 'rb') as f:
                audio_data = f.read()
            
            files = {
                'photo': (os.path.basename(self.photo_path), photo_data, 'image/jpeg'),
                'audio': (os.path.basename(self.audio_path), audio_data, 'audio/wav')
            }
            
            response = requests.post(f'{self.api_url}/generate', files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    gif_data = base64.b64decode(result['animation'])
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as f:
                        f.write(gif_data)
                        temp_path = f.name
                    
                    self._show_gif(temp_path)
                    self.update_status("✅ Анимация создана успешно!")
                    
                    if messagebox.askyesno("Успех", "Анимация создана!\nСохранить файл?"):
                        save_path = filedialog.asksaveasfilename(defaultextension=".gif",
                                                                 filetypes=[("GIF files", "*.gif")],
                                                                 initialfile="animation.gif")
                        if save_path:
                            shutil.copy(temp_path, save_path)
                            self.update_status(f"💾 Анимация сохранена: {os.path.basename(save_path)}")
                            messagebox.showinfo("Сохранено", f"Файл сохранен:\n{save_path}")
                    
                    os.unlink(temp_path)
                else:
                    raise Exception(result.get('message', 'Неизвестная ошибка'))
            else:
                raise Exception(f"Ошибка API: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.update_status("❌ Ошибка: API сервер не запущен")
            messagebox.showerror("Ошибка", "API сервер не запущен!")
        except Exception as e:
            self.update_status(f"❌ Ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось создать анимацию:\n{str(e)}")
        finally:
            self.progress.stop()
            self.progress.pack_forget()
            self.check_ready()
    
    def _show_gif(self, gif_path):
        try:
            gif = Image.open(gif_path)
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
        self.status_label.config(text=message)
        self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().theme_use('clam')
    app = DigitalAvatarGUI(root)
    root.mainloop()