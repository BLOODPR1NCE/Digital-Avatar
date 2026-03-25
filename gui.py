# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import requests
import numpy as np
import base64
from io import BytesIO
import threading
import os

class AvatarGeneratorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Avatar Generator - CelebA")
        self.root.geometry("1200x800")
        
        self.api_url = "http://localhost:5000"
        self.attributes = []
        self.last_generated_image = None
        
        # Загрузка списка атрибутов
        self.load_attributes()
        
        self.setup_ui()
        
        # Проверка API
        self.check_api()
        
    def check_api(self):
        """Проверка доступности API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('model_loaded'):
                    self.status_label.config(text="Статус: ✅ API готов (модель загружена)", foreground="green")
                else:
                    self.status_label.config(text="Статус: ⚠️ API работает, но модель не загружена", foreground="orange")
            else:
                self.status_label.config(text="Статус: ❌ API недоступен", foreground="red")
        except:
            self.status_label.config(text="Статус: ❌ API недоступен (запустите api.py)", foreground="red")
    
    def load_attributes(self):
        """Загрузка списка атрибутов"""
        try:
            response = requests.get(f"{self.api_url}/attributes", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.attributes = data['attributes']
                print(f"Загружено {len(self.attributes)} атрибутов")
            else:
                print(f"Ошибка загрузки атрибутов: {response.status_code}")
                self.attributes = [f"attr_{i}" for i in range(40)]
        except Exception as e:
            print(f"Не удалось загрузить атрибуты: {e}")
            self.attributes = [f"attr_{i}" for i in range(40)]
    
    def setup_ui(self):
        """Настройка интерфейса"""
        # Основной контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Верхняя панель - результат
        result_frame = ttk.LabelFrame(main_frame, text="Сгенерированный аватар", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_label = tk.Label(result_frame, text="Нажмите\n'Сгенерировать'",
                                     width=40, height=20, relief=tk.SUNKEN,
                                     bg='gray90', fg='gray40')
        self.result_label.pack(pady=10)
        
        # Средняя панель - управление атрибутами
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Левая часть - список атрибутов
        attrs_frame = ttk.LabelFrame(middle_frame, text="Атрибуты (отметьте нужные)", padding="10")
        attrs_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        # Создание Canvas с прокруткой
        canvas_frame = ttk.Frame(attrs_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame, height=400)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Переменные для атрибутов
        self.attr_vars = {}
        
        # Создание чекбоксов для атрибутов
        for i, attr in enumerate(self.attributes):
            var = tk.BooleanVar()
            self.attr_vars[attr] = var
            cb = ttk.Checkbutton(scrollable_frame, text=attr, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Правая часть - кнопки управления
        control_frame = ttk.LabelFrame(middle_frame, text="Управление", padding="10")
        control_frame.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)
        
        # Кнопка генерации
        generate_btn = ttk.Button(control_frame, text="Сгенерировать аватар", 
                                  command=self.generate_avatar)
        generate_btn.pack(pady=10, fill=tk.X)
        
        # Кнопка сброса атрибутов
        reset_btn = ttk.Button(control_frame, text="Сбросить все атрибуты", 
                               command=self.reset_attributes)
        reset_btn.pack(pady=5, fill=tk.X)
        
        # Кнопка сохранения
        save_btn = ttk.Button(control_frame, text="Сохранить результат", 
                              command=self.save_result)
        save_btn.pack(pady=5, fill=tk.X)
        
        # Кнопка тестовой генерации
        test_btn = ttk.Button(control_frame, text="Случайный аватар", 
                              command=self.test_generation)
        test_btn.pack(pady=5, fill=tk.X)
        
        # Кнопка проверки API
        check_btn = ttk.Button(control_frame, text="Проверить API", 
                               command=self.check_api)
        check_btn.pack(pady=5, fill=tk.X)
        
        # Индикатор загрузки
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(pady=10, fill=tk.X)
        
        # Статус
        self.status_label = ttk.Label(control_frame, text="Проверка API...")
        self.status_label.pack(pady=5)
        
        # Информация
        info_frame = ttk.LabelFrame(control_frame, text="Информация", padding="10")
        info_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=35)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.insert(tk.END, "Готов к работе\n\n")
        self.info_text.insert(tk.END, "1. Отметьте желаемые атрибуты\n")
        self.info_text.insert(tk.END, "2. Нажмите 'Сгенерировать аватар'\n")
        self.info_text.insert(tk.END, "3. Дождитесь генерации\n")
        self.info_text.insert(tk.END, "4. Сохраните результат\n")
    
    def get_attributes_vector(self):
        """Получение вектора атрибутов из UI"""
        attrs = []
        for attr in self.attributes:
            attrs.append(1 if self.attr_vars.get(attr, False).get() else 0)
        return attrs
    
    def reset_attributes(self):
        """Сброс всех атрибутов"""
        for var in self.attr_vars.values():
            var.set(False)
        self.info_text.insert(tk.END, "\n✅ Все атрибуты сброшены\n")
    
    def generate_avatar(self):
        """Генерация аватара по выбранным атрибутам"""
        selected_attrs = [attr for attr, var in self.attr_vars.items() if var.get()]
        
        if not selected_attrs:
            self.info_text.insert(tk.END, "\n⚠️ Выберите хотя бы один атрибут\n")
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один атрибут")
            return
        
        def generate():
            self.progress.start()
            self.result_label.config(text="Генерация...\nПожалуйста, подождите")
            self.info_text.insert(tk.END, f"\n🔄 Генерация с атрибутами: {', '.join(selected_attrs[:5])}...\n")
            
            try:
                attrs = self.get_attributes_vector()
                
                response = requests.post(
                    f"{self.api_url}/generate",
                    json={'attributes': attrs},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        # Отображение результата
                        img_data = base64.b64decode(data['image'])
                        img = Image.open(BytesIO(img_data))
                        img = img.resize((256, 256), Image.Resampling.LANCZOS)
                        img_tk = ImageTk.PhotoImage(img)
                        self.result_label.config(image=img_tk, text="")
                        self.result_label.image = img_tk
                        
                        self.info_text.insert(tk.END, f"✅ Генерация успешна!\n")
                        self.last_generated_image = img
                    else:
                        self.info_text.insert(tk.END, f"❌ Ошибка: {data}\n")
                else:
                    self.info_text.insert(tk.END, f"❌ Ошибка API: {response.status_code}\n")
                    
            except Exception as e:
                self.info_text.insert(tk.END, f"❌ Ошибка: {str(e)}\n")
                messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
            finally:
                self.progress.stop()
        
        threading.Thread(target=generate, daemon=True).start()
    
    def test_generation(self):
        """Тестовая генерация случайного аватара"""
        def test():
            self.progress.start()
            self.result_label.config(text="Генерация...")
            self.info_text.insert(tk.END, "\n🔄 Генерация случайного аватара...\n")
            
            try:
                response = requests.get(f"{self.api_url}/test", timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        img_data = base64.b64decode(data['image'])
                        img = Image.open(BytesIO(img_data))
                        img = img.resize((256, 256), Image.Resampling.LANCZOS)
                        img_tk = ImageTk.PhotoImage(img)
                        self.result_label.config(image=img_tk, text="")
                        self.result_label.image = img_tk
                        
                        self.info_text.insert(tk.END, "✅ Случайный аватар сгенерирован\n")
                        self.last_generated_image = img
                    else:
                        self.info_text.insert(tk.END, f"❌ Ошибка: {data}\n")
                else:
                    self.info_text.insert(tk.END, f"❌ Ошибка API: {response.status_code}\n")
                    
            except Exception as e:
                self.info_text.insert(tk.END, f"❌ Ошибка: {str(e)}\n")
                messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
            finally:
                self.progress.stop()
        
        threading.Thread(target=test, daemon=True).start()
    
    def save_result(self):
        """Сохранение сгенерированного аватара"""
        if hasattr(self, 'last_generated_image') and self.last_generated_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            )
            if file_path:
                try:
                    self.last_generated_image.save(file_path)
                    messagebox.showinfo("Успех", f"Изображение сохранено:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить: {e}")
        else:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте аватар")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("Запуск GUI...")
    app = AvatarGeneratorGUI()
    app.run()