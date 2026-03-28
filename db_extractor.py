# db_extractor.py
import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class PostgresDataExtractor:
    def __init__(self, host='localhost', port=5432, database='postgres', 
                 user='postgres', password='postgres'):
        """
        Инициализация экстрактора данных из PostgreSQL
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.engine = None
        self.connection = None
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        
    def connect(self):
        """Установка соединения с БД"""
        try:
            # Создание connection string для SQLAlchemy
            conn_str = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
            self.engine = create_engine(conn_str)
            self.connection = self.engine.connect()
            print("✅ Успешное подключение к PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def get_table_list(self):
        """Получение списка таблиц в БД"""
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """
        try:
            tables = pd.read_sql(query, self.connection)
            print(f"\n📋 Найденные таблицы:")
            for table in tables['table_name']:
                print(f"   - {table}")
            return tables['table_name'].tolist()
        except Exception as e:
            print(f"Ошибка получения списка таблиц: {e}")
            return []
    
    def get_table_info(self, table_name):
        """Получение информации о структуре таблицы"""
        query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        try:
            info = pd.read_sql(query, self.connection)
            return info
        except Exception as e:
            print(f"Ошибка получения информации о таблице {table_name}: {e}")
            return None
    
    def extract_table_data(self, table_name, limit=None):
        """Извлечение данных из таблицы"""
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            start_time = time.time()
            df = pd.read_sql(query, self.connection)
            elapsed_time = time.time() - start_time
            
            print(f"\n📊 Таблица '{table_name}':")
            print(f"   - Записей: {len(df)}")
            print(f"   - Столбцов: {len(df.columns)}")
            print(f"   - Время загрузки: {elapsed_time:.2f} сек")
            print(f"   - Размер в памяти: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            return df
        except Exception as e:
            print(f"Ошибка извлечения данных из {table_name}: {e}")
            return None
    
    def get_table_statistics(self, df, table_name):
        """Получение статистики по таблице"""
        stats = {
            'table_name': table_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {},
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Статистика для числовых столбцов
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            stats['columns'][col] = {
                'type': 'numeric',
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None,
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None,
                'unique_count': df[col].nunique()
            }
        
        # Статистика для категориальных столбцов
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            value_counts = df[col].value_counts().head(10)
            stats['columns'][col] = {
                'type': 'categorical',
                'unique_count': df[col].nunique(),
                'top_values': value_counts.to_dict(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None
            }
        
        return stats
    
    def extract_all_data(self):
        """Извлечение всех данных из БД"""
        if not self.connection:
            if not self.connect():
                return None
        
        tables = self.get_table_list()
        if not tables:
            print("Таблицы не найдены")
            return None
        
        all_data = {}
        all_stats = {}
        
        for table in tables:
            print(f"\n{'='*50}")
            print(f"Извлечение данных из таблицы: {table}")
            print('='*50)
            
            # Получаем информацию о структуре
            table_info = self.get_table_info(table)
            if table_info is not None:
                print(f"\nСтруктура таблицы:")
                print(table_info.to_string(index=False))
            
            # Извлекаем данные
            df = self.extract_table_data(table)
            if df is not None:
                all_data[table] = df
                
                # Получаем статистику
                stats = self.get_table_statistics(df, table)
                all_stats[table] = stats
                
                # Сохраняем данные
                self.save_data(df, table)
        
        # Сохраняем статистику
        self.save_statistics(all_stats)
        
        return all_data, all_stats
    
    def save_data(self, df, table_name):
        """Сохранение данных в файл"""
        # CSV
        csv_path = self.data_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 Данные сохранены в {csv_path}")
        
        # Parquet (более эффективный формат)
        parquet_path = self.data_dir / f"{table_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"💾 Данные сохранены в {parquet_path}")
    
    def save_statistics(self, stats):
        """Сохранение статистики"""
        stats_path = self.data_dir / "database_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n📊 Статистика сохранена в {stats_path}")
    
    def disconnect(self):
        """Закрытие соединения"""
        if self.connection:
            self.connection.close()
            print("🔌 Соединение с БД закрыто")

def create_sample_data():
    """Создание тестовых данных в PostgreSQL (для демонстрации)"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password='postgres'
        )
        cursor = conn.cursor()
        
        # Создание тестовых таблиц
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                username VARCHAR(100),
                email VARCHAR(255),
                age INTEGER,
                registration_date DATE,
                country VARCHAR(50)
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_recordings (
                recording_id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id),
                recording_date TIMESTAMP,
                duration FLOAT,
                emotion VARCHAR(50),
                file_path VARCHAR(500)
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facial_landmarks (
                landmark_id SERIAL PRIMARY KEY,
                recording_id INTEGER REFERENCES audio_recordings(recording_id),
                frame_number INTEGER,
                point_index INTEGER,
                x_coordinate FLOAT,
                y_coordinate FLOAT
            );
        """)
        
        # Вставка тестовых данных
        cursor.execute("""
            INSERT INTO users (username, email, age, registration_date, country)
            VALUES 
                ('john_doe', 'john@example.com', 28, '2023-01-15', 'USA'),
                ('jane_smith', 'jane@example.com', 32, '2023-02-20', 'UK'),
                ('bob_wilson', 'bob@example.com', 25, '2023-03-10', 'Canada'),
                ('alice_brown', 'alice@example.com', 35, '2023-04-05', 'Australia'),
                ('charlie_davis', 'charlie@example.com', 29, '2023-05-12', 'USA')
            ON CONFLICT DO NOTHING;
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Тестовые данные созданы")
        
    except Exception as e:
        print(f"Ошибка создания тестовых данных: {e}")

if __name__ == "__main__":
    print("🚀 Запуск экстрактора данных из PostgreSQL")
    print("="*50)
    
    # Создание тестовых данных (для демонстрации)
    create_sample_data()
    
    # Инициализация экстрактора
    extractor = PostgresDataExtractor()
    
    # Подключение и извлечение данных
    if extractor.connect():
        all_data, all_stats = extractor.extract_all_data()
        
        if all_data:
            print("\n" + "="*50)
            print("Итоговый отчет:")
            print("="*50)
            for table_name, df in all_data.items():
                print(f"\n📁 {table_name}:")
                print(f"   - Записей: {len(df)}")
                print(f"   - Столбцов: {len(df.columns)}")
                print(f"   - Столбцы: {', '.join(df.columns.tolist())}")
        
        extractor.disconnect()
    else:
        print("Не удалось подключиться к БД. Проверьте параметры подключения.")