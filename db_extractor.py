# db_extractor.py
import pandas as pd
from pathlib import Path
import time
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class PostgresDataExtractor:
    def __init__(self, host='localhost', port=5432, database='postgres', 
                 user='postgres', password='postgres'):
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
        try:
            conn_str = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
            self.engine = create_engine(conn_str)
            self.connection = self.engine.connect()
            print("✅ Успешное подключение к PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def get_table_list(self):
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
    
    def extract_all_data(self):
        if not self.connection:
            if not self.connect():
                return None
        
        tables = self.get_table_list()
        if not tables:
            print("Таблицы не найдены")
            return None
        
        all_data = {}
        
        for table in tables:
            print(f"\n{'='*50}")
            print(f"Извлечение данных из таблицы: {table}")
            print('='*50)
            
            table_info = self.get_table_info(table)
            if table_info is not None:
                print(f"\nСтруктура таблицы:")
                print(table_info.to_string(index=False))
            
            df = self.extract_table_data(table)
            if df is not None:
                all_data[table] = df
                self.save_data(df, table)
        
        return all_data, all_stats
    
    def save_data(self, df, table_name):
        csv_path = self.data_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 Данные сохранены в {csv_path}")
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("🔌 Соединение с БД закрыто")


if __name__ == "__main__":
    print("🚀 Запуск экстрактора данных из PostgreSQL")
    print("="*50)
    
    extractor = PostgresDataExtractor()
    
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