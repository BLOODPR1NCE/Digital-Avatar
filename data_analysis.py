# query_optimization.py
import pandas as pd
import numpy as np
import time
import sqlite3
import os

class QueryOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.conn = None
        
    def create_database(self):
        """Создание оптимизированной базы данных"""
        print("Создание оптимизированной базы данных...")
        
        # Загрузка данных
        df_attr = pd.read_excel(os.path.join(self.data_path, 'list_attr_celeba_orig.xlsx'), index_col=0)
        df_attr = df_attr.replace({-1: 0, 1: 1})
        
        df_bbox = pd.read_excel(os.path.join(self.data_path, 'list_bbox_celeba_orig.xlsx'), index_col=0)
        df_landmarks = pd.read_excel(os.path.join(self.data_path, 'list_landmarks_align_celeba_orig.xlsx'), index_col=0)
        df_partition = pd.read_excel(os.path.join(self.data_path, 'list_eval_partition_orig.xlsx'), index_col=0)
        
        # Создание SQLite базы данных
        self.conn = sqlite3.connect('celeba_optimized.db')
        
        # Сохранение таблиц с индексами
        df_attr.to_sql('attributes', self.conn, if_exists='replace')
        df_bbox.to_sql('bbox', self.conn, if_exists='replace')
        df_landmarks.to_sql('landmarks', self.conn, if_exists='replace')
        df_partition.to_sql('partition', self.conn, if_exists='replace')
        
        # Создание индексов для оптимизации
        cursor = self.conn.cursor()
        
        # Индексы для атрибутов
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_male ON attributes(Male)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_young ON attributes(Young)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_smiling ON attributes(Smiling)")
        
        # Составные индексы для частых комбинаций
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr_combo ON attributes(Male, Young, Smiling)")
        
        # Индексы для координат
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bbox_width ON bbox(width)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bbox_height ON bbox(height)")
        
        self.conn.commit()
        print("База данных создана с индексами")
    
    def compare_query_performance(self):
        """Сравнение производительности запросов"""
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ЗАПРОСОВ")
        print("=" * 60)
        
        # Загрузка данных в DataFrame для сравнения
        df_attr = pd.read_excel(os.path.join(self.data_path, 'list_attr_celeba_orig.xlsx'), index_col=0)
        df_attr = df_attr.replace({-1: 0, 1: 1})
        
        # Запросы для тестирования
        queries = [
            {
                'name': 'Фильтрация по полу и возрасту',
                'condition': lambda df: df[(df['Male'] == 1) & (df['Young'] == 1)],
                'sql': "SELECT * FROM attributes WHERE Male = 1 AND Young = 1"
            },
            {
                'name': 'Подсчет атрибутов',
                'condition': lambda df: df['Smiling'].value_counts(),
                'sql': "SELECT Smiling, COUNT(*) FROM attributes GROUP BY Smiling"
            },
            {
                'name': 'Сложная фильтрация',
                'condition': lambda df: df[(df['Male'] == 1) & (df['Young'] == 1) & (df['Smiling'] == 1)],
                'sql': "SELECT * FROM attributes WHERE Male = 1 AND Young = 1 AND Smiling = 1"
            }
        ]
        
        results = []
        
        for query in queries:
            print(f"\n{query['name']}:")
            
            # DataFrame (без оптимизации)
            start_time = time.time()
            result_df = query['condition'](df_attr)
            df_time = time.time() - start_time
            
            # SQLite (с индексами)
            start_time = time.time()
            cursor = self.conn.cursor()
            result_sql = cursor.execute(query['sql']).fetchall()
            sql_time = time.time() - start_time
            
            print(f"  DataFrame: {df_time:.4f} сек")
            print(f"  SQLite: {sql_time:.4f} сек")
            print(f"  Ускорение: {df_time/sql_time:.2f}x")
            
            results.append({
                'query': query['name'],
                'df_time': df_time,
                'sql_time': sql_time,
                'speedup': df_time/sql_time
            })
        
        # Создание отчета
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("СВОДНЫЙ ОТЧЕТ")
        print("=" * 60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def close(self):
        if self.conn:
            self.conn.close()

def main():
    data_path = r"C:\Users\prince\Downloads\archive"
    optimizer = QueryOptimizer(data_path)
    
    optimizer.create_database()
    results = optimizer.compare_query_performance()
    
    optimizer.close()
    
    print("\nОптимизация завершена!")

if __name__ == "__main__":
    main()