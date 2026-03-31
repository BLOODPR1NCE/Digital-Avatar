# db_extractor.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class PostgresDataExtractor:
    def __init__(self, host='localhost', port=5432, database='postgres', user='postgres', password='postgres'):
        self.conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        self.engine = None
        
    def connect(self):
        self.engine = create_engine(self.conn_str)
        print("✅ Успешное подключение к PostgreSQL")
        return True
    
    def get_table_list(self):
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """
        try:
            tables = pd.read_sql(query, self.engine)
            print(f"\n📋 Найденные таблицы:")
            for table in tables['table_name']:
                print(f"   - {table}")
            return tables['table_name'].tolist()
        except Exception as e:
            print(f"Ошибка получения списка таблиц: {e}")
            return []
    
    def extract_and_save_table(self, table_name):
        print(f"\n📊 Извлечение таблицы: {table_name}")
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        
        csv_path = self.data_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
            
        print(f"   ✅ Записей: {len(df)}, столбцов: {len(df.columns)}")
        print(f"   💾 Сохранено в {csv_path}")
            
        return df
    
    def create_visualizations(self, df, table_name):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Определяем количество графиков
        n_cols = len(numeric_cols)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'Анализ данных: {table_name}', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols[:n_rows*n_cols]):
            ax = axes[idx]
            
            # Гистограмма распределения
            ax.hist(df[col].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_title(f'Распределение: {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Частота')
            ax.legend(fontsize=8)
        
        # Скрываем лишние подграфики
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / f'{table_name}_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Графики сохранены в {self.data_dir}/{table_name}_analysis.png")
        
        # Корреляционная матрица для числовых колонок (если их больше 1)
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1, ax=ax)
            ax.set_title(f'Корреляционная матрица: {table_name}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.data_dir / f'{table_name}_correlation.png', dpi=150, bbox_inches='tight')
            plt.show()
            print(f"📊 Корреляционная матрица сохранена в {self.data_dir}/{table_name}_correlation.png")
    
    def extract_all_data(self):
        if not self.engine:
            if not self.connect():
                return False
        
        tables = self.get_table_list()
        
        all_data = {}
        
        for table in tables:
            df = self.extract_and_save_table(table)
            if df is not None:
                all_data[table] = df
                self.create_visualizations(df, table)
        
        return all_data
    
    def disconnect(self):
        if self.engine:
            self.engine.dispose()
            print("🔌 Соединение с БД закрыто")

if __name__ == "__main__":
    print("🚀 Запуск экстрактора данных из PostgreSQL")
    print("="*50)
    
    extractor = PostgresDataExtractor()
    
    if extractor.connect():
        all_data = extractor.extract_all_data()
        
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
        print("❌ Не удалось подключиться к БД. Проверьте параметры подключения.")