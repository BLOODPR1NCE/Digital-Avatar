# query_optimization.py
import pandas as pd
import time
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
from pathlib import Path

class QueryOptimizer:
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
        self.results_dir = Path("./query_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def connect(self):
        try:
            conn_str = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
            self.engine = create_engine(conn_str)
            self.connection = self.engine.connect()
            print("✅ Подключение к PostgreSQL установлено")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def execute_query(self, query, description):
        try:
            start_time = time.time()
            result = pd.read_sql(query, self.connection)
            elapsed_time = time.time() - start_time
            
            print(f"   ⏱️ Время выполнения: {elapsed_time:.4f} сек")
            print(f"   📊 Результатов: {len(result)}")
            
            return elapsed_time, result
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return None, None
    
    def optimize_query_1(self):
        print("\n" + "="*60)
        print("ЗАПРОС 1: Анализ активности пользователей")
        print("="*60)
        
        print("\n🔴 НЕОПТИМИЗИРОВАННЫЙ ЗАПРОС:")
        slow_query = """
        SELECT 
            u.user_id,
            u.username,
            u.country,
            COUNT(r.recording_id) as total_recordings,
            AVG(r.duration) as avg_duration,
            (
                SELECT COUNT(*)
                FROM user_transactions t
                WHERE t.user_id = u.user_id
                AND t.transaction_date > CURRENT_DATE - INTERVAL '30 days'
            ) as recent_transactions
        FROM users u
        LEFT JOIN audio_recordings r ON u.user_id = r.user_id
        WHERE u.age BETWEEN 25 AND 35
        GROUP BY u.user_id, u.username, u.country
        ORDER BY total_recordings DESC
        LIMIT 100;
        """
        
        print("Запрос использует подзапрос для каждого пользователя")
        time_slow, _ = self.execute_query(slow_query, "Неоптимизированный запрос")
        
        print("\n🟢 ОПТИМИЗИРОВАННЫЙ ЗАПРОС:")
        optimized_query = """
        WITH user_activity AS (
            SELECT 
                u.user_id,
                u.username,
                u.country,
                COUNT(r.recording_id) as total_recordings,
                AVG(r.duration) as avg_duration
            FROM users u
            LEFT JOIN audio_recordings r ON u.user_id = r.user_id
            WHERE u.age BETWEEN 25 AND 35
            GROUP BY u.user_id, u.username, u.country
        ),
        user_transactions_recent AS (
            SELECT 
                user_id,
                COUNT(*) as recent_transactions
            FROM user_transactions
            WHERE transaction_date > CURRENT_DATE - INTERVAL '30 days'
            GROUP BY user_id
        )
        SELECT 
            ua.*,
            COALESCE(utr.recent_transactions, 0) as recent_transactions
        FROM user_activity ua
        LEFT JOIN user_transactions_recent utr ON ua.user_id = utr.user_id
        ORDER BY ua.total_recordings DESC
        LIMIT 100;
        """
        
        print("Использование CTE и предварительной агрегации")
        time_opt, _ = self.execute_query(optimized_query, "Оптимизированный запрос")
        
        if time_slow and time_opt:
            improvement = (time_slow - time_opt) / time_slow * 100
            print(f"\n📈 Улучшение: {improvement:.1f}% быстрее")
            return time_slow, time_opt, improvement
        
        return None, None, None
    def create_indexes(self):
        """Создание индексов для оптимизации"""
        print("\n🔧 Создание индексов для оптимизации...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_audio_recordings_date ON audio_recordings(recording_date)",
            "CREATE INDEX IF NOT EXISTS idx_audio_recordings_emotion_date ON audio_recordings(emotion, recording_date)",
            "CREATE INDEX IF NOT EXISTS idx_users_age_country ON users(age, country)",
            "CREATE INDEX IF NOT EXISTS idx_user_transactions_date_user ON user_transactions(transaction_date, user_id)"
        ]
        
        for idx in indexes:
            try:
                self.connection.execute(text(idx))
                self.connection.commit()
                print(f"   ✅ {idx.split('ON')[1].split('(')[0]}")
            except Exception as e:
                print(f"   ⚠️ Ошибка: {e}")
    
    def run_all_optimizations(self):
        print("\n" + "="*70)
        print("ОПТИМИЗАЦИЯ ЗАПРОСОВ К БАЗЕ ДАННЫХ")
        print("="*70)
        
        if not self.connect():
            return
        
        # Создание индексов
        self.create_indexes()
        
        results = []
        
        # Запуск оптимизации запросов
        results.append(self.optimize_query_1())
        
        # Визуализация результатов
        self.plot_results(results)
        
        self.connection.close()
        print("\n🔌 Соединение закрыто")
    
    def plot_results(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        queries = ['Query 1']
        slow_times = []
        opt_times = []
        
        for result in results:
            if result[0] and result[1]:
                slow_times.append(result[0])
                opt_times.append(result[1])
        
        if slow_times and opt_times:
            x = range(len(slow_times))
            width = 0.35
            
            axes[0].bar([i - width/2 for i in x], slow_times, width, label='Неоптимизированный', color='red', alpha=0.7)
            axes[0].bar([i + width/2 for i in x], opt_times, width, label='Оптимизированный', color='green', alpha=0.7)
            axes[0].set_xlabel('Запросы', fontsize=12)
            axes[0].set_ylabel('Время выполнения (сек)', fontsize=12)
            axes[0].set_title('Сравнение времени выполнения запросов', fontsize=12, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(queries[:len(slow_times)])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'query_optimization_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n📊 Результаты сохранены в {self.results_dir / 'query_optimization_results.png'}")

if __name__ == "__main__":
    optimizer = QueryOptimizer()
    optimizer.run_all_optimizations()