# query_optimization.py
import psycopg2
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
        """Установка соединения"""
        try:
            conn_str = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
            self.engine = create_engine(conn_str)
            self.connection = self.engine.connect()
            print("✅ Подключение к PostgreSQL установлено")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def create_test_tables(self):
        """Создание тестовых таблиц для демонстрации оптимизации"""
        print("\n📋 Создание тестовых таблиц...")
        
        # Создание таблиц
        create_tables_sql = """
        -- Таблица пользователей
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(100),
            email VARCHAR(255),
            age INTEGER,
            registration_date DATE,
            country VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Таблица аудиозаписей
        CREATE TABLE IF NOT EXISTS audio_recordings (
            recording_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            recording_date TIMESTAMP,
            duration FLOAT,
            emotion VARCHAR(50),
            intensity VARCHAR(20),
            file_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Таблица транзакций (для демонстрации)
        CREATE TABLE IF NOT EXISTS user_transactions (
            transaction_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            amount DECIMAL(10,2),
            transaction_date DATE,
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Создание индексов для оптимизации
        CREATE INDEX IF NOT EXISTS idx_audio_recordings_user_id ON audio_recordings(user_id);
        CREATE INDEX IF NOT EXISTS idx_audio_recordings_emotion ON audio_recordings(emotion);
        CREATE INDEX IF NOT EXISTS idx_user_transactions_user_id ON user_transactions(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_transactions_date ON user_transactions(transaction_date);
        """
        
        try:
            self.connection.execute(text(create_tables_sql))
            self.connection.commit()
            print("✅ Таблицы созданы")
            return True
        except Exception as e:
            print(f"Ошибка создания таблиц: {e}")
            return False
    
    def populate_test_data(self, num_users=1000, num_recordings=50000):
        """Заполнение тестовыми данными"""
        print(f"\n📊 Заполнение тестовыми данными ({num_users} пользователей, {num_recordings} записей)...")
        
        import numpy as np
        from datetime import datetime, timedelta
        
        try:
            # Очистка существующих данных
            self.connection.execute(text("TRUNCATE TABLE user_transactions CASCADE"))
            self.connection.execute(text("TRUNCATE TABLE audio_recordings CASCADE"))
            self.connection.execute(text("TRUNCATE TABLE users CASCADE"))
            self.connection.commit()
            
            # Вставка пользователей
            users = []
            for i in range(num_users):
                users.append({
                    'username': f'user_{i}',
                    'email': f'user_{i}@example.com',
                    'age': np.random.randint(18, 80),
                    'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Japan']),
                    'registration_date': datetime.now() - timedelta(days=np.random.randint(0, 1000))
                })
            
            users_df = pd.DataFrame(users)
            users_df.to_sql('users', self.engine, if_exists='append', index=False)
            print(f"   ✅ Добавлено {len(users_df)} пользователей")
            
            # Вставка аудиозаписей
            emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgust']
            intensities = ['normal', 'strong']
            
            recordings = []
            for i in range(num_recordings):
                user_id = np.random.randint(1, num_users + 1)
                recordings.append({
                    'user_id': user_id,
                    'recording_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                    'duration': np.random.uniform(1.0, 10.0),
                    'emotion': np.random.choice(emotions),
                    'intensity': np.random.choice(intensities),
                    'file_path': f'/recordings/rec_{i}.wav'
                })
            
            # Вставка пачками
            batch_size = 10000
            for i in range(0, len(recordings), batch_size):
                batch = recordings[i:i+batch_size]
                batch_df = pd.DataFrame(batch)
                batch_df.to_sql('audio_recordings', self.engine, if_exists='append', index=False)
                print(f"   ✅ Добавлено записей: {min(i+batch_size, len(recordings))}")
            
            # Вставка транзакций
            transactions = []
            for i in range(num_recordings // 10):  # 5000 транзакций
                user_id = np.random.randint(1, num_users + 1)
                transactions.append({
                    'user_id': user_id,
                    'amount': np.random.uniform(10, 500),
                    'transaction_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                    'status': np.random.choice(['completed', 'pending', 'failed'], p=[0.8, 0.15, 0.05])
                })
            
            transactions_df = pd.DataFrame(transactions)
            transactions_df.to_sql('user_transactions', self.engine, if_exists='append', index=False)
            print(f"   ✅ Добавлено {len(transactions_df)} транзакций")
            
            self.connection.commit()
            print("✅ Данные успешно добавлены")
            return True
            
        except Exception as e:
            print(f"Ошибка заполнения данных: {e}")
            return False
    
    def execute_query(self, query, description):
        """Выполнение запроса с замером времени"""
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
        """Оптимизация запроса 1: Анализ активности пользователей с оконными функциями"""
        print("\n" + "="*60)
        print("ЗАПРОС 1: Анализ активности пользователей")
        print("="*60)
        
        # Неоптимизированный запрос
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
        
        # Оптимизированный запрос
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
        
        # Сравнение
        if time_slow and time_opt:
            improvement = (time_slow - time_opt) / time_slow * 100
            print(f"\n📈 Улучшение: {improvement:.1f}% быстрее")
            return time_slow, time_opt, improvement
        
        return None, None, None
    
    def optimize_query_2(self):
        """Оптимизация запроса 2: Ранжирование пользователей с оконными функциями"""
        print("\n" + "="*60)
        print("ЗАПРОС 2: Ранжирование пользователей")
        print("="*60)
        
        # Неоптимизированный запрос
        print("\n🔴 НЕОПТИМИЗИРОВАННЫЙ ЗАПРОС:")
        slow_query = """
        SELECT 
            u.user_id,
            u.username,
            u.country,
            COUNT(r.recording_id) as total_recordings,
            (
                SELECT AVG(duration)
                FROM audio_recordings r2
                WHERE r2.user_id = u.user_id
                AND r2.emotion = 'happy'
            ) as avg_happy_duration,
            RANK() OVER (ORDER BY COUNT(r.recording_id) DESC) as recording_rank
        FROM users u
        LEFT JOIN audio_recordings r ON u.user_id = r.user_id
        GROUP BY u.user_id, u.username, u.country
        HAVING COUNT(r.recording_id) > 0
        ORDER BY recording_rank
        LIMIT 50;
        """
        
        print("Запрос использует подзапрос для вычисления среднего")
        time_slow, _ = self.execute_query(slow_query, "Неоптимизированный запрос")
        
        # Оптимизированный запрос
        print("\n🟢 ОПТИМИЗИРОВАННЫЙ ЗАПРОС:")
        optimized_query = """
        WITH user_stats AS (
            SELECT 
                user_id,
                COUNT(*) as total_recordings,
                AVG(CASE WHEN emotion = 'happy' THEN duration END) as avg_happy_duration
            FROM audio_recordings
            GROUP BY user_id
        )
        SELECT 
            u.user_id,
            u.username,
            u.country,
            us.total_recordings,
            us.avg_happy_duration,
            RANK() OVER (ORDER BY us.total_recordings DESC) as recording_rank
        FROM users u
        INNER JOIN user_stats us ON u.user_id = us.user_id
        ORDER BY recording_rank
        LIMIT 50;
        """
        
        print("Предварительная агрегация с CASE выражением")
        time_opt, _ = self.execute_query(optimized_query, "Оптимизированный запрос")
        
        if time_slow and time_opt:
            improvement = (time_slow - time_opt) / time_slow * 100
            print(f"\n📈 Улучшение: {improvement:.1f}% быстрее")
            return time_slow, time_opt, improvement
        
        return None, None, None
    
    def optimize_query_3(self):
        """Оптимизация запроса 3: Анализ временных рядов"""
        print("\n" + "="*60)
        print("ЗАПРОС 3: Анализ временных рядов")
        print("="*60)
        
        # Неоптимизированный запрос
        print("\n🔴 НЕОПТИМИЗИРОВАННЫЙ ЗАПРОС:")
        slow_query = """
        SELECT 
            DATE(r.recording_date) as date,
            u.country,
            COUNT(*) as total_recordings,
            SUM(COUNT(*)) OVER (
                PARTITION BY u.country 
                ORDER BY DATE(r.recording_date)
            ) as cumulative_recordings
        FROM audio_recordings r
        JOIN users u ON r.user_id = u.user_id
        WHERE r.recording_date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY DATE(r.recording_date), u.country
        ORDER BY date DESC, u.country;
        """
        
        print("Запрос с оконной функцией без оптимизации")
        time_slow, _ = self.execute_query(slow_query, "Неоптимизированный запрос")
        
        # Оптимизированный запрос
        print("\n🟢 ОПТИМИЗИРОВАННЫЙ ЗАПРОС:")
        optimized_query = """
        WITH daily_stats AS (
            SELECT 
                DATE(r.recording_date) as date,
                u.country,
                COUNT(*) as total_recordings
            FROM audio_recordings r
            JOIN users u ON r.user_id = u.user_id
            WHERE r.recording_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY DATE(r.recording_date), u.country
        )
        SELECT 
            date,
            country,
            total_recordings,
            SUM(total_recordings) OVER (
                PARTITION BY country 
                ORDER BY date
            ) as cumulative_recordings
        FROM daily_stats
        ORDER BY date DESC, country;
        """
        
        print("Предварительная агрегация с CTE")
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
        """Запуск всех оптимизаций"""
        print("\n" + "="*70)
        print("ОПТИМИЗАЦИЯ ЗАПРОСОВ К БАЗЕ ДАННЫХ")
        print("="*70)
        
        if not self.connect():
            return
        
        # Создание таблиц и данных
        if self.create_test_tables():
            self.populate_test_data(num_users=1000, num_recordings=50000)
        
        # Создание индексов
        self.create_indexes()
        
        results = []
        
        # Запуск оптимизации запросов
        results.append(self.optimize_query_1())
        results.append(self.optimize_query_2())
        results.append(self.optimize_query_3())
        
        # Визуализация результатов
        self.plot_results(results)
        
        self.connection.close()
        print("\n🔌 Соединение закрыто")
    
    def plot_results(self, results):
        """Визуализация результатов оптимизации"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График сравнения времени выполнения
        queries = ['Query 1', 'Query 2', 'Query 3']
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
            
            # График улучшения
            improvements = [r[2] for r in results if r[2] is not None]
            if improvements:
                axes[1].bar(queries[:len(improvements)], improvements, color='blue', alpha=0.7)
                axes[1].set_xlabel('Запросы', fontsize=12)
                axes[1].set_ylabel('Улучшение (%)', fontsize=12)
                axes[1].set_title('Процент улучшения после оптимизации', fontsize=12, fontweight='bold')
                axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[1].grid(True, alpha=0.3)
                
                for i, imp in enumerate(improvements):
                    axes[1].text(i, imp + 1, f'{imp:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'query_optimization_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n📊 Результаты сохранены в {self.results_dir / 'query_optimization_results.png'}")
        
        # Сохранение отчета
        self.save_report(results)
    
    def save_report(self, results):
        """Сохранение отчета"""
        report_path = self.results_dir / 'optimization_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ОТЧЕТ ПО ОПТИМИЗАЦИИ ЗАПРОСОВ\n")
            f.write("="*70 + "\n\n")
            
            for i, result in enumerate(results, 1):
                if result[0] and result[1]:
                    f.write(f"Запрос {i}:\n")
                    f.write(f"  - Время выполнения (неоптимизированный): {result[0]:.4f} сек\n")
                    f.write(f"  - Время выполнения (оптимизированный): {result[1]:.4f} сек\n")
                    f.write(f"  - Улучшение: {result[2]:.1f}%\n\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("ПРИМЕНЕННЫЕ ОПТИМИЗАЦИИ:\n")
            f.write("="*70 + "\n")
            f.write("1. Использование CTE (Common Table Expressions) для предварительной агрегации\n")
            f.write("2. Замена подзапросов на JOIN с предварительно агрегированными данными\n")
            f.write("3. Использование CASE выражений вместо отдельных подзапросов\n")
            f.write("4. Создание индексов на часто используемых полях\n")
            f.write("5. Оптимизация оконных функций через предварительную фильтрацию\n")
            f.write("6. Уменьшение количества сканирований таблиц\n")
        
        print(f"📄 Отчет сохранен в {report_path}")

if __name__ == "__main__":
    optimizer = QueryOptimizer()
    optimizer.run_all_optimizations()