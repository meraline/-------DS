"""
train_main.py - Главный файл для запуска процесса обучения модели RWKV
"""
from train_process import train_poker_model

if __name__ == "__main__":
    import argparse
    
    # Аргументы командной строки с установленными значениями по умолчанию
    parser = argparse.ArgumentParser(description="Обучение RWKV модели на покерных данных")
    parser.add_argument("--file", type=str, default="/home/tofan/data1/csv/split_data/buyin_type_MTT_1000.csv", help="Путь к CSV-файлу с данными")
    parser.add_argument("--output", type=str, default="/home/tofan/Документы/GitLab_grace/ДИПЛОМ DS/evaluation_results/model_dir", help="Директория для сохранения результатов")
    parser.add_argument("--hidden_dim", type=int, default=192, help="Размерность скрытого состояния")
    parser.add_argument("--num_layers", type=int, default=6, help="Количество слоев RWKV")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох обучения")
    parser.add_argument("--lr", type=float, default=0.0001, help="Скорость обучения")
    parser.add_argument("--max_rows", type=int, default=1500000, help="Максимальное число строк для загрузки")
    
    args = parser.parse_args()
    
    print("=== Запуск обучения модели RWKV для покерных данных ===")
    print("Параметры:")
    print(f"  Файл данных: {args.file}")
    print(f"  Директория вывода: {args.output}")
    print(f"  Размерность скрытого состояния: {args.hidden_dim}")
    print(f"  Количество слоев: {args.num_layers}")
    print(f"  Эпохи: {args.epochs}")
    print(f"  Скорость обучения: {args.lr}")
    print(f"  Максимальное число строк: {args.max_rows if args.max_rows else 'все'}")
    print("=" * 60)
    
    # Запуск процесса обучения
    result = train_poker_model(
        file_path=args.file,
        output_dir=args.output,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_rows=args.max_rows
    )
    
    print("=" * 60)
    print(f"Обучение завершено. Модель сохранена в: {result['output_dir']}")
    print(f"Файлы модели:")
    print(f"  - Веса модели: {result['model_path']}")
    print(f"  - Скейлер для нормализации: {result['scaler_path']}")
    print(f"  - Метаданные модели: {result['model_info_path']}")
    print("=" * 60)