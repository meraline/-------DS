"""
train_main.py - Главный файл для запуска процесса обучения модели RWKV
"""
import sys
import warnings

warnings.filterwarnings('ignore',
                        category=RuntimeWarning,
                        message='Mean of empty slice')
from train_process import train_poker_model

if __name__ == "__main__":
    import argparse

    # Аргументы командной строки с установленными значениями по умолчанию
    parser = argparse.ArgumentParser(
        description="Обучение RWKV модели на покерных данных")
    parser.add_argument("--file",
                        type=str,
                        default="./data/buyin_type_MTT_500.csv",
                        help="Путь к CSV-файлу с данными")
    parser.add_argument("--output",
                        type=str,
                        default="./model_dir",
                        help="Директория для сохранения результатов")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=192,
                        help="Размерность скрытого состояния")
    parser.add_argument("--num_layers",
                        type=int,
                        default=6,
                        help="Количество слоев RWKV")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Количество эпох обучения")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0001,
                        help="Скорость обучения")
    parser.add_argument("--max_rows",
                        type=int,
                        default=1500000,
                        help="Максимальное число строк для загрузки")

    args = parser.parse_args()

    print("=== Запуск обучения модели RWKV для покерных данных ===")
    print("Параметры:")
    print(f"  Файл данных: {args.file}")
    print(f"  Директория вывода: {args.output}")
    print(f"  Размерность скрытого состояния: {args.hidden_dim}")
    print(f"  Количество слоев: {args.num_layers}")
    print(f"  Эпохи: {args.epochs}")
    print(f"  Скорость обучения: {args.lr}")
    print(
        f"  Максимальное число строк: {args.max_rows if args.max_rows else 'все'}"
    )
    print("=" * 60)

    # Запуск обучения первой модели (базовые действия)
    result_actions = train_poker_model(file_path=args.file,
                                       output_dir=args.output,
                                       hidden_dim=args.hidden_dim,
                                       num_layers=args.num_layers,
                                       epochs=args.epochs,
                                       learning_rate=args.lr,
                                       max_rows=args.max_rows)

    # Запуск обучения второй модели (размеры ставок)
    from train_size_model import train_size_model
    result_sizes = train_size_model(file_path=args.file,
                                    output_dir=args.output + "_size",
                                    hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers,
                                    epochs=args.epochs,
                                    learning_rate=args.lr,
                                    max_rows=args.max_rows)

    # Запуск обучения третьей модели (all-in)
    from train_allin_model import train_allin_model
    result_allin = train_allin_model(file_path=args.file,
                                     output_dir=args.output + "_allin",
                                     hidden_dim=args.hidden_dim,
                                     num_layers=args.num_layers,
                                     epochs=args.epochs,
                                     learning_rate=args.lr,
                                     max_rows=args.max_rows)

    print("=" * 60)
    print(
        f"Обучение завершено. Модель (действия) сохранена в: {result_actions['output_dir']}"
    )
    print(f"Файлы модели (действия):")
    print(f"  - Веса модели: {result_actions['model_path']}")
    print(f"  - Скейлер для нормализации: {result_actions['scaler_path']}")
    print(f"  - Метаданные модели: {result_actions['model_info_path']}")
    print("=" * 60)
    print(
        f"Обучение завершено. Модель (размеры ставок) сохранена в: {result_sizes['output_dir']}"
    )
    print(f"Файлы модели (размеры ставок):")
    print(f"  - Веса модели: {result_sizes['model_path']}")
    print(f"  - Скейлер для нормализации: {result_sizes['scaler_path']}")
    print(f"  - Метаданные модели: {result_sizes['model_info_path']}")
    print("=" * 60)

    # Запуск оценки модели
    print("\n=== Этап 1: Обучение модели ===")
    print("Обучение модели на тренировочных данных...")

    print("\n=== Этап 2: Валидация модели ===")
    from evaluate_model import main as evaluate_main
    sys.argv = [
        "evaluate_model.py", "--model_dir", args.output, "--test", args.file,
        "--output", args.output, "--max_rows",
        str(min(500000, args.max_rows))
    ]
    evaluate_main()

    print("\n=== Этап 3: Генерация визуализаций ===")
    print("Все визуализации сохранены в директории:", args.output)
    print(
        "Запустите 'python app.py' для просмотра результатов через веб-интерфейс"
    )
