from graphviz import Digraph

# Создаем диаграмму с детальной информацией о данных
dot = Digraph(comment='Detailed RWKV Poker Pipeline', format='png')
dot.attr(rankdir='LR', size='12,10', dpi='300', nodesep='0.5', ranksep='1.5')
dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='14', margin='0.3,0.2')
dot.attr('edge', penwidth='1.5', arrowsize='1.2', fontname='Arial', fontsize='12')

# Определяем цвета для каждого этапа
colors = {
    'raw': '#D6EAF8',       # Светло-голубой
    'extract': '#D5F5E3',   # Светло-зеленый
    'process': '#FCF3CF',   # Светло-желтый
    'train': '#FAE5D3',     # Светло-оранжевый
    'predict': '#EBDEF0',   # Светло-фиолетовый
    'data': '#F2F3F4',      # Светло-серый для данных
    'code': '#E5E8E8'       # Серый для кода/модулей
}

# Создаем кластеры для каждого этапа
with dot.subgraph(name='cluster_raw') as c:
    c.attr(label='ИСХОДНЫЕ ДАННЫЕ', style='filled', fillcolor=colors['raw'], fontcolor='#2C3E50', fontsize='16')
    c.node('ohh_files', '.ohh файлы', fillcolor=colors['data'])
    c.node('json_data', 'JSON структуры\nпокерных раздач', fillcolor=colors['data'])
    c.edge('ohh_files', 'json_data')

with dot.subgraph(name='cluster_extract') as c:
    c.attr(label='ЭТАП 1: ИЗВЛЕЧЕНИЕ', style='filled', fillcolor=colors['extract'], fontcolor='#2C3E50', fontsize='16')
    c.node('main_py', 'main.py', shape='folder', fillcolor=colors['code'])
    c.node('parse_module', 'parse_ohh_new()', shape='component', fillcolor=colors['code'])
    c.node('raw_csv', 'Сырые CSV\nс базовой структурой', fillcolor=colors['data'])
    c.edge('main_py', 'parse_module')
    c.edge('parse_module', 'raw_csv')

with dot.subgraph(name='cluster_process') as c:
    c.attr(label='ЭТАП 2: ОБРАБОТКА', style='filled', fillcolor=colors['process'], fontcolor='#2C3E50', fontsize='16')
    c.node('splyt_module', 'splyt_buyin.py', shape='folder', fillcolor=colors['code'])
    c.node('transform_module', 'transform_data.py', shape='folder', fillcolor=colors['code'])
    c.node('filtered_data', 'Отфильтрованные\nданные по buy-in', fillcolor=colors['data'])
    c.node('features_data', 'Данные с\nдоп. признаками', fillcolor=colors['data'])
    c.node('normalized_data', 'Нормализованные\nданные', fillcolor=colors['data'])
    c.edge('splyt_module', 'filtered_data')
    c.edge('filtered_data', 'transform_module')
    c.edge('transform_module', 'features_data')
    c.edge('features_data', 'normalized_data')

with dot.subgraph(name='cluster_train') as c:
    c.attr(label='ЭТАП 3: ОБУЧЕНИЕ', style='filled', fillcolor=colors['train'], fontcolor='#2C3E50', fontsize='16')
    c.node('train_main', 'train_main.py', shape='folder', fillcolor=colors['code'])
    c.node('model_py', 'model.py\n(RWKV блоки)', shape='component', fillcolor=colors['code'])
    c.node('train_process', 'train_process.py', shape='component', fillcolor=colors['code'])
    c.node('trained_model', 'Обученная модель\nRWKV', fillcolor=colors['data'])
    c.edge('train_main', 'train_process')
    c.edge('model_py', 'train_process')
    c.edge('train_process', 'trained_model')

with dot.subgraph(name='cluster_predict') as c:
    c.attr(label='ЭТАП 4: ПРОГНОЗИРОВАНИЕ', style='filled', fillcolor=colors['predict'], fontcolor='#2C3E50', fontsize='16')
    c.node('evaluate_model', 'evaluate_model.py', shape='folder', fillcolor=colors['code'])
    c.node('visualizations', 'visualizations.py', shape='component', fillcolor=colors['code'])
    c.node('predictions', 'Предсказанные\nдействия', fillcolor=colors['data'])
    c.node('metrics', 'Метрики качества\n(матрица ошибок)', fillcolor=colors['data'])
    c.node('tsne_vis', 't-SNE\nвизуализация', fillcolor=colors['data'])
    c.edge('evaluate_model', 'predictions')
    c.edge('predictions', 'metrics')
    c.edge('visualizations', 'tsne_vis')
    c.edge('predictions', 'tsne_vis')

# Связи между кластерами
dot.edge('json_data', 'main_py')
dot.edge('raw_csv', 'splyt_module')
dot.edge('normalized_data', 'train_main')
dot.edge('trained_model', 'evaluate_model')

# Добавляем аннотации с деталями процессов
dot.node('parse_details', 'Извлечение:\n• Парсинг JSON структур\n• Выделение последовательности действий\n• Фиксация данных о картах, ставках, позициях', 
         shape='note', fillcolor='white')
dot.node('process_details', 'Обработка:\n• Фильтрация по типу турнира\n• Создание >20 дополнительных признаков\n• Нормализация и масштабирование', 
         shape='note', fillcolor='white')
dot.node('train_details', 'Обучение:\n• Инициализация модели RWKV\n• Оптимизация на обучающей выборке\n• Валидация и сохранение лучшей модели', 
         shape='note', fillcolor='white')
dot.node('predict_details', 'Прогнозирование:\n• Предсказание действий в тестовых ситуациях\n• Визуализация результатов\n• Оценка точности по классам', 
         shape='note', fillcolor='white')

# Связи с аннотациями
dot.edge('parse_module', 'parse_details', style='dashed', dir='none')
dot.edge('transform_module', 'process_details', style='dashed', dir='none')
dot.edge('train_process', 'train_details', style='dashed', dir='none')
dot.edge('evaluate_model', 'predict_details', style='dashed', dir='none')

# Добавляем подпись
dot.attr(label='Детальный поток данных в системе предсказания покерных действий с RWKV\n', 
         fontsize='18', fontname='Arial')

# Сохраняем диаграмму
dot.render('rwkv_poker_pipeline_detailed', view=True, cleanup=True)

# Вернем код для создания SVG версии с другими настройками форматирования
dot.format = 'svg'
dot.render('rwkv_poker_pipeline_detailed_svg', view=False, cleanup=True)

print("Детальная схема создана в файлах 'rwkv_poker_pipeline_detailed.png' и 'rwkv_poker_pipeline_detailed_svg.svg'")