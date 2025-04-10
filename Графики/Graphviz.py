from graphviz import Digraph

# Создаем диаграмму
dot = Digraph(comment='RWKV Poker Pipeline', format='png')
dot.attr(rankdir='TB', size='8,10', dpi='300')
dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='16', margin='0.3,0.2')
dot.attr('edge', penwidth='2.0', arrowsize='1.5', fontname='Arial', fontsize='14')

# Определяем цвета для каждого этапа
colors = {
    'raw': '#D6EAF8',  # Светло-голубой
    'extract': '#D5F5E3',  # Светло-зеленый
    'process': '#FCF3CF',  # Светло-желтый
    'train': '#FAE5D3',    # Светло-оранжевый
    'predict': '#EBDEF0'   # Светло-фиолетовый
}

# Добавляем узлы и связи

# Исходные данные
dot.node('raw_data', 'ИСХОДНЫЕ ДАННЫЕ\n(.ohh файлы)', fillcolor=colors['raw'], fontcolor='#2C3E50')

# Этап 1: Извлечение
dot.node('extract', 'ЭТАП 1: ИЗВЛЕЧЕНИЕ', fillcolor=colors['extract'], fontcolor='#2C3E50')
dot.node('extract_details', '• Парсинг JSON\n• Базовая структуризация', 
         shape='note', fillcolor='white', fontcolor='#2C3E50')

# Этап 2: Обработка
dot.node('process', 'ЭТАП 2: ОБРАБОТКА', fillcolor=colors['process'], fontcolor='#2C3E50')
dot.node('process_details', '• Фильтрация и обогащение\n• Создание признаков\n• Нормализация данных', 
         shape='note', fillcolor='white', fontcolor='#2C3E50')

# Этап 3: Обучение
dot.node('train', 'ЭТАП 3: ОБУЧЕНИЕ', fillcolor=colors['train'], fontcolor='#2C3E50')
dot.node('train_details', '• Модель RWKV\n• Оптимизация параметров\n• Валидация', 
         shape='note', fillcolor='white', fontcolor='#2C3E50')

# Этап 4: Прогнозирование
dot.node('predict', 'ЭТАП 4: ПРОГНОЗИРОВАНИЕ', fillcolor=colors['predict'], fontcolor='#2C3E50')
dot.node('predict_details', '• Предсказание действий\n• Визуализация результатов\n• Оценка точности', 
         shape='note', fillcolor='white', fontcolor='#2C3E50')

# Связи между основными этапами
dot.edge('raw_data', 'extract', color='#3498DB', label='')
dot.edge('extract', 'process', color='#3498DB', label='')
dot.edge('process', 'train', color='#3498DB', label='')
dot.edge('train', 'predict', color='#3498DB', label='')

# Связи с деталями
dot.edge('extract', 'extract_details', style='dashed', color='#7F8C8D', dir='none')
dot.edge('process', 'process_details', style='dashed', color='#7F8C8D', dir='none')
dot.edge('train', 'train_details', style='dashed', color='#7F8C8D', dir='none')
dot.edge('predict', 'predict_details', style='dashed', color='#7F8C8D', dir='none')

# Создаем невидимые связи для улучшения компоновки
dot.edge('extract_details', 'process_details', style='invis')
dot.edge('process_details', 'train_details', style='invis')
dot.edge('train_details', 'predict_details', style='invis')

# Добавляем подпись
dot.attr(label='Поток данных в системе предсказания покерных действий на основе RWKV\n', fontsize='20', fontname='Arial')

# Сохраняем диаграмму
dot.render('rwkv_poker_pipeline', view=True, cleanup=True)

# Вернем код для создания SVG версии с другими настройками форматирования
dot.format = 'svg'
dot.render('rwkv_poker_pipeline_svg', view=False, cleanup=True)

print("Схема создана в файлах 'rwkv_poker_pipeline.png' и 'rwkv_poker_pipeline_svg.svg'")