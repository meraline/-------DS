from graphviz import Digraph

# Создаем диаграмму, оптимизированную для презентации
dot = Digraph(comment='Presentation RWKV Poker Pipeline', format='png')
dot.attr(rankdir='LR', size='10,4', dpi='300')
dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='18', 
         margin='0.4,0.3', height='1.2', width='3.5')
dot.attr('edge', penwidth='3.0', arrowsize='1.8', fontname='Arial', fontsize='16')

# Определяем цвета для этапов с высокой контрастностью и читаемостью
colors = {
    'raw': '#AED6F1',      # Голубой
    'extract': '#A3E4D7',  # Зеленый
    'process': '#F9E79F',  # Желтый
    'train': '#F5CBA7',    # Оранжевый
    'predict': '#D7BDE2'   # Фиолетовый
}

edge_color = '#34495E'  # Темный цвет для стрелок

# Основные блоки
dot.node('raw_data', 'ИСХОДНЫЕ ДАННЫЕ\n(.ohh файлы)', fillcolor=colors['raw'], fontcolor='#2C3E50')
dot.node('extract', 'ЭТАП 1\nИЗВЛЕЧЕНИЕ', fillcolor=colors['extract'], fontcolor='#2C3E50')
dot.node('process', 'ЭТАП 2\nОБРАБОТКА', fillcolor=colors['process'], fontcolor='#2C3E50')
dot.node('train', 'ЭТАП 3\nОБУЧЕНИЕ', fillcolor=colors['train'], fontcolor='#2C3E50')
dot.node('predict', 'ЭТАП 4\nПРОГНОЗИРОВАНИЕ', fillcolor=colors['predict'], fontcolor='#2C3E50')

# Связи между блоками
dot.edge('raw_data', 'extract', color=edge_color)
dot.edge('extract', 'process', color=edge_color)
dot.edge('process', 'train', color=edge_color)
dot.edge('train', 'predict', color=edge_color)

# Аннотации для потока данных
dot.edge('raw_data', 'extract', label='JSON структуры', fontcolor='#2C3E50')
dot.edge('extract', 'process', label='Сырые CSV', fontcolor='#2C3E50')
dot.edge('process', 'train', label='Признаки', fontcolor='#2C3E50')
dot.edge('train', 'predict', label='Модель RWKV', fontcolor='#2C3E50')

# Добавляем подпись
dot.attr(label='Поток данных в системе RWKV для покера', fontsize='24', fontname='Arial Bold')

# Сохраняем диаграмму
dot.render('rwkv_poker_pipeline_presentation', view=True, cleanup=True)

# Создаем SVG версию
dot.format = 'svg'
dot.render('rwkv_poker_pipeline_presentation_svg', view=False, cleanup=True)

print("Презентационная схема создана в файлах 'rwkv_poker_pipeline_presentation.png' и 'rwkv_poker_pipeline_presentation_svg.svg'")