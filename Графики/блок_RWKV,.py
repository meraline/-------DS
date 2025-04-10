from graphviz import Digraph

# Создаем диаграмму RWKV блока
dot = Digraph(comment='RWKV Block Structure', format='png')
dot.attr(rankdir='TB', size='8,10', dpi='300')
dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='16', margin='0.3,0.2')
dot.attr('edge', penwidth='2.0', arrowsize='1.5', fontname='Arial', fontsize='14')

# Определяем цвета
colors = {
    'bg': '#F5F5F5',       # Фон
    'input': '#D6EAF8',    # Вход
    'state': '#FADBD8',    # Состояние
    'key': '#D5F5E3',      # Ключ
    'value': '#FCF3CF',    # Значение
    'recept': '#EBDEF0',   # Рецептивность
    'output': '#FDEBD0',   # Выход
    'formula': '#EBF5FB'   # Формулы
}

# Создаем кластер для RWKV блока
with dot.subgraph(name='cluster_rwkv') as c:
    c.attr(label='Структура RWKV блока', style='filled', fillcolor=colors['bg'], fontcolor='#2C3E50', fontsize='20')
    
    # Входные данные и состояние
    c.node('input', 'Текущий вход (x_t)', shape='ellipse', fillcolor=colors['input'])
    c.node('state', 'Предыдущее состояние', shape='box', fillcolor=colors['state'])
    
    # Компоненты RWKV
    c.node('key', 'Key\nформирование ключа', shape='component', fillcolor=colors['key'])
    c.node('value', 'Value\nформирование значения', shape='component', fillcolor=colors['value'])
    c.node('recept', 'Receptance\nрецептивность', shape='component', fillcolor=colors['recept'])
    
    # Выходные данные
    c.node('new_state', 'Обновленное состояние', shape='box', fillcolor=colors['state'])
    c.node('output', 'Выход', shape='ellipse', fillcolor=colors['output'])
    
    # Формулы
    c.node('key_formula', 'k = key(x_t * time_mix_k + state * (1-time_mix_k))', 
           shape='note', fillcolor=colors['formula'])
    c.node('value_formula', 'v = value(x_t * time_mix_v + state * (1-time_mix_v))', 
           shape='note', fillcolor=colors['formula'])
    c.node('recept_formula', 'r = sigmoid(receptance(x_t * time_mix_r + state * (1-time_mix_r)))', 
           shape='note', fillcolor=colors['formula'])
    c.node('state_formula', 'state = x_t + state * exp(-exp(time_decay))', 
           shape='note', fillcolor=colors['formula'])
    c.node('output_formula', 'out = r * output(v)', 
           shape='note', fillcolor=colors['formula'])
    
    # Связи для потока данных
    c.edge('input', 'key', color='#3498DB')
    c.edge('input', 'value', color='#3498DB')
    c.edge('input', 'recept', color='#3498DB')
    c.edge('state', 'key', color='#E74C3C')
    c.edge('state', 'value', color='#E74C3C')
    c.edge('state', 'recept', color='#E74C3C')
    
    c.edge('key', 'output', style='invis')  # Невидимая связь для выравнивания
    c.edge('value', 'output', color='#F39C12')
    c.edge('recept', 'output', color='#8E44AD')
    
    c.edge('input', 'new_state', color='#3498DB')
    c.edge('state', 'new_state', color='#E74C3C')
    
    # Связи для формул
    c.edge('key', 'key_formula', style='dashed', color='#7F8C8D', dir='none')
    c.edge('value', 'value_formula', style='dashed', color='#7F8C8D', dir='none')
    c.edge('recept', 'recept_formula', style='dashed', color='#7F8C8D', dir='none')
    c.edge('new_state', 'state_formula', style='dashed', color='#7F8C8D', dir='none')
    c.edge('output', 'output_formula', style='dashed', color='#7F8C8D', dir='none')

# Добавляем объяснение
dot.node('explanation', 
         "Особенности RWKV в контексте покера:\n\n"\
         "• Рецептивность определяет, насколько важен текущий ход\n"\
         "• Механизм состояния накапливает историю предыдущих действий\n"\
         "• time_decay регулирует 'забывание' устаревшей информации\n"\
         "• Это позволяет модели адаптировать предсказания на основе\n"\
         "  контекста всей раздачи, а не только текущей ситуации",
         shape='note', fillcolor='white', fontsize='14')

dot.edge('cluster_rwkv', 'explanation', style='invis')

# Сохраняем диаграмму
dot.render('rwkv_block_structure', view=True, cleanup=True)

# Создаем SVG версию
dot.format = 'svg'
dot.render('rwkv_block_structure_svg', view=False, cleanup=True)

print("Схема RWKV блока создана в файлах 'rwkv_block_structure.png' и 'rwkv_block_structure_svg.svg'")