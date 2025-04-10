from graphviz import Digraph

dot = Digraph(comment='RWKV Architecture')
dot.attr(rankdir='TB', size='11,8')

# Определение узлов
dot.node('data', 'Входные данные покерного турнира\nПозиция, Стек, SPR, История действий...', shape='box', style='filled', fillcolor='lightblue')
dot.node('time', 'Time-Mixing блок\nАнализ истории действий', shape='box', style='filled', fillcolor='lightgreen')
dot.node('channel', 'Channel-Mixing блок\nАнализ покерных признаков', shape='box', style='filled', fillcolor='lightyellow')
dot.node('output', 'Выходные вероятности\nFold, Check, Call, Bet, Raise', shape='box', style='filled', fillcolor='lightpink')

# Определение связей
dot.edge('data', 'time', label='История')
dot.edge('data', 'channel', label='Признаки')
dot.edge('time', 'channel', label='Временной контекст')
dot.edge('channel', 'output', label='Интегрированная информация')

# Сохранение
dot.render('rwkv_simple', format='png')