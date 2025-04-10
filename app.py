
from flask import Flask, send_file, render_template_string
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Результаты модели RWKV</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .image-container { margin: 20px 0; }
        img { max-width: 100%; border: 1px solid #ddd; }
        .tabs { margin-bottom: 20px; }
        .tab-button {
            padding: 10px 20px;
            border: none;
            background: #f0f0f0;
            cursor: pointer;
            margin-right: 5px;
        }
        .tab-button.active {
            background: #007bff;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
    <script>
        function refreshImages() {
            const images = document.getElementsByTagName('img');
            for(let img of images) {
                img.src = img.src.split('?')[0] + '?' + new Date().getTime();
            }
        }
        
        function openTab(evt, tabName) {
            const tabContents = document.getElementsByClassName("tab-content");
            for (let content of tabContents) {
                content.classList.remove("active");
            }
            
            const tabButtons = document.getElementsByClassName("tab-button");
            for (let button of tabButtons) {
                button.classList.remove("active");
            }
            
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }
        
        // Обновлять каждые 5 секунд
        setInterval(refreshImages, 5000);
    </script>
</head>
<body>
    <h1>Результаты модели RWKV</h1>
    
    <div class="tabs">
        <button class="tab-button active" onclick="openTab(event, 'training')">Обучение</button>
        <button class="tab-button" onclick="openTab(event, 'evaluation')">Проверка модели</button>
    </div>
    
    <div id="training" class="tab-content active">
        <div class="image-container">
            <h2>История обучения</h2>
            <img src="/plot/training_history.png" alt="История обучения">
        </div>
        
        <div class="image-container">
            <h2>Распределение классов</h2>
            <img src="/plot/class_distribution.png" alt="Распределение классов">
        </div>
    </div>
    
    <div id="evaluation" class="tab-content">
        <div class="image-container">
            <h2>Матрица ошибок</h2>
            <img src="/plot/confusion_matrix.png" alt="Матрица ошибок">
        </div>
        
        <div class="image-container">
            <h2>t-SNE визуализация</h2>
            <img src="/plot/tsne_visualization.png" alt="t-SNE визуализация">
        </div>
        
        <div class="image-container">
            <h2>Уверенность предсказаний</h2>
            <img src="/plot/prediction_confidence.png" alt="Уверенность предсказаний">
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/plot/<filename>')
def serve_plot(filename):
    plot_path = os.path.join('model_dir', filename)
    if os.path.exists(plot_path):
        return send_file(plot_path, max_age=0)
    return "Plot not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
