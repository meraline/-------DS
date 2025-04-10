
from flask import Flask, send_file, render_template_string
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Результаты обучения модели</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .image-container { margin: 20px 0; }
        img { max-width: 100%; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Результаты обучения модели RWKV</h1>
    
    <div class="image-container">
        <h2>История обучения</h2>
        <img src="/plot/training_history.png" alt="История обучения">
    </div>
    
    <div class="image-container">
        <h2>Распределение классов</h2>
        <img src="/plot/class_distribution.png" alt="Распределение классов">
    </div>
    
    <div class="image-container">
        <h2>t-SNE визуализация</h2>
        <img src="/plot/tsne_visualization.png" alt="t-SNE визуализация">
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
        return send_file(plot_path)
    return "Plot not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
