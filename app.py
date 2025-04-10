from flask import Flask, render_template_string, send_file
import os

def get_latest_file(prefix, directory):
    """Get the latest file with given prefix from directory"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))

def get_file_path(filename, directory):
    """Get the file path"""
    return os.path.join(directory, filename)



app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Результаты оценки модели</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background-color: #f5f5f5;
        }
        .tab { 
            display: none;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tab-active { display: block; }
        .tabs { 
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .tab-button { 
            padding: 10px 20px; 
            margin-right: 5px;
            cursor: pointer;
            border: none;
            background: #f0f0f0;
        }
        .tab-button.active {
            background: #007bff;
            color: white;
        }
        .image-container {
            margin: 20px 0;
            text-align: center;
        }
        img {
            max-width: 100%;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Результаты оценки модели RWKV</h1>

    <div class="tabs">
        <button class="tab-button active" onclick="showTab('actions')">Действия</button>
        <button class="tab-button" onclick="showTab('sizes')">Размеры ставок</button>
        <button class="tab-button" onclick="showTab('tsne')">t-SNE</button>
        <button class="tab-button" onclick="showTab('allin')">All-in</button>
    </div>

    <div id="actions" class="tab tab-active">
        <div class="image-container">
            <h2>Матрица ошибок</h2>
            <img src="/plot/confusion_matrix.png?v={{timestamp}}" alt="Матрица ошибок">
        </div>

        <div class="image-container">
            <h2>Распределение классов</h2>
            <img src="/plot/class_distribution.png" alt="Распределение классов">
        </div>

        <div class="image-container">
            <h2>Динамика обучения модели</h2>
            <img src="/plot/training_history.png" alt="Динамика обучения">
        </div>

        <div class="image-container">
            <h2>Уверенность в предсказаниях</h2>
            <img src="/plot/prediction_confidence.png" alt="Уверенность в предсказаниях">
        </div>
    </div>

    <div id="sizes" class="tab">
        <div class="image-container">
            <h2>Матрица ошибок размеров ставок</h2>
            <img src="/plot_size/bet_size_confusion_matrix.png" alt="Матрица ошибок размеров ставок">
        </div>

        <div class="image-container">
            <h2>Динамика обучения модели размеров ставок</h2>
            <img src="/plot_size/training_history.png" alt="Динамика обучения размеров ставок">
        </div>

        <div class="image-container">
            <h2>Распределение размеров ставок</h2>
            <img src="/plot_size/bet_size_distribution.png" alt="Распределение размеров ставок">
        </div>

        <div class="image-container">
            <h2>Распределение классов</h2>
            <img src="/plot_size/class_distribution.png" alt="Распределение классов">
        </div>
    </div>

    <div id="tsne" class="tab">
        <div class="image-container">
            <h2>t-SNE визуализация</h2>
            <img src="/plot/tsne_visualization.png" alt="t-SNE визуализация">
        </div>
    </div>

    <div id="allin" class="tab">
        <div class="image-container">
            <h2>Матрица ошибок размеров ставок All-in</h2>
            <img src="/plot_allin/allin_bet_size_confusion_matrix.png" alt="Матрица ошибок размеров ставок All-in">
        </div>
        
        <div class="image-container">
            <h2>Распределение All-in</h2>
            <img src="/plot_allin/allin_distribution.png" alt="Распределение All-in">
        </div>

        <div class="image-container">
            <h2>Распределение стеков при All-in</h2>
            <img src="/plot_allin/allin_stack_distribution.png" alt="Распределение стеков при All-in">
        </div>

        <div class="image-container">
            <h2>ROC-кривая для All-in</h2>
            <img src="/plot_allin/allin_roc_curve.png" alt="ROC-кривая All-in">
        </div>

        <div class="image-container">
            <h2>Динамика обучения модели All-in</h2>
            <img src="/plot_allin/training_history.png" alt="Динамика обучения All-in">
        </div>
    </div>

    <script>
        function showTab(tabId) {
            // Hide all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('tab-active');
            });

            // Show selected tab
            document.getElementById(tabId).classList.add('tab-active');

            // Update buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            document.querySelector(`[onclick="showTab('${tabId}')"]`).classList.add('active');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/plot/<filename>')
def serve_plot(filename):
    plot_path = get_file_path(filename, 'model_dir')
        
    if os.path.exists(plot_path):
        timestamp = str(os.path.getmtime(plot_path))
        response = send_file(plot_path, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Version'] = timestamp
        return response
    return "File not found", 404

@app.route('/plot_size/<filename>')
def serve_plot_size(filename):
    plot_path = os.path.join('model_dir_size', filename)
    if os.path.exists(plot_path):
        response = send_file(plot_path, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return "File not found", 404

@app.route('/plot_allin/<filename>')
def serve_plot_allin(filename):
    plot_path = os.path.join('model_dir_allin', filename)  # Changed path to model_dir_allin
    if os.path.exists(plot_path):
        response = send_file(plot_path, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)