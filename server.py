from flask import Flask, request, jsonify, send_file
import pickle
import gzip
from io import BytesIO
import torch

app = Flask(__name__)

model_parameters = []
avg_params = None
clients_received = set()
epoch = -1

def average_model_parameters(param_list):
    if not param_list:
        print(param_list)
        return None
    # 假設param_list是一個列表，其中每個元素都是字典型的模型參數
    avg_params = {}
    for key in param_list[0].keys():
        # 堆疊同一鍵的所有參數，沿新的0維，然後計算平均值
        avg_params[key] = torch.mean(torch.stack([p[key] for p in param_list]), dim=0)
    return avg_params

@app.route('/upload', methods=['POST'])
def upload_parameters():
    global avg_params,model_parameters,epoch  # 使用全局變量來存儲平均參數
    if 'model' not in request.files:
        return "No model file in request", 400

    model_file = request.files['model']
    
    try:
        with gzip.GzipFile(fileobj=model_file.stream, mode='rb') as f:
            parameters = pickle.load(f)
    except (OSError, pickle.UnpicklingError) as e:
        return f"Error handling the file: {str(e)}", 400

    model_parameters.append(parameters)
    if len(model_parameters) >= 2:  # 當收集到2個或更多模型參數時
        avg_params = average_model_parameters(model_parameters)
        epoch +=1
        model_parameters.clear()  # 清空收集到的參數以便下一次收集
        clients_received.clear()  # 清空客戶端集合
        return jsonify({"status": "Parameters averaged"}), 200

    return jsonify({"status": "Parameter received"}), 200

@app.route('/model', methods=['GET'])
def get_model():
    global avg_params,epoch
    client_epoch = int(request.args.get('epoch'))
    
    if avg_params is None or epoch != client_epoch:
        return "No averaged model parameters available.", 404
    
    memfile = BytesIO()
    with gzip.GzipFile(fileobj=memfile, mode='wb') as f:
        pickle.dump(avg_params, f)
    memfile.seek(0)
    
    return send_file(memfile, download_name='model.pkl.gz', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
