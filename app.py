from flask import Flask, render_template, request, jsonify
import json
import subprocess
import os
import atexit
import signal

app = Flask(__name__)

# --- GERENCIAMENTO DE PROCESSO ---
detector_process = None
detector_status = {
    "running": False,
    "mode": None,
    "pid": None,
    "showwindow": False,
    "sensitivity": 0.5  # valor padrão
}

# Path to the shared data file
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True) # Ensure the data directory exists
DATA_FILE = os.path.join(DATA_DIR, 'data.json')
ROI_CONFIG_FILE = os.path.join(DATA_DIR, 'roi_config.json') # New ROI config file

def load_roi_config():
    """Carrega as configurações de ROI de um arquivo."""
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[AVISO] Falha ao carregar ROI de {ROI_CONFIG_FILE}. Usando padrão.")
    return {"x": 0, "y": 0, "width": 0, "height": 0} # Default empty ROI

def save_roi_config(roi_data):
    """Salva as configurações de ROI em um arquivo."""
    with open(ROI_CONFIG_FILE, 'w') as f:
        json.dump(roi_data, f)
    print(f"[INFO] ROI salva em {ROI_CONFIG_FILE}: {roi_data}")


def cleanup_detector():
    """Para o processo do detector se estiver em execução e limpa o status."""
    global detector_process, detector_status

    if detector_process:
        print("[INFO] Limpando processo do detector...")
        try:
            # Envia o sinal SIGTERM para todo o grupo de processos
            os.killpg(os.getpgid(detector_process.pid), signal.SIGTERM)
            detector_process.wait(timeout=5)
            print("[INFO] Processo do detector parado.")
        except ProcessLookupError:
            print("[INFO] Processo do detector já não existia.")
        except subprocess.TimeoutExpired:
            print("[AVISO] Processo do detector não respondeu, forçando o encerramento.")
            os.killpg(os.getpgid(detector_process.pid), signal.SIGKILL)

    detector_process = None

    # Aqui entra o reset explícito:
    detector_status["running"] = False
    detector_status["mode"] = None
    detector_status["pid"] = None
    # mantém showwindow e sensitivity como estão

def start_detector(mode='balanced', sensitivity='0.5', delay='0.1', show=False,
                   distancia='2.0', volume_tanque='100', volume_irrigacao='5',
                   max_corr='10'):
    """Inicia o processo do detector com os parâmetros fornecidos."""
    global detector_process, detector_status
    cleanup_detector() # Garante que qualquer processo antigo seja parado

    command = [
        "python3", "detector_unificado.py",
        "--mode", mode,
        "--limiar", str(sensitivity),
        "--delay", str(delay),
        "--distancia", str(distancia),
        "--volume_tanque", str(volume_tanque),
        "--volume_irrigacao", str(volume_irrigacao),
        "--max_corr", str(max_corr)
    ]
    if show:
        command.append("--show")
    
    # Load current normalized ROI config
    normalized_roi = load_roi_config()
    
    # Get frame dimensions from the latest data.json, or use defaults
    current_data = get_data_from_file()
    framewidth = current_data.get('framewidth', 640) # Default to 640 if not yet available
    frameheight = current_data.get('frameheight', 480) # Default to 480 if not yet available
    
    # Check if ROI is valid (width and height > 0 in normalized terms)
    if normalized_roi and normalized_roi["width"] > 0 and normalized_roi["height"] > 0:
        # Scale normalized ROI to original frame resolution
        det_x = int(normalized_roi["x"] * framewidth)
        det_y = int(normalized_roi["y"] * frameheight)
        det_w = int(normalized_roi["width"] * framewidth)
        det_h = int(normalized_roi["height"] * frameheight)

        command.extend([
            "--roix", str(det_x),
            "--roiy", str(det_y),
            "--roiw", str(det_w),
            "--roih", str(det_h)
        ])
        print(f"[INFO] Passando ROI (pixels do frame original) para o detector: X={det_x}, Y={det_y}, W={det_w}, H={det_h}", flush=True)

    print(f"[INFO] Iniciando detector com comando: {' '.join(command)}")
    print("[INFO] A saída do detector está sendo registrada em 'detector_stdout.log' e 'detector_stderr.log'")
    
    # Abrir arquivos de log para capturar a saída do processo filho
    stdout_log = open('detector_stdout.log', 'a')
    stderr_log = open('detector_stderr.log', 'a')
    
    # Inicia o processo em um novo grupo de sessão para podermos matar todos os seus filhos
    detector_process = subprocess.Popen(
        command, 
        preexec_fn=os.setsid,
        stdout=stdout_log,
        stderr=stderr_log
    )
    
    detector_status["running"] = True
    detector_status["mode"] = mode
    detector_status["sensitivity"] = float(sensitivity)
    detector_status["showwindow"] = bool(show)
    detector_status["pid"] = detector_process.pid
    print(f"[INFO] Detector iniciado com PID: {detector_process.pid}")

# --- ROTAS DA APLICAÇÃO ---
@app.route('/')
def index():
    data = get_data_from_file()
    data['detector_status'] = detector_status
    data['saved_roi'] = load_roi_config() # This is already normalized
    return render_template('index.html', data=data)

@app.route('/start', methods=['POST'])
def start_route():
    try:
        delay = request.form.get('delay', '100')
        model = request.form.get('model', 'balanced')
        sensitivity = request.form.get('sensitivity', '0.5')
        showwindow = request.form.get('showwindow')
        distancia = request.form.get('distancia', '2.0')
        volume_tanque = request.form.get('volume_tanque', '100')
        volume_irrigacao = request.form.get('volume_irrigacao', '5')
        max_corr = request.form.get('max_corr', '20')

        # Converte delay de ms (web) para segundos (script)
        delay_sec = str(float(delay) / 1000.0)

        start_detector(
            mode=model,
            sensitivity=sensitivity,
            delay=delay_sec,
            show=bool(showwindow),
            distancia=distancia,
            volume_tanque=volume_tanque,
            volume_irrigacao=volume_irrigacao,
            max_corr=max_corr
        )
        
        return jsonify({'status': 'success', 'message': f'Detector iniciado no modo {model}.'})
    except Exception as e:
        print(f"[ERRO] Falha ao iniciar o detector: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop', methods=['POST'])
def stop_route():
    try:
        cleanup_detector()
        return jsonify({'status': 'success', 'message': 'Detector parado.'})
    except Exception as e:
        print(f"[ERRO] Falha ao parar o detector: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/set_roi', methods=['POST']) # NEW ROI ENDPOINT
def set_roi_route():
    try:
        normalized_roi_data = request.get_json()
        required_keys = ["x", "y", "width", "height"]
        if not all(k in normalized_roi_data for k in required_keys):
            raise ValueError("Dados de ROI incompletos.")
        
        # Validate values (e.g., non-negative, within 0-1 range)
        for k in required_keys:
            if not isinstance(normalized_roi_data[k], (int, float)) or \
               normalized_roi_data[k] < 0 or normalized_roi_data[k] > 1:
                raise ValueError(f"Valor inválido para {k} (deve estar entre 0 e 1): {normalized_roi_data[k]}")

        save_roi_config(normalized_roi_data)
        return jsonify({'status': 'success', 'message': 'ROI salva com sucesso!'})
    except Exception as e:
        print(f"[ERRO] Falha ao salvar ROI: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/clear_roi', methods=['POST'])
def clear_roi_route():
    try:
        if os.path.exists(ROI_CONFIG_FILE):
            os.remove(ROI_CONFIG_FILE)
            print(f"[INFO] ROI config file {ROI_CONFIG_FILE} removido.", flush=True)
        else:
            print(f"[INFO] Nenhum arquivo de configuração ROI encontrado para remover: {ROI_CONFIG_FILE}", flush=True)
        return jsonify({'status': 'success', 'message': 'ROI limpa com sucesso!'})
    except Exception as e:
        print(f"[ERRO] Falha ao limpar ROI: {e}", flush=True)
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_data')
def get_data_route():
    data = get_data_from_file() # Renamed to avoid recursion
    data['detector_status'] = detector_status
    data['saved_roi'] = load_roi_config() # Include saved ROI in data
    return jsonify(data)

def get_data_from_file(): # Renamed to avoid recursion
    """Lê os dados do arquivo JSON compartilhado."""
    if not os.path.exists(DATA_FILE):
        return {
            "seedling_count": 0,
            "tank_volume": 100,
            "tractor_speed": 0.0,
            "framewidth": 640, # Default if not found
            "frameheight": 480 # Default if not found
        }
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            # Ensure framewidth/height defaults are present if file exists but lacks them
            data.setdefault("framewidth", 640)
            data.setdefault("frameheight", 480)
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        # Retorna dados padrão se o arquivo estiver vazio, corrompido ou não existir ainda
        return {
            "seedling_count": 0,
            "tank_volume": 100,
            "tractor_speed": 0.0,
            "framewidth": 640, # Default if file is corrupt
            "frameheight": 480 # Default if file is corrupt
        }

# --- ROTA PARA MODO MANUAL ---
@app.route('/manual_valve', methods=['POST'])
def manual_valve_route():
    """Ativa a válvula manualmente por um curto período."""
    h = None
    try:
        # Tenta abrir o GPIO da mesma forma que o detector_unificado.py
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, 17) # Pino 17
        print("[INFO] GPIO chip 4 aberto para acionamento manual.", flush=True)
    except Exception as e:
        print(f"[AVISO] Falha ao abrir GPIO chip 4 para acionamento manual: {e}", flush=True)
        try:
            h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(h, 17)
            print("[INFO] GPIO chip 0 aberto para acionamento manual.", flush=True)
        except Exception as e_inner:
            print(f"[ERRO] Não foi possível abrir o GPIO para acionamento manual: {e_inner}", flush=True)
            return jsonify({'status': 'error', 'message': 'Falha ao acessar GPIO.'})

    try:
        from detector_unificado import activate_valve
        activate_valve(h, 17)
        return jsonify({'status': 'success', 'message': 'Válvula ativada manualmente!'})
    except Exception as e:
        print(f"[ERRO] Falha ao chamar activate_valve: {e}", flush=True)
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        if h is not None:
            lgpio.gpiochip_close(h)
            print("[INFO] GPIO fechado após acionamento manual.", flush=True)

# --- INICIALIZAÇÃO E LIMPEZA ---
if __name__ == '__main__':
    # Garante que o processo do detector seja parado quando a aplicação Flask for encerrada
    atexit.register(cleanup_detector)
    
    # Inicia a aplicação Flask
    # use_reloader=False é importante para evitar que o atexit seja chamado em duplicidade durante o debug
    app.run(debug=True, host='0.0.0.0', use_reloader=False)