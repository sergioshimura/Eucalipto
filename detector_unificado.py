#!/usr/bin/env python3
import os
print("DEBUG: Iniciando script...", flush=True)

# --- CONFIGURAÇÃO DE AMBIENTE (EVITA TRAVAMENTOS) ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import cv2
import time
import argparse
import numpy as np
import lgpio
import json
from threading import Thread, Lock, Timer
import traceback

cv2.setNumThreads(0)
print("DEBUG: Imports concluídos.", flush=True)
# Desativa threads internas do OpenCV para evitar conflito com ONNX no Pi

# --- PATHS E ARQUIVOS ---
DATADIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATADIR, exist_ok=True)
DATAFILE = os.path.join(DATADIR, 'data.json')
print(f"DEBUG: DATAFILE definido como {DATAFILE}", flush=True)

def save_data(data):
    with open(DATAFILE, 'w') as f:
        json.dump(data, f)

# --- FUNÇÃO LETTERBOX (PADRONIZAÇÃO DE IMAGEM) ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # H, W
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

# --- IMPORTS CONDICIONAIS ---
try:
    from ultralytics import YOLO
    print("DEBUG: Import YOLO bem-sucedido.", flush=True)
except ImportError:
    YOLO = None
    print("DEBUG: Falha ao importar YOLO. YOLO=None.", flush=True)

try:
    import onnxruntime as ort
    print("DEBUG: Import ONNXRuntime bem-sucedido.", flush=True)
except ImportError:
    ort = None
    print("DEBUG: Falha ao importar ONNXRuntime. ort=None.", flush=True)

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='balanced', choices=['fast', 'balanced', 'gst', 'onnx', 'int8'],
                    help='Modo de operação')
parser.add_argument('--limiar', type=float, default=0.5, help='Limiar de confiança (0.1 a 1.0). Padrão 0.5')
parser.add_argument('--delay', type=float, default=0.1, help='Atraso para acionamento da válvula em segundos. Padrão 0.1')
parser.add_argument('--show', action='store_true', help='Exibe a janela de detecção no terminal.')
parser.add_argument('--roix', type=float, default=0.0, help='Coordenada X inicial da ROI.')
parser.add_argument('--roiy', type=float, default=0.0, help='Coordenada Y inicial da ROI.')
parser.add_argument('--roiw', type=float, default=0.0, help='Largura da ROI.')
parser.add_argument('--roih', type=float, default=0.0, help='Altura da ROI.')
parser.add_argument('--distancia', type=float, default=2.0, help='Distância média entre mudas em metros. Padrão 2.0')
parser.add_argument('--volume_tanque', type=float, default=100.0, help='Volume inicial do tanque em litros. Padrão 100')
parser.add_argument('--volume_irrigacao', type=float, default=5.0, help='Volume gasto por irrigação em litros. Padrão 5')
parser.add_argument('--max_corr', type=float, default=10.0, help='Limite máximo de correção do período do PLL por detecção, em %%. Padrão 10')
args = parser.parse_args()
print(f"DEBUG: Argumentos parseados {args}", flush=True)

# --- ROI CONFIGURAÇÃO ---
ROI = (int(args.roix), int(args.roiy), int(args.roiw), int(args.roih))
ROIACTIVE = ROI[2] > 0 and ROI[3] > 0
if ROIACTIVE:
    print(f"DEBUG: ROI ativa X{ROI[0]}, Y{ROI[1]}, W{ROI[2]}, H{ROI[3]}", flush=True)
else:
    print("DEBUG: Nenhuma ROI ativa.", flush=True)

# --- CONFIGURAÇÕES DA CÂMERA ---
CAMIP = "192.168.1.64"
USER = "admin"
PASS = "$S559612s$"
PASSESC = PASS.replace('$', '%24')
RTSPMAIN = f"rtsp://{USER}:{PASSESC}@{CAMIP}/Streaming/Channels/101"
RTSPSUB = f"rtsp://{USER}:{PASSESC}@{CAMIP}/Streaming/Channels/102"
print("DEBUG: Configurações de câmera definidas.", flush=True)

# --- GPIO E PARÂMETROS ---
VALVEPIN = 17
CONFTHRESHOLD = args.limiar
VALVEDELAY = args.delay
h = None  # Inicializa h como None
print("DEBUG: Prestes a configurar GPIO...", flush=True)

try:
    h = lgpio.gpiochip_open(4)
    lgpio.gpio_claim_output(h, VALVEPIN)
    print("DEBUG: GPIO chip 4 aberto com sucesso.", flush=True)
except Exception as e:
    print(f"DEBUG: Falha ao abrir GPIO chip 4 {e}", flush=True)
    try:
        h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(h, VALVEPIN)
        print("DEBUG: GPIO chip 0 aberto com sucesso.", flush=True)
    except Exception as e_inner:
        print(f"ERRO: Não foi possível abrir o GPIO em chip 4 ou 0 {e_inner}", flush=True)
        h = None

print("DEBUG: Configuração GPIO concluída. Handle:", h, flush=True)

# --- THREAD DE CAPTURA DE VÍDEO ---
class StreamThread:
    def __init__(self, source, apipref=cv2.CAP_ANY):
        print(f"DEBUG: StreamThread Iniciando captura de {source}", flush=True)
        self.source_url = source  # Guardar para possível reabertura
        self.api_pref = apipref  # Guardar para possível reabertura
        self.cap = cv2.VideoCapture(source, apipref)
        print(f"DEBUG: StreamThread cv2.VideoCapture chamado. isOpened: {self.cap.isOpened}", flush=True)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # THREAD DE CAPTURA
        print("DEBUG: StreamThread Buffer size definido.", flush=True)
        self.success = False
        self.frame = None
        # Removido self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000) # Timeout de 5 segundos para leitura

        if self.cap.isOpened():
            print("DEBUG: StreamThread Câmera aberta. Tentando ler frame inicial...", flush=True)
            self.success, self.frame = self.cap.read()
            print(f"DEBUG: StreamThread Leitura inicial do frame. Sucesso: {self.success}", flush=True)
        else:
            print("ERRO: StreamThread Falha ao abrir a câmera. Verifique a URL e a conectividade.", flush=True)

        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                print("AVISO: StreamThread Câmera não está aberta no update. Tentando reabrir...", flush=True)
                self.cap = cv2.VideoCapture(self.source_url, self.api_pref)  # Tenta reabrir
                time.sleep(1)  # Espera um pouco antes de tentar novamente
                continue

            # Tenta ler um frame para verificar se a câmera está funcionando
            grabbed = self.cap.grab()
            if not grabbed:  # Usar grab+retrieve mais eficiente e robusto para streams RTSP
                time.sleep(0.05)  # Pequena espera antes de tentar novamente
                continue

            success, frame = self.cap.retrieve()
            with self.lock:
                self.success = success
                if success:
                    self.frame = frame
                else:
                    if time.time() % 5 < 0.1:  # Para não ser muito verboso
                        print("AVISO: StreamThread Falha ao recuperar frame (retrieve).", flush=True)
                if not success:  # Se falhou, espera um pouco mais
                    time.sleep(0.05)

    def read(self):
        with self.lock:
            if not self.success:
                print("AVISO: StreamThread.read Câmera não tem frames de sucesso. Retornando None.", flush=True)
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
        print("DEBUG: StreamThread Captura parada e liberada.", flush=True)

# --- SETUP DO MODELO ---
session = None
model = None
input_name = None
print("DEBUG: Prestes a configurar o modelo...", flush=True)

if args.mode in ['onnx', 'int8', 'fast']:
    if ort is None:
        raise RuntimeError("ONNXRuntime não instalado.")
    if args.mode == 'fast':
        onnx_file = 'eucalipto_yolov8n.onnx'
    else:  # onnx or int8
        onnx_file = 'eucalipto_yolov8n_INT8.onnx' if args.mode == 'int8' else 'eucalipto_yolov8n.onnx'
    print(f"INFO: Carregando ONNX {onnx_file}", flush=True)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(onnx_file, sess_options, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        print(f"DEBUG: Modelo ONNX carregado {onnx_file}", flush=True)
    except Exception as e:
        print(f"ERRO: Falha ao carregar modelo ONNX {onnx_file} {e}", flush=True)
        session = None
elif args.mode == 'balanced':
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO não instalado.")
    print("INFO: Carregando modelo YOLO PyTorch...", flush=True)
    try:
        model = YOLO('eucalipto_yolov8n.pt')
        print("DEBUG: Modelo YOLO PyTorch carregado.", flush=True)
    except Exception as e:
        print(f"ERRO: Falha ao carregar modelo YOLO PyTorch {e}", flush=True)
        model = None

print("DEBUG: Configuração do modelo concluída.", flush=True)

# --- SETUP DO STREAM E PARÂMETROS ---
SHOW = args.show
IMGSIZE = 640
FRAMESKIP = 1

apipref = cv2.CAP_FFMPEG
source_url = RTSPSUB

if args.mode == 'fast':
    source_url = RTSPSUB
    IMGSIZE = 320
elif args.mode == 'balanced':
    source_url = RTSPMAIN
    IMGSIZE = 640
elif args.mode in ['onnx', 'int8']:
    source_url = RTSPSUB
    IMGSIZE = 640

print(f"INFO: Modo {args.mode.upper()} Limiar {CONFTHRESHOLD} Size {IMGSIZE} Atraso {VALVEDELAY}s", flush=True)
print("DEBUG: Prestes a iniciar o stream de vídeo...", flush=True)
stream = StreamThread(source_url, apipref).start()
print("DEBUG: Stream de vídeo iniciado.", flush=True)

# --- VARIÁVEIS GLOBAIS ---
framecounter = 0
seedlingcount = 0
tankvolume = 100
tractorspeed = 0.0
lastdetectiontime = time.time()
avginterval = 0.0

def activate_valve(h, pin):
    """Ativa a válvula por um curto período."""
    if h is not None:
        try:
            print(f"DEBUG: Ativando válvula no pino {pin}", flush=True)
            lgpio.gpio_write(h, pin, 1)
            time.sleep(0.2)  # Manter a válvula aberta por 200ms
            lgpio.gpio_write(h, pin, 0)
            print(f"DEBUG: Válvula desativada no pino {pin}", flush=True)
        except Exception as e:
            print(f"ERRO ao tentar ativar a válvula: {e}", flush=True)

class PLLController:
    """Gerador de pulsos sincronizado (PLL) com as detecções de mudas.

    Mantém um intervalo constante entre acionamentos da válvula.
    Mesmo que uma detecção seja perdida, o pulso é gerado na hora esperada.
    As detecções reais ajustam a frequência/fase do gerador.
    Após 3 disparos consecutivos sem detecção real, sinaliza 'perda de sincronismo'.
    """
    ALPHA = 0.3       # Peso da nova amostra no filtro IIR do período
    MIN_PERIOD = 0.3  # Período mínimo aceitável (segundos)
    MAX_PERIOD = 60.0 # Período máximo aceitável (segundos)

    def __init__(self, h, valve_pin, valve_delay, distancia_media=2.0,
                 volume_tanque=100.0, volume_irrigacao=5.0, max_corr_pct=0.20):
        self.h = h
        self.valve_pin = valve_pin
        self.valve_delay = valve_delay
        self.distancia_media = distancia_media  # metros
        self.tankvolume = volume_tanque
        self.volume_irrigacao = volume_irrigacao
        self.max_corr_pct = max_corr_pct  # fração máxima de correção por ciclo (ex: 0.20 = 20%)
        self.period = 2.0       # estimativa inicial do período
        self.missed_count = 0
        self.sync_lost = False
        self.active = False
        self._timer = None
        self._lock = Lock()
        self._started = False   # True após a primeira detecção real
        self._last_fire_time = 0.0
        self.tractorspeed = 0.0

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def on_detection(self, current_time):
        """Notifica o PLL sobre uma detecção real de muda.

        Aciona a válvula, atualiza o período estimado e reinicia o temporizador.
        Retorna a velocidade estimada do trator em km/h.
        """
        with self._lock:
            if self._started and self._last_fire_time > 0:
                elapsed = current_time - self._last_fire_time

                # Debounce: ignora se chegou cedo demais (mesma muda em frames consecutivos)
                if elapsed < self.period * 0.4:
                    print(f"PLL: Detecção ignorada (debounce {elapsed:.2f}s < "
                          f"{self.period * 0.4:.2f}s)", flush=True)
                    return self.tractorspeed

                if self.MIN_PERIOD <= elapsed <= self.MAX_PERIOD:
                    # Slew rate limiter: limita o intervalo aceito a ±max_corr_pct do período atual
                    # Evita que um falso positivo ou detecção espúria mude bruscamente a frequência
                    max_delta = self.period * self.max_corr_pct
                    clamped = max(self.period - max_delta, min(self.period + max_delta, elapsed))
                    self.period = (1 - self.ALPHA) * self.period + self.ALPHA * clamped
                    if self.period > 0 and self.distancia_media > 0:
                        self.tractorspeed = 3.6 * self.distancia_media / self.period

            self.missed_count = 0
            self.sync_lost = False
            self._started = True
            self._last_fire_time = current_time

            # Aciona a válvula e decrementa volume do tanque
            self._trigger_valve()

            # Reinicia o temporizador PLL a partir de agora
            if self._timer:
                self._timer.cancel()
            self._timer = Timer(self.period, self._pll_fire)
            self._timer.start()

        print(f"PLL: Detecção recebida. Período={self.period:.2f}s, "
              f"Velocidade={self.tractorspeed:.2f}km/h", flush=True)
        return self.tractorspeed

    def _pll_fire(self):
        """Chamado pelo temporizador quando a próxima detecção era esperada mas não chegou."""
        if not self.active:
            return

        with self._lock:
            self.missed_count += 1

            if self.missed_count >= 3:
                # Perda de sincronismo: para de acionar e para de reagendar
                self.sync_lost = True
                print(f"PLL: PERDA DE SINCRONISMO após {self.missed_count} falhas. "
                      f"Válvula parada.", flush=True)
                return

            print(f"PLL: Disparo sem detecção. Falhas consecutivas={self.missed_count}", flush=True)

            self._last_fire_time = time.time()

            # Aciona a válvula mesmo sem detecção real e decrementa volume do tanque
            self._trigger_valve()

            # Reagenda o próximo disparo
            if self.active:
                self._timer = Timer(self.period, self._pll_fire)
                self._timer.start()

    def _trigger_valve(self):
        """Decrementa o volume do tanque e agenda o acionamento da válvula.
        Deve ser chamado com self._lock já adquirido.
        """
        self.tankvolume = max(0.0, self.tankvolume - self.volume_irrigacao)
        Timer(self.valve_delay, lambda: activate_valve(self.h, self.valve_pin)).start()

    def get_status(self):
        """Retorna o estado atual do PLL para a interface web."""
        with self._lock:
            return {
                'pll_sync_lost': self.sync_lost,
                'pll_missed_count': self.missed_count,
                'pll_period': round(self.period, 2),
                'tankvolume': round(self.tankvolume, 2)
            }


def run_detector(args):
    """Executa o loop principal de detecção."""
    # --- SETUP DO MODELO ---
    session = None
    model = None
    input_name = None
    print("DEBUG: Prestes a configurar o modelo...", flush=True)

    if args.mode in ['onnx', 'int8', 'fast']:
        if ort is None:
            raise RuntimeError("ONNXRuntime não instalado.")
        if args.mode == 'fast':
            onnx_file = 'eucalipto_yolov8n.onnx'
        else:  # onnx or int8
            onnx_file = 'eucalipto_yolov8n_INT8.onnx' if args.mode == 'int8' else 'eucalipto_yolov8n.onnx'
        print(f"INFO: Carregando ONNX {onnx_file}", flush=True)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            session = ort.InferenceSession(onnx_file, sess_options, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            print(f"DEBUG: Modelo ONNX carregado {onnx_file}", flush=True)
        except Exception as e:
            print(f"ERRO: Falha ao carregar modelo ONNX {onnx_file} {e}", flush=True)
            session = None
    elif args.mode == 'balanced':
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO não instalado.")
        print("INFO: Carregando modelo YOLO PyTorch...", flush=True)
        try:
            model = YOLO('eucalipto_yolov8n.pt')
            print("DEBUG: Modelo YOLO PyTorch carregado.", flush=True)
        except Exception as e:
            print(f"ERRO: Falha ao carregar modelo YOLO PyTorch {e}", flush=True)
            model = None

    print("DEBUG: Configuração do modelo concluída.", flush=True)

    # --- SETUP DO STREAM E PARÂMETROS ---
    SHOW = args.show
    IMGSIZE = 640
    FRAMESKIP = 1

    apipref = cv2.CAP_FFMPEG
    source_url = RTSPSUB

    if args.mode == 'fast':
        source_url = RTSPSUB
        IMGSIZE = 320
    elif args.mode == 'balanced':
        source_url = RTSPMAIN
        IMGSIZE = 640
    elif args.mode in ['onnx', 'int8']:
        source_url = RTSPSUB
        IMGSIZE = 640

    print(f"INFO: Modo {args.mode.upper()} Limiar {CONFTHRESHOLD} Size {IMGSIZE} Atraso {VALVEDELAY}s", flush=True)
    print("DEBUG: Prestes a iniciar o stream de vídeo...", flush=True)
    stream = StreamThread(source_url, apipref).start()
    print("DEBUG: Stream de vídeo iniciado.", flush=True)

    # --- VARIÁVEIS GLOBAIS ---
    framecounter = 0
    seedlingcount = 0
    tractorspeed = 0.0

    # --- PLL ---
    pll = PLLController(h, VALVEPIN, VALVEDELAY,
                        distancia_media=args.distancia,
                        volume_tanque=args.volume_tanque,
                        volume_irrigacao=args.volume_irrigacao,
                        max_corr_pct=args.max_corr / 100.0)
    pll.start()

    print("DEBUG: Entrando no loop principal...", flush=True)

    outputpath = 'static/live.jpg'

    # --- LOOP PRINCIPAL ---
    try:
        while True:
            print(f"DEBUG: Loop principal Tentando ler frame {framecounter}", flush=True)
            frame = stream.read()
            if frame is None:
                if h is not None:
                    lgpio.gpio_write(h, VALVEPIN, 0)
                if framecounter % 20 == 0:
                    print(f"AVISO: Aguardando frame... Modo {args.mode}. isOpened: {stream.cap.isOpened}. "
                          f"Sucesso anterior do StreamThread: {stream.success}", flush=True)
                time.sleep(0.1)
                framecounter += 1
                continue

            print(f"DEBUG: Frame lido com sucesso shape {frame.shape}", flush=True)
            framecounter = 0

            if framecounter % FRAMESKIP != 0:
                continue

            detectionmade = False
            outputframe = frame.copy()  # Frame para desenhar detecções e ROI

            # --- INFERÊNCIA ONNX ---
            if session is not None:
                print("DEBUG: Iniciando inferência ONNX.", flush=True)
                img_padded, ratio, (dw, dh) = letterbox(frame, new_shape=IMGSIZE)
                img_input = img_padded.transpose(2, 0, 1)  # HWC to CHW
                img_input = np.ascontiguousarray(img_input, dtype=np.float32) / 255.0
                img_input = img_input[None]

                outputs = session.run(None, {input_name: img_input})
                pred = outputs[0][0]
                if pred.shape[0] > pred.shape[1]:
                    pred = pred.T

                boxes_raw = pred[:, :4]
                scores = pred[:, 4].max(axis=1)
                mask = scores > CONFTHRESHOLD
                boxes = boxes_raw[mask]
                scores = scores[mask]

                filtered_boxes = []
                filtered_scores = []
                print(f"DEBUG: ONNX Encontradas {len(boxes)} caixas antes do filtro ROI.", flush=True)

                for i, box in enumerate(boxes):
                    # Rescale boxes from img_padded size to original frame size
                    box = box.astype(np.float32)
                    box[0] = (box[0] - dw) / ratio[0]  # x1
                    box[1] = (box[1] - dh) / ratio[1]  # y1
                    box[2] = (box[2] - dw) / ratio[0]  # x2
                    box[3] = (box[3] - dh) / ratio[1]  # y2
                    x1, y1, x2, y2 = np.round(box).astype(int)

                    # Ensure box coordinates are floats for calculations
                    if ROIACTIVE:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        if not (ROI[0] <= cx <= ROI[0] + ROI[2] and ROI[1] <= cy <= ROI[1] + ROI[3]):
                            continue  # Skip this box if its center is outside ROI

                    filtered_boxes.append((x1, y1, x2, y2))
                    filtered_scores.append(scores[i])

                for i, (x1, y1, x2, y2) in enumerate(filtered_boxes):
                    cv2.rectangle(outputframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{filtered_scores[i]:.2f}'
                    cv2.putText(outputframe, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"DEBUG: ONNX {len(filtered_boxes)} caixas após o filtro ROI.", flush=True)
                if len(filtered_boxes) > 0:
                    detectionmade = True

            # --- INFERÊNCIA YOLO TORCH ---
            elif model is not None:
                print("DEBUG: Iniciando inferência YOLO PyTorch.", flush=True)
                results = model(frame, imgsz=IMGSIZE, verbose=False, conf=CONFTHRESHOLD)
                filtered_results_boxes = []

                if len(results[0].boxes) > 0:
                    print(f"DEBUG: YOLO Encontradas {len(results[0].boxes)} caixas antes do filtro ROI.", flush=True)
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Process results to filter by ROI before plotting
                        if ROIACTIVE:
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            if not (ROI[0] <= cx <= ROI[0] + ROI[2] and ROI[1] <= cy <= ROI[1] + ROI[3]):
                                continue  # Skip this box if its center is outside ROI

                        filtered_results_boxes.append(box)

                for box in filtered_results_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cv2.rectangle(outputframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{conf:.2f}'
                    cv2.putText(outputframe, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if len(filtered_results_boxes) > 0:
                    detectionmade = True
                print("DEBUG: Processamento de detecção concluído. detectionmade:", detectionmade, flush=True)

            # --- DESENHO DA ROI ---
            if ROIACTIVE:
                print(f"DEBUG: Desenhando ROI X{ROI[0]}, Y{ROI[1]}, W{ROI[2]}, H{ROI[3]}", flush=True)
                cv2.rectangle(outputframe, (ROI[0], ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (255, 0, 0), 2)  # Blue rectangle for ROI
                cv2.putText(outputframe, 'ROI', (ROI[0], ROI[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # --- LÓGICA DE ATIVAÇÃO DA VÁLVULA (via PLL) ---
            if detectionmade:
                seedlingcount += 1
                print("INFO: Muda detectada! Contagem:", seedlingcount, flush=True)
                currenttime = time.time()
                tractorspeed = pll.on_detection(currenttime)

            # --- ATUALIZAÇÃO DOS DADOS PARA A INTERFACE ---
            pll_status = pll.get_status()
            data = {
                'seedlingcount': seedlingcount,
                'tankvolume': pll_status['tankvolume'],
                'tractorspeed': tractorspeed,
                'pll_sync_lost': pll_status['pll_sync_lost'],
                'pll_missed_count': pll_status['pll_missed_count'],
                'pll_period': pll_status['pll_period']
            }
            
            # CORREÇÃO: Detectar dimensões reais do frame na primeira execução
            if 'framewidth' not in data or data['framewidth'] == 640:
                data['framewidth'] = frame.shape[1]  # Largura real
                data['frameheight'] = frame.shape[0]  # Altura real
                print(f"DEBUG: Dimensões reais do frame detectadas: {data['framewidth']}x{data['frameheight']}", flush=True)
            
            save_data(data)
            print("DEBUG: Dados salvos em", DATAFILE, flush=True)

            # Salva o frame para a interface web
            cv2.imwrite(outputpath, outputframe)
            print("DEBUG: Frame salvo em", outputpath, ". Próxima iteração.", flush=True)

            if SHOW:
                cv2.imshow('Detector', outputframe)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("INFO: Parando...", flush=True)
    except Exception as e:
        print(f"ERRO FATAL: Uma exceção não tratada ocorreu {e}", flush=True)
        traceback.print_exc()
    finally:
        print("DEBUG: Bloco finally alcançado.", flush=True)
        pll.stop()
        stream.stop()
        if h is not None:
            lgpio.gpio_write(h, VALVEPIN, 0)
            lgpio.gpiochip_close(h)
            print("DEBUG: GPIO liberado.", flush=True)
        if SHOW:
            cv2.destroyAllWindows()
        # if os.path.exists(DATAFILE): os.remove(DATAFILE)  # Comentado para não limpar em caso de crash
        pass
        print("DEBUG: Fim do script.", flush=True)

if __name__ == '__main__':
    run_detector(args)
