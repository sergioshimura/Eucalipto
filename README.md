# Sistema de Irrigação Seletiva Linear v1.0

**Status:** Sistema refatorado — câmera Hikvision RTSP, acumulador de água, comando RETOMAR, serviço systemd instalado na Pi. Aguardando atualização do HMI no PIStudio e teste de campo.

## Descrição

Sistema embarcado para irrigação seletiva de mudas de eucalipto em plantio linear. Roda em Raspberry Pi 5. Detecta mudas por visão computacional (YOLOv8n) via câmera IP RTSP Hikvision. Aciona válvula solenóide via GPIO no momento correto usando um Phase-Locked Loop (PLL) sincronizado com as detecções. Interface web Flask para monitoramento e controle. Servidor Modbus RTU para operação via HMI touchscreen WECON PI3070ig sem necessidade de PC.

## Componentes Principais

- `app.py` — servidor Flask + integração Modbus RTU + lógica de retomada de sessão
- `modbus_server.py` — servidor Modbus RTU serial para HMI (pymodbus 3.12.1), comando 4=RETOMAR
- `detector_unificado.py` — loop de detecção YOLOv8 + PLLController + GPIO, água consumida acumulada
- `templates/index.html` + `static/style.css` — interface web
- `irrigacao.service` — serviço systemd para auto-start no boot da Pi

## Como Executar

```bash
# Na Raspberry Pi (via systemd — recomendado):
sudo systemctl start irrigacao

# Manualmente:
cd ~/yolo
source py313env/bin/activate
python app.py
# Acesso campo (hotspot Shimura-cel): http://172.16.148.73:5000
# Acesso Ethernet: http://192.168.1.1:5000
# Modbus RTU: /dev/ttyUSB0, 9600 bps, slave ID 1
```

## Hardware

- Raspberry Pi 5 — eth0: 192.168.1.1 (câmera), wlan0: 172.16.148.73 (hotspot campo)
- Câmera Hikvision DS-2CD1021G0-I — IP 192.168.1.64, RTSP
- Válvula solenóide via GPIO chip 4, pino 17, pulso 200ms
- HMI WECON PI3070ig via USB-RS232 (Modbus RTU slave ID 1)

## Ultima Atualizacao

2026-03-08
