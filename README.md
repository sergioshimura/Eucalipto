# Sistema de Irrigação Seletiva Linear v1.0

**Status:** Servidor Modbus RTU implementado — aguardando configuração HMI (PIStudio) e deploy no Raspberry Pi.

## Descrição

Sistema embarcado para irrigação seletiva de mudas de eucalipto em plantio linear. Roda em Raspberry Pi 5. Detecta mudas por visão computacional (YOLOv8n) via câmera IP RTSP. Aciona válvula solenóide via GPIO no momento correto usando um Phase-Locked Loop (PLL) sincronizado com as detecções. Interface web Flask para monitoramento e controle. Servidor Modbus RTU para operação via HMI touchscreen WECON PI3070ig sem necessidade de PC.

## Componentes Principais

- `app.py` — servidor Flask + integração Modbus RTU
- `modbus_server.py` — servidor Modbus RTU serial para HMI (pymodbus 3.12.1)
- `detector_unificado.py` — loop de detecção YOLOv8 + PLLController + GPIO
- `templates/index.html` + `static/style.css` — interface web

## Como Executar

```bash
cd /home/sergio/Linear
source venv/bin/activate
python app.py
# Interface web: http://<IP>:5000
# Modbus RTU: /dev/ttyUSB0, 9600 bps, slave ID 1
```

## Hardware

- Raspberry Pi 5
- Câmera IP RTSP (192.168.1.64)
- Válvula solenóide via GPIO pino 17
- HMI WECON PI3070ig via USB-RS232 (Modbus RTU)

## Última Atualização

2026-02-22
