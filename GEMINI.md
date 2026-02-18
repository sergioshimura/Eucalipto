# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sistema de Irrigação Seletiva Linear — detecção de mudas de eucalipto via câmera IP (RTSP) em Raspberry Pi 5, com acionamento automático de válvula solenóide via GPIO e interface web Flask para monitoramento e controle.

## Running the Application

```bash
cd /home/sergio/Linear
source venv/bin/activate   # ambiente virtual py313env na Raspberry Pi
python app.py
# Acesso: http://<IP>:5000
```

O `app.py` gerencia o `detector_unificado.py` como subprocesso. Não execute o detector diretamente em produção.

## File Structure

```
Linear/
├── app.py                    # Servidor Flask — inicia/para detector, serve interface
├── detector_unificado.py     # Loop de detecção YOLO + PLLController + GPIO
├── gpio_handler.py           # Utilitário standalone para testar GPIO
├── templates/index.html      # Interface web (câmera ao vivo, controles, alertas)
├── static/style.css          # Estilos da interface
├── data/
│   ├── data.json             # Estado compartilhado (gerado em runtime)
│   └── roi_config.json       # ROI salva (gerado em runtime)
└── static/live.jpg           # Frame atual da câmera (gerado em runtime)
```

Modelos YOLO na raiz de `Linear/`:
- `eucalipto_yolov8n.pt` — modo balanced (PyTorch)
- `eucalipto_yolov8n.onnx` — modos fast/onnx
- `eucalipto_yolov8n_INT8.onnx` — modo int8

## Architecture

```
app.py (Flask)
  ├── GET  /          → serve index.html com dados atuais
  ├── POST /start     → chama start_detector(), lança subprocesso
  ├── POST /stop      → mata subprocesso (SIGTERM ao grupo de processos)
  ├── GET  /get_data  → lê data/data.json + detector_status
  ├── POST /set_roi   → salva ROI normalizada em data/roi_config.json
  ├── POST /clear_roi → remove roi_config.json
  └── POST /manual_valve → aciona GPIO diretamente

detector_unificado.py (subprocesso)
  ├── StreamThread    → captura RTSP em thread separada (grab+retrieve)
  ├── PLLController   → gerador de pulsos sincronizado com detecções
  │     ├── on_detection()  → debounce + atualiza período + aciona válvula
  │     └── _pll_fire()     → disparo autônomo se detecção perdida (para em 3 falhas)
  └── run_detector()  → loop principal: frame → inferência → PLL → salva data.json
```

**Comunicação app ↔ detector:** arquivo `data/data.json` (escrito pelo detector, lido pelo Flask a cada `/get_data`).

## PLLController — Comportamento

- O PLL mantém um temporizador que dispara no intervalo esperado entre mudas
- Detecção real → aciona válvula + reinicia timer + atualiza período via filtro IIR (α=0.3)
- **Debounce:** detecções que chegam em menos de 40% do período estimado são ignoradas (mesma muda em frames consecutivos)
- Timer dispara sem detecção → aciona válvula (1ª e 2ª falha); na 3ª falha consecutiva: para de acionar e sinaliza `pll_sync_lost = true`
- Re-sincronização automática quando uma detecção real chega após perda de sincronismo
- Velocidade calculada: `3.6 × distancia_media / period` (km/h)
- Volume do tanque decrementado a cada acionamento da válvula

## Parameters (passed via form → subprocess args)

| Parâmetro | Default | Descrição |
|---|---|---|
| `--mode` | balanced | fast / balanced / onnx / int8 |
| `--limiar` | 0.5 | Confiança mínima de detecção |
| `--delay` | 0.1s | Atraso entre detecção e acionamento da válvula |
| `--distancia` | 2.0m | Distância média entre mudas |
| `--volume_tanque` | 100L | Volume inicial do tanque |
| `--volume_irrigacao` | 5L | Volume gasto por acionamento |
| `--max_corr` | 20% | Variação máxima do período aceita por detecção (slew rate limiter) |
| `--roix/y/w/h` | 0 | ROI em pixels (calculada a partir da ROI normalizada salva) |
| `--show` | off | Exibe janela OpenCV no terminal |

## Hardware

- **Câmera:** IP `192.168.1.64`, streams RTSP canal 101 (main) e 102 (sub)
- **GPIO:** chip 4 (Raspberry Pi 5), pino 17, válvula solenóide 200ms
- **Fallback GPIO:** chip 0 se chip 4 falhar

## data.json Schema

```json
{
  "seedlingcount": 0,
  "tankvolume": 100.0,
  "tractorspeed": 0.0,
  "pll_sync_lost": false,
  "pll_missed_count": 0,
  "pll_period": 2.0,
  "framewidth": 640,
  "frameheight": 480
}
```

## Session History

### Session 2026-02-18
- Phase: Implementation complete — ready for field testing
- Accomplishments: Flask web application (app.py) created from scratch; PLLController implemented in detector_unificado.py with debounce, slew rate limiter, and sync-loss stop; web interface redesigned with 3-column data layout, action buttons below camera, and PLL sync loss alert; .claude/agents/script-session-closer.md added; CLAUDE.md documentation written
- Key Decisions: Subprocess architecture for detector isolation; PLLController as dedicated class; debounce at 40% of estimated period; max_corr slew limiter at 20%; valve stops on 3 consecutive missed detections with auto-resync
- Next Steps: Deploy and test on Raspberry Pi 5 with real RTSP camera and GPIO; calibrate PLL parameters from field data; evaluate CPU usage and consider int8 mode as default
