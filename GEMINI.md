# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sistema de Irrigação Seletiva Linear — detecção de mudas de eucalipto via câmera IP (RTSP) em Raspberry Pi 5, com acionamento automático de válvula solenóide via GPIO e interface web Flask para monitoramento e controle.

## Running the Application

```bash
# No PC de desenvolvimento:
cd /home/sergio/Linear

# Na Raspberry Pi (diretório diferente!):
cd ~/yolo
source py313env/bin/activate
python app.py
# Acesso web local: http://192.168.1.1:5000  (Ethernet, câmera)
# Acesso campo (hotspot): http://172.16.148.73:5000
```

O `app.py` gerencia o `detector_unificado.py` como subprocesso. Não execute o detector diretamente em produção.

O sistema é gerenciado pelo serviço systemd `irrigacao.service` (instalado em `/etc/systemd/system/`, enabled, auto-start no boot).

## File Structure

```
Linear/
├── app.py                    # Servidor Flask — inicia/para detector, serve interface; integra Modbus
├── modbus_server.py          # Servidor Modbus RTU serial para HMI WECON PI3070ig
├── detector_unificado.py     # Loop de detecção YOLO + PLLController + GPIO
├── gpio_handler.py           # Utilitário standalone para testar GPIO
├── templates/index.html      # Interface web (câmera ao vivo, controles, alertas)
├── static/style.css          # Estilos da interface
├── data/
│   ├── data.json             # Estado compartilhado (gerado em runtime)
│   └── roi_config.json       # ROI salva (gerado em runtime)
└── static/live.jpg           # Frame atual da câmera (gerado em runtime)
```

Documentação HMI:
- `Wecon-PI3070ig-programação.pptx` — guia de programação do HMI WECON PI3070ig (PIStudio, upload via Backstage, mapa de registradores)

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

modbus_server.py (thread daemon no mesmo processo Flask)
  ├── _CallbackBlock  → ModbusSequentialDataBlock com hook de escrita do HMI
  ├── ModbusRTUServer
  │     ├── _handle_write() → processa comandos do HMI (start/stop/válvula)
  │     ├── _update_loop()  → thread: lê data.json a cada 0.5s → atualiza registradores
  │     └── _run_loop()     → asyncio em thread separada: servidor serial RTU
  └── Integração via callbacks injetados pelo app.py

detector_unificado.py (subprocesso)
  ├── StreamThread    → captura RTSP em thread separada (grab+retrieve)
  ├── PLLController   → gerador de pulsos sincronizado com detecções
  │     ├── on_detection()  → debounce + atualiza período + aciona válvula
  │     └── _pll_fire()     → disparo autônomo se detecção perdida (para em 3 falhas)
  └── run_detector()  → loop principal: frame → inferência → PLL → salva data.json
```

**Comunicação app ↔ detector:** arquivo `data/data.json` (escrito pelo detector, lido pelo Flask a cada `/get_data`).
**Comunicação Pi ↔ HMI:** Modbus RTU serial via USB-RS232 (`/dev/ttyUSB0`), slave ID=1, 9600 bps.

## PLLController — Comportamento

- O PLL mantém um temporizador que dispara no intervalo esperado entre mudas
- Detecção real → aciona válvula + reinicia timer + atualiza período via filtro IIR (α=0.3)
- **Debounce:** detecções que chegam em menos de 40% do período estimado são ignoradas (mesma muda em frames consecutivos)
- Timer dispara sem detecção → aciona válvula (1ª e 2ª falha); na 3ª falha consecutiva: para de acionar e sinaliza `pll_sync_lost = true`
- Re-sincronização automática quando uma detecção real chega após perda de sincronismo
- Velocidade calculada: `3.6 × distancia_media / period` (km/h)
- `agua_consumida` acumulado (começa em 0, incrementado a cada acionamento); substitui `tankvolume` do data.json
- `manual_valve()` público no PLLController — chamado por valve_trigger e pelo Modbus (comando 3)
- Comando 4 via Modbus = RETOMAR: chama `on_retomar` callback no app.py → reinicia detector preservando estado (seedlingcount, agua_consumida, pll_period)

## Parameters (passed via form → subprocess args)

| Parâmetro | Default | Descrição |
|---|---|---|
| `--mode` | balanced | fast / balanced / onnx / int8 |
| `--limiar` | 0.5 | Confiança mínima de detecção |
| `--delay` | 0.1s | Atraso entre detecção e acionamento da válvula |
| `--distancia` | 2.5m | Distância média entre mudas |
| `--volume_irrigacao` | 5L | Volume gasto por acionamento |
| `--max_corr` | 20% | Variação máxima do período aceita por detecção (slew rate limiter) |
| `--agua_consumida_inicial` | 0L | Água consumida acumulada inicial (para RETOMAR) |
| `--seedlingcount_inicial` | 0 | Contagem inicial de mudas (para RETOMAR) |
| `--pll_period_inicial` | 1.5s | Período inicial do PLL (para RETOMAR) |
| `--roix/y/w/h` | 0 | ROI em pixels (calculada a partir da ROI normalizada salva) |
| `--show` | off | Exibe janela OpenCV no terminal |

## Hardware

- **Câmera:** Hikvision DS-2CD1021G0-I, IP `192.168.1.64` (Ethernet eth0, rede dedicada), RTSP: `rtsp://192.168.1.64/...`
- **Raspberry Pi:** eth0=`192.168.1.1/24` (câmera), wlan0=`172.16.148.73/24` (hotspot Shimura-cel), diretório `~/yolo`, venv: `py313env`
- **GPIO:** chip 4 (Raspberry Pi 5), pino 17, válvula solenóide 200ms
- **Fallback GPIO:** chip 0 se chip 4 falhar
- **HMI:** WECON PI3070ig, 800×480, touch — software: PIStudio V9.5.9
- **Serial:** conversor USB-RS232 em `/dev/ttyUSB0`, Modbus RTU slave ID=1, 9600 bps, 8-N-1
- **Systemd:** `irrigacao.service` instalado em `/etc/systemd/system/`, enabled, auto-start no boot

## Modbus Register Map (Holding Registers)

**IMPORTANTE — pymodbus 3.x offset:** pymodbus aplica `+1` ao endereço PDU ao chamar `setValues`/`getValues` no bloco. Por isso, os índices no código (constantes R_*) são sempre `PDU_address + 1`. O HMI PIStudio usa endereços 4xxxx (40001 = PDU 0 → índice no bloco = 1, mas pymodbus chama `setValues(2)` → R_SEEDLING = 2).

Mapa correto (constantes em modbus_server.py, confirmado em produção):

| Constante | Índice bloco | PDU | HMI | Descrição | Escala |
|---|---|---|---|---|---|
| R_SEEDLING | 2 | 1 | 40001 | seedlingcount | x1 |
| R_TANK | 3 | 2 | 40002 | agua_consumida | x1 (raw litros) |
| R_SPEED | 4 | 3 | 40003 | tractorspeed | x100 |
| R_PLL_LOST | 5 | 4 | 40004 | pll_sync_lost | 0/1 |
| R_PLL_MISSED | 6 | 5 | 40005 | pll_missed_count | x1 |
| R_PLL_PERIOD | 7 | 6 | 40006 | pll_period | x100 |
| R_RUNNING | 8 | 7 | 40007 | detector_running | 0/1 |
| R_CMD | 12 | 11 | 40011 | comando (1=start, 2=stop, 3=válvula, 4=retomar) | auto-reset 0 |
| R_MODE | 13 | 12 | 40012 | modo (0=fast,1=balanced,2=onnx,3=int8) | x1 |
| R_LIMIAR | 14 | 13 | 40013 | limiar | x100 |
| R_DELAY | 15 | 14 | 40014 | delay | ms |
| R_DISTANCIA | 16 | 15 | 40015 | distancia | x10 (default 25 = 2.5m) |
| R_VOLTANQUE | 17 | 16 | 40016 | volume_tanque (parâmetro inicial) | L |
| R_VOLIRRG | 18 | 17 | 40017 | volume_irrigacao | x10 |
| R_MAXCORR | 19 | 18 | 40018 | max_corr | % |

## data.json Schema

```json
{
  "seedlingcount": 0,
  "agua_consumida": 0.0,
  "tractorspeed": 0.0,
  "pll_sync_lost": false,
  "pll_missed_count": 0,
  "pll_period": 1.5,
  "framewidth": 640,
  "frameheight": 480
}
```

Nota: `tankvolume` foi substituído por `agua_consumida` (acumulador que começa em 0 e incrementa a cada acionamento de válvula). O campo `pll_period` default mudou de 2.0 para 1.5s (= 6 km/h com 2.5m entre mudas).

## Session History

### Session 2026-03-08 (parte 4 — câmera Hikvision, agua_consumida, RETOMAR, systemd)
- Phase: Sistema refatorado com nova câmera RTSP, acumulador de água e suporte a retomada de sessão; serviço systemd instalado
- Accomplishments:
  1. Câmera trocada: Wanscam JW0004 (MJPEG HTTP) → Hikvision DS-2CD1021G0-I (RTSP, 192.168.1.64); Pi eth0 = 192.168.1.1/24; wlan0 = 172.16.148.73/24 (hotspot campo)
  2. detector_unificado.py: distancia default 2.0→2.5m; MIN_PERIOD 0.3→1.5s; tankvolume substituído por agua_consumida (acumulador, começa em 0); PLLController aceita agua_consumida_inicial, pll_period_inicial; seedlingcount inicializa de args.seedlingcount_inicial; manual_valve() público no PLLController; valve_trigger chama pll.manual_valve()
  3. modbus_server.py: R_TANK lê agua_consumida (raw litros, sem ×10); R_SPEED: tractorspeed × 100 (HMI divisor 100); R_DISTANCIA default 20→25; cooldown 3s apenas para comandos 1 e 2 (não para IRRIGAR=3); comando 4 = RETOMAR com callback on_retomar
  4. app.py: _increment_agua_consumida() para modo OFF; manual_valve e _do_valve incrementam agua_consumida; start_detector(resume=False/True): resume=True preserva seedlingcount, agua_consumida, pll_period; resume=False zera em data.json imediatamente; rota /retomar adicionada; on_retomar registrado no Modbus; args iniciais passados ao subprocesso
  5. templates/index.html: "Tanque" → "Água Consumida", id tankvolume → agua_consumida
  6. irrigacao.service criado e instalado na Pi (/etc/systemd/system/, enabled, auto-start no boot)
- Key Decisions: agua_consumida como acumulador (começa em 0) é mais intuitivo para operador de campo que "volume restante do tanque"; MIN_PERIOD 1.5s evita disparos falsos em alta velocidade; resume=True permite retomar contagem após parada sem perder dados; cooldown removido para IRRIGAR (comando 3) pois operador precisa acionar multiple vezes; systemd garante auto-restart em caso de falha
- Next Steps: Atualizar HMI no PIStudio: label 40002 de "Tanque" para "Agua Cons.", divisor 40003 de /10 para /100, adicionar botão RETOMAR (escreve 4 em 40011); fazer git pull na Pi e reiniciar serviço; teste de campo completo com câmera Hikvision e trator em movimento

### Session 2026-02-22 (parte 1)
- Phase: Modbus RTU integration complete — awaiting PIStudio configuration and Raspberry Pi deploy
- Accomplishments: modbus_server.py created from scratch (ModbusRTUServer class with callback-based HMI command handling, async serial server, data.json sync loop); app.py integrated with Modbus server (import + instance + start in __main__); pymodbus 3.12.1 installed in venv; register map fully defined (7 read registers + 8 write registers); NameError bug on Pi identified (code on Pi is outdated, fix already present in Linear/app.py)
- Key Decisions: Modbus RTU over RS232 (not Ethernet — camera uses the network port); pymodbus 3.12.1 uses ModbusDeviceContext not ModbusSlaveContext; register index is 1-based in the block (pymodbus applies +1 offset internally); command register auto-resets to 0 after processing; integration via injected callbacks to keep modbus_server.py independent of app.py
- Next Steps: Install PIStudio V9.5.9 on Windows PC; create HMI project for PI3070ig; deploy updated code to Raspberry Pi (git pull + pip install pymodbus); test Modbus communication between Pi and HMI via USB-RS232

### Session 2026-03-08 (parte 3 — fix offset Modbus e IPs)
- Phase: Sistema totalmente funcional — INICIAR/PARAR via HMI confirmados; câmera aparecendo na web
- Accomplishments:
  1. IP da câmera atualizado: 192.168.15.11 (antes 192.168.15.12); Pi agora é 192.168.15.12
  2. Câmera não aparecia na web: StreamThread estava com URL antiga (192.168.15.12) → corrigido para 192.168.15.11
  3. Bug crítico Modbus corrigido: pymodbus 3.x adiciona +1 ao endereço PDU ao chamar setValues/getValues no bloco. R_CMD era 11 mas pymodbus chamava setValues(12) → callback verificava address==R_CMD (11) e retornava sem fazer nada. Corrigido incrementando todas as constantes R_* em +1 (R_CMD=12, R_SEEDLING=2, etc.) e _NREGS de 20 para 21
  4. Detector iniciava com parâmetros zerados: registradores R_MODE..R_MAXCORR eram 0 antes do HMI escrever. Corrigido com inicialização de defaults em start()
  5. Botão HMI pressionado → múltiplos INICIAR: adicionado cooldown de 3s em _handle_write
  6. gpio_handler.py ausente na Pi (~/yolo): copiado via scp
  7. venv da Pi é py313env, não venv/
  8. Logging pymodbus revertido para WARNING (estava DEBUG para diagnóstico)
- Key Decisions: pymodbus 3.x offset +1 é definitivo — todos os índices R_* devem ser PDU+1; INICIAR/PARAR via HMI confirmados em produção com parâmetros corretos; câmera e Pi têm IPs trocados em relação à sessão anterior
- Next Steps: Testar botão IRRIGAR (valor=3) → gpio_handler.py pulso 200ms; verificar display HMI mostrando valores corretos dos registradores de leitura (seedlingcount, tankvolume etc.); teste de campo completo

### Session 2026-02-24 (parte 2 — diagnóstico e deploy Modbus)
- Phase: Modbus RTU comunicação bidirecional confirmada — aguardando teste dos botões de controle
- Accomplishments:
  1. app.py copiado para Raspberry Pi (~/yolo) — versão com integração Modbus não estava na Pi
  2. pymodbus e pyserial instalados no venv da Pi — dependências faltavam
  3. Diagnóstico completo da falha Modbus: causa raiz foi cabo RS232 fisicamente solto; após reconectar, Pi recebe FC03 reads do HMI e responde corretamente; comunicação bidirecional confirmada via logs debug do pymodbus
  4. PIStudio corrigido: botões INICIAR/PARAR/IRRIGAR reconfigurados com PLC Station No.=1 (antes estava como Default=0, causando envio de comandos para escravo ID=0 em vez de ID=1)
  5. Procedimento de upload HMI via Backstage documentado (pressionar canto superior direito 3-4s)
  6. modbus_server.py: logging revertido para WARNING (StreamHandler de debug removido antes do commit)
- Key Decisions: Causa raiz de toda falha Modbus foi cabo solto — não software; PLC Station No. em Word Switch deve ser explicitamente 1, não "Default"; logs debug do pymodbus são úteis para diagnóstico mas devem ser revertidos para WARNING em produção
- Next Steps: Testar botões INICIAR/PARAR/IRRIGAR após atualização do HMI com PIStudio corrigido (PLC Station No.=1); verificar se FC06 write chega ao Pi para cada botão; executar fluxo completo (start via HMI → detector rodando → registradores atualizando no display → válvula manual)

### Session 2026-02-23
- Phase: HMI screen design complete in PIStudio — awaiting UDisk upload to HMI hardware
- Accomplishments: PIStudio V9.5.9 installed on Windows PC (supports PI3070ig / ig series); Modbus RTU communication configured in PIStudio (COM1, RS232, 9600 bps, 8-N-1, Modbus RTU Slave All Function, Device No. 1); complete HMI screen created with monitoring section (seedlingcount 40001, tankvolume 40002/10, tractorspeed 40003/10, PLL period 40006/100, missed count 40005), Word Lamp indicators (SYNC OK/LOST on 40004, DETECTOR ON/OFF on 40007), configuration section (mode dropdown 40012, threshold 40013/100, delay 40014, distance 40015/10, tank volume 40016, irrigation volume 40017/10, max correction 40018), and action buttons (INICIAR writes 1 to 40011, PARAR writes 2 to 40011, PURGAR writes 3 to 40011); CP210x USB driver installed on Windows PC
- Key Decisions: PIStudio V9.5.9 confirmed compatible with PI3070ig (ig series); Word Lamp (not Bit Lamp) required for holding registers — Bit Lamp only works with coils; direct USB cable download does not work (Download button non-functional in PIStudio for this model); UDisk download via FAT32 pendrive is the correct method; UDisk format must be WMT3 using the "HMI V2.0" tab (not V1.0) inside PIStudio; PURGAR button = temporary manual valve trigger (equivalent to flush/purge)
- HMI Upload Procedure (PI3070ig Backstage mode): inserir pendrive FAT32 com arquivo .wmt3 na raiz → pressionar e segurar canto superior direito da tela por 3-4s até entrar no "Backstage" → selecionar idioma English → Finish → na barra direita escolher "Update" → aparece "HMI Project" com o novo projeto → selecionar e clicar Download. NOTA: o método UDisk do PIStudio apenas gera o arquivo no pendrive; o upload real é feito pelo Backstage do HMI. Se HMI diz "no update" ao inserir o pendrive normalmente, usar o Backstage.
- Next Steps: Generate UDisk file in WMT3 format (HMI V2.0 tab) from PIStudio; copy to FAT32 pendrive; upload to PI3070ig via USB port; deploy updated code to Raspberry Pi (git pull + pip install pymodbus); connect USB-RS232 cable between Pi and HMI; test full Modbus communication loop

### Session 2026-02-18
- Phase: Implementation complete — ready for field testing
- Accomplishments: Flask web application (app.py) created from scratch; PLLController implemented in detector_unificado.py with debounce, slew rate limiter, and sync-loss stop; web interface redesigned with 3-column data layout, action buttons below camera, and PLL sync loss alert; .claude/agents/script-session-closer.md added; CLAUDE.md documentation written
- Key Decisions: Subprocess architecture for detector isolation; PLLController as dedicated class; debounce at 40% of estimated period; max_corr slew limiter at 20%; valve stops on 3 consecutive missed detections with auto-resync
- Next Steps: Deploy and test on Raspberry Pi 5 with real RTSP camera and GPIO; calibrate PLL parameters from field data; evaluate CPU usage and consider int8 mode as default
