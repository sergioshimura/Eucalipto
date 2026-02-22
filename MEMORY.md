# MEMORY.md — Sistema de Irrigação Seletiva Linear v1.0

Arquivo de memória persistente do projeto. Atualizado ao final de cada sessão de desenvolvimento.

---

## Estado Atual do Projeto

**Data da ultima atualizacao:** 2026-02-22
**Fase:** Integracao Modbus RTU completa — aguardando configuracao do PIStudio e deploy no Raspberry Pi
**Versao:** v1.0

---

## Resumo Executivo

Sistema embarcado para irrigacao seletiva de mudas de eucalipto em plantio linear. Roda em Raspberry Pi 5. Detecta mudas por visao computacional (YOLOv8) via camera IP RTSP. Aciona valvula solenoide via GPIO no momento correto usando um PLL (Phase-Locked Loop) sincronizado com as deteccoes. Interface web Flask para monitoramento e controle. Servidor Modbus RTU para operacao via HMI WECON PI3070ig.

---

## Decisoes de Arquitetura (nao reverter)

1. **app.py como orquestrador:** Flask + subprocesso para detector_unificado.py. Comunicacao via data/data.json. Nao executar o detector diretamente em producao.

2. **PLLController:** classe dentro de detector_unificado.py. Debounce 40% do periodo. Filtro IIR alpha=0.3. Slew rate limiter max_corr=20%. Para na 3a falha consecutiva, ressincroniza automaticamente na proxima deteccao real.

3. **Modbus RTU via RS232:** USB-RS232 em /dev/ttyUSB0. Nao usar Ethernet (a porta ja esta ocupada pela camera). pymodbus 3.12.1 — usa ModbusDeviceContext (nao ModbusSlaveContext). Registradores 1-based no bloco (pymodbus aplica +1 internamente). Registrador de comando (40011) faz auto-reset para 0 apos processamento.

4. **GPIO:** chip 4 no Raspberry Pi 5, pino 17, 200ms. Fallback chip 0.

---

## Mapa de Registradores Modbus (definitivo)

Holding Registers. Indice no bloco = HMI 40000 + indice.

Leitura pelo HMI:
- 1 (40001): seedlingcount
- 2 (40002): tankvolume x10
- 3 (40003): tractorspeed x10
- 4 (40004): pll_sync_lost 0/1
- 5 (40005): pll_missed_count
- 6 (40006): pll_period x100
- 7 (40007): detector_running 0/1

Escrita pelo HMI:
- 11 (40011): comando — 1=start, 2=stop, 3=valvula manual (auto-reset)
- 12 (40012): modo — 0=fast, 1=balanced, 2=onnx, 3=int8
- 13 (40013): limiar x100
- 14 (40014): delay em ms
- 15 (40015): distancia x10
- 16 (40016): volume_tanque em L
- 17 (40017): volume_irrigacao x10
- 18 (40018): max_corr em %

---

## Hardware

- Raspberry Pi 5 — IP 192.168.15.10 (campo) / IP de desenvolvimento varia
- Camera IP — IP 192.168.1.64, RTSP canal 101 (main), 102 (sub)
- HMI WECON PI3070ig — 800x480 touch, software PIStudio V9.5.9
- Conversor USB-RS232 em /dev/ttyUSB0

---

## Dependencias Python (venv)

- Flask
- ultralytics (YOLOv8)
- opencv-python
- lgpio (GPIO Raspberry Pi 5)
- pymodbus 3.12.1 (instalado na sessao 2026-02-22)

---

## Bugs Conhecidos e Pendencias

1. **Raspberry Pi rodando codigo antigo:** O Pi em 192.168.15.10 ainda tem o codigo de /home/sergio/yolo/app.py com o bug NameError `get_data`. O codigo correto esta em /home/sergio/Linear/. Solucao: git pull no Pi + pip install pymodbus.

2. **Volume negativo do tanque:** Nao ha protecao contra tankvolume < 0 se o sistema rodar com tanque vazio. Pendente desde sessao 2026-02-18.

3. **PIStudio nao configurado:** A tela HMI ainda nao existe. Nenhum teste Pi <-> HMI foi realizado.

---

## Proximas Tarefas (ordem de prioridade)

1. Instalar PIStudio V9.5.9 no PC Windows
   - Download: https://ftp.we-con.com.cn/Download/Software/PIStudio_9.5.9.25091702.zip
2. Criar projeto PIStudio para PI3070ig (800x480)
3. Configurar dispositivo Modbus RTU no PIStudio (COM1, 9600, 8-N-1, slave 1)
4. Criar Tela 1 — Monitoramento + Controles
5. Criar Tela 2 — Parametros
6. Gravar no HMI via cabo USB-B
7. Deploy do codigo atualizado no Raspberry Pi (git pull + pip install pymodbus)
8. Testar comunicacao Pi <-> HMI
9. Calibrar parametros PLL com dados de campo
10. Adicionar protecao contra volume negativo

---

## Historico de Sessoes

### 2026-02-22
Modbus RTU: modbus_server.py criado do zero. app.py integrado. pymodbus 3.12.1 instalado. Mapa de registradores definido. Bug no Pi identificado.

### 2026-02-18
Versao inicial: Flask app, PLLController, interface web 3 colunas, ROI configuravel, GPIO com fallback.
