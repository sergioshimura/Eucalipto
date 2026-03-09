"""Servidor Modbus RTU Serial para HMI WECON PI3070ig.

Roda em thread daemon junto com o Flask (app.py).
Comunicação via conversor USB-RS232 em /dev/ttyUSB0.

Mapa de Holding Registers (pymodbus 3.x aplica offset +1 internamente):
  Índice no bloco  PDU Modbus  HMI (PIStudio)  Descrição
  ─────────────────────────────────────────────────────────
  1                0           40001           seedlingcount
  2                1           40002           agua_consumida em L (inteiro, ex: 5 = 5 L)
  3                2           40003           tractorspeed × 10 (ex: 52 = 5.2 km/h)
  4                3           40004           pll_sync_lost (0 ou 1)
  5                4           40005           pll_missed_count
  6                5           40006           pll_period × 100 (ex: 201 = 2.01 s)
  7                6           40007           detector_running (0 ou 1)
  ─── Escrita pelo HMI ─────────────────────────────────────
  11               10          40011           comando: 1=start 2=stop 3=válvula manual
  12               11          40012           modo: 0=fast 1=balanced 2=onnx 3=int8
  13               12          40013           limiar × 100 (ex: 50 = 0.50)
  14               13          40014           delay em ms  (ex: 100 = 0.1 s)
  15               14          40015           distancia × 10 (ex: 20 = 2.0 m)
  16               15          40016           volume_tanque em L
  17               16          40017           volume_irrigacao × 10 (ex: 50 = 5.0 L)
  18               17          40018           max_corr em % (ex: 20 = 20%)
"""

import asyncio
import threading
import json
import os
import time
import logging

_pm_logger = logging.getLogger('pymodbus')
_pm_logger.setLevel(logging.WARNING)

try:
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusServerContext,
        ModbusDeviceContext,
    )
    from pymodbus.server import ModbusSerialServer
    from pymodbus.framer import FramerType
    _PYMODBUS_OK = True
except ImportError:
    _PYMODBUS_OK = False

_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'data.json')
_MODES = ['fast', 'balanced', 'onnx', 'int8']
_NREGS = 21  # tamanho do bloco (índices 0-20, R_MAXCORR=19)

# Índices no bloco. pymodbus 3.x adiciona +1 ao endereço PDU ao chamar
# setValues/getValues, portanto cada índice = endereço_HMI - 40000 + 1.
# Exemplo: HMI 40001 (PDU 1) → setValues/getValues com index 2.
R_SEEDLING   = 2   # HMI 40001
R_TANK       = 3   # HMI 40002
R_SPEED      = 4   # HMI 40003
R_PLL_LOST   = 5   # HMI 40004
R_PLL_MISSED = 6   # HMI 40005
R_PLL_PERIOD = 7   # HMI 40006
R_RUNNING    = 8   # HMI 40007

R_CMD        = 12  # HMI 40011
R_MODE       = 13  # HMI 40012
R_LIMIAR     = 14  # HMI 40013
R_DELAY      = 15  # HMI 40014
R_DISTANCIA  = 16  # HMI 40015
R_VOLTANQUE  = 17  # HMI 40016
R_VOLIRRG    = 18  # HMI 40017
R_MAXCORR    = 19  # HMI 40018


class _CallbackBlock(ModbusSequentialDataBlock):
    """Holding Register com callback quando o HMI escreve um registrador."""

    def __init__(self, on_write=None):
        super().__init__(0x00, [0] * _NREGS)
        self._on_write = on_write

    def setValues(self, address, values):
        super().setValues(address, values)
        if self._on_write:
            for i, val in enumerate(values):
                self._on_write(address + i, val)


class ModbusRTUServer:
    """Servidor Modbus RTU serial integrado ao projeto de irrigação.

    Uso em app.py:
        modbus = ModbusRTUServer(port='/dev/ttyUSB0')
        modbus.on_start  = start_detector   # fn(**kwargs)
        modbus.on_stop   = cleanup_detector # fn()
        modbus.on_valve  = _do_manual_valve # fn()
        modbus.get_status = lambda: detector_status
        modbus.start()
    """

    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, slave_id=1):
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self._block = None
        self._context = None
        self._server = None
        self._loop = None
        self._running = False
        self._last_cmd_time = 0  # cooldown para evitar comando repetido

        # Callbacks injetados pelo app.py
        self.on_start   = None  # fn(**kwargs) inicia detector (zera contadores)
        self.on_retomar = None  # fn(**kwargs) retoma detector (preserva contadores)
        self.on_stop    = None  # fn() para detector
        self.on_valve   = None  # fn() aciona válvula manual
        self.get_status = None  # fn() → dict com chave 'running'

    # ------------------------------------------------------------------ #
    # Acesso interno ao bloco (bypassa o offset +1 do ModbusDeviceContext) #
    # ------------------------------------------------------------------ #

    def _reg(self, idx):
        """Lê valor do bloco pelo índice direto."""
        vals = self._block.getValues(idx, 1)
        return int(vals[0]) if vals else 0

    def _set_reg(self, idx, value):
        """Escreve valor no bloco pelo índice direto (sem disparar callback)."""
        ModbusSequentialDataBlock.setValues(self._block, idx, [max(0, min(int(value), 65535))])

    # ------------------------------------------------------------------ #
    # Callback de escrita do HMI                                          #
    # ------------------------------------------------------------------ #

    def _handle_write(self, address, value):
        """Chamado quando o HMI escreve um registrador. address = índice no bloco."""
        if value != 0:
            print(f"[MODBUS] Escrita do HMI: address={address} value={value}", flush=True)
        if address != R_CMD or value == 0:
            return

        print(f"[MODBUS] Comando recebido do HMI: {value}", flush=True)

        # Cooldown de 3s apenas para INICIAR/PARAR (evita re-disparo com botão pressionado)
        # IRRIGAR (value=3) não tem cooldown — resposta imediata
        if value in (1, 2):
            now = time.time()
            if now - self._last_cmd_time < 3.0:
                self._set_reg(R_CMD, 0)
                return
            self._last_cmd_time = now

        if value in (1, 4):
            mode_idx = self._reg(R_MODE)
            mode = _MODES[mode_idx] if 0 <= mode_idx < len(_MODES) else 'balanced'
            kwargs = dict(
                mode=mode,
                sensitivity=str(self._reg(R_LIMIAR) / 100.0),
                delay=str(self._reg(R_DELAY) / 1000.0),
                distancia=str(self._reg(R_DISTANCIA) / 10.0),
                volume_tanque=str(self._reg(R_VOLTANQUE)),
                volume_irrigacao=str(self._reg(R_VOLIRRG) / 10.0),
                max_corr=str(self._reg(R_MAXCORR)),
            )
            if value == 1 and self.on_start:
                self.on_start(**kwargs)
            elif value == 4 and self.on_retomar:
                self.on_retomar(**kwargs)

        elif value == 2 and self.on_stop:
            self.on_stop()

        elif value == 3 and self.on_valve:
            threading.Thread(target=self.on_valve, daemon=True).start()

        # Zera o registrador de comando sem re-disparar o callback
        self._set_reg(R_CMD, 0)

    # ------------------------------------------------------------------ #
    # Thread de sincronização data.json → registradores                   #
    # ------------------------------------------------------------------ #

    def _update_loop(self):
        while self._running:
            try:
                data = {}
                if os.path.exists(_DATA_FILE):
                    with open(_DATA_FILE) as f:
                        data = json.load(f)

                running = 0
                if self.get_status:
                    running = 1 if self.get_status().get('running') else 0

                self._set_reg(R_SEEDLING,   data.get('seedlingcount', 0))
                self._set_reg(R_TANK,       data.get('agua_consumida', 0))
                self._set_reg(R_SPEED,      data.get('tractorspeed', 0.0) * 100)
                self._set_reg(R_PLL_LOST,   1 if data.get('pll_sync_lost') else 0)
                self._set_reg(R_PLL_MISSED, data.get('pll_missed_count', 0))
                self._set_reg(R_PLL_PERIOD, data.get('pll_period', 0) * 100)
                self._set_reg(R_RUNNING,    running)

            except (json.JSONDecodeError, OSError):
                pass
            except Exception as e:
                print(f"[MODBUS] Erro no update: {e}", flush=True)

            time.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Servidor serial (asyncio em thread daemon)                          #
    # ------------------------------------------------------------------ #

    async def _serve(self):
        self._server = ModbusSerialServer(
            context=self._context,
            framer=FramerType.RTU,
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1,
        )
        await self._server.serve_forever()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[MODBUS] Servidor encerrado: {e}", flush=True)

    # ------------------------------------------------------------------ #
    # API pública                                                         #
    # ------------------------------------------------------------------ #

    def start(self):
        if not _PYMODBUS_OK:
            print("[MODBUS] pymodbus não disponível — servidor desativado.", flush=True)
            return False

        self._block = _CallbackBlock(on_write=self._handle_write)
        store = ModbusDeviceContext(hr=self._block)
        self._context = ModbusServerContext(devices=store, single=True)

        # Inicializa registradores de configuração com valores padrão
        # (HMI pode sobrescrever antes de pressionar INICIAR)
        self._set_reg(R_MODE,      1)    # balanced
        self._set_reg(R_LIMIAR,   50)    # 0.50
        self._set_reg(R_DELAY,   100)    # 100ms
        self._set_reg(R_DISTANCIA, 25)   # 2.5m
        self._set_reg(R_VOLTANQUE, 100)  # 100L
        self._set_reg(R_VOLIRRG,   50)   # 5.0L
        self._set_reg(R_MAXCORR,   20)   # 20%

        self._running = True
        threading.Thread(target=self._update_loop, daemon=True).start()
        threading.Thread(target=self._run_loop, daemon=True).start()

        print(f"[MODBUS] Servidor RTU iniciado — {self.port} @ {self.baudrate}bps "
              f"slave_id={self.slave_id}", flush=True)
        return True

    def stop(self):
        self._running = False
        if self._server and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._server.server_close)
        print("[MODBUS] Servidor Modbus encerrado.", flush=True)
