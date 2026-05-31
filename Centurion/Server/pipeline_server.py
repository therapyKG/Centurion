#!/usr/bin/env python3
"""
Pipeline-parallel training server: orchestrates multiple iOS workers,
each owning a contiguous slice of GPT-2 layers.

The server:
  - Listens on a single port (9998) for both workers and orchestrator
  - After HMAC auth, the first message determines the client type:
    - PIPELINE_REGISTER (0x40) → worker
    - ORCH_IDENTIFY (0x94) → orchestrator
  - Workers register and stay connected between training runs
  - An orchestrator (iOS app) configures and starts/stops training
  - The server is persistent: it never exits, waiting for commands between runs

Protocol: Length-prefixed binary frames over TCP.
Authentication: HMAC-SHA256 challenge-response (single shared secret).

Requires: pip3 install numpy safetensors
"""

import asyncio
import struct
import time
import logging
import argparse
import os
import hmac
import hashlib
import traceback
import numpy as np
import tiktoken
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline-server")

# ── Authentication message types (shared with checkpoint_server) ──
MSG_AUTH_CHALLENGE = 0x30
MSG_AUTH_RESPONSE = 0x31
MSG_AUTH_RESULT = 0x32

# ── Pipeline message types (0x40+) ──
MSG_PIPELINE_REGISTER = 0x40
MSG_PIPELINE_CONFIG = 0x41
MSG_PIPELINE_CONFIG_ACK = 0x42
MSG_PIPELINE_START = 0x43
MSG_PIPELINE_STOP = 0x44

MSG_PIPELINE_DATA_BATCH = 0x50

MSG_PIPELINE_ACTIVATION = 0x60
MSG_PIPELINE_GRADIENT = 0x61

MSG_PIPELINE_SYNC_BARRIER = 0x70
MSG_PIPELINE_SYNC_ACK = 0x71

MSG_PIPELINE_LOSS_REPORT = 0x80

# ── Orchestrator message types (0x90+ orch→server, 0xA0+ server→orch) ──
MSG_ORCH_UPDATE_CONFIG = 0x90
MSG_ORCH_START_TRAINING = 0x91
MSG_ORCH_STOP_TRAINING = 0x92
MSG_ORCH_GET_STATUS = 0x93
MSG_ORCH_IDENTIFY = 0x94  # Orchestrator sends this as first message after auth
MSG_ORCH_ALLOW_WORKER = 0x95  # Orchestrator requests a worker-auth bypass for its IP
MSG_ORCH_ALLOW_WORKER_ACK = 0xA6  # Server confirms the bypass is ready

MSG_ORCH_STATUS_REPORT = 0xA0
MSG_ORCH_CONFIG_ACK = 0xA1
MSG_ORCH_TRAINING_STARTED = 0xA2
MSG_ORCH_TRAINING_STOPPED = 0xA3
MSG_ORCH_LOSS_UPDATE = 0xA4
MSG_ORCH_ERROR = 0xA5

# ── Secrets ──
WORKER_SECRET = b""
ORCH_SECRET = b""

# IPs that are allowed to skip worker auth (one-time use, set by orchestrator)
# Maps IP -> expiry timestamp (monotonic)
worker_auth_bypass: Dict[str, float] = {}
BYPASS_EXPIRY_SECONDS = 30.0


# ── Frame I/O ──

async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise asyncio.IncompleteReadError(data, n)
        data += chunk
    return data


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    length_bytes = await read_exactly(reader, 4)
    msg_len = struct.unpack(">I", length_bytes)[0]
    return await read_exactly(reader, msg_len)


async def write_frame(writer: asyncio.StreamWriter, payload: bytes):
    frame = struct.pack(">I", len(payload)) + payload
    writer.write(frame)
    await writer.drain()


# ── Authentication ──

async def authenticate_client(reader, writer, addr) -> Optional[str]:
    """Authenticate a client. Returns 'worker', 'orchestrator', or None on failure."""
    client_ip = addr[0] if addr else None

    nonce = os.urandom(32)
    challenge = struct.pack(">B", MSG_AUTH_CHALLENGE) + nonce
    await write_frame(writer, challenge)

    try:
        response = await asyncio.wait_for(read_frame(reader), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning(f"Auth timeout from {addr}")
        return None

    if len(response) != 33 or response[0] != MSG_AUTH_RESPONSE:
        log.warning(f"Invalid auth response from {addr}: len={len(response)}")
        return None

    client_hmac = response[1:33]

    # Check worker secret
    worker_expected = hmac.new(WORKER_SECRET, nonce, hashlib.sha256).digest()
    if hmac.compare_digest(client_hmac, worker_expected):
        result = struct.pack(">BB", MSG_AUTH_RESULT, 0)
        await write_frame(writer, result)
        log.info(f"Auth OK from {addr} (worker secret)")
        return "worker"

    # Check orchestrator secret
    orch_expected = hmac.new(ORCH_SECRET, nonce, hashlib.sha256).digest()
    if hmac.compare_digest(client_hmac, orch_expected):
        result = struct.pack(">BB", MSG_AUTH_RESULT, 0)
        await write_frame(writer, result)
        log.info(f"Auth OK from {addr} (orchestrator secret)")
        return "orchestrator"

    # Check worker auth bypass (orchestrator vouched for this IP)
    if client_ip and client_ip in worker_auth_bypass:
        expiry = worker_auth_bypass.pop(client_ip)
        if time.monotonic() < expiry:
            result = struct.pack(">BB", MSG_AUTH_RESULT, 0)
            await write_frame(writer, result)
            log.info(f"Auth OK from {addr} (worker bypass, vouched by orchestrator)")
            return "worker"
        else:
            log.warning(f"Worker bypass for {client_ip} expired")

    log.warning(f"Auth FAILED from {addr}")
    result = struct.pack(">BB", MSG_AUTH_RESULT, 1)
    await write_frame(writer, result)
    return None


# ── Training Dataset ──

class TextDataset:
    """Tokenized text corpus served as sequential windows."""

    def __init__(self, path: str, seq_len: int):
        enc = tiktoken.get_encoding("gpt2")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = np.array(enc.encode(text), dtype=np.int32)
        self.seq_len = seq_len
        self.cursor = 0
        log.info(f"Dataset loaded: {len(self.tokens)} tokens from {path}")

    def get_batch(self, batch_size: int):
        """Return (tokens, targets) each of shape (B, S) as sequential windows."""
        S = self.seq_len
        all_tokens = []
        all_targets = []
        for _ in range(batch_size):
            # Wrap around if near end
            if self.cursor + S + 1 > len(self.tokens):
                self.cursor = 0
            window = self.tokens[self.cursor : self.cursor + S + 1]
            all_tokens.append(window[:S])
            all_targets.append(window[1:S + 1])
            self.cursor += S  # stride by seq_len (no overlap)
        return np.stack(all_tokens), np.stack(all_targets)


# ── Pipeline State Machine ──

class WorkerDisconnectedError(Exception):
    """Raised when a worker disconnects during an active training loop."""
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        super().__init__(f"Worker {worker_id} disconnected during training")


class PipelineState(Enum):
    IDLE = "idle"
    CONFIGURING = "configuring"
    TRAINING = "training"


@dataclass
class WorkerInfo:
    worker_id: int
    device_type: int  # 1=iPhone, 2=iPad
    memory_mb: int
    reader: asyncio.StreamReader = field(repr=False)
    writer: asyncio.StreamWriter = field(repr=False)
    addr: tuple = field(default=())
    stage_index: int = -1
    first_layer: int = 0
    last_layer: int = 0
    is_head: bool = False
    is_tail: bool = False
    rtt_ms: float = 0.0


class PipelineOrchestrator:
    """Persistent pipeline-parallel training orchestrator."""

    def __init__(self, config: dict, dataset: Optional[TextDataset] = None):
        self.state = PipelineState.IDLE
        self.workers: Dict[int, WorkerInfo] = {}
        self.config = config
        self.dataset = dataset
        self.min_workers = config.get("min_workers", 2)
        self._next_worker_id = 1  # Sequential worker ID counter
        self.loss_history: List[float] = []
        self.current_step: int = 0
        self.total_steps: int = config.get("total_steps", 200)
        self.stop_requested: bool = False
        # Orchestrator connection (only one allowed)
        self.orch_writer: Optional[asyncio.StreamWriter] = None
        # Event to signal workers that a new training run is starting
        self.training_start_event = asyncio.Event()
        # Event to signal that training has ended (workers should loop back)
        self.training_done_event = asyncio.Event()
        # Counts how many worker idle loops have yielded their readers
        self._readers_yielded_count = 0
        self._readers_yielded_event = asyncio.Event()

    def next_worker_id(self) -> int:
        """Return the next sequential worker ID."""
        wid = self._next_worker_id
        self._next_worker_id += 1
        return wid

    def register_worker(self, worker: WorkerInfo):
        self.workers[worker.worker_id] = worker
        count = len(self.workers)
        log.info(
            f"Worker {worker.worker_id} registered "
            f"(device_type={worker.device_type}, mem={worker.memory_mb} MB). "
            f"Total: {count}"
        )

    def remove_worker(self, worker_id: int):
        if worker_id in self.workers:
            w = self.workers.pop(worker_id)
            try:
                w.writer.close()
            except Exception:
                pass
            log.info(f"Worker {worker_id} removed. Total: {len(self.workers)}")

    def reset_assignments(self):
        """Clear all stage assignments and training metrics.

        Called between training runs and when the orchestrator disconnects
        so that status reports show workers as 'Unassigned' and metrics
        don't carry over from a previous run.

        Note: does NOT touch total_steps — that is a config value set
        before this method is called in start_training().
        """
        for w in self.workers.values():
            w.stage_index = -1
            w.first_layer = 0
            w.last_layer = 0
            w.is_head = False
            w.is_tail = False
        self.current_step = 0
        self.loss_history.clear()
        log.info(f"Reset assignments and metrics for {len(self.workers)} worker(s)")

    async def probe_workers(self):
        """Probe all registered workers to detect dead connections.

        Sends a PIPELINE_STOP to each worker (which is harmless — workers
        in their idle loop simply ignore it or treat it as a no-op between
        runs) and checks if the write succeeds.  drain() alone cannot detect
        a remotely-closed socket; an actual write is required.
        """
        probe_payload = struct.pack(">B", MSG_PIPELINE_STOP)
        dead: List[int] = []
        for w in list(self.workers.values()):
            try:
                await asyncio.wait_for(
                    write_frame(w.writer, probe_payload), timeout=5.0
                )
            except (ConnectionResetError, BrokenPipeError, OSError,
                    asyncio.TimeoutError, Exception) as e:
                log.info(f"Probe: worker {w.worker_id} is dead ({e}), removing")
                dead.append(w.worker_id)
        for wid in dead:
            self.remove_worker(wid)
        if dead:
            log.info(f"Probe complete: removed {len(dead)} dead worker(s), "
                     f"{len(self.workers)} remaining")
        else:
            log.info(f"Probe complete: all {len(self.workers)} worker(s) alive")

    def assign_stages(self):
        """Assign pipeline stages: split layers evenly across workers."""
        n_workers = len(self.workers)
        n_layers = self.config["n_layers_total"]

        sorted_workers = sorted(self.workers.values(), key=lambda w: w.worker_id)
        layers_per_worker = n_layers // n_workers
        remainder = n_layers % n_workers

        current_layer = 0
        for i, worker in enumerate(sorted_workers):
            worker.stage_index = i
            worker.first_layer = current_layer
            extra = 1 if i < remainder else 0
            worker.last_layer = current_layer + layers_per_worker + extra
            worker.is_head = (i == 0)
            worker.is_tail = (i == n_workers - 1)
            current_layer = worker.last_layer

            log.info(
                f"  Stage {i}: worker {worker.worker_id} — "
                f"layers [{worker.first_layer}, {worker.last_layer}) "
                f"{'HEAD' if worker.is_head else ''}"
                f"{'TAIL' if worker.is_tail else ''}"
            )

    async def send_pipeline_config(self):
        """Send PIPELINE_CONFIG to each worker. Returns list of dead worker IDs."""
        cfg = self.config
        num_micro_batches = cfg.get("num_micro_batches", 4)
        dead: List[int] = []
        for worker in list(self.workers.values()):
            payload = struct.pack(
                ">B"         # type
                "IIII"       # stage_index, total_stages, first_layer, last_layer
                "BB"         # is_head, is_tail
                "I"          # num_micro_batches
                "IIIII"      # vocab_size, d_model, n_heads, n_layers_total, seq_len
                "II"         # batch_size, ffn_hidden_mul
                "ff",        # learning_rate, dropout
                MSG_PIPELINE_CONFIG,
                worker.stage_index, len(self.workers),
                worker.first_layer, worker.last_layer,
                1 if worker.is_head else 0,
                1 if worker.is_tail else 0,
                num_micro_batches,
                cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                cfg["n_layers_total"], cfg["seq_len"],
                cfg["batch_size"], cfg["ffn_hidden_mul"],
                cfg["learning_rate"], cfg["dropout"],
            )
            try:
                await write_frame(worker.writer, payload)
                log.info(f"Sent PIPELINE_CONFIG to worker {worker.worker_id}")
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                log.error(f"Failed to send CONFIG to worker {worker.worker_id}: {e}")
                dead.append(worker.worker_id)
        for wid in dead:
            self.remove_worker(wid)
        return dead

    async def wait_for_config_acks(self, timeout=30.0):
        """Wait for CONFIG_ACK from all workers.

        Workers that disconnect during the wait are removed.
        Returns True if all remaining workers ACK'd successfully.
        """
        acked: set = set()
        dead: List[int] = []

        async def wait_ack(worker: WorkerInfo):
            try:
                resp = await asyncio.wait_for(
                    read_frame(worker.reader), timeout=timeout
                )
                if resp[0] == MSG_PIPELINE_CONFIG_ACK:
                    status = resp[5] if len(resp) > 5 else resp[1]
                    if status == 0:
                        log.info(f"CONFIG_ACK OK from worker {worker.worker_id}")
                        acked.add(worker.worker_id)
                    else:
                        log.error(f"CONFIG_ACK FAIL from worker {worker.worker_id}")
            except (asyncio.IncompleteReadError, ConnectionResetError,
                    BrokenPipeError, OSError, asyncio.TimeoutError) as e:
                log.error(f"Worker {worker.worker_id} disconnected during CONFIG_ACK: {e}")
                dead.append(worker.worker_id)

        tasks = [wait_ack(w) for w in list(self.workers.values())]
        await asyncio.gather(*tasks)

        for wid in dead:
            self.remove_worker(wid)

        if dead:
            log.error(f"Lost {len(dead)} worker(s) during config phase")
            return False

        expected = set(self.workers.keys())
        if acked != expected:
            missing = expected - acked
            log.error(f"Missing ACKs from workers: {missing}")
            return False
        return True

    async def send_pipeline_start(self):
        """Broadcast PIPELINE_START to all workers."""
        payload = struct.pack(">BI", MSG_PIPELINE_START, self.total_steps)
        dead: List[int] = []
        for worker in list(self.workers.values()):
            try:
                await write_frame(worker.writer, payload)
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                log.error(f"Failed to send START to worker {worker.worker_id}: {e}")
                dead.append(worker.worker_id)
        for wid in dead:
            self.remove_worker(wid)
        if dead:
            raise WorkerDisconnectedError(dead[0])
        log.info(f"PIPELINE_START sent to all workers ({self.total_steps} steps)")

    async def send_pipeline_stop(self):
        """Broadcast PIPELINE_STOP to all workers so they exit their training loops."""
        payload = struct.pack(">B", MSG_PIPELINE_STOP)
        for worker in self.workers.values():
            try:
                await write_frame(worker.writer, payload)
            except Exception:
                pass  # worker may have disconnected
        log.info("PIPELINE_STOP sent to all workers")

    def generate_batch(self, mini_batch_id: int, micro_batch_id: int):
        """Return (tokens, targets) each of shape (B, S) from the dataset."""
        B = self.config["batch_size"]
        S = self.config["seq_len"]
        V = self.config["vocab_size"]

        if self.dataset is not None:
            return self.dataset.get_batch(B)
        else:
            # Fallback: random data (no learning will occur)
            tokens = np.random.randint(0, V, size=(B, S), dtype=np.int32)
            targets = np.roll(tokens, -1, axis=1)
            targets[:, -1] = np.random.randint(0, V, size=B, dtype=np.int32)
            return tokens, targets

    def build_data_batch(self, micro_batch_id: int, mini_batch_id: int,
                         tokens: np.ndarray, targets: np.ndarray) -> bytes:
        """Build PIPELINE_DATA_BATCH message."""
        B, S = tokens.shape
        header = struct.pack(
            ">BIIII",
            MSG_PIPELINE_DATA_BATCH,
            micro_batch_id, mini_batch_id, B, S,
        )
        tokens_be = tokens.astype(">i4").tobytes()
        targets_be = targets.astype(">i4").tobytes()
        return header + tokens_be + targets_be

    async def notify_orch_loss(self, step: int, loss: float):
        """Send loss update to orchestrator if connected."""
        if self.orch_writer is not None:
            try:
                payload = struct.pack(">BIf", MSG_ORCH_LOSS_UPDATE, step, loss)
                await write_frame(self.orch_writer, payload)
            except Exception:
                pass  # orchestrator may have disconnected

    async def notify_orch_error(self, msg: str):
        """Send error message to orchestrator if connected."""
        if self.orch_writer is not None:
            try:
                msg_bytes = msg.encode("utf-8")
                payload = struct.pack(">BI", MSG_ORCH_ERROR, len(msg_bytes)) + msg_bytes
                await write_frame(self.orch_writer, payload)
            except Exception:
                pass

    async def start_training(self):
        """Full training run: probe → reset → assign stages → config → start → loop → idle."""
        self.stop_requested = False
        self.total_steps = self.config.get("total_steps", 200)

        try:
            # Probe for dead workers before we try to use them
            await self.probe_workers()

            # Clear stale assignments and metrics from previous run
            self.reset_assignments()

            n_workers = len(self.workers)
            if n_workers < self.min_workers:
                err = f"Need at least {self.min_workers} workers, have {n_workers}"
                log.error(err)
                await self.notify_orch_error(err)
                return

            # Assign stages
            self.assign_stages()

            # Signal worker idle loops to yield the reader before we send config.
            self._readers_yielded_count = 0
            self._readers_yielded_event.clear()
            self.training_start_event.set()
            self.training_done_event.clear()

            # Wait for all workers to confirm they've yielded their readers
            n = len(self.workers)
            try:
                await asyncio.wait_for(self._readers_yielded_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                log.warning(
                    f"Only {self._readers_yielded_count}/{n} workers yielded readers"
                )

            # Send config to all workers
            self.state = PipelineState.CONFIGURING
            log.info("Sending PIPELINE_CONFIG to all workers...")
            dead_on_config = await self.send_pipeline_config()
            if dead_on_config:
                # Workers died during config — abort if not enough remain
                if len(self.workers) < self.min_workers:
                    err = (f"Workers died during config. "
                           f"Need {self.min_workers}, have {len(self.workers)}")
                    log.error(err)
                    await self.notify_orch_error(err)
                    self.state = PipelineState.IDLE
                    self.training_start_event.clear()
                    self.training_done_event.set()
                    return

            # Wait for ACKs
            log.info("Waiting for CONFIG_ACK from all workers...")
            if not await self.wait_for_config_acks():
                err = f"Failed to configure all workers (have {len(self.workers)} remaining)"
                log.error(err)
                await self.notify_orch_error(err)
                self.state = PipelineState.IDLE
                self.training_start_event.clear()
                self.training_done_event.set()
                return

            # Send start
            log.info("All workers configured. Sending PIPELINE_START...")
            await self.send_pipeline_start()

            # Notify orchestrator
            if self.orch_writer:
                await write_frame(self.orch_writer,
                                  struct.pack(">B", MSG_ORCH_TRAINING_STARTED))

            # Run training loop
            await self.run_training_loop()

            # Training done — notify orchestrator
            final_loss = float(np.mean(self.loss_history[-4:])) if self.loss_history else 0.0
            if self.orch_writer:
                payload = struct.pack(">BIf", MSG_ORCH_TRAINING_STOPPED,
                                      self.current_step, final_loss)
                try:
                    await write_frame(self.orch_writer, payload)
                except Exception:
                    pass

        except WorkerDisconnectedError as e:
            log.error(f"Training aborted: {e}")
            self.remove_worker(e.worker_id)
            await self.notify_orch_error(str(e))
        except Exception as e:
            log.error(f"Training error: {e}", exc_info=True)
            await self.notify_orch_error(str(e))
        finally:
            # Tell remaining workers to exit their training loops and wait for next CONFIG
            await self.send_pipeline_stop()
            self.state = PipelineState.IDLE
            self.training_start_event.clear()
            self.training_done_event.set()
            # Clear stale assignments so status reports show "Unassigned"
            self.reset_assignments()
            log.info("Server state → IDLE. Workers stay connected.")

    async def _worker_read(self, worker: "WorkerInfo") -> bytes:
        """Read a frame from a worker, raising WorkerDisconnectedError on failure."""
        try:
            return await asyncio.wait_for(read_frame(worker.reader), timeout=60.0)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError,
                OSError, asyncio.TimeoutError) as e:
            log.error(f"Worker {worker.worker_id} read failed: {e}")
            raise WorkerDisconnectedError(worker.worker_id) from e

    async def _worker_write(self, worker: "WorkerInfo", payload: bytes):
        """Write a frame to a worker, raising WorkerDisconnectedError on failure."""
        try:
            await write_frame(worker.writer, payload)
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            log.error(f"Worker {worker.worker_id} write failed: {e}")
            raise WorkerDisconnectedError(worker.worker_id) from e

    async def run_training_loop(self):
        """Main training loop: stream data, relay activations/gradients."""
        self.state = PipelineState.TRAINING

        sorted_workers = sorted(self.workers.values(), key=lambda w: w.stage_index)
        head_worker = sorted_workers[0]
        tail_worker = sorted_workers[-1]

        M = self.config.get("num_micro_batches", 4)

        log.info(f"Starting training: {self.total_steps} mini-batches, {M} micro-batches each")

        for mini_batch in range(self.total_steps):
            if self.stop_requested:
                log.info(f"Stop requested at mini-batch {mini_batch}")
                break

            t0 = time.monotonic()
            self.current_step = mini_batch

            batch_targets: Dict[int, np.ndarray] = {}
            batch_tokens: Dict[int, np.ndarray] = {}

            # ── Phase 1: Send data batches to head ──
            if mini_batch < 3:
                log.info(f"[mb {mini_batch}] Phase 1: Sending {M} data batches to head...")
            for m in range(M):
                tokens, targets = self.generate_batch(mini_batch, m)
                batch_targets[m] = targets
                batch_tokens[m] = tokens
                data_msg = self.build_data_batch(m, mini_batch, tokens, targets)
                await self._worker_write(head_worker, data_msg)

            # ── Phase 2: Relay activations forward ──
            if mini_batch < 3:
                log.info(f"[mb {mini_batch}] Phase 2: Relaying {M} activations head→tail...")
            for m in range(M):
                act_frame = await self._worker_read(head_worker)
                if act_frame[0] != MSG_PIPELINE_ACTIVATION:
                    log.warning(f"Expected ACTIVATION, got 0x{act_frame[0]:02x}")
                    continue

                _, mb_id, src_stage, dst_stage = struct.unpack_from(">BIII", act_frame, 0)
                offset = 13
                has_targets = act_frame[offset]
                offset += 1
                existing_tgt_len = struct.unpack_from(">I", act_frame, offset)[0]
                offset += 4 + existing_tgt_len
                st_len = struct.unpack_from(">I", act_frame, offset)[0]
                offset += 4
                st_data = act_frame[offset:offset + st_len]

                tgt_np = batch_targets.get(mb_id, batch_targets.get(m))
                tgt_be = tgt_np.astype(">i4").tobytes()

                relay_msg = struct.pack(
                    ">BIII",
                    MSG_PIPELINE_ACTIVATION,
                    mb_id, src_stage, src_stage + 1,
                )
                relay_msg += struct.pack(">B", 1)  # has_targets = true
                relay_msg += struct.pack(">I", len(tgt_be)) + tgt_be
                relay_msg += struct.pack(">I", st_len) + st_data

                await self._worker_write(tail_worker, relay_msg)

            # ── Phase 3: Relay gradients backward ──
            if mini_batch < 3:
                log.info(f"[mb {mini_batch}] Phase 3: Relaying {M} gradients tail→head...")
            for m in range(M):
                grad_frame = await self._worker_read(tail_worker)
                if grad_frame[0] == MSG_PIPELINE_LOSS_REPORT:
                    _, mb_id, ub_id, loss_val, step = struct.unpack_from(">BIIfI", grad_frame, 0)
                    self.loss_history.append(loss_val)
                    # Forward loss to orchestrator
                    await self.notify_orch_loss(mini_batch, loss_val)
                    grad_frame = await self._worker_read(tail_worker)

                if grad_frame[0] != MSG_PIPELINE_GRADIENT:
                    log.warning(f"Expected GRADIENT, got 0x{grad_frame[0]:02x}")
                    continue

                await self._worker_write(head_worker, grad_frame)

            # ── Phase 4: Sync barrier ──
            barrier = struct.pack(">BI", MSG_PIPELINE_SYNC_BARRIER, mini_batch)
            for w in sorted_workers:
                await self._worker_write(w, barrier)

            for w in sorted_workers:
                ack = await self._worker_read(w)
                if ack[0] != MSG_PIPELINE_SYNC_ACK:
                    log.warning(f"Expected SYNC_ACK from worker {w.worker_id}, got 0x{ack[0]:02x}")

            elapsed_ms = (time.monotonic() - t0) * 1000

            if mini_batch % 10 == 0 or mini_batch < 5:
                avg_loss = (
                    np.mean(self.loss_history[-M:])
                    if self.loss_history else float("nan")
                )
                log.info(
                    f"[mini-batch {mini_batch + 1}/{self.total_steps}] "
                    f"loss={avg_loss:.4f} elapsed={elapsed_ms:.0f}ms"
                )

        self.current_step = min(mini_batch + 1, self.total_steps) if self.total_steps > 0 else 0
        log.info(
            f"Training complete: {self.current_step} mini-batches. "
            f"Final avg loss: {np.mean(self.loss_history[-M:]) if self.loss_history else 0:.4f}"
        )

    def build_status_report(self) -> bytes:
        """Build ORCH_STATUS_REPORT message."""
        state_byte = {
            PipelineState.IDLE: 0,
            PipelineState.CONFIGURING: 1,
            PipelineState.TRAINING: 2,
        }[self.state]

        latest_loss = float(self.loss_history[-1]) if self.loss_history else 0.0
        n_workers = len(self.workers)

        payload = struct.pack(
            ">BB I II f",
            MSG_ORCH_STATUS_REPORT,
            state_byte,
            n_workers,
            self.current_step, self.total_steps,
            latest_loss,
        )

        for w in self.workers.values():
            stage = w.stage_index if w.stage_index >= 0 else 0xFFFFFFFF
            payload += struct.pack(">IIII", w.worker_id, w.device_type, w.memory_mb, stage)

        return payload


# ── Global orchestrator ──
orchestrator: Optional[PipelineOrchestrator] = None


# ── Unified client handler ──

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle any incoming connection. Auth determines role, first message confirms it."""
    global orchestrator
    addr = writer.get_extra_info("peername")
    log.info(f"Client connected: {addr}")

    role = await authenticate_client(reader, writer, addr)
    if role is None:
        log.warning(f"Rejecting unauthenticated client {addr}")
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return

    try:
        # Read first message to confirm client type
        first_frame = await asyncio.wait_for(read_frame(reader), timeout=30.0)
        msg_type = first_frame[0]

        if msg_type == MSG_PIPELINE_REGISTER and role == "worker":
            await handle_worker_session(reader, writer, addr, first_frame)
        elif msg_type == MSG_ORCH_IDENTIFY and role == "orchestrator":
            await handle_orchestrator_session(reader, writer, addr)
        else:
            log.warning(f"Role mismatch from {addr}: auth={role}, msg=0x{msg_type:02x}")

    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError,
            OSError, asyncio.TimeoutError):
        log.info(f"Client {addr} disconnected during identification")
    except Exception as e:
        log.error(f"Error identifying client {addr}: {e}", exc_info=True)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def handle_worker_session(reader, writer, addr, reg_frame):
    """Handle a worker connection (already authenticated, first frame received)."""
    global orchestrator
    worker_info: Optional[WorkerInfo] = None

    try:
        _client_id, device_type, memory_mb = struct.unpack_from(">III", reg_frame, 1)
        worker_id = orchestrator.next_worker_id()
        worker_info = WorkerInfo(
            worker_id=worker_id,
            device_type=device_type,
            memory_mb=memory_mb,
            reader=reader,
            writer=writer,
            addr=addr,
        )
        orchestrator.register_worker(worker_info)

        # Idle loop: just wait. No socket reads here — the reader is exclusively
        # used by the training coroutines (send_pipeline_config, wait_for_config_acks,
        # run_training_loop). We detect dead connections by attempting a write.
        while True:
            try:
                if orchestrator.training_start_event.is_set():
                    # Only participate if this worker was included in the current training run
                    if worker_info.stage_index >= 0:
                        # Signal that this worker has yielded its reader
                        orchestrator._readers_yielded_count += 1
                        if orchestrator._readers_yielded_count >= len(orchestrator.workers):
                            orchestrator._readers_yielded_event.set()
                    # Wait for training to finish before resuming idle loop
                    await orchestrator.training_done_event.wait()
                else:
                    # Sleep, but wake immediately if training starts
                    start_waiter = asyncio.create_task(
                        orchestrator.training_start_event.wait()
                    )
                    sleep_task = asyncio.create_task(asyncio.sleep(3.0))
                    done, pending = await asyncio.wait(
                        {start_waiter, sleep_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    if start_waiter in done:
                        # Training is starting — loop back to yield reader
                        continue
                    # Sleep finished — probe connection liveness.
                    # drain() alone cannot detect a remotely-closed socket;
                    # we must write actual data. PIPELINE_STOP is harmless —
                    # idle workers simply ignore it ("Received STOP (between
                    # runs), continuing to wait for CONFIG...").
                    try:
                        probe = struct.pack(">B", MSG_PIPELINE_STOP)
                        await asyncio.wait_for(
                            write_frame(writer, probe), timeout=5.0
                        )
                    except (ConnectionResetError, BrokenPipeError, OSError,
                            asyncio.TimeoutError):
                        log.info(f"Worker {worker_info.worker_id} connection lost (probe failed)")
                        break
            except asyncio.CancelledError:
                break

    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError,
            OSError, asyncio.TimeoutError):
        log.info(f"Worker {addr} disconnected")
    except Exception as e:
        log.error(f"Error handling worker {addr}: {e}", exc_info=True)
    finally:
        if worker_info:
            orchestrator.remove_worker(worker_info.worker_id)


async def handle_orchestrator_session(reader, writer, addr):
    """Handle an orchestrator connection (already authenticated, ORCH_IDENTIFY received)."""
    global orchestrator

    # Only one orchestrator at a time
    if orchestrator.orch_writer is not None:
        log.warning(f"Rejecting second orchestrator from {addr}")
        err_msg = b"Another orchestrator is already connected"
        payload = struct.pack(">BI", MSG_ORCH_ERROR, len(err_msg)) + err_msg
        await write_frame(writer, payload)
        return

    orchestrator.orch_writer = writer
    log.info(f"Orchestrator registered from {addr}")

    # Send initial status
    try:
        status = orchestrator.build_status_report()
        await write_frame(writer, status)
    except Exception:
        pass

    try:
        while True:
            frame = await read_frame(reader)
            msg_type = frame[0]

            if msg_type == MSG_ORCH_UPDATE_CONFIG:
                # Parse config update
                (vocab_size, d_model, n_heads, n_layers, seq_len,
                 batch_size, ffn_hidden_mul, micro_batches, total_steps,
                 lr, dropout) = struct.unpack_from(">IIIIIIIIIff", frame, 1)

                orchestrator.config.update({
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers_total": n_layers,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "ffn_hidden_mul": ffn_hidden_mul,
                    "num_micro_batches": micro_batches,
                    "total_steps": total_steps,
                    "learning_rate": lr,
                    "dropout": dropout,
                })
                orchestrator.total_steps = total_steps
                log.info(f"Config updated by orchestrator: {orchestrator.config}")
                ack = struct.pack(">BB", MSG_ORCH_CONFIG_ACK, 0)
                await write_frame(writer, ack)

            elif msg_type == MSG_ORCH_START_TRAINING:
                if orchestrator.state != PipelineState.IDLE:
                    err = f"Cannot start: server is {orchestrator.state.value}"
                    log.warning(err)
                    await orchestrator.notify_orch_error(err)
                else:
                    log.info("Orchestrator requested training start")
                    asyncio.create_task(orchestrator.start_training())

            elif msg_type == MSG_ORCH_STOP_TRAINING:
                if orchestrator.state == PipelineState.TRAINING:
                    log.info("Orchestrator requested training stop")
                    orchestrator.stop_requested = True
                else:
                    log.warning("Stop requested but not currently training")

            elif msg_type == MSG_ORCH_GET_STATUS:
                status = orchestrator.build_status_report()
                await write_frame(writer, status)

            elif msg_type == MSG_ORCH_ALLOW_WORKER:
                # Orchestrator wants its device to also connect as a worker.
                # Whitelist the orchestrator's IP for one worker auth bypass.
                orch_ip = addr[0] if addr else None
                if orch_ip:
                    worker_auth_bypass[orch_ip] = time.monotonic() + BYPASS_EXPIRY_SECONDS
                    log.info(f"Worker auth bypass set for IP {orch_ip} (expires in {BYPASS_EXPIRY_SECONDS}s)")
                    ack = struct.pack(">BB", MSG_ORCH_ALLOW_WORKER_ACK, 0)
                    await write_frame(writer, ack)
                else:
                    log.warning("Cannot determine orchestrator IP for bypass")
                    ack = struct.pack(">BB", MSG_ORCH_ALLOW_WORKER_ACK, 1)
                    await write_frame(writer, ack)

            else:
                log.warning(f"Unknown orchestrator message: 0x{msg_type:02x}")

    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError,
            OSError, asyncio.TimeoutError):
        log.info(f"Orchestrator {addr} disconnected")
    except Exception as e:
        log.error(f"Orchestrator error: {e}", exc_info=True)
    finally:
        orchestrator.orch_writer = None
        # If training is active, request a stop so the run winds down
        if orchestrator.state == PipelineState.TRAINING:
            log.info("Orchestrator disconnected during training — requesting stop")
            orchestrator.stop_requested = True
        else:
            # Clear stale assignments when orchestrator disconnects outside training
            orchestrator.reset_assignments()
        log.info("Orchestrator slot freed")


async def main(host: str, port: int, worker_secret: str, orch_secret: str,
               config: dict, data_path: str = None):
    global orchestrator, WORKER_SECRET, ORCH_SECRET
    WORKER_SECRET = worker_secret.encode("utf-8")
    ORCH_SECRET = orch_secret.encode("utf-8")

    dataset = None
    if data_path:
        dataset = TextDataset(data_path, config["seq_len"])
    else:
        log.warning("No --data provided; training will use random data (loss won't decrease)")

    orchestrator = PipelineOrchestrator(config, dataset=dataset)

    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f"Listening on {addr[0]}:{addr[1]} (workers + orchestrator)")
    log.info(f"Auth: HMAC-SHA256 (worker secret len={len(WORKER_SECRET)}, orch secret len={len(ORCH_SECRET)})")
    log.info(f"Default config: {config}")
    log.info("Server IDLE — waiting for orchestrator + workers...")

    async with server:
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline-parallel training server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9998, help="Server port")
    parser.add_argument("--secret", type=str, required=True,
                        help="Worker secret for HMAC authentication")
    parser.add_argument("--orch-secret", type=str, required=True,
                        help="Orchestrator secret for HMAC authentication")
    parser.add_argument("--min-workers", type=int, default=2,
                        help="Minimum workers before allowing training start")
    parser.add_argument("--total-steps", type=int, default=200,
                        help="Default total mini-batches to train")
    parser.add_argument("--micro-batches", type=int, default=4,
                        help="Micro-batches per mini-batch for pipeline fill")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ffn-hidden-mul", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training text file (tokenized with GPT-2 BPE)")
    args = parser.parse_args()

    config = {
        "min_workers": args.min_workers,
        "total_steps": args.total_steps,
        "num_micro_batches": args.micro_batches,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers_total": args.n_layers,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "ffn_hidden_mul": args.ffn_hidden_mul,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
    }

    asyncio.run(main(args.host, args.port, args.secret, args.orch_secret, config,
                     data_path=args.data))
