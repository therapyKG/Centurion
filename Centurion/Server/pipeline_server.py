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
MSG_ORCH_RESTART_SERVER = 0x96  # Orchestrator requests full server reset
MSG_ORCH_ALLOW_WORKER_ACK = 0xA6  # Server confirms the bypass is ready
MSG_ORCH_RESTART_ACK = 0xA7  # Server confirms restart complete

MSG_ORCH_STATUS_REPORT = 0xA0
MSG_ORCH_CONFIG_ACK = 0xA1
MSG_ORCH_TRAINING_STARTED = 0xA2
MSG_ORCH_TRAINING_STOPPED = 0xA3
MSG_ORCH_LOSS_UPDATE = 0xA4
MSG_ORCH_ERROR = 0xA5

# ── Profiling message types ──
MSG_PROFILE_REQUEST = 0x48
MSG_PROFILE_RESULT = 0x49
MSG_ORCH_PROFILE_REPORT = 0xA8

STALE_WORKER_FRAME_TYPES = {
    MSG_PIPELINE_STOP,
    MSG_PIPELINE_CONFIG_ACK,
    MSG_PIPELINE_ACTIVATION,
    MSG_PIPELINE_GRADIENT,
    MSG_PIPELINE_SYNC_ACK,
    MSG_PIPELINE_LOSS_REPORT,
}

# ── Secrets ──
WORKER_SECRET = b""
ORCH_SECRET = b""

# IPs that are allowed to skip worker auth (one-time use, set by orchestrator)
# Maps IP -> expiry timestamp (monotonic)
worker_auth_bypass: Dict[str, float] = {}
BYPASS_EXPIRY_SECONDS = 30.0
PROFILE_TIMEOUT_SECONDS = 45.0
WORKER_READ_TIMEOUT_SECONDS = 120.0
MEMORY_RESERVE_BYTES = 512 * 1024 * 1024
MEMORY_SAFETY_FRACTION = 0.55
PHONE_MEMORY_SAFETY_FRACTION = 0.45
LAYER_MEMORY_MULTIPLIER = 2.0
ROLE_MEMORY_MULTIPLIER = 1.5
PARAM_TRAINING_MULTIPLIER = 3.5
UNUSED_PARAM_MULTIPLIER = 1.3
PER_LAYER_TRANSIENT_BYTES = 32 * 1024 * 1024
HEAD_TAIL_EXTRA_MEMORY_BYTES = 256 * 1024 * 1024


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

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.cursor = 0

    @classmethod
    def from_file(cls, path: str, seq_len: int) -> "TextDataset":
        enc = tiktoken.get_encoding("gpt2")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = np.array(enc.encode(text), dtype=np.int32)
        log.info(f"Dataset loaded: {len(tokens)} tokens from {path}")
        return cls(tokens, seq_len)

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


def load_wikitext103(seq_len: int, split: str = "train") -> TextDataset:
    """Load wikitext-103 dataset. Downloads and caches tokenized numpy on first use."""
    cache_dir = os.path.expanduser("~/.cache/centurion/wikitext-103")
    cache_path = os.path.join(cache_dir, f"{split}_tokens.npy")

    if os.path.exists(cache_path):
        log.info(f"Loading cached wikitext-103 {split} from {cache_path}")
        tokens = np.load(cache_path)
        log.info(f"Loaded {len(tokens)} tokens from cache")
        return TextDataset(tokens, seq_len)

    log.info(f"Downloading wikitext-103 {split} (first time, will be cached)...")
    os.makedirs(cache_dir, exist_ok=True)

    # Download Parquet file from HuggingFace and extract text
    import urllib.request
    import pyarrow.parquet as pq

    base = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet"
    url = f"{base}/wikitext-103-raw-v1/{split}/0000.parquet"
    log.info(f"Downloading wikitext-103 {split} parquet from HuggingFace...")
    try:
        tmp = os.path.join(cache_dir, f"{split}_tmp.parquet")
        req = urllib.request.Request(url, headers={"User-Agent": "Centurion/1.0"})
        resp = urllib.request.urlopen(req)
        with open(tmp, "wb") as f:
            f.write(resp.read())
        log.info(f"Downloaded parquet, reading...")
        table = pq.read_table(tmp)
        text = "\n".join(table["text"].to_pylist())
        os.remove(tmp)
        log.info(f"Extracted wikitext-103 {split}: {len(text)} chars")
    except Exception as e:
        log.error(f"Failed to download wikitext-103: {e}")
        raise

    # Tokenize with GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    log.info("Tokenizing with GPT-2 BPE...")
    tokens = np.array(enc.encode(text), dtype=np.int32)
    log.info(f"Tokenized: {len(tokens)} tokens")

    np.save(cache_path, tokens)
    log.info(f"Cached tokenized data to {cache_path}")

    return TextDataset(tokens, seq_len)


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


@dataclass
class ProfileResult:
    worker_id: int
    device_type: int
    layer_forward_ms: float = 0.0
    layer_backward_ms: float = 0.0
    layer_peak_memory_bytes: int = 0
    head_forward_ms: float = 0.0
    head_backward_ms: float = 0.0
    head_peak_memory_bytes: int = 0
    tail_forward_ms: float = 0.0
    tail_backward_ms: float = 0.0
    tail_peak_memory_bytes: int = 0
    available_memory_bytes: int = 0
    rtt_ms: float = 0.0

    @property
    def compute_speed(self) -> float:
        """Layers per second based on single-layer fwd+bwd time."""
        total_ms = self.layer_forward_ms + self.layer_backward_ms
        return 1000.0 / total_ms if total_ms > 0 else 0.0

    @property
    def available_memory_mb(self) -> int:
        return self.available_memory_bytes // (1024 * 1024)

    def estimated_layer_bytes(self, config: dict) -> int:
        d_model = config["d_model"]
        ffn_hidden = d_model * config["ffn_hidden_mul"]
        param_count = (
            4 * d_model * d_model +      # Q, K, V, O projections
            2 * d_model * ffn_hidden +   # FFN up/down projections
            4 * d_model                  # two LayerNorms
        )
        param_bytes = param_count * 4
        profile_bytes = int(self.layer_peak_memory_bytes * LAYER_MEMORY_MULTIPLIER)
        formula_bytes = int(param_bytes * PARAM_TRAINING_MULTIPLIER) + PER_LAYER_TRANSIENT_BYTES
        return max(profile_bytes, formula_bytes)

    def estimated_embedding_bytes(self, config: dict, role: str) -> int:
        d_model = config["d_model"]
        vocab_size = config["vocab_size"]
        seq_len = config["seq_len"]
        embedding_param_bytes = (vocab_size * d_model + seq_len * d_model) * 4
        multiplier = PARAM_TRAINING_MULTIPLIER if role in ("head", "tail") else UNUSED_PARAM_MULTIPLIER
        return int(embedding_param_bytes * multiplier)

    def memory_safety_fraction(self) -> float:
        # iPhones have shown less practical MLX headroom than iPads at the same
        # nominal RAM, so let available memory count for less on phones.
        if self.device_type == 1:
            return PHONE_MEMORY_SAFETY_FRACTION
        return MEMORY_SAFETY_FRACTION

    def estimated_role_extra_bytes(self, role: str) -> int:
        if role in ("head", "tail"):
            return HEAD_TAIL_EXTRA_MEMORY_BYTES
        return 0

    def estimated_fixed_overhead_bytes(self, role: str, config: dict) -> int:
        base_role_mem = max(self.head_peak_memory_bytes, self.tail_peak_memory_bytes)
        if role == "head":
            role_mem = self.head_peak_memory_bytes
        elif role == "tail":
            role_mem = self.tail_peak_memory_bytes
        else:
            role_mem = base_role_mem

        return (
            int(role_mem * ROLE_MEMORY_MULTIPLIER) +
            self.estimated_embedding_bytes(config, role) +
            self.estimated_role_extra_bytes(role)
        )

    def max_layers(self, role: str, config: dict) -> int:
        """Conservative max layers this worker can hold for a given role.

        The worker-side profile measures a tiny probe model. Full training also
        has optimizer state, gradients, MLX cache pressure, and transient arrays,
        so keep a large reserve and multiply observed layer/role costs.
        """
        safety_fraction = self.memory_safety_fraction()
        usable = min(
            int(self.available_memory_bytes * safety_fraction),
            max(0, self.available_memory_bytes - MEMORY_RESERVE_BYTES),
        )
        layer_mem = self.estimated_layer_bytes(config)
        if layer_mem <= 0:
            return 0
        usable -= self.estimated_fixed_overhead_bytes(role, config)
        return max(0, usable // layer_mem)


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
        self._training_task: Optional[asyncio.Task] = None

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

    async def drain_stale_worker_frames(
        self,
        worker: WorkerInfo,
        reason: str,
        max_frames: int = 32,
        idle_timeout: float = 0.05,
    ):
        """Drop queued frames from a previous run before reusing a worker reader."""
        drained = 0
        while drained < max_frames:
            try:
                frame = await asyncio.wait_for(
                    read_frame(worker.reader), timeout=idle_timeout
                )
            except asyncio.TimeoutError:
                break
            except (asyncio.IncompleteReadError, ConnectionResetError,
                    BrokenPipeError, OSError) as e:
                raise WorkerDisconnectedError(worker.worker_id) from e

            drained += 1
            msg_type = frame[0] if frame else -1
            log.warning(
                f"Drained stale worker {worker.worker_id} frame while {reason}: "
                f"type=0x{msg_type:02x} len={len(frame)}"
            )

        if drained >= max_frames:
            log.warning(
                f"Stopped draining worker {worker.worker_id} after {drained} "
                f"queued frame(s) while {reason}"
            )

    async def settle_workers_before_profiling(self):
        """Ask workers to stop any old loop, then drain stale frames."""
        stop_payload = struct.pack(">B", MSG_PIPELINE_STOP)
        dead: List[int] = []

        for w in list(self.workers.values()):
            try:
                await write_frame(w.writer, stop_payload)
            except Exception:
                dead.append(w.worker_id)

        if dead:
            for wid in dead:
                self.remove_worker(wid)

        await asyncio.sleep(0.25)

        for w in list(self.workers.values()):
            try:
                await self.drain_stale_worker_frames(w, "settling before profiling")
            except WorkerDisconnectedError as e:
                log.info(f"Worker {e.worker_id} disconnected while settling")
                self.remove_worker(e.worker_id)

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

    async def run_profiling_phase(self) -> Dict[int, ProfileResult]:
        """Send PROFILE_REQUEST to each worker, collect PROFILE_RESULT.

        Returns dict mapping worker_id -> ProfileResult.
        Workers that fail profiling are removed.
        """
        cfg = self.config
        profile_req = struct.pack(
            ">B IIIIIII",
            MSG_PROFILE_REQUEST,
            cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
            cfg["n_layers_total"], cfg["seq_len"],
            cfg["batch_size"], cfg["ffn_hidden_mul"],
        )

        results: Dict[int, ProfileResult] = {}
        dead: List[int] = []

        for w in list(self.workers.values()):
            t0 = time.monotonic()
            try:
                await self.drain_stale_worker_frames(w, "starting profiling")
                await write_frame(w.writer, profile_req)
                log.info(f"Sent PROFILE_REQUEST to worker {w.worker_id}")

                deadline = time.monotonic() + PROFILE_TIMEOUT_SECONDS
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    resp = await asyncio.wait_for(
                        read_frame(w.reader), timeout=remaining
                    )
                    wall_time = (time.monotonic() - t0) * 1000  # ms

                    if resp and resp[0] == MSG_PROFILE_RESULT and len(resp) >= 49:
                        break

                    msg_type = resp[0] if resp else -1
                    if msg_type in STALE_WORKER_FRAME_TYPES:
                        log.warning(
                            f"Ignoring stale worker {w.worker_id} frame while "
                            f"waiting for PROFILE_RESULT: type=0x{msg_type:02x} "
                            f"len={len(resp)}"
                        )
                        continue

                    log.error(f"Bad PROFILE_RESULT from worker {w.worker_id}: "
                              f"type=0x{msg_type:02x} len={len(resp)}")
                    dead.append(w.worker_id)
                    break

                if w.worker_id in dead:
                    continue

                offset = 1
                wid = struct.unpack_from(">I", resp, offset)[0]; offset += 4
                layer_fwd = struct.unpack_from(">f", resp, offset)[0]; offset += 4
                layer_bwd = struct.unpack_from(">f", resp, offset)[0]; offset += 4
                layer_mem = struct.unpack_from(">I", resp, offset)[0]; offset += 4
                head_fwd = struct.unpack_from(">f", resp, offset)[0]; offset += 4
                head_bwd = struct.unpack_from(">f", resp, offset)[0]; offset += 4
                head_mem = struct.unpack_from(">I", resp, offset)[0]; offset += 4
                tail_fwd = struct.unpack_from(">f", resp, offset)[0]; offset += 4
                tail_bwd = struct.unpack_from(">f", resp, offset)[0]; offset += 4
                tail_mem = struct.unpack_from(">I", resp, offset)[0]; offset += 4
                avail_mem_raw = struct.unpack_from(">I", resp, offset)[0]; offset += 4
                # New workers send MB so 8+ GB devices fit in 32 bits.
                # Older workers sent bytes, so keep byte-based values compatible.
                if avail_mem_raw > 1024 * 1024:
                    avail_mem = avail_mem_raw
                else:
                    avail_mem = avail_mem_raw * 1024 * 1024
                dev_type = struct.unpack_from(">I", resp, offset)[0]; offset += 4

                # RTT = wall_time - worker's total compute time
                worker_compute = layer_fwd + layer_bwd + head_fwd + head_bwd + tail_fwd + tail_bwd
                rtt = max(0, wall_time - worker_compute)

                pr = ProfileResult(
                    worker_id=w.worker_id,
                    device_type=dev_type,
                    layer_forward_ms=layer_fwd,
                    layer_backward_ms=layer_bwd,
                    layer_peak_memory_bytes=layer_mem,
                    head_forward_ms=head_fwd,
                    head_backward_ms=head_bwd,
                    head_peak_memory_bytes=head_mem,
                    tail_forward_ms=tail_fwd,
                    tail_backward_ms=tail_bwd,
                    tail_peak_memory_bytes=tail_mem,
                    available_memory_bytes=avail_mem,
                    rtt_ms=rtt,
                )
                results[w.worker_id] = pr
                w.rtt_ms = rtt

                log.info(
                    f"Profile worker {w.worker_id}: "
                    f"layer={layer_fwd:.0f}+{layer_bwd:.0f}ms, "
                    f"head_mem={head_mem // (1024*1024)}MB, "
                    f"tail_mem={tail_mem // (1024*1024)}MB, "
                    f"layer_mem={layer_mem // (1024*1024)}MB, "
                    f"avail={avail_mem // (1024*1024)}MB, "
                    f"speed={pr.compute_speed:.1f} L/s, "
                    f"rtt={rtt:.0f}ms"
                )

            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError,
                    OSError, asyncio.TimeoutError) as e:
                log.error(f"Worker {w.worker_id} failed profiling: {e}")
                await self.notify_orch_error(
                    f"Worker {w.worker_id} disconnected during profiling"
                )
                dead.append(w.worker_id)

        for wid in dead:
            self.remove_worker(wid)

        return results

    def assign_stages_smart(self, profiles: Dict[int, ProfileResult]) -> bool:
        """Assign pipeline stages proportional to compute speed, capped by memory.

        Falls back to even split when profiling is incomplete. Returns False
        when profiled capacity is insufficient for the requested model.
        """
        n_workers = len(self.workers)
        n_layers = self.config["n_layers_total"]

        if not profiles or len(profiles) < n_workers:
            log.warning("Incomplete profiling data, falling back to even split")
            self.assign_stages()
            return True

        workers = list(self.workers.values())
        if n_workers >= 2:
            head_worker = max(
                workers,
                key=lambda w: (
                    profiles[w.worker_id].max_layers("head", self.config),
                    profiles[w.worker_id].available_memory_bytes,
                )
            )
            remaining = [w for w in workers if w.worker_id != head_worker.worker_id]
            tail_worker = max(
                remaining,
                key=lambda w: (
                    profiles[w.worker_id].max_layers("tail", self.config),
                    profiles[w.worker_id].available_memory_bytes,
                )
            )
            middle_workers = [
                w for w in remaining
                if w.worker_id != tail_worker.worker_id
            ]
            middle_workers.sort(
                key=lambda w: profiles[w.worker_id].compute_speed,
                reverse=True,
            )
            sorted_workers = [head_worker] + middle_workers + [tail_worker]
        else:
            sorted_workers = workers

        if all(profiles[w.worker_id].compute_speed <= 0 for w in sorted_workers):
            log.warning("All workers reported zero speed, falling back to even split")
            self.assign_stages()
            return True

        raw_layers = []
        for i, w in enumerate(sorted_workers):
            pr = profiles[w.worker_id]

            # Determine role for memory cap
            if i == 0:
                role = "head"
            elif i == n_workers - 1:
                role = "tail"
            else:
                role = "middle"

            cap = pr.max_layers(role, self.config)
            raw_layers.append((w, cap, role, pr))

        no_capacity = [
            w.worker_id for w, cap, _, _ in raw_layers
            if cap < 1
        ]
        if no_capacity:
            log.error(f"Workers have no safe layer capacity: {no_capacity}")
            return False

        total_capacity = sum(cap for _, cap, _, _ in raw_layers)
        if total_capacity < n_layers:
            log.error(
                f"Insufficient profiled memory capacity: {total_capacity} "
                f"layers fit, {n_layers} requested"
            )
            return False

        hard_caps = [cap for _, cap, _, _ in raw_layers]
        soft_caps = [max(1, cap - 1) for cap in hard_caps]
        if sum(soft_caps) >= n_layers:
            assignment_caps = soft_caps
            log.info("Using one-layer memory headroom for assignment caps")
        else:
            assignment_caps = hard_caps
            log.info("Using full memory caps; requested layers exceed soft capacity")

        def role_overhead_ms(pr: ProfileResult, role: str) -> float:
            if role == "head":
                return pr.head_forward_ms + pr.head_backward_ms
            if role == "tail":
                return pr.tail_forward_ms + pr.tail_backward_ms
            return 0.0

        def estimated_stage_ms(index: int, layers: int) -> float:
            _, _, role, pr = raw_layers[index]
            layer_ms = pr.layer_forward_ms + pr.layer_backward_ms
            return layer_ms * layers + role_overhead_ms(pr, role) + pr.rtt_ms

        assigned = [1] * n_workers
        remaining = n_layers - sum(assigned)
        while remaining > 0:
            candidates = []
            for i in range(n_workers):
                if assigned[i] >= assignment_caps[i]:
                    continue

                trial_layers = assigned[i] + 1
                trial_stage_times = [
                    estimated_stage_ms(j, trial_layers if j == i else assigned[j])
                    for j in range(n_workers)
                ]
                _, hard_cap, _, _ = raw_layers[i]
                candidates.append((
                    max(trial_stage_times),
                    trial_stage_times[i],
                    -(assignment_caps[i] - trial_layers),
                    -(hard_cap - trial_layers),
                    i,
                ))

            if not candidates:
                log.error("No remaining memory headroom while assigning layers")
                return False

            idx = min(candidates)[-1]
            assigned[idx] += 1
            remaining -= 1

        # Apply assignments
        current_layer = 0
        for i, (w, cap, role, pr) in enumerate(raw_layers):
            w.stage_index = i
            w.first_layer = current_layer
            w.last_layer = current_layer + assigned[i]
            w.is_head = (i == 0)
            w.is_tail = (i == n_workers - 1)
            current_layer = w.last_layer

            # Estimate step time for this worker
            layer_time = (pr.layer_forward_ms + pr.layer_backward_ms) * assigned[i]
            overhead = 0
            if w.is_head:
                overhead = pr.head_forward_ms + pr.head_backward_ms
            if w.is_tail:
                overhead = pr.tail_forward_ms + pr.tail_backward_ms
            est_step = layer_time + overhead + pr.rtt_ms
            est_layer_mb = pr.estimated_layer_bytes(self.config) // (1024 * 1024)
            est_embed_mb = pr.estimated_embedding_bytes(self.config, role) // (1024 * 1024)
            est_fixed_mb = pr.estimated_fixed_overhead_bytes(role, self.config) // (1024 * 1024)
            safety_fraction = pr.memory_safety_fraction()

            log.info(
                f"  Stage {i} ({'HEAD' if w.is_head else 'TAIL' if w.is_tail else 'MID'}): "
                f"worker {w.worker_id} — layers [{w.first_layer}, {w.last_layer}) "
                f"[{assigned[i]} layers], speed={pr.compute_speed:.1f} L/s, "
                f"est_step={est_step:.0f}ms, avail={pr.available_memory_mb}MB, "
                f"max={cap}, assign_cap={assignment_caps[i]}, "
                f"safety={safety_fraction:.2f}, "
                f"est_layer={est_layer_mb}MB, est_embed={est_embed_mb}MB, "
                f"est_fixed={est_fixed_mb}MB"
            )
        return True

    async def notify_orch_profile_report(self, profiles: Dict[int, ProfileResult]):
        """Send ORCH_PROFILE_REPORT to orchestrator with per-worker profiling results."""
        if self.orch_writer is None:
            return

        sorted_workers = sorted(self.workers.values(), key=lambda w: w.stage_index)
        n = len(sorted_workers)

        payload = struct.pack(">BI", MSG_ORCH_PROFILE_REPORT, n)

        for w in sorted_workers:
            pr = profiles.get(w.worker_id)
            if pr is None:
                continue

            assigned_layers = w.last_layer - w.first_layer
            role = "head" if w.is_head else ("tail" if w.is_tail else "middle")
            cap = pr.max_layers(role, self.config)

            layer_time = (pr.layer_forward_ms + pr.layer_backward_ms) * assigned_layers
            overhead = 0
            if w.is_head:
                overhead = pr.head_forward_ms + pr.head_backward_ms
            if w.is_tail:
                overhead = pr.tail_forward_ms + pr.tail_backward_ms
            est_step = layer_time + overhead + pr.rtt_ms

            payload += struct.pack(
                ">II I f I I I B B f I f",
                w.worker_id, pr.device_type,
                pr.available_memory_mb,
                pr.compute_speed,
                assigned_layers,
                w.first_layer, w.last_layer,
                1 if w.is_head else 0,
                1 if w.is_tail else 0,
                est_step,
                cap,
                pr.rtt_ms,
            )

        try:
            await write_frame(self.orch_writer, payload)
            log.info(f"Sent ORCH_PROFILE_REPORT ({n} workers) to orchestrator")
        except Exception:
            pass

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
        training_started = False

        try:
            # Probe for dead workers before we try to use them
            await self.probe_workers()

            # Clear stale assignments and metrics from previous run
            self.reset_assignments()
            self.state = PipelineState.CONFIGURING

            n_workers = len(self.workers)
            if n_workers < self.min_workers:
                err = f"Need at least {self.min_workers} workers, have {n_workers}"
                log.error(err)
                await self.notify_orch_error(err)
                self.state = PipelineState.IDLE
                return

            # Signal worker idle loops to yield their readers for profiling
            self._readers_yielded_count = 0
            self._readers_yielded_event.clear()
            self.training_start_event.set()
            self.training_done_event.clear()

            n = len(self.workers)
            try:
                await asyncio.wait_for(self._readers_yielded_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                log.warning(
                    f"Only {self._readers_yielded_count}/{n} workers yielded readers for profiling"
                )

            # A cancelled previous run can leave activation/gradient/loss frames
            # queued on the socket.  Quiesce workers and drain those frames before
            # sending PROFILE_REQUEST, otherwise the profiler may mistake stale
            # training traffic for a failed worker.
            await self.settle_workers_before_profiling()

            # Run profiling phase
            log.info("Starting profiling phase...")
            profiles = await self.run_profiling_phase()

            # Check we still have enough workers after profiling
            if len(self.workers) < self.min_workers:
                err = f"Lost workers during profiling. Need {self.min_workers}, have {len(self.workers)}"
                log.error(err)
                await self.notify_orch_error(err)
                self.state = PipelineState.IDLE
                self.training_start_event.clear()
                self.training_done_event.set()
                return

            # Smart assignment based on profiling results
            if not self.assign_stages_smart(profiles):
                err = (
                    f"Not enough available device memory for "
                    f"{self.config['n_layers_total']} layers"
                )
                log.error(err)
                await self.notify_orch_error(err)
                self.state = PipelineState.IDLE
                self.training_start_event.clear()
                self.training_done_event.set()
                return

            # Notify orchestrator with profiling results
            await self.notify_orch_profile_report(profiles)

            # Reset events so worker idle loops re-yield for the training phase
            self.training_start_event.clear()
            self.training_done_event.set()
            await asyncio.sleep(0.1)  # let workers loop back

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
            training_started = True

            # Run training loop
            await self.run_training_loop()

        except WorkerDisconnectedError as e:
            log.error(f"Training aborted: {e}")
            self.remove_worker(e.worker_id)
            await self.notify_orch_error(str(e))
        except asyncio.CancelledError:
            log.info("Training task cancelled (stop requested)")
        except Exception as e:
            log.error(f"Training error: {e}", exc_info=True)
            await self.notify_orch_error(str(e))
        finally:
            # Tell remaining workers to exit their training loops and wait for next CONFIG
            await self.send_pipeline_stop()
            # Only report training stopped if workers actually reached START.
            # Profiling/config failures already send a concrete error.
            if training_started and self.orch_writer:
                try:
                    final_loss = float(np.mean(self.loss_history[-4:])) if self.loss_history else 0.0
                    payload = struct.pack(">BIf", MSG_ORCH_TRAINING_STOPPED,
                                          self.current_step, final_loss)
                    await write_frame(self.orch_writer, payload)
                except Exception:
                    pass
            self.state = PipelineState.IDLE
            self.training_start_event.clear()
            self.training_done_event.set()
            self._training_task = None
            # Clear stale assignments so status reports show "Unassigned"
            self.reset_assignments()
            log.info("Server state → IDLE. Workers stay connected.")

    async def _worker_read(self, worker: "WorkerInfo") -> bytes:
        """Read a frame from a worker, raising WorkerDisconnectedError on failure."""
        try:
            return await asyncio.wait_for(
                read_frame(worker.reader),
                timeout=WORKER_READ_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as e:
            log.error(
                f"Worker {worker.worker_id} read timed out after "
                f"{WORKER_READ_TIMEOUT_SECONDS:.0f}s"
            )
            raise WorkerDisconnectedError(worker.worker_id) from e
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError,
                OSError) as e:
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
        """Main training loop: stream data, relay activations/gradients.

        Supports N>=2 pipeline stages.  The server relays activations
        forward through every consecutive pair (stage 0→1, 1→2, …, N-2→N-1)
        and gradients backward through each pair in reverse order.
        """
        self.state = PipelineState.TRAINING

        sorted_workers = sorted(self.workers.values(), key=lambda w: w.stage_index)
        head_worker = sorted_workers[0]
        tail_worker = sorted_workers[-1]
        N = len(sorted_workers)

        M = self.config.get("num_micro_batches", 4)

        log.info(f"Starting training: {self.total_steps} mini-batches, {M} micro-batches each, {N} stages")

        for mini_batch in range(self.total_steps):
            if self.stop_requested:
                log.info(f"Stop requested at mini-batch {mini_batch}")
                break

            t0 = time.monotonic()
            self.current_step = mini_batch

            batch_targets: Dict[int, np.ndarray] = {}
            batch_tokens: Dict[int, np.ndarray] = {}

            # ── Phase 1: Send M data batches to head ──
            if mini_batch < 3:
                log.info(f"[mb {mini_batch}] Phase 1: Sending {M} data batches to head...")
            for m in range(M):
                tokens, targets = self.generate_batch(mini_batch, m)
                batch_targets[m] = targets
                batch_tokens[m] = tokens
                data_msg = self.build_data_batch(m, mini_batch, tokens, targets)
                await self._worker_write(head_worker, data_msg)

            # ── Phase 2: Async relay activations and gradients as workers produce them ──
            if mini_batch < 3:
                log.info(f"[mb {mini_batch}] Phase 2: Async 1F1B relay across {N} stages...")

            write_locks = {w.worker_id: asyncio.Lock() for w in sorted_workers}

            async def locked_worker_write(worker: WorkerInfo, payload: bytes):
                async with write_locks[worker.worker_id]:
                    await self._worker_write(worker, payload)

            async def relay_worker_outputs(worker: WorkerInfo):
                expected = 0
                if worker.stage_index < N - 1:
                    expected += M  # activations to downstream
                if worker.stage_index > 0:
                    expected += M  # gradients to upstream
                if worker.is_tail:
                    expected += M  # loss reports

                seen = 0
                while seen < expected:
                    frame = await self._worker_read(worker)
                    if not frame:
                        continue

                    msg_type = frame[0]
                    if msg_type == MSG_PIPELINE_ACTIVATION:
                        _, mb_id, src_stage, dst_stage = struct.unpack_from(">BIII", frame, 0)
                        offset = 13
                        _has_targets = frame[offset]
                        offset += 1
                        existing_tgt_len = struct.unpack_from(">I", frame, offset)[0]
                        offset += 4 + existing_tgt_len
                        st_len = struct.unpack_from(">I", frame, offset)[0]
                        offset += 4
                        st_data = frame[offset:offset + st_len]

                        tgt_np = batch_targets.get(mb_id)
                        if tgt_np is None:
                            log.warning(f"No targets for activation micro-batch {mb_id}")
                            continue
                        tgt_be = tgt_np.astype(">i4").tobytes()

                        relay_msg = struct.pack(
                            ">BIII",
                            MSG_PIPELINE_ACTIVATION,
                            mb_id, src_stage, src_stage + 1,
                        )
                        relay_msg += struct.pack(">B", 1)
                        relay_msg += struct.pack(">I", len(tgt_be)) + tgt_be
                        relay_msg += struct.pack(">I", st_len) + st_data

                        receiver = sorted_workers[src_stage + 1]
                        await locked_worker_write(receiver, relay_msg)
                        seen += 1

                    elif msg_type == MSG_PIPELINE_GRADIENT:
                        _, mb_id, src_stage, dst_stage = struct.unpack_from(">BIII", frame, 0)
                        receiver = sorted_workers[dst_stage]
                        await locked_worker_write(receiver, frame)
                        seen += 1

                    elif msg_type == MSG_PIPELINE_LOSS_REPORT:
                        _, mb_id, ub_id, loss_val, step = struct.unpack_from(">BIIfI", frame, 0)
                        self.loss_history.append(loss_val)
                        await self.notify_orch_loss(mini_batch, loss_val)
                        seen += 1

                    else:
                        log.warning(
                            f"Unexpected frame from worker {worker.worker_id} "
                            f"during relay: 0x{msg_type:02x}"
                        )

            await asyncio.gather(*(relay_worker_outputs(w) for w in sorted_workers))

            # ── Phase 4: Sync barrier ──
            barrier = struct.pack(">BI", MSG_PIPELINE_SYNC_BARRIER, mini_batch)
            for w in sorted_workers:
                await locked_worker_write(w, barrier)

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
                    # Signal that this worker has yielded its reader to the
                    # orchestrator-owned profiling/config/training coroutines.
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
                    orchestrator._training_task = asyncio.create_task(orchestrator.start_training())

            elif msg_type == MSG_ORCH_STOP_TRAINING:
                if orchestrator.state == PipelineState.TRAINING:
                    log.info("Orchestrator requested training stop")
                    orchestrator.stop_requested = True
                    await orchestrator.send_pipeline_stop()
                    if orchestrator._training_task and not orchestrator._training_task.done():
                        orchestrator._training_task.cancel()
                else:
                    log.warning("Stop requested but not currently training")

            elif msg_type == MSG_ORCH_GET_STATUS:
                status = orchestrator.build_status_report()
                await write_frame(writer, status)

            elif msg_type == MSG_ORCH_RESTART_SERVER:
                log.info("Orchestrator requested server restart")
                # Stop training if active
                if orchestrator.state == PipelineState.TRAINING:
                    orchestrator.stop_requested = True
                    if orchestrator._training_task and not orchestrator._training_task.done():
                        orchestrator._training_task.cancel()
                        try:
                            await asyncio.wait_for(orchestrator._training_task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                    else:
                        await orchestrator.send_pipeline_stop()
                # Disconnect all workers
                for w in list(orchestrator.workers.values()):
                    try:
                        w.writer.close()
                    except Exception:
                        pass
                orchestrator.workers.clear()
                # Reset all state
                orchestrator.state = PipelineState.IDLE
                orchestrator.stop_requested = False
                orchestrator._training_task = None
                orchestrator.training_start_event.clear()
                orchestrator.training_done_event.set()
                orchestrator._readers_yielded_count = 0
                orchestrator.current_step = 0
                orchestrator.loss_history.clear()
                orchestrator._next_worker_id = 1
                worker_auth_bypass.clear()
                log.info("Server restart complete — all workers disconnected, state reset")
                ack = struct.pack(">B", MSG_ORCH_RESTART_ACK)
                await write_frame(writer, ack)

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
            await orchestrator.send_pipeline_stop()
            if orchestrator._training_task and not orchestrator._training_task.done():
                orchestrator._training_task.cancel()
        else:
            # Clear stale assignments when orchestrator disconnects outside training
            orchestrator.reset_assignments()
        log.info("Orchestrator slot freed")


async def main(host: str, port: int, worker_secret: str, orch_secret: str,
               config: dict, data_path: str = None, dataset_name: str = "wikitext-103"):
    global orchestrator, WORKER_SECRET, ORCH_SECRET
    WORKER_SECRET = worker_secret.encode("utf-8")
    ORCH_SECRET = orch_secret.encode("utf-8")

    dataset = None
    if dataset_name == "wikitext-103":
        dataset = load_wikitext103(config["seq_len"])
    elif dataset_name == "custom" and data_path:
        dataset = TextDataset.from_file(data_path, config["seq_len"])
    else:
        log.warning("No dataset selected; training will use random data (loss won't decrease)")

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
    parser.add_argument("--d-model", type=int, default=1024,
                        help="Model dimension (default: 1024 for GPT-2 Medium)")
    parser.add_argument("--n-heads", type=int, default=16,
                        help="Attention heads (default: 16 for GPT-2 Medium)")
    parser.add_argument("--n-layers", type=int, default=24,
                        help="Transformer layers (default: 24 for GPT-2 Medium)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1 for memory safety at d=1024)")
    parser.add_argument("--ffn-hidden-mul", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        choices=["wikitext-103", "custom", "none"],
                        help="Dataset to use (default: wikitext-103)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training text file (for --dataset custom)")
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
                     data_path=args.data, dataset_name=args.dataset))
