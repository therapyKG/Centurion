#!/usr/bin/env python3
"""
Checkpoint aggregation server for federated training.

Receives model checkpoints (safetensors format) from worker devices,
averages them via exponential moving average, and serves the aggregated
model back on request.

Also handles legacy latency-probe messages for backward compatibility.

Protocol: Length-prefixed binary frames over TCP.
  Frame: [4 bytes big-endian uint32 payload_length][payload bytes]

Authentication: HMAC-SHA256 challenge-response on connect.
  1. Server sends AUTH_CHALLENGE: [1B type=0x30][32B random nonce]
  2. Client sends AUTH_RESPONSE: [1B type=0x31][32B HMAC-SHA256(secret, nonce)]
  3. Server verifies and sends AUTH_RESULT: [1B type=0x32][1B status (0=ok, 1=fail)]
  Unauthenticated clients are disconnected immediately.

Message types (first byte of payload):
  0x20  CHECKPOINT_UPLOAD    worker -> server
  0x21  CHECKPOINT_ACK       server -> worker
  0x22  CHECKPOINT_REQUEST   worker -> server
  0x23  CHECKPOINT_RESPONSE  server -> worker
  0x30  AUTH_CHALLENGE       server -> worker (on connect)
  0x31  AUTH_RESPONSE        worker -> server
  0x32  AUTH_RESULT          server -> worker
  Other: Legacy probe (backward compat with latency_probe_server.py)

Requires: pip install safetensors numpy
"""

import asyncio
import struct
import array
import time
import random
import logging
import argparse
import sys
import os
import hmac
import hashlib
import numpy as np
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("checkpoint-server")

# ── Message types ──
MSG_CHECKPOINT_UPLOAD = 0x20
MSG_CHECKPOINT_ACK = 0x21
MSG_CHECKPOINT_REQUEST = 0x22
MSG_CHECKPOINT_RESPONSE = 0x23
MSG_AUTH_CHALLENGE = 0x30
MSG_AUTH_RESPONSE = 0x31
MSG_AUTH_RESULT = 0x32

# ── Shared secret ──
SHARED_SECRET = b""


# ── Aggregator State ──

class AggregatorState:
    """Holds the global averaged model and merges incoming checkpoints."""

    def __init__(self, alpha: float = 0.5):
        self.global_params: Dict[str, np.ndarray] = {}
        self.global_step: int = 0
        self.num_contributors: int = 0
        self.alpha = alpha  # EMA mixing: new = (1-alpha)*old + alpha*incoming

    def merge(self, incoming: Dict[str, np.ndarray], worker_id: int, local_step: int):
        """Merge incoming checkpoint into global model via EMA."""
        if not self.global_params:
            # First contribution: just copy
            self.global_params = {k: v.copy() for k, v in incoming.items()}
            log.info(
                f"  First checkpoint stored: {len(incoming)} params, "
                f"{sum(v.nbytes for v in incoming.values()) / 1024:.0f} KB"
            )
        else:
            merged_count = 0
            for key in self.global_params:
                if key in incoming:
                    self.global_params[key] = (
                        (1 - self.alpha) * self.global_params[key]
                        + self.alpha * incoming[key]
                    )
                    merged_count += 1
            log.info(f"  Merged {merged_count}/{len(self.global_params)} params (alpha={self.alpha})")

        self.global_step += 1
        self.num_contributors += 1

    def serialize(self) -> bytes:
        """Serialize global model to safetensors bytes."""
        from safetensors.numpy import save
        return save(self.global_params)

    def total_bytes(self) -> int:
        return sum(v.nbytes for v in self.global_params.values())


# Shared state across all connections
aggregator = AggregatorState()


# ── Frame I/O ──

async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes, raising on EOF."""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise asyncio.IncompleteReadError(data, n)
        data += chunk
    return data


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one length-prefixed frame."""
    length_bytes = await read_exactly(reader, 4)
    msg_len = struct.unpack(">I", length_bytes)[0]
    return await read_exactly(reader, msg_len)


async def write_frame(writer: asyncio.StreamWriter, payload: bytes):
    """Write one length-prefixed frame."""
    frame = struct.pack(">I", len(payload)) + payload
    writer.write(frame)
    await writer.drain()


# ── Authentication ──

async def authenticate_client(reader, writer, addr) -> bool:
    """Challenge-response auth using HMAC-SHA256.
    Returns True if the client is authenticated.
    """
    # Generate 32-byte random nonce
    nonce = os.urandom(32)

    # Send AUTH_CHALLENGE: [1B type=0x30][32B nonce]
    challenge = struct.pack(">B", MSG_AUTH_CHALLENGE) + nonce
    await write_frame(writer, challenge)

    # Read AUTH_RESPONSE: [1B type=0x31][32B hmac]
    try:
        response = await asyncio.wait_for(read_frame(reader), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning(f"Auth timeout from {addr}")
        return False

    if len(response) != 33 or response[0] != MSG_AUTH_RESPONSE:
        log.warning(f"Invalid auth response from {addr}: len={len(response)}")
        return False

    client_hmac = response[1:33]
    expected_hmac = hmac.new(SHARED_SECRET, nonce, hashlib.sha256).digest()

    if not hmac.compare_digest(client_hmac, expected_hmac):
        log.warning(f"Auth FAILED from {addr} — bad HMAC")
        # Send failure
        result = struct.pack(">BB", MSG_AUTH_RESULT, 1)
        await write_frame(writer, result)
        return False

    # Send success
    result = struct.pack(">BB", MSG_AUTH_RESULT, 0)
    await write_frame(writer, result)
    log.info(f"Auth OK from {addr}")
    return True


# ── Checkpoint message handling ──

def parse_checkpoint_upload(payload: bytes):
    """Parse CHECKPOINT_UPLOAD (0x20).
    Returns (worker_id, local_step, safetensors_data).
    """
    offset = 1  # skip msg_type
    worker_id, local_step, st_len = struct.unpack_from(">III", payload, offset)
    offset += 12
    st_data = payload[offset: offset + st_len]
    return worker_id, local_step, st_data


def build_checkpoint_ack(global_step: int, num_contributors: int, status: int = 0) -> bytes:
    """Build CHECKPOINT_ACK (0x21)."""
    return struct.pack(">BIIB", MSG_CHECKPOINT_ACK, global_step, num_contributors, status)


def parse_checkpoint_request(payload: bytes):
    """Parse CHECKPOINT_REQUEST (0x22). Returns worker_id."""
    worker_id = struct.unpack_from(">I", payload, 1)[0]
    return worker_id


def build_checkpoint_response(global_step: int, num_contributors: int, st_data: bytes) -> bytes:
    """Build CHECKPOINT_RESPONSE (0x23)."""
    header = struct.pack(">BIII", MSG_CHECKPOINT_RESPONSE, global_step, num_contributors, len(st_data))
    return header + st_data


# ── Legacy probe handling ──

def parse_legacy_probe(payload: bytes) -> dict:
    """Parse a legacy latency probe request."""
    offset = 0
    request_id = struct.unpack_from(">I", payload, offset)[0]
    offset += 4
    client_ts = struct.unpack_from(">Q", payload, offset)[0]
    offset += 8
    ndims = struct.unpack_from(">B", payload, offset)[0]
    offset += 1

    shape = []
    for _ in range(ndims):
        dim = struct.unpack_from(">I", payload, offset)[0]
        shape.append(dim)
        offset += 4

    nelements = 1
    for d in shape:
        nelements *= d

    tensor_bytes = payload[offset: offset + nelements * 4]
    tensor = array.array("f")
    tensor.frombytes(tensor_bytes)
    if sys.byteorder == "little":
        tensor.byteswap()

    return {
        "request_id": request_id,
        "client_ts": client_ts,
        "shape": shape,
        "nelements": nelements,
        "tensor": tensor,
    }


def build_legacy_response(request_id, client_ts, server_ts, processing_us, shape, tensor) -> bytes:
    """Build a legacy probe response."""
    header = struct.pack(">IQQI", request_id, server_ts, client_ts, processing_us)
    shape_header = struct.pack(">B", len(shape))
    for dim in shape:
        shape_header += struct.pack(">I", dim)

    out_tensor = array.array("f", tensor)
    if sys.byteorder == "little":
        out_tensor.byteswap()

    return header + shape_header + out_tensor.tobytes()


def simulate_peer_processing(tensor: array.array) -> array.array:
    """Simulate peer processing (add small noise)."""
    result = array.array("f")
    for v in tensor:
        result.append(v * 0.99 + random.gauss(0, 0.01))
    return result


# ── Client handler ──

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    log.info(f"Client connected: {addr}")

    # ── Authentication gate ──
    if not await authenticate_client(reader, writer, addr):
        log.warning(f"Rejecting unauthenticated client {addr}")
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return

    request_count = 0

    try:
        while True:
            payload = await read_frame(reader)
            msg_type = payload[0]
            request_count += 1

            if msg_type == MSG_CHECKPOINT_UPLOAD:
                worker_id, local_step, st_data = parse_checkpoint_upload(payload)
                log.info(
                    f"[UPLOAD] worker={worker_id} step={local_step} "
                    f"size={len(st_data) / 1024:.0f} KB"
                )

                t0 = time.monotonic()
                from safetensors.numpy import load
                incoming = load(st_data)
                aggregator.merge(incoming, worker_id, local_step)
                proc_ms = (time.monotonic() - t0) * 1000

                log.info(
                    f"  -> Merged in {proc_ms:.1f}ms. "
                    f"Global step={aggregator.global_step}, "
                    f"contributors={aggregator.num_contributors}, "
                    f"model={aggregator.total_bytes() / 1024:.0f} KB"
                )

                ack = build_checkpoint_ack(aggregator.global_step, aggregator.num_contributors)
                await write_frame(writer, ack)

            elif msg_type == MSG_CHECKPOINT_REQUEST:
                worker_id = parse_checkpoint_request(payload)
                log.info(f"[REQUEST] worker={worker_id}")

                if not aggregator.global_params:
                    log.warning("  No checkpoints available yet")
                    ack = build_checkpoint_ack(0, 0, status=1)
                    await write_frame(writer, ack)
                else:
                    t0 = time.monotonic()
                    st_bytes = aggregator.serialize()
                    proc_ms = (time.monotonic() - t0) * 1000
                    log.info(
                        f"  -> Sending {len(st_bytes) / 1024:.0f} KB "
                        f"(serialized in {proc_ms:.1f}ms)"
                    )
                    resp = build_checkpoint_response(
                        aggregator.global_step, aggregator.num_contributors, st_bytes
                    )
                    await write_frame(writer, resp)

            elif msg_type not in (MSG_CHECKPOINT_ACK, MSG_CHECKPOINT_RESPONSE,
                                   MSG_AUTH_CHALLENGE, MSG_AUTH_RESPONSE, MSG_AUTH_RESULT):
                # Legacy probe message — first byte is part of request_id
                proc_start = time.monotonic_ns()
                request = parse_legacy_probe(payload)

                data_kb = request["nelements"] * 4 / 1024
                log.info(
                    f"[PROBE #{request_count}] req_id={request['request_id']} "
                    f"shape={request['shape']} payload={data_kb:.1f} KB"
                )

                response_tensor = simulate_peer_processing(request["tensor"])
                proc_end = time.monotonic_ns()
                processing_us = (proc_end - proc_start) // 1000
                server_ts = time.time_ns()

                response_payload = build_legacy_response(
                    request_id=request["request_id"],
                    client_ts=request["client_ts"],
                    server_ts=server_ts,
                    processing_us=processing_us,
                    shape=request["shape"],
                    tensor=response_tensor,
                )
                await write_frame(writer, response_payload)

                log.info(
                    f"  -> probe response: {len(response_payload)} bytes, "
                    f"processing={processing_us} us"
                )

            else:
                log.warning(f"Unexpected message type: 0x{msg_type:02x}")

    except (asyncio.IncompleteReadError, ConnectionResetError):
        log.info(f"Client {addr} disconnected (after {request_count} messages)")
    except Exception as e:
        log.error(f"Error handling {addr}: {e}", exc_info=True)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main(host: str, port: int, alpha: float, secret: str):
    global aggregator, SHARED_SECRET
    aggregator = AggregatorState(alpha=alpha)
    SHARED_SECRET = secret.encode("utf-8")

    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f"Checkpoint aggregation server listening on {addr[0]}:{addr[1]}")
    log.info(f"EMA alpha={alpha} (mixing weight for incoming checkpoints)")
    log.info(f"Auth: HMAC-SHA256 challenge-response (secret length={len(SHARED_SECRET)})")
    log.info("Waiting for connections...")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint aggregation server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9999, help="Bind port")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="EMA mixing weight (0=keep old, 1=replace with new)")
    parser.add_argument("--secret", type=str, required=True,
                        help="Shared secret for HMAC authentication")
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port, args.alpha, args.secret))
