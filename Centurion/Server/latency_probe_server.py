#!/usr/bin/env python3
"""
Latency probe coordinator for distributed training over iOS devices.

Protocol: Length-prefixed binary frames over TCP.
  Frame: [4 bytes big-endian uint32 payload_length][payload bytes]

Request payload:
  [4B uint32 request_id]
  [8B uint64 client_timestamp_ns]
  [1B uint8  ndims]
  [4B*ndims uint32 shape...]
  [4B*nelements float32 tensor_data...]

Response payload:
  [4B uint32 request_id]
  [8B uint64 server_timestamp_ns]
  [8B uint64 client_timestamp_ns (echo)]
  [4B uint32 server_processing_us]
  [1B uint8  ndims]
  [4B*ndims uint32 shape...]
  [4B*nelements float32 tensor_data...]

No external dependencies — pure Python stdlib.
"""

import asyncio
import struct
import array
import time
import random
import logging
import argparse
import sys
from typing import Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("probe-server")


def parse_request(payload: bytes) -> dict:
    """Parse a binary probe request."""
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

    # Read float32 tensor data using array module (fast)
    tensor_bytes = payload[offset : offset + nelements * 4]
    tensor = array.array("f")
    tensor.frombytes(tensor_bytes)
    # Convert from big-endian if needed
    if sys.byteorder == "little":
        tensor.byteswap()

    return {
        "request_id": request_id,
        "client_ts": client_ts,
        "shape": shape,
        "nelements": nelements,
        "tensor": tensor,
    }


def build_response(
    request_id: int,
    client_ts: int,
    server_ts: int,
    processing_us: int,
    shape: List[int],
    tensor: array.array,
) -> bytes:
    """Build a binary probe response."""
    header = struct.pack(
        ">IQQI",
        request_id,
        server_ts,
        client_ts,
        processing_us,
    )

    shape_header = struct.pack(">B", len(shape))
    for dim in shape:
        shape_header += struct.pack(">I", dim)

    # Swap to big-endian for network transmission
    out_tensor = array.array("f", tensor)
    if sys.byteorder == "little":
        out_tensor.byteswap()

    return header + shape_header + out_tensor.tobytes()


def simulate_peer_processing(tensor: array.array) -> array.array:
    """Simulate a peer device processing the tensor (add small noise)."""
    result = array.array("f")
    for v in tensor:
        result.append(v * 0.99 + random.gauss(0, 0.01))
    return result


async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes, raising on EOF."""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise asyncio.IncompleteReadError(data, n)
        data += chunk
    return data


async def handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
):
    addr = writer.get_extra_info("peername")
    log.info(f"Client connected: {addr}")
    request_count = 0

    try:
        while True:
            # Read 4-byte length header
            length_bytes = await read_exactly(reader, 4)
            msg_len = struct.unpack(">I", length_bytes)[0]

            # Read payload
            payload = await read_exactly(reader, msg_len)

            proc_start = time.monotonic_ns()
            request = parse_request(payload)
            request_count += 1

            data_kb = request["nelements"] * 4 / 1024
            log.info(
                f"[#{request_count}] req_id={request['request_id']} "
                f"shape={request['shape']} "
                f"payload={data_kb:.1f} KB"
            )

            # Simulate peer processing
            response_tensor = simulate_peer_processing(request["tensor"])

            proc_end = time.monotonic_ns()
            processing_us = (proc_end - proc_start) // 1000

            server_ts = time.time_ns()

            # Build response
            response_payload = build_response(
                request_id=request["request_id"],
                client_ts=request["client_ts"],
                server_ts=server_ts,
                processing_us=processing_us,
                shape=request["shape"],
                tensor=response_tensor,
            )

            # Send with length prefix
            frame = struct.pack(">I", len(response_payload)) + response_payload
            writer.write(frame)
            await writer.drain()

            log.info(
                f"  -> response sent: {len(response_payload)} bytes, "
                f"server_processing={processing_us} us"
            )

    except (asyncio.IncompleteReadError, ConnectionResetError):
        log.info(f"Client {addr} disconnected (after {request_count} requests)")
    except Exception as e:
        log.error(f"Error handling {addr}: {e}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main(host: str, port: int):
    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f"Latency probe server listening on {addr[0]}:{addr[1]}")
    log.info("Protocol: length-prefixed binary frames over TCP")
    log.info("Waiting for connections...")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency probe coordinator")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9999, help="Bind port")
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port))
