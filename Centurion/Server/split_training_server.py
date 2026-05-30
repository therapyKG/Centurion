#!/usr/bin/env python3
"""
Split-training server: holds the back half of a GPT-2 model
(blocks[K..N] + final LayerNorm + LM head).

Receives front-half activations + targets from the iOS device,
runs forward -> cross-entropy loss -> backward, and sends
activation gradients back.

Protocol: Length-prefixed binary frames over TCP (same framing
as latency_probe_server.py). Port 9998 by default.

Requires: torch (CPU-only is fine for the coordinator role)
  pip3 install torch --index-url https://download.pytorch.org/whl/cpu
"""

import asyncio
import struct
import math
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("split-server")

# ── Message types ──
MSG_TRAIN_FORWARD = 0x01
MSG_TRAIN_BACKWARD = 0x02
MSG_CONFIG = 0x10
MSG_CONFIG_ACK = 0x11


# ── GPT-2 Architecture (matching Swift/MLX implementation) ──

class GPT2Block(nn.Module):
    """Pre-norm GPT-2 block with separate Q/K/V/O projections."""

    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.wQ = nn.Linear(d_model, d_model)
        self.wK = nn.Linear(d_model, d_model)
        self.wV = nn.Linear(d_model, d_model)
        self.wO = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x, mask=None):
        # Pre-norm attention
        normed = self.ln1(x)
        attn_out = self._attention(normed, mask)
        x = x + self.resid_drop(attn_out)

        # Pre-norm FFN with approximate GELU (matching MLX geluApproximate)
        normed_ffn = self.ln2(x)
        ffn_out = self.fc2(F.gelu(self.fc1(normed_ffn), approximate="tanh"))
        return x + self.resid_drop(ffn_out)

    def _attention(self, x, mask):
        B, S, D = x.shape
        q = self.wQ(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wK(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wV(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wO(out)


class GPT2BackHalf(nn.Module):
    """Back half of GPT-2: blocks[K..N] + final LayerNorm + LM head."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers_total,
                 split_layer, ffn_hidden_mul=4, dropout=0.1):
        super().__init__()
        n_back_layers = n_layers_total - split_layer
        ffn_hidden = d_model * ffn_hidden_mul

        self.blocks = nn.ModuleList([
            GPT2Block(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_back_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model, eps=1e-5)

        # Untied LM head (embedding lives on device side)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, activation, targets, seq_len):
        """
        activation: [B, S, D] (requires_grad=True)
        targets: [B, S] (long)
        seq_len: int (for mask generation)
        Returns: (loss scalar, logits [B, S, V])
        """
        B, S, D = activation.shape

        # Causal mask: upper-triangular filled with -1e9
        mask = torch.full((S, S), -1e9, device=activation.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        x = activation
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)  # [B, S, vocab_size]

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(B * S, -1),
            targets.view(B * S),
            reduction="mean",
        )
        return loss, logits


class SplitTrainingSession:
    """Holds the back-half model and optimizer for one client."""

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.device = torch.device("cpu")
        self.configured = False
        self.step_count = 0

    def configure(self, vocab_size, d_model, n_heads, n_layers_total,
                  split_layer, seq_len, ffn_hidden_mul, learning_rate, dropout):
        self.model = GPT2BackHalf(
            vocab_size, d_model, n_heads, n_layers_total, split_layer,
            ffn_hidden_mul, dropout,
        ).to(self.device)
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        self.configured = True
        self.step_count = 0

        param_count = sum(p.numel() for p in self.model.parameters())
        n_back = n_layers_total - split_layer
        log.info(
            f"Model built: {n_back} back layers, {param_count:,} params, "
            f"d={d_model} h={n_heads} ffn={d_model * ffn_hidden_mul} "
            f"vocab={vocab_size} lr={learning_rate}"
        )
        return param_count

    def train_step(self, activation_np, targets_np, seq_len):
        """
        activation_np: float32 [B, S, D]
        targets_np: int32 [B, S]
        Returns: (loss_float, grad_np float32 [B, S, D])
        """
        activation = (
            torch.from_numpy(activation_np)
            .to(self.device)
            .requires_grad_(True)
        )
        targets = torch.from_numpy(targets_np).long().to(self.device)

        self.optimizer.zero_grad()

        loss, _ = self.model(activation, targets, seq_len)
        loss.backward()

        # Extract gradient w.r.t. the input activation
        activation_grad = activation.grad.detach().cpu().numpy()
        loss_value = loss.item()

        # Update server-side parameters
        self.optimizer.step()
        self.step_count += 1

        return loss_value, activation_grad


# ── Protocol parsing ──

def parse_config(payload):
    """Parse CONFIG message (0x10)."""
    offset = 1  # skip msg_type byte
    fields = struct.unpack_from(">IIIIIII", payload, offset)
    offset += 7 * 4
    lr, dropout = struct.unpack_from(">ff", payload, offset)
    return {
        "vocab_size": fields[0],
        "d_model": fields[1],
        "n_heads": fields[2],
        "n_layers_total": fields[3],
        "split_layer": fields[4],
        "seq_len": fields[5],
        "ffn_hidden_mul": fields[6],
        "learning_rate": lr,
        "dropout": dropout,
    }


def build_config_ack(param_count, status=0):
    """Build CONFIG_ACK message (0x11)."""
    return struct.pack(">BIB", MSG_CONFIG_ACK, param_count, status)


def parse_train_forward(payload):
    """Parse TRAIN_FORWARD message (0x01).
    Returns (request_id, step, B, S, D, activation_np, targets_np).
    """
    offset = 1  # skip msg_type
    request_id, step, B, S, D = struct.unpack_from(">IIIII", payload, offset)
    offset += 5 * 4

    # Activation: B*S*D float32 big-endian
    n_act = B * S * D
    act_bytes = payload[offset: offset + n_act * 4]
    activation = np.frombuffer(act_bytes, dtype=">f4").astype(np.float32)
    activation = activation.reshape(B, S, D)
    offset += n_act * 4

    # Targets: B*S int32 big-endian
    n_tgt = B * S
    tgt_bytes = payload[offset: offset + n_tgt * 4]
    targets = np.frombuffer(tgt_bytes, dtype=">i4").astype(np.int32)
    targets = targets.reshape(B, S)

    return request_id, step, B, S, D, activation, targets


def build_train_backward(request_id, loss_value, B, S, D, grad_np):
    """Build TRAIN_BACKWARD message (0x02)."""
    header = struct.pack(">BIfIII",
                         MSG_TRAIN_BACKWARD,
                         request_id,
                         loss_value,
                         B, S, D)
    grad_be = grad_np.astype(">f4").tobytes()
    return header + grad_be


# ── Frame I/O ──

async def read_exactly(reader, n):
    """Read exactly n bytes."""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise asyncio.IncompleteReadError(data, n)
        data += chunk
    return data


async def read_frame(reader):
    """Read one length-prefixed frame."""
    length_bytes = await read_exactly(reader, 4)
    msg_len = struct.unpack(">I", length_bytes)[0]
    return await read_exactly(reader, msg_len)


async def write_frame(writer, payload):
    """Write one length-prefixed frame."""
    frame = struct.pack(">I", len(payload)) + payload
    writer.write(frame)
    await writer.drain()


# ── Client handler ──

async def handle_client(reader, writer):
    addr = writer.get_extra_info("peername")
    log.info(f"Client connected: {addr}")
    session = SplitTrainingSession()

    try:
        while True:
            payload = await read_frame(reader)
            msg_type = payload[0]

            if msg_type == MSG_CONFIG:
                cfg = parse_config(payload)
                log.info(f"CONFIG received: {cfg}")
                param_count = session.configure(**cfg)
                ack = build_config_ack(param_count, status=0)
                await write_frame(writer, ack)
                log.info(f"CONFIG_ACK sent: {param_count:,} params")

            elif msg_type == MSG_TRAIN_FORWARD:
                if not session.configured:
                    log.warning("TRAIN_FORWARD before CONFIG — ignoring")
                    continue

                request_id, step, B, S, D, activation, targets = (
                    parse_train_forward(payload)
                )

                t0 = time.monotonic()
                loss_val, grad = session.train_step(activation, targets, S)
                proc_ms = (time.monotonic() - t0) * 1000

                resp = build_train_backward(request_id, loss_val, B, S, D, grad)
                await write_frame(writer, resp)

                if session.step_count % 10 == 1 or session.step_count <= 5:
                    log.info(
                        f"[step {session.step_count}] "
                        f"req={request_id} loss={loss_val:.4f} "
                        f"proc={proc_ms:.1f}ms "
                        f"grad_norm={np.linalg.norm(grad):.4f}"
                    )

            else:
                log.warning(f"Unknown message type: 0x{msg_type:02x}")

    except (asyncio.IncompleteReadError, ConnectionResetError):
        log.info(
            f"Client {addr} disconnected "
            f"(after {session.step_count} train steps)"
        )
    except Exception as e:
        log.error(f"Error handling {addr}: {e}", exc_info=True)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main(host, port):
    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    log.info(f"Split training server listening on {addr[0]}:{addr[1]}")
    log.info("Waiting for connections...")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-training coordinator")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9998, help="Bind port")
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port))
