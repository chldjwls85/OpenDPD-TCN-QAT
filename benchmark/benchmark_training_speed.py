#!/usr/bin/env python3
"""Wall-clock training-speed benchmark for OpenDPD.

For every combination of:
    role        : PA  (CoreModel)
                  DPD (CascadedModel = trainable DPD CoreModel + frozen PA CoreModel)
    backbone    : GRU, TCN
    target #P   : 200, 500, 1000  (trainable parameters)
    device      : cpu, cuda (if available), mps (if available)
    batch size  : 16, 32, 64, 128, 256

train one epoch on a synthetic IQ tensor and record the wall-clock time and
sample throughput.  Results are rendered as rich tables in the terminal and
written to a markdown report.

Run:
    python benchmark/benchmark_training_speed.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import CascadedModel, CoreModel


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROLES = ("PA", "DPD")
BACKBONES = ("gru", "tcn")
TARGET_PARAMS = (200, 500, 1000)
BATCH_SIZES = (16, 32, 64, 128, 256)

# Hidden sizes chosen so the *trainable* parameter count lands close to the
# target.  Exact counts (also reported in the output):
#   GRU(input=2, hidden=H, output=2, num_layers=1)   -> 3*H^2 + 14*H + 2
#   TCN(hidden_channels=H)                            -> 29*H
HIDDEN_SIZES = {
    "gru": {200: 6, 500: 11, 1000: 16},
    "tcn": {200: 7, 500: 17, 1000: 34},
}

# DPD runs use a CascadedModel(DPD + frozen PA).  The frozen PA matches the
# DPD backbone with a fixed mid-size hidden so the trainable parameter count
# corresponds purely to the DPD model.
FROZEN_PA_HIDDEN = {"gru": 11, "tcn": 17}

INPUT_SIZE = 2
FRAME_LENGTH = 200
N_FRAMES = 2048
SEED = 0


# -----------------------------------------------------------------------------
# Devices
# -----------------------------------------------------------------------------
def discover_devices(force: list[str] | None = None) -> list[torch.device]:
    if force:
        return [torch.device(d) for d in force]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        devices.append(torch.device("mps"))
    return devices


def device_short(device: torch.device) -> str:
    return {"cuda": "GPU", "mps": "MPS", "cpu": "CPU"}[device.type]


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        try:
            return f"GPU (CUDA): {torch.cuda.get_device_name(idx)}"
        except Exception:
            return "GPU (CUDA)"
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------
def build_model(role: str, backbone: str, target_p: int,
                device: torch.device) -> nn.Module:
    """Build a PA or DPD-cascaded model for the given config and move to device."""
    hidden = HIDDEN_SIZES[backbone][target_p]
    # Suppress the "Backbone Initialized..." print from CoreModel.__init__.
    with contextlib.redirect_stdout(io.StringIO()):
        primary = CoreModel(input_size=INPUT_SIZE, hidden_size=hidden,
                            num_layers=1, backbone_type=backbone)
        if role == "PA":
            return primary.to(device)
        pa = CoreModel(input_size=INPUT_SIZE,
                       hidden_size=FROZEN_PA_HIDDEN[backbone],
                       num_layers=1, backbone_type=backbone)
        cascaded = CascadedModel(dpd_model=primary, pa_model=pa)
        cascaded.freeze_pa_model()
        return cascaded.to(device)


def trainable_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Single benchmark run
# -----------------------------------------------------------------------------
@dataclass
class RunResult:
    role: str
    backbone: str
    target_params: int
    actual_params: int
    device: str
    batch_size: int
    n_batches: int
    epoch_seconds: float
    samples_per_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_one_epoch(role: str, backbone: str, target_p: int,
                  device: torch.device, batch_size: int,
                  features: torch.Tensor, targets: torch.Tensor,
                  outer_pbar: tqdm) -> RunResult:
    torch.manual_seed(SEED)
    net = build_model(role, backbone, target_p, device)
    n_params = trainable_params(net)

    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad), lr=5e-4)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(features, targets),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    n_batches = len(loader)
    net.train()

    # Warmup (one batch, not timed).  Stabilises CUDA allocator / cuDNN
    # autotune / first MPS kernel compile so the timed epoch is representative.
    warm_x, warm_y = next(iter(loader))
    warm_x, warm_y = warm_x.to(device), warm_y.to(device)
    optimizer.zero_grad()
    criterion(net(warm_x), warm_y).backward()
    optimizer.step()
    device_sync(device)

    desc = f"{role:<3s} {backbone.upper():<3s} ~{target_p:>4d}P {device_short(device):<3s} bs={batch_size:<3d}"
    inner = tqdm(loader, desc=desc, leave=False, position=1,
                 dynamic_ncols=True, file=sys.stdout)

    device_sync(device)
    t0 = time.perf_counter()
    n_samples = 0
    for x, y in inner:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()
        n_samples += x.size(0)
    device_sync(device)
    elapsed = time.perf_counter() - t0
    inner.close()

    # Free memory before the next config.
    del net, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    outer_pbar.update(1)
    return RunResult(
        role=role, backbone=backbone, target_params=target_p,
        actual_params=n_params, device=device_short(device),
        batch_size=batch_size, n_batches=n_batches,
        epoch_seconds=elapsed,
        samples_per_sec=n_samples / elapsed if elapsed > 0 else float("nan"),
    )


# -----------------------------------------------------------------------------
# Console rendering
# -----------------------------------------------------------------------------
def _row_keys(results: list[RunResult]):
    return sorted({(r.role, r.backbone, r.target_params, r.actual_params)
                   for r in results})


def _find(results: list[RunResult], role: str, backbone: str,
          target_p: int, batch_size: int) -> RunResult | None:
    for r in results:
        if (r.role == role and r.backbone == backbone
                and r.target_params == target_p and r.batch_size == batch_size):
            return r
    return None


def make_results_table(results: list[RunResult], device_name: str) -> Table:
    table = Table(
        title=(f"[bold cyan]{device_name}[/bold cyan]  "
               f"[dim]— seconds / epoch  •  samples / sec[/dim]"),
        box=box.SIMPLE_HEAVY,
        title_style="bold",
        header_style="bold magenta",
        pad_edge=False,
    )
    table.add_column("Role", justify="center", style="bold cyan")
    table.add_column("Backbone", justify="center", style="bold")
    table.add_column("Target #P", justify="right", style="dim")
    table.add_column("Actual #P", justify="right")
    for bs in BATCH_SIZES:
        table.add_column(f"bs={bs}", justify="right")

    last_role: str | None = None
    for role, backbone, tgt, act in _row_keys(results):
        if last_role is not None and role != last_role:
            table.add_section()
        last_role = role
        cells = [role, backbone.upper(), str(tgt), f"{act:,}"]
        for bs in BATCH_SIZES:
            r = _find(results, role, backbone, tgt, bs)
            if r is None:
                cells.append("[red]—[/red]")
            else:
                cells.append(f"[bold]{r.epoch_seconds:.3f}s[/bold]\n"
                             f"[green]{r.samples_per_sec:,.0f}[/green]")
        table.add_row(*cells)
    return table


def render_console(console: Console, all_results: list[RunResult],
                   device_descs: dict[str, str]) -> None:
    by_device: dict[str, list[RunResult]] = {}
    for r in all_results:
        by_device.setdefault(r.device, []).append(r)
    for device_short_name, rs in by_device.items():
        console.print()
        console.print(make_results_table(rs, device_descs.get(
            device_short_name, device_short_name)))


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------
def render_markdown(all_results: list[RunResult],
                    device_descs: dict[str, str],
                    sys_info: dict[str, str], total_seconds: float,
                    output: Path) -> None:
    lines: list[str] = []
    lines.append("# OpenDPD Training-Speed Benchmark")
    lines.append("")
    lines.append(
        "Wall-clock benchmark of one training epoch for **PA** and **DPD** "
        "models with **GRU** and **TCN** backbones at three parameter scales "
        "(~200, ~500, ~1000), across batch sizes "
        f"{{{', '.join(str(b) for b in BATCH_SIZES)}}} on each available device."
    )
    lines.append("")

    # System info
    lines.append("## System")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k, v in sys_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"| Total benchmark wall-clock | {total_seconds:.1f} s |")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Synthetic dataset: **{N_FRAMES}** IQ frames of length "
                 f"**{FRAME_LENGTH}** (random, seed={SEED})")
    lines.append("- Optimiser: **AdamW** (lr=5e-4), MSE loss")
    lines.append("- 1 warmup batch (untimed) + 1 full epoch (timed) per configuration")
    lines.append(
        "- DPD runs use **CascadedModel(DPD + frozen PA)**; the frozen PA "
        f"uses the same backbone as the DPD with hidden_size = "
        f"{FROZEN_PA_HIDDEN['gru']} (GRU) / {FROZEN_PA_HIDDEN['tcn']} (TCN)."
    )
    lines.append("")
    lines.append("**Hidden sizes used to hit each parameter target**")
    lines.append("")
    lines.append("| Backbone | ~200 P | ~500 P | ~1000 P |")
    lines.append("|---|---|---|---|")
    for bb in BACKBONES:
        cells = [bb.upper()] + [str(HIDDEN_SIZES[bb][p]) for p in TARGET_PARAMS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Results
    lines.append("## Results")
    lines.append("")
    by_device: dict[str, list[RunResult]] = {}
    for r in all_results:
        by_device.setdefault(r.device, []).append(r)

    for device_short_name, rs in by_device.items():
        lines.append(f"### {device_descs.get(device_short_name, device_short_name)}")
        lines.append("")
        header = (["Role", "Backbone", "Target #P", "Actual #P"]
                  + [f"bs={bs}" for bs in BATCH_SIZES])

        lines.append("**Wall-clock per epoch (seconds)**")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for role, backbone, tgt, act in _row_keys(rs):
            row = [role, backbone.upper(), str(tgt), f"{act:,}"]
            for bs in BATCH_SIZES:
                r = _find(rs, role, backbone, tgt, bs)
                row.append(f"{r.epoch_seconds:.3f}" if r else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("**Throughput (samples / sec)**")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for role, backbone, tgt, act in _row_keys(rs):
            row = [role, backbone.upper(), str(tgt), f"{act:,}"]
            for bs in BATCH_SIZES:
                r = _find(rs, role, backbone, tgt, bs)
                row.append(f"{r.samples_per_sec:,.0f}" if r else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    output.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", nargs="*", default=None,
                   choices=["cpu", "cuda", "mps"],
                   help="Override which devices to benchmark (default: auto).")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent
                   / "benchmark_training_speed_report.md",
                   help="Markdown report output path.")
    p.add_argument("--json-output", type=Path, default=None,
                   help="Optional path to also dump raw results as JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    devices = discover_devices(args.devices)
    device_descs = {device_short(d): device_label(d) for d in devices}

    g = torch.Generator().manual_seed(SEED)
    features = torch.randn(N_FRAMES, FRAME_LENGTH, INPUT_SIZE, generator=g)
    targets = torch.randn(N_FRAMES, FRAME_LENGTH, INPUT_SIZE, generator=g)

    sys_info = {
        "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "Devices": " | ".join(device_descs.values()),
        "CPU threads (torch)": str(torch.get_num_threads()),
    }
    panel_text = "\n".join(f"[bold]{k:<22}[/bold] {v}" for k, v in sys_info.items())
    console.print(Panel(
        panel_text,
        title="[bold cyan]OpenDPD Training-Speed Benchmark[/bold cyan]",
        border_style="cyan", expand=False,
    ))

    configs = [
        (role, backbone, tp, dev, bs)
        for dev in devices
        for role in ROLES
        for backbone in BACKBONES
        for tp in TARGET_PARAMS
        for bs in BATCH_SIZES
    ]
    console.print(f"[dim]Total configurations: {len(configs)}[/dim]\n")

    results: list[RunResult] = []
    bench_t0 = time.perf_counter()

    overall = tqdm(total=len(configs), desc="overall", position=0,
                   leave=True, dynamic_ncols=True, file=sys.stdout)
    try:
        for role, backbone, tp, dev, bs in configs:
            try:
                results.append(run_one_epoch(
                    role, backbone, tp, dev, bs, features, targets, overall))
            except RuntimeError as e:
                overall.update(1)
                console.print(
                    f"[red]Failed[/red] {role}/{backbone}/p{tp}/"
                    f"{device_short(dev)}/bs{bs}: {e}")
    finally:
        overall.close()

    total_seconds = time.perf_counter() - bench_t0

    render_console(console, results, device_descs)
    render_markdown(results, device_descs, sys_info, total_seconds, args.output)
    console.print(f"\n[green]Markdown report written to:[/green] {args.output}")

    if args.json_output:
        args.json_output.write_text(
            json.dumps([r.to_dict() for r in results], indent=2))
        console.print(f"[green]JSON results written to:[/green] {args.json_output}")


if __name__ == "__main__":
    main()
