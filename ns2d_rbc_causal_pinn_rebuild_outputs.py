

import os
import csv
import math
import time
import platform
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

@dataclass
class RBCConfig:
    case_name: str = "RBC"
    model_name: str = "Causal-PINN"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Physical parameters
    nu: float = 0.01
    kappa: float = 0.01
    beta: float = 1.0

    # Domain
    x0: float = 0.0
    x1: float = 1.0
    y0: float = 0.0
    y1: float = 1.0
    t0: float = 0.0
    t1: float = 2.0

    # Dataset sizes
    n_col: int = 22000
    n_ic: int = 2400
    n_bc_left: int = 1800
    n_bc_right: int = 1800
    n_bc_bottom: int = 1800
    n_bc_top: int = 1800
    train_ratio: float = 0.7

    # Training
    iters: int = 80000
    lr: float = 1e-4
    batch_col: int = 1024
    batch_ic: int = 256
    batch_bc: int = 256

    val_batch_col: int = 512
    val_batch_ic: int = 256
    val_batch_bc: int = 256

    eval_every: int = 500
    grad_clip: float = 1.0

    # Loss weights
    w_pde: float = 1.0
    w_div: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 2.0

    # PINN architecture
    hidden: int = 128
    layers: int = 8

    # Causality-aware residual weighting
    causal_chunks: int = 10
    causal_epsilon: float = 1.0

    # Evaluation grid and time slices
    nx_eval: int = 120
    ny_eval: int = 120
    eval_times: Tuple[float, ...] = (0.0, 1.0, 2.0)

    # Thresholds for computational cost comparison
    rel_l2_thresholds: Tuple[float, ...] = (1e-2, 1e-3)

    out_dir: str = "./outputs/RBC/Causal-PINN"
    seed: int = 0


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_col(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(1)
    return x


def sample_uniform(n: int, low: float, high: float, device: str, dtype: torch.dtype):
    return low + (high - low) * torch.rand(n, 1, device=device, dtype=dtype)


def split_indices(n: int, ratio: float, seed: int, device: str):
    if n < 2:
        tr = torch.arange(n, device=device)
        va = torch.zeros(0, dtype=torch.long, device=device)
        return tr, va
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=device)
    ntr = max(1, min(int(ratio * n), n - 1))
    return perm[:ntr], perm[ntr:]


def safe_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def time_tag(t: float) -> str:
    return f"t{str(round(float(t), 4)).replace('.', 'p')}"


def make_dirs(cfg: RBCConfig):
    subdirs = [
        cfg.out_dir,
        os.path.join(cfg.out_dir, "field_data"),
        os.path.join(cfg.out_dir, "metrics"),
        os.path.join(cfg.out_dir, "cost"),
        os.path.join(cfg.out_dir, "loss"),
        os.path.join(cfg.out_dir, "figures", "reference"),
        os.path.join(cfg.out_dir, "figures", "prediction"),
        os.path.join(cfg.out_dir, "figures", "error"),
        os.path.join(cfg.out_dir, "figures", "loss"),
    ]
    for ti in cfg.eval_times:
        subdirs.append(os.path.join(cfg.out_dir, "field_data", time_tag(ti)))
    for d in subdirs:
        os.makedirs(d, exist_ok=True)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_hardware_info(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_name": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.version.cuda is not None else "None",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": "None",
        "gpu_total_memory_MB": 0.0,
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = prop.name
        info["gpu_total_memory_MB"] = prop.total_memory / (1024 ** 2)
    return info


def current_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        return (
            torch.cuda.memory_allocated() / (1024 ** 2),
            torch.cuda.memory_reserved() / (1024 ** 2),
        )
    return 0.0, 0.0


def peak_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        return (
            torch.cuda.max_memory_allocated() / (1024 ** 2),
            torch.cuda.max_memory_reserved() / (1024 ** 2),
        )
    return 0.0, 0.0


# ============================================================
# Manufactured solution and forcing
# ============================================================

def uvpt_mms(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    """Manufactured reference solution for RBC: u, v, p, theta."""
    x = ensure_col(x)
    y = ensure_col(y)
    t = ensure_col(t)

    # Use stream function to make velocity divergence-free.
    with torch.enable_grad():
        xg = x.clone().detach().requires_grad_(True)
        yg = y.clone().detach().requires_grad_(True)
        tg = t.clone().detach()
        psi = (
            torch.sin(math.pi * xg) ** 2
            * torch.sin(math.pi * yg) ** 2
            * torch.sin(math.pi * tg)
        )
        u = torch.autograd.grad(psi, yg, grad_outputs=torch.ones_like(psi), create_graph=False, retain_graph=True)[0].detach()
        v = -torch.autograd.grad(psi, xg, grad_outputs=torch.ones_like(psi), create_graph=False, retain_graph=True)[0].detach()

    p = 0.1 * torch.cos(math.pi * x) * torch.cos(math.pi * y) * torch.cos(math.pi * t)
    theta = torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(math.pi * t)
    return ensure_col(u), ensure_col(v), ensure_col(p), ensure_col(theta)


def forcing_mms_detached(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, cfg: RBCConfig):

    x = ensure_col(x)
    y = ensure_col(y)
    t = ensure_col(t)

    with torch.enable_grad():
        xg = x.clone().detach().requires_grad_(True)
        yg = y.clone().detach().requires_grad_(True)
        tg = t.clone().detach().requires_grad_(True)

        def g1(u, z):
            return torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        psi = (
            torch.sin(math.pi * xg) ** 2
            * torch.sin(math.pi * yg) ** 2
            * torch.sin(math.pi * tg)
        )
        u = g1(psi, yg)
        v = -g1(psi, xg)
        p = 0.1 * torch.cos(math.pi * xg) * torch.cos(math.pi * yg) * torch.cos(math.pi * tg)
        th = torch.sin(math.pi * xg) * torch.sin(math.pi * yg) * torch.cos(math.pi * tg)

        u_t = g1(u, tg); v_t = g1(v, tg); th_t = g1(th, tg)
        u_x = g1(u, xg); u_y = g1(u, yg)
        v_x = g1(v, xg); v_y = g1(v, yg)
        th_x = g1(th, xg); th_y = g1(th, yg)
        u_xx = g1(u_x, xg); u_yy = g1(u_y, yg)
        v_xx = g1(v_x, xg); v_yy = g1(v_y, yg)
        th_xx = g1(th_x, xg); th_yy = g1(th_y, yg)
        p_x = g1(p, xg); p_y = g1(p, yg)

        fu = u_t + u * u_x + v * u_y + p_x - cfg.nu * (u_xx + u_yy)
        fv = v_t + u * v_x + v * v_y + p_y - cfg.nu * (v_xx + v_yy) - cfg.beta * th
        fth = th_t + u * th_x + v * th_y - cfg.kappa * (th_xx + th_yy)
        return fu.detach(), fv.detach(), fth.detach()


# ============================================================
# Dataset
# ============================================================

def build_dataset(cfg: RBCConfig):
    x_col = sample_uniform(cfg.n_col, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y_col = sample_uniform(cfg.n_col, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    t_col = sample_uniform(cfg.n_col, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    tr, va = split_indices(cfg.n_col, cfg.train_ratio, cfg.seed + 101, cfg.device)
    col_tr = (x_col[tr], y_col[tr], t_col[tr])
    col_va = (x_col[va], y_col[va], t_col[va])

    x_ic = sample_uniform(cfg.n_ic, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y_ic = sample_uniform(cfg.n_ic, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    t_ic = torch.full_like(x_ic, cfg.t0)
    tr, va = split_indices(cfg.n_ic, cfg.train_ratio, cfg.seed + 202, cfg.device)
    u_tr, v_tr, p_tr, th_tr = uvpt_mms(x_ic[tr], y_ic[tr], t_ic[tr])
    u_va, v_va, p_va, th_va = uvpt_mms(x_ic[va], y_ic[va], t_ic[va])
    ic_tr = (x_ic[tr], y_ic[tr], t_ic[tr], u_tr.detach(), v_tr.detach(), p_tr.detach(), th_tr.detach())
    ic_va = (x_ic[va], y_ic[va], t_ic[va], u_va.detach(), v_va.detach(), p_va.detach(), th_va.detach())

    def make_side(n: int, seed: int, kind: str):
        if kind == "left":
            y = sample_uniform(n, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
            x = torch.full_like(y, cfg.x0)
        elif kind == "right":
            y = sample_uniform(n, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
            x = torch.full_like(y, cfg.x1)
        elif kind == "bottom":
            x = sample_uniform(n, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
            y = torch.full_like(x, cfg.y0)
        elif kind == "top":
            x = sample_uniform(n, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
            y = torch.full_like(x, cfg.y1)
        else:
            raise ValueError(f"Unknown boundary kind: {kind}")
        t = sample_uniform(n, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
        tr, va = split_indices(n, cfg.train_ratio, seed, cfg.device)
        u_tr, v_tr, p_tr, th_tr = uvpt_mms(x[tr], y[tr], t[tr])
        u_va, v_va, p_va, th_va = uvpt_mms(x[va], y[va], t[va])
        return (
            x[tr], y[tr], t[tr], u_tr.detach(), v_tr.detach(), p_tr.detach(), th_tr.detach()
        ), (
            x[va], y[va], t[va], u_va.detach(), v_va.detach(), p_va.detach(), th_va.detach()
        )

    bc_left_tr, bc_left_va = make_side(cfg.n_bc_left, cfg.seed + 301, "left")
    bc_right_tr, bc_right_va = make_side(cfg.n_bc_right, cfg.seed + 302, "right")
    bc_bottom_tr, bc_bottom_va = make_side(cfg.n_bc_bottom, cfg.seed + 303, "bottom")
    bc_top_tr, bc_top_va = make_side(cfg.n_bc_top, cfg.seed + 304, "top")

    return {
        "col_tr": col_tr, "col_va": col_va,
        "ic_tr": ic_tr, "ic_va": ic_va,
        "bc_left_tr": bc_left_tr, "bc_left_va": bc_left_va,
        "bc_right_tr": bc_right_tr, "bc_right_va": bc_right_va,
        "bc_bottom_tr": bc_bottom_tr, "bc_bottom_va": bc_bottom_va,
        "bc_top_tr": bc_top_tr, "bc_top_va": bc_top_va,
    }


# ============================================================
# Model
# ============================================================

class PINNModel(nn.Module):
    def __init__(self, cfg: RBCConfig):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(3, cfg.hidden))
        layers.append(nn.Tanh())
        for _ in range(cfg.layers - 1):
            layers.append(nn.Linear(cfg.hidden, cfg.hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(cfg.hidden, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        x = ensure_col(x); y = ensure_col(y); t = ensure_col(t)
        inp = torch.cat([x, y, t], dim=1)
        o = self.net(inp)
        return o[:, 0:1], o[:, 1:2], o[:, 2:3], o[:, 3:4]


# ============================================================
# Losses and metrics
# ============================================================

def grad1(u: torch.Tensor, x: torch.Tensor):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]


def laplacian_2d(u: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    ux = grad1(u, x)
    uy = grad1(u, y)
    uxx = grad1(ux, x)
    uyy = grad1(uy, y)
    return ux, uy, uxx + uyy


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def causal_time_weighted_losses(
    point_mom: torch.Tensor,
    point_theta: torch.Tensor,
    point_div: torch.Tensor,
    t: torch.Tensor,
    cfg: RBCConfig,
):

    mom_flat = point_mom.reshape(-1)
    theta_flat = point_theta.reshape(-1)
    div_flat = point_div.reshape(-1)
    t_flat = ensure_col(t).reshape(-1)

    edges = torch.linspace(
        cfg.t0,
        cfg.t1,
        cfg.causal_chunks + 1,
        device=t.device,
        dtype=t.dtype,
    )

    chunk_mom_losses = []
    chunk_theta_losses = []
    chunk_div_losses = []

    for i in range(cfg.causal_chunks):
        left = edges[i]
        right = edges[i + 1]
        if i == cfg.causal_chunks - 1:
            mask = (t_flat >= left) & (t_flat <= right)
        else:
            mask = (t_flat >= left) & (t_flat < right)
        if torch.any(mask):
            chunk_mom_losses.append(torch.mean(mom_flat[mask]))
            chunk_theta_losses.append(torch.mean(theta_flat[mask]))
            chunk_div_losses.append(torch.mean(div_flat[mask]))

    if len(chunk_mom_losses) == 0:
        loss_mom = torch.mean(point_mom)
        loss_theta_pde = torch.mean(point_theta)
        loss_div = torch.mean(point_div)
        return loss_mom + loss_theta_pde, loss_mom, loss_theta_pde, loss_div

    weighted_mom = []
    weighted_theta = []
    weighted_div = []
    cumulative = torch.zeros((), device=t.device, dtype=t.dtype)

    for lm, lt, ld in zip(chunk_mom_losses, chunk_theta_losses, chunk_div_losses):
        w = torch.exp(-cfg.causal_epsilon * cumulative.detach())
        weighted_mom.append(w * lm)
        weighted_theta.append(w * lt)
        weighted_div.append(w * ld)
        cumulative = cumulative + lm + lt + ld

    loss_mom = torch.stack(weighted_mom).mean()
    loss_theta_pde = torch.stack(weighted_theta).mean()
    loss_div = torch.stack(weighted_div).mean()
    return loss_mom + loss_theta_pde, loss_mom, loss_theta_pde, loss_div


def batch_losses(cfg: RBCConfig, model: PINNModel, col, ic, bc_left, bc_right, bc_bottom, bc_top):
    with torch.enable_grad():
        x_col, y_col, t_col = col
        x = ensure_col(x_col.clone().detach()).requires_grad_(True)
        y = ensure_col(y_col.clone().detach()).requires_grad_(True)
        t = ensure_col(t_col.clone().detach()).requires_grad_(True)

        u, v, p, th = model(x, y, t)
        u_t = grad1(u, t); v_t = grad1(v, t); th_t = grad1(th, t)
        u_x, u_y, lap_u = laplacian_2d(u, x, y)
        v_x, v_y, lap_v = laplacian_2d(v, x, y)
        th_x, th_y, lap_th = laplacian_2d(th, x, y)
        p_x = grad1(p, x); p_y = grad1(p, y)

        fu, fv, fth = forcing_mms_detached(x_col, y_col, t_col, cfg)
        r_u = u_t + u * u_x + v * u_y + p_x - cfg.nu * lap_u - fu
        r_v = v_t + u * v_x + v * v_y + p_y - cfg.nu * lap_v - cfg.beta * th - fv
        r_th = th_t + u * th_x + v * th_y - cfg.kappa * lap_th - fth
        r_div = u_x + v_y

        point_mom = r_u ** 2 + r_v ** 2
        point_theta = r_th ** 2
        point_div = r_div ** 2
        loss_pde, loss_mom, loss_theta_pde, loss_div = causal_time_weighted_losses(
            point_mom, point_theta, point_div, t, cfg
        )

        x_ic, y_ic, t_ic, u_ic, v_ic, p_ic, th_ic = ic
        u0, v0, p0, th0 = model(x_ic, y_ic, t_ic)
        loss_ic = mse(u0, u_ic) + mse(v0, v_ic) + mse(p0, p_ic) + mse(th0, th_ic)

        def bc_loss(bundle):
            xb, yb, tb, ub, vb, pb, thb = bundle
            up, vp, pp, tp = model(xb, yb, tb)
            return mse(up, ub) + mse(vp, vb) + mse(pp, pb) + mse(tp, thb)

        loss_bc = bc_loss(bc_left) + bc_loss(bc_right) + bc_loss(bc_bottom) + bc_loss(bc_top)
        loss_total = cfg.w_pde * loss_pde + cfg.w_div * loss_div + cfg.w_ic * loss_ic + cfg.w_bc * loss_bc

        return loss_total, {
            "pde": loss_pde,
            "mom": loss_mom,
            "theta_pde": loss_theta_pde,
            "div": loss_div,
            "ic": loss_ic,
            "bc": loss_bc,
        }


def compute_tensor_metrics(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach()
    ref = ref.detach()
    diff = pred - ref
    mse_v = torch.mean(diff ** 2)
    rmse_v = torch.sqrt(mse_v)
    mae_v = torch.mean(torch.abs(diff))
    l2_v = torch.linalg.norm(diff.reshape(-1), ord=2)
    rel_l2_v = l2_v / (torch.linalg.norm(ref.reshape(-1), ord=2) + 1e-12)
    max_abs = torch.max(torch.abs(diff))
    mean_abs = torch.mean(torch.abs(diff))
    return {
        "MSE": safe_float(mse_v),
        "RMSE": safe_float(rmse_v),
        "MAE": safe_float(mae_v),
        "L2": safe_float(l2_v),
        "RelL2": safe_float(rel_l2_v),
        "MaxAbsError": safe_float(max_abs),
        "MeanAbsError": safe_float(mean_abs),
    }


@torch.no_grad()
def data_loss_fields(cfg: RBCConfig, model: PINNModel, batch) -> Dict[str, float]:
    x, y, t = batch
    u_ref, v_ref, p_ref, th_ref = uvpt_mms(x, y, t)
    u, v, p, th = model(x, y, t)
    sp_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
    sp_pred = torch.sqrt(u ** 2 + v ** 2)
    losses = {
        "u": safe_float(torch.mean((u - u_ref) ** 2)),
        "v": safe_float(torch.mean((v - v_ref) ** 2)),
        "p": safe_float(torch.mean((p - p_ref) ** 2)),
        "theta": safe_float(torch.mean((th - th_ref) ** 2)),
        "speed": safe_float(torch.mean((sp_pred - sp_ref) ** 2)),
    }
    losses["total"] = losses["u"] + losses["v"] + losses["p"] + losses["theta"]
    return losses


@torch.no_grad()
def evaluate_val_metrics(cfg: RBCConfig, model: PINNModel, data, n: int = 3000) -> Dict[str, float]:
    x_va, y_va, t_va = data["col_va"]
    n = min(n, x_va.shape[0])
    idx = torch.randperm(x_va.shape[0], device=cfg.device)[:n]
    x = x_va[idx]; y = y_va[idx]; t = t_va[idx]
    u_ref, v_ref, p_ref, th_ref = uvpt_mms(x, y, t)
    u, v, p, th = model(x, y, t)
    pred = torch.cat([u, v, p, th], dim=1)
    ref = torch.cat([u_ref, v_ref, p_ref, th_ref], dim=1)
    return compute_tensor_metrics(pred, ref)


# ============================================================
# Minibatches
# ============================================================

def minibatch3(a, b, c, batch: int, seed: int):
    n = a.shape[0]
    g = torch.Generator(device=a.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=a.device)[:min(batch, n)]
    return a[idx], b[idx], c[idx]


def minibatch7(a, b, c, d, e, f, g0, batch: int, seed: int):
    n = a.shape[0]
    gen = torch.Generator(device=a.device)
    gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen, device=a.device)[:min(batch, n)]
    return a[idx], b[idx], c[idx], d[idx], e[idx], f[idx], g0[idx]


# ============================================================
# Evaluation grid, field data, metrics, and figures
# ============================================================

@torch.no_grad()
def eval_on_grid(cfg: RBCConfig, model: PINNModel):
    xs = torch.linspace(cfg.x0, cfg.x1, cfg.nx_eval, device=cfg.device, dtype=cfg.dtype)
    ys = torch.linspace(cfg.y0, cfg.y1, cfg.ny_eval, device=cfg.device, dtype=cfg.dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    X = XX.reshape(-1, 1)
    Y = YY.reshape(-1, 1)
    grid: Dict[str, Any] = {
        "x": xs.cpu().numpy(),
        "y": ys.cpu().numpy(),
        "times": np.array(cfg.eval_times, dtype=np.float32),
    }

    for ti in cfg.eval_times:
        T = torch.full_like(X, float(ti))
        u_ref, v_ref, p_ref, th_ref = uvpt_mms(X, Y, T)
        u_pred, v_pred, p_pred, th_pred = model(X, Y, T)
        speed_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
        speed_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)
        tag = time_tag(ti)
        fields = {
            "u": (u_pred, u_ref),
            "v": (v_pred, v_ref),
            "p": (p_pred, p_ref),
            "theta": (th_pred, th_ref),
            "speed": (speed_pred, speed_ref),
        }
        for name, (pred, ref) in fields.items():
            grid[f"{name}_pred_{tag}"] = ensure_col(pred).view(cfg.nx_eval, cfg.ny_eval).cpu().numpy()
            grid[f"{name}_true_{tag}"] = ensure_col(ref).view(cfg.nx_eval, cfg.ny_eval).cpu().numpy()
            grid[f"{name}_err_{tag}"] = ensure_col(pred - ref).view(cfg.nx_eval, cfg.ny_eval).cpu().numpy()
    return grid


def write_field_txt(cfg: RBCConfig, grid: Dict[str, Any]):
    xs = grid["x"]
    ys = grid["y"]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    fields = ["u", "v", "p", "theta", "speed"]
    for ti in cfg.eval_times:
        tag = time_tag(ti)
        folder = os.path.join(cfg.out_dir, "field_data", tag)
        os.makedirs(folder, exist_ok=True)
        for fld in fields:
            pred = grid[f"{fld}_pred_{tag}"]
            ref = grid[f"{fld}_true_{tag}"]
            err = pred - ref
            arr = np.column_stack([
                XX.reshape(-1),
                YY.reshape(-1),
                pred.reshape(-1),
                ref.reshape(-1),
                err.reshape(-1),
                np.abs(err).reshape(-1),
            ])
            header = "x\ty\tPrediction\tReference\tError\tAbsError"
            fname = os.path.join(folder, f"{fld}_field_data_{tag}.txt")
            np.savetxt(fname, arr, header=header, comments="", delimiter="\t", fmt="%.10e")


def write_metrics(cfg: RBCConfig, grid: Dict[str, Any]):
    fields = ["u", "v", "p", "theta", "speed"]
    field_rows: List[Dict[str, Any]] = []
    all_pred = []
    all_ref = []

    for ti in cfg.eval_times:
        tag = time_tag(ti)
        for fld in fields:
            pred_np = grid[f"{fld}_pred_{tag}"]
            ref_np = grid[f"{fld}_true_{tag}"]
            pred = torch.tensor(pred_np)
            ref = torch.tensor(ref_np)
            m = compute_tensor_metrics(pred, ref)
            row = {"case": cfg.case_name, "model": cfg.model_name, "time": float(ti), "field": fld}
            row.update(m)
            field_rows.append(row)
            if fld != "speed":
                all_pred.append(pred.reshape(-1))
                all_ref.append(ref.reshape(-1))


    pred_all = torch.cat(all_pred)
    ref_all = torch.cat(all_ref)
    overall = compute_tensor_metrics(pred_all, ref_all)
    overall_row = {"case": cfg.case_name, "model": cfg.model_name}
    overall_row.update(overall)

    metrics_dir = os.path.join(cfg.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    overall_path = os.path.join(metrics_dir, "overall_metrics.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_row.keys()))
        writer.writeheader()
        writer.writerow(overall_row)

    field_path = os.path.join(metrics_dir, "field_metrics.csv")
    fieldnames = ["case", "model", "time", "field", "MSE", "RMSE", "MAE", "L2", "RelL2", "MaxAbsError", "MeanAbsError"]
    with open(field_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in field_rows:
            writer.writerow(row)

    return overall_row, field_rows


def save_map(cfg: RBCConfig, data: np.ndarray, title: str, fname: str, subfolder: str):
    xs = None
    ys = None
    plt.figure(figsize=(5.6, 4.6))
    plt.imshow(data.T, origin="lower", extent=[cfg.x0, cfg.x1, cfg.y0, cfg.y1], aspect="auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(cfg.out_dir, "figures", subfolder, fname)
    plt.savefig(out, dpi=220)
    plt.close()


def save_figures(cfg: RBCConfig, grid: Dict[str, Any], history: Dict[str, List[float]]):
    fields = ["u", "v", "p", "theta", "speed"]
    for ti in cfg.eval_times:
        tag = time_tag(ti)
        for fld in fields:
            save_map(cfg, grid[f"{fld}_true_{tag}"], f"Reference {fld}, {tag}", f"{fld}_reference_{tag}.png", "reference")
            save_map(cfg, grid[f"{fld}_pred_{tag}"], f"Prediction {fld}, {tag}", f"{fld}_prediction_{tag}.png", "prediction")
            save_map(cfg, np.abs(grid[f"{fld}_err_{tag}"]), f"|Error| {fld}, {tag}", f"{fld}_error_{tag}.png", "error")

    curves = [
        ("loss_total_train", "loss_total_val", "Total loss", "loss_total.png"),
        ("loss_pde_train", "loss_pde_val", "PDE loss", "loss_pde.png"),
        ("loss_mom_train", "loss_mom_val", "Momentum loss", "loss_mom.png"),
        ("loss_theta_pde_train", "loss_theta_pde_val", "Theta PDE loss", "loss_theta_pde.png"),
        ("loss_div_train", "loss_div_val", "Divergence loss", "loss_div.png"),
        ("loss_ic_train", "loss_ic_val", "IC loss", "loss_ic.png"),
        ("loss_bc_train", "loss_bc_val", "BC loss", "loss_bc.png"),
        ("loss_data_train", "loss_data_val", "Data loss", "loss_data.png"),
    ]
    for tr_key, va_key, title, fname in curves:
        if len(history.get("iter", [])) == 0:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(history["iter"], history[tr_key], label="train")
        plt.plot(history["iter"], history[va_key], label="val")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "figures", "loss", fname), dpi=220)
        plt.close()


def save_grid(cfg: RBCConfig, grid: Dict[str, Any]):
    np.savez(os.path.join(cfg.out_dir, "grid.npz"), **grid)


# ============================================================
# CSV writers
# ============================================================

def init_history() -> Dict[str, List[float]]:
    keys = [
        "iter",
        "loss_total_train", "loss_total_val",
        "loss_pde_train", "loss_pde_val",
        "loss_mom_train", "loss_mom_val",
        "loss_div_train", "loss_div_val",
        "loss_theta_pde_train", "loss_theta_pde_val",
        "loss_ic_train", "loss_ic_val",
        "loss_bc_train", "loss_bc_val",
        "loss_data_train", "loss_data_val",
        "loss_data_u_train", "loss_data_u_val",
        "loss_data_v_train", "loss_data_v_val",
        "loss_data_p_train", "loss_data_p_val",
        "loss_data_theta_train", "loss_data_theta_val",
        "loss_data_speed_train", "loss_data_speed_val",
        "MSE_val", "RMSE_val", "MAE_val", "L2_val", "RelL2_val",
        "time_elapsed_sec",
    ]
    return {k: [] for k in keys}


def append_history_row(history: Dict[str, List[float]], row: Dict[str, float]):
    for key in history.keys():
        history[key].append(row.get(key, 0.0))


def write_history(cfg: RBCConfig, history: Dict[str, List[float]]):
    path = os.path.join(cfg.out_dir, "loss", "train_history.csv")
    keys = list(history.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(history[keys[0]]) if keys else 0
        for i in range(n):
            writer.writerow([history[k][i] for k in keys])


def write_model_info(cfg: RBCConfig, model: nn.Module, trainable_params: int, total_params: int):
    path = os.path.join(cfg.out_dir, "model_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Case: {cfg.case_name}\n")
        f.write(f"Model: {cfg.model_name}\n")
        f.write("Architecture: Causal-PINN with fully connected MLP backbone\n")
        f.write("Input dimension: 3 (x, y, t)\n")
        f.write("Output dimension: 4 (u, v, p, theta)\n")
        f.write(f"Hidden width: {cfg.hidden}\n")
        f.write(f"Hidden layers: {cfg.layers}\n")
        f.write("Activation: Tanh\n")
        f.write(f"Causal chunks: {cfg.causal_chunks}\n")
        f.write(f"Causal epsilon: {cfg.causal_epsilon}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Iterations: {cfg.iters}\n")
        f.write(f"Learning rate: {cfg.lr}\n")
        f.write(f"Batch collocation: {cfg.batch_col}\n")
        f.write(f"Batch IC: {cfg.batch_ic}\n")
        f.write(f"Batch BC: {cfg.batch_bc}\n")
        f.write(f"Seed: {cfg.seed}\n")


def write_cost_summary(
    cfg: RBCConfig,
    trainable_params: int,
    total_params: int,
    hardware: Dict[str, Any],
    total_wall_time: float,
    iter_times: List[float],
    best_iter: int,
    best_rel_l2: float,
    final_rel_l2: float,
):
    peak_alloc, peak_reserved = peak_gpu_memory(cfg.device)
    mean_iter = float(np.mean(iter_times)) if iter_times else 0.0
    median_iter = float(np.median(iter_times)) if iter_times else 0.0
    row = {
        "case": cfg.case_name,
        "model": cfg.model_name,
        "device": cfg.device,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "total_wall_time_sec": total_wall_time,
        "total_wall_time_min": total_wall_time / 60.0,
        "mean_time_per_iter_sec": mean_iter,
        "median_time_per_iter_sec": median_iter,
        "gpu_name": hardware.get("gpu_name", "None"),
        "gpu_count": hardware.get("gpu_count", 0),
        "cuda_available": hardware.get("cuda_available", False),
        "cuda_version": hardware.get("cuda_version", "None"),
        "torch_version": hardware.get("torch_version", "None"),
        "gpu_total_memory_MB": hardware.get("gpu_total_memory_MB", 0.0),
        "peak_gpu_memory_allocated_MB": peak_alloc,
        "peak_gpu_memory_reserved_MB": peak_reserved,
        "cpu_name": hardware.get("cpu_name", "None"),
        "best_iter": best_iter,
        "best_RelL2": best_rel_l2,
        "final_RelL2": final_rel_l2,
    }
    path = os.path.join(cfg.out_dir, "cost", "model_cost_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_threshold_cost(cfg: RBCConfig, history: Dict[str, List[float]]):
    rows = []
    iters = history.get("iter", [])
    rels = history.get("RelL2_val", [])
    times = history.get("time_elapsed_sec", [])
    for th in cfg.rel_l2_thresholds:
        reached = False
        first_iter = -1
        first_time = -1.0
        for it, rel, tm in zip(iters, rels, times):
            if rel <= th:
                reached = True
                first_iter = int(it)
                first_time = float(tm)
                break
        rows.append({
            "case": cfg.case_name,
            "model": cfg.model_name,
            "metric": "RelL2",
            "threshold": th,
            "reached": reached,
            "iter_to_threshold": first_iter,
            "time_to_threshold_sec": first_time,
        })
    path = os.path.join(cfg.out_dir, "cost", "threshold_cost.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["case", "model", "metric", "threshold", "reached", "iter_to_threshold", "time_to_threshold_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Training
# ============================================================

def train(cfg: RBCConfig):
    make_dirs(cfg)
    set_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    hardware = get_hardware_info(cfg.device)
    model = PINNModel(cfg).to(cfg.device, cfg.dtype)
    trainable_params, total_params = count_parameters(model)
    write_model_info(cfg, model, trainable_params, total_params)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    data = build_dataset(cfg)
    history = init_history()

    best_rel_l2 = float("inf")
    best_iter = -1
    final_rel_l2 = float("nan")
    iter_times: List[float] = []
    t_start = time.time()

    print(f"[INFO] Case: {cfg.case_name} | Model: {cfg.model_name}")
    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Output directory: {cfg.out_dir}")

    for it in range(1, cfg.iters + 1):
        iter_start = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)

        col_tr = minibatch3(*data["col_tr"], cfg.batch_col, cfg.seed + it)
        ic_tr = minibatch7(*data["ic_tr"], cfg.batch_ic, cfg.seed + 10000 + it)
        bc_left_tr = minibatch7(*data["bc_left_tr"], cfg.batch_bc, cfg.seed + 20000 + it)
        bc_right_tr = minibatch7(*data["bc_right_tr"], cfg.batch_bc, cfg.seed + 30000 + it)
        bc_bottom_tr = minibatch7(*data["bc_bottom_tr"], cfg.batch_bc, cfg.seed + 40000 + it)
        bc_top_tr = minibatch7(*data["bc_top_tr"], cfg.batch_bc, cfg.seed + 50000 + it)

        loss, parts = batch_losses(cfg, model, col_tr, ic_tr, bc_left_tr, bc_right_tr, bc_bottom_tr, bc_top_tr)
        if not torch.isfinite(loss):
            print(f"[iter {it}] Non-finite loss detected. Stop training.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        iter_times.append(time.time() - iter_start)

        if it == 1 or it % cfg.eval_every == 0:
            model.eval()
            elapsed = time.time() - t_start

            # Validation physics losses
            col_va = minibatch3(*data["col_va"], cfg.val_batch_col, cfg.seed + 60000 + it)
            ic_va = minibatch7(*data["ic_va"], cfg.val_batch_ic, cfg.seed + 70000 + it)
            bc_left_va = minibatch7(*data["bc_left_va"], cfg.val_batch_bc, cfg.seed + 80000 + it)
            bc_right_va = minibatch7(*data["bc_right_va"], cfg.val_batch_bc, cfg.seed + 90000 + it)
            bc_bottom_va = minibatch7(*data["bc_bottom_va"], cfg.val_batch_bc, cfg.seed + 100000 + it)
            bc_top_va = minibatch7(*data["bc_top_va"], cfg.val_batch_bc, cfg.seed + 110000 + it)
            loss_va, parts_va = batch_losses(cfg, model, col_va, ic_va, bc_left_va, bc_right_va, bc_bottom_va, bc_top_va)

            # Data losses on train and validation collocation minibatches
            data_tr = data_loss_fields(cfg, model, col_tr)
            data_va = data_loss_fields(cfg, model, col_va)
            val_metrics = evaluate_val_metrics(cfg, model, data)
            final_rel_l2 = val_metrics["RelL2"]

            if final_rel_l2 < best_rel_l2:
                best_rel_l2 = final_rel_l2
                best_iter = it
                torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_model.pt"))

            row = {
                "iter": it,
                "loss_total_train": safe_float(loss),
                "loss_total_val": safe_float(loss_va),
                "loss_pde_train": safe_float(parts["pde"]),
                "loss_pde_val": safe_float(parts_va["pde"]),
                "loss_mom_train": safe_float(parts["mom"]),
                "loss_mom_val": safe_float(parts_va["mom"]),
                "loss_div_train": safe_float(parts["div"]),
                "loss_div_val": safe_float(parts_va["div"]),
                "loss_theta_pde_train": safe_float(parts["theta_pde"]),
                "loss_theta_pde_val": safe_float(parts_va["theta_pde"]),
                "loss_ic_train": safe_float(parts["ic"]),
                "loss_ic_val": safe_float(parts_va["ic"]),
                "loss_bc_train": safe_float(parts["bc"]),
                "loss_bc_val": safe_float(parts_va["bc"]),
                "loss_data_train": data_tr["total"],
                "loss_data_val": data_va["total"],
                "loss_data_u_train": data_tr["u"],
                "loss_data_u_val": data_va["u"],
                "loss_data_v_train": data_tr["v"],
                "loss_data_v_val": data_va["v"],
                "loss_data_p_train": data_tr["p"],
                "loss_data_p_val": data_va["p"],
                "loss_data_theta_train": data_tr["theta"],
                "loss_data_theta_val": data_va["theta"],
                "loss_data_speed_train": data_tr["speed"],
                "loss_data_speed_val": data_va["speed"],
                "MSE_val": val_metrics["MSE"],
                "RMSE_val": val_metrics["RMSE"],
                "MAE_val": val_metrics["MAE"],
                "L2_val": val_metrics["L2"],
                "RelL2_val": val_metrics["RelL2"],
                "time_elapsed_sec": elapsed,
            }
            append_history_row(history, row)
            write_history(cfg, history)

            print(
                f"[iter {it:06d}] loss={row['loss_total_train']:.3e} "
                f"val_RelL2={row['RelL2_val']:.3e} "
                f"best={best_rel_l2:.3e} elapsed={elapsed/60:.2f} min"
            )

    total_wall_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "final_model.pt"))

    # Final evaluation outputs
    model.eval()
    grid = eval_on_grid(cfg, model)
    save_grid(cfg, grid)
    write_field_txt(cfg, grid)
    write_metrics(cfg, grid)
    save_figures(cfg, grid, history)
    write_threshold_cost(cfg, history)
    write_cost_summary(
        cfg=cfg,
        trainable_params=trainable_params,
        total_params=total_params,
        hardware=hardware,
        total_wall_time=total_wall_time,
        iter_times=iter_times,
        best_iter=best_iter,
        best_rel_l2=best_rel_l2,
        final_rel_l2=final_rel_l2,
    )

    print("[DONE] Training and unified outputs completed.")
    print(f"[DONE] Results saved to: {cfg.out_dir}")
    return model, history


if __name__ == "__main__":
    cfg = RBCConfig()
    train(cfg)
