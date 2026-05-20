

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


# =========================
# 1. Configuration
# =========================

@dataclass
class NS2DConfig:
    case_name: str = "TGV"
    model_name: str = "LSTM-PINN"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Physical parameter
    nu: float = 0.1

    # Domain
    x0: float = 0.0
    x1: float = 2.0 * math.pi
    y0: float = 0.0
    y1: float = 2.0 * math.pi
    t0: float = 0.0
    t1: float = 5.0

    # Dataset size
    n_col: int = 20000
    n_ic: int = 2400
    n_bcx: int = 2400
    n_bcy: int = 2400
    train_ratio: float = 0.7

    # Training
    iters: int = 80000
    lr: float = 1e-4
    batch_col: int = 1024
    batch_ic: int = 256
    batch_bcx: int = 256
    batch_bcy: int = 256

    # Validation batches
    val_batch_col: int = 512
    val_batch_ic: int = 256
    val_batch_bcx: int = 256
    val_batch_bcy: int = 256

    eval_every: int = 500
    grad_clip: float = 1.0

    # Loss weights
    w_mom: float = 1.0
    w_div: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 1.0

    # LSTM-PINN architecture
    hidden: int = 192
    lstm_layers: int = 2

    # Evaluation grid
    eval_nx: int = 80
    eval_ny: int = 80
    eval_times: Tuple[float, float, float] = (0.0, 2.5, 5.0)

    # Cost threshold tracking
    rel_l2_thresholds: Tuple[float, float] = (1e-2, 1e-3)

    # Output
    out_dir: str = "./outputs/TGV/LSTM-PINN"
    seed: int = 0


# =========================
# 2. Basic utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def time_tag(t: float) -> str:
    return f"t{str(round(float(t), 4)).replace('.', 'p')}"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def get_cpu_name() -> str:
    name = platform.processor()
    if name:
        return name
    # Linux fallback
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def get_hardware_info(device: str) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    info = {
        "device": device,
        "cpu_name": get_cpu_name(),
        "gpu_name": "None",
        "gpu_count": gpu_count,
        "cuda_available": bool(cuda_available),
        "cuda_version": torch.version.cuda if torch.version.cuda is not None else "None",
        "torch_version": torch.__version__,
        "gpu_total_memory_MB": 0.0,
    }
    if cuda_available and device.startswith("cuda"):
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_memory_MB"] = props.total_memory / (1024 ** 2)
    return info


def current_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return allocated, reserved
    return 0.0, 0.0


def peak_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        return allocated, reserved
    return 0.0, 0.0


# =========================
# 3. Taylor-Green MMS functions
# =========================

def uvp_mms(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    e1 = torch.exp(-t)
    e2 = torch.exp(-2.0 * t)
    u = torch.sin(x) * torch.cos(y) * e1
    v = -torch.cos(x) * torch.sin(y) * e1
    p = 0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * e2
    return u, v, p


def forcing_mms(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: float):
    e1 = torch.exp(-t)
    fu = (2.0 * nu - 1.0) * e1 * torch.sin(x) * torch.cos(y)
    fv = (1.0 - 2.0 * nu) * e1 * torch.cos(x) * torch.sin(y)
    return fu, fv


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


def build_dataset(cfg: NS2DConfig):
    # Collocation
    x_col = sample_uniform(cfg.n_col, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y_col = sample_uniform(cfg.n_col, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    t_col = sample_uniform(cfg.n_col, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    tr, va = split_indices(cfg.n_col, cfg.train_ratio, cfg.seed + 101, cfg.device)
    col_tr = (x_col[tr], y_col[tr], t_col[tr])
    col_va = (x_col[va], y_col[va], t_col[va])

    # Initial condition
    x_ic = sample_uniform(cfg.n_ic, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y_ic = sample_uniform(cfg.n_ic, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    t_ic = torch.full_like(x_ic, cfg.t0)
    tr, va = split_indices(cfg.n_ic, cfg.train_ratio, cfg.seed + 202, cfg.device)
    u_tr, v_tr, p_tr = uvp_mms(x_ic[tr], y_ic[tr], t_ic[tr])
    u_va, v_va, p_va = uvp_mms(x_ic[va], y_ic[va], t_ic[va])
    ic_tr = (x_ic[tr], y_ic[tr], t_ic[tr], u_tr.detach(), v_tr.detach(), p_tr.detach())
    ic_va = (x_ic[va], y_ic[va], t_ic[va], u_va.detach(), v_va.detach(), p_va.detach())

    # Periodic BC in x direction
    y_bcx = sample_uniform(cfg.n_bcx, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    t_bcx = sample_uniform(cfg.n_bcx, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    xL = torch.full_like(y_bcx, cfg.x0)
    xR = torch.full_like(y_bcx, cfg.x1)
    tr, va = split_indices(cfg.n_bcx, cfg.train_ratio, cfg.seed + 303, cfg.device)
    bcx_tr = (xL[tr], xR[tr], y_bcx[tr], t_bcx[tr])
    bcx_va = (xL[va], xR[va], y_bcx[va], t_bcx[va])

    # Periodic BC in y direction
    x_bcy = sample_uniform(cfg.n_bcy, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    t_bcy = sample_uniform(cfg.n_bcy, cfg.t0, cfg.t1, cfg.device, cfg.dtype)
    yB = torch.full_like(x_bcy, cfg.y0)
    yT = torch.full_like(x_bcy, cfg.y1)
    tr, va = split_indices(cfg.n_bcy, cfg.train_ratio, cfg.seed + 404, cfg.device)
    bcy_tr = (x_bcy[tr], yB[tr], yT[tr], t_bcy[tr])
    bcy_va = (x_bcy[va], yB[va], yT[va], t_bcy[va])

    return {
        "col_tr": col_tr, "col_va": col_va,
        "ic_tr": ic_tr, "ic_va": ic_va,
        "bcx_tr": bcx_tr, "bcx_va": bcx_va,
        "bcy_tr": bcy_tr, "bcy_va": bcy_va,
    }


# =========================
# 4. Model
# =========================

class BaselineModel(nn.Module):
    def __init__(self, cfg: NS2DConfig):
        super().__init__()
        self.embed = nn.Linear(1, cfg.hidden)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden,
            hidden_size=cfg.hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, 3),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        seq = torch.stack([x, y, t], dim=1)   # [B, 3, 1]
        seq = self.embed(seq)                 # [B, 3, H]
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.lstm(seq)
        o = self.head(out[:, -1, :])
        return o[:, 0:1], o[:, 1:2], o[:, 2:3]


# =========================
# 5. Loss functions
# =========================

def grad1(u: torch.Tensor, x: torch.Tensor):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian_2d(u: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    ux = grad1(u, x)
    uy = grad1(u, y)
    uxx = grad1(ux, x)
    uyy = grad1(uy, y)
    return ux, uy, uxx + uyy


def field_data_losses_from_points(model: BaselineModel, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
    u_ref, v_ref, p_ref = uvp_mms(x, y, t)
    u_pred, v_pred, p_pred = model(x, y, t)
    sp_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
    sp_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)
    loss_u = torch.mean((u_pred - u_ref) ** 2)
    loss_v = torch.mean((v_pred - v_ref) ** 2)
    loss_p = torch.mean((p_pred - p_ref) ** 2)
    loss_speed = torch.mean((sp_pred - sp_ref) ** 2)
    return {
        "data_u": loss_u,
        "data_v": loss_v,
        "data_p": loss_p,
        "data_speed": loss_speed,
        "data_total": loss_u + loss_v + loss_p + loss_speed,
    }


def batch_losses(cfg: NS2DConfig, model: BaselineModel, col, ic, bcx, bcy):
    with torch.enable_grad():
        x_col, y_col, t_col = col
        x = x_col.clone().detach().requires_grad_(True)
        y = y_col.clone().detach().requires_grad_(True)
        t = t_col.clone().detach().requires_grad_(True)

        u, v, p = model(x, y, t)
        u_t = grad1(u, t)
        v_t = grad1(v, t)
        u_x, u_y, lap_u = laplacian_2d(u, x, y)
        v_x, v_y, lap_v = laplacian_2d(v, x, y)
        p_x = grad1(p, x)
        p_y = grad1(p, y)

        fu, fv = forcing_mms(x_col, y_col, t_col, cfg.nu)
        r_u = u_t + u * u_x + v * u_y + p_x - cfg.nu * lap_u - fu
        r_v = v_t + u * v_x + v * v_y + p_y - cfg.nu * lap_v - fv
        r_div = u_x + v_y

        loss_mom = torch.mean(r_u ** 2) + torch.mean(r_v ** 2)
        loss_div = torch.mean(r_div ** 2)
        loss_pde = loss_mom + loss_div

        x_ic, y_ic, t_ic, u_ic, v_ic, p_ic = ic
        u0, v0, p0 = model(x_ic, y_ic, t_ic)
        loss_ic = (
            torch.mean((u0 - u_ic) ** 2)
            + torch.mean((v0 - v_ic) ** 2)
            + torch.mean((p0 - p_ic) ** 2)
        )

        xL, xR, y_bcx, t_bcx = bcx
        xLr = xL.clone().detach().requires_grad_(True)
        xRr = xR.clone().detach().requires_grad_(True)
        yx = y_bcx.clone().detach().requires_grad_(True)
        tt = t_bcx.clone().detach().requires_grad_(True)
        uL, vL, pL = model(xLr, yx, tt)
        uR, vR, pR = model(xRr, yx, tt)
        uL_x = grad1(uL, xLr); uR_x = grad1(uR, xRr)
        vL_x = grad1(vL, xLr); vR_x = grad1(vR, xRr)
        pL_x = grad1(pL, xLr); pR_x = grad1(pR, xRr)
        loss_bcx = (
            torch.mean((uL - uR) ** 2) + torch.mean((vL - vR) ** 2) + torch.mean((pL - pR) ** 2)
            + torch.mean((uL_x - uR_x) ** 2) + torch.mean((vL_x - vR_x) ** 2) + torch.mean((pL_x - pR_x) ** 2)
        )

        x_bcy, yB, yT, t_bcy = bcy
        xx = x_bcy.clone().detach().requires_grad_(True)
        yBr = yB.clone().detach().requires_grad_(True)
        yTr = yT.clone().detach().requires_grad_(True)
        tt2 = t_bcy.clone().detach().requires_grad_(True)
        uB, vB, pB = model(xx, yBr, tt2)
        uT, vT, pT = model(xx, yTr, tt2)
        uB_y = grad1(uB, yBr); uT_y = grad1(uT, yTr)
        vB_y = grad1(vB, yBr); vT_y = grad1(vT, yTr)
        pB_y = grad1(pB, yBr); pT_y = grad1(pT, yTr)
        loss_bcy = (
            torch.mean((uB - uT) ** 2) + torch.mean((vB - vT) ** 2) + torch.mean((pB - pT) ** 2)
            + torch.mean((uB_y - uT_y) ** 2) + torch.mean((vB_y - vT_y) ** 2) + torch.mean((pB_y - pT_y) ** 2)
        )

        loss_bc = loss_bcx + loss_bcy

        # Data losses are not necessarily optimized, but are logged for every field.
        data_parts = field_data_losses_from_points(model, x_col, y_col, t_col)

        loss = cfg.w_mom * loss_mom + cfg.w_div * loss_div + cfg.w_ic * loss_ic + cfg.w_bc * loss_bc

        parts = {
            "pde": loss_pde,
            "mom": loss_mom,
            "div": loss_div,
            "ic": loss_ic,
            "bc": loss_bc,
            "data": data_parts["data_total"],
            "data_u": data_parts["data_u"],
            "data_v": data_parts["data_v"],
            "data_p": data_parts["data_p"],
            "data_speed": data_parts["data_speed"],
        }
        return loss, parts


# =========================
# 6. Mini-batching
# =========================

def minibatch3(a, b, c, batch: int, seed: int):
    n = a.shape[0]
    g = torch.Generator(device=a.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=a.device)[:min(batch, n)]
    return a[idx], b[idx], c[idx]


def minibatch4(a, b, c, d, batch: int, seed: int):
    n = a.shape[0]
    g = torch.Generator(device=a.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=a.device)[:min(batch, n)]
    return a[idx], b[idx], c[idx], d[idx]


def minibatch6(a, b, c, d, e, f, batch: int, seed: int):
    n = a.shape[0]
    g = torch.Generator(device=a.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=a.device)[:min(batch, n)]
    return a[idx], b[idx], c[idx], d[idx], e[idx], f[idx]


# =========================
# 7. Metrics and evaluation
# =========================

@torch.no_grad()
def compute_metrics_np(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred).reshape(-1)
    ref = np.asarray(ref).reshape(-1)
    diff = pred - ref
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff ** 2)))
    rel_l2 = float(l2 / (np.sqrt(np.sum(ref ** 2)) + 1e-12))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "L2": l2,
        "RelL2": rel_l2,
        "MaxAbsError": max_abs,
        "MeanAbsError": mean_abs,
    }


@torch.no_grad()
def compute_metrics_torch(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = pred - ref
    mse = torch.mean(diff ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()
    l2 = torch.sqrt(torch.sum(diff ** 2)).item()
    rel_l2 = (torch.sqrt(torch.sum(diff ** 2)) / (torch.sqrt(torch.sum(ref ** 2)) + 1e-12)).item()
    max_abs = torch.max(torch.abs(diff)).item()
    mean_abs = torch.mean(torch.abs(diff)).item()
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "L2": l2,
        "RelL2": rel_l2,
        "MaxAbsError": max_abs,
        "MeanAbsError": mean_abs,
    }


@torch.no_grad()
def evaluate_val_metrics(cfg: NS2DConfig, model: BaselineModel, data, n: int = 3000):
    x_va, y_va, t_va = data["col_va"]
    n = min(n, x_va.shape[0])
    idx = torch.randperm(x_va.shape[0], device=cfg.device)[:n]
    x = x_va[idx]
    y = y_va[idx]
    t = t_va[idx]
    u_ref, v_ref, p_ref = uvp_mms(x, y, t)
    sp_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
    u_pred, v_pred, p_pred = model(x, y, t)
    sp_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)
    ref = torch.cat([u_ref, v_ref, p_ref, sp_ref], dim=1)
    pred = torch.cat([u_pred, v_pred, p_pred, sp_pred], dim=1)
    return compute_metrics_torch(pred, ref)


@torch.no_grad()
def eval_on_grid(cfg: NS2DConfig, model: BaselineModel):
    nx, ny = cfg.eval_nx, cfg.eval_ny
    xs = torch.linspace(cfg.x0, cfg.x1, nx, device=cfg.device, dtype=cfg.dtype)
    ys = torch.linspace(cfg.y0, cfg.y1, ny, device=cfg.device, dtype=cfg.dtype)
    result = {
        "x": xs.cpu().numpy(),
        "y": ys.cpu().numpy(),
        "times": np.array(cfg.eval_times, dtype=np.float32),
    }

    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    X = XX.reshape(-1, 1)
    Y = YY.reshape(-1, 1)

    for ti in cfg.eval_times:
        T = torch.full_like(X, float(ti))
        u_ref, v_ref, p_ref = uvp_mms(X, Y, T)

        u_list, v_list, p_list = [], [], []
        chunk = 4096
        for i in range(0, X.shape[0], chunk):
            up, vp, pp = model(X[i:i+chunk], Y[i:i+chunk], T[i:i+chunk])
            u_list.append(up)
            v_list.append(vp)
            p_list.append(pp)

        u_pred = torch.cat(u_list, 0)
        v_pred = torch.cat(v_list, 0)
        p_pred = torch.cat(p_list, 0)
        sp_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
        sp_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)

        tag = time_tag(ti)
        values = {
            "u_ref": u_ref,
            "v_ref": v_ref,
            "p_ref": p_ref,
            "speed_ref": sp_ref,
            "u_pred": u_pred,
            "v_pred": v_pred,
            "p_pred": p_pred,
            "speed_pred": sp_pred,
        }
        for name, val in values.items():
            result[f"{name}_{tag}"] = val.view(nx, ny).cpu().numpy()

        for fld in ["u", "v", "p", "speed"]:
            result[f"{fld}_err_{tag}"] = result[f"{fld}_pred_{tag}"] - result[f"{fld}_ref_{tag}"]

    return result


# =========================
# 8. Standardized outputs
# =========================

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_history_csv(path: str, history: Dict[str, List[Any]]):
    ensure_dir(os.path.dirname(path))
    keys = list(history.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(history[keys[0]]) if keys else 0
        for i in range(n):
            writer.writerow([history[k][i] for k in keys])


def save_field_txt_files(cfg: NS2DConfig, grid: Dict[str, np.ndarray]):
    xs = grid["x"]
    ys = grid["y"]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    for ti in grid["times"]:
        tag = time_tag(float(ti))
        out_sub = os.path.join(cfg.out_dir, "field_data", tag)
        ensure_dir(out_sub)
        for fld in ["u", "v", "p", "speed"]:
            pred = grid[f"{fld}_pred_{tag}"]
            ref = grid[f"{fld}_ref_{tag}"]
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
            fname = os.path.join(out_sub, f"{fld}_field_data_{tag}.txt")
            np.savetxt(fname, arr, fmt="%.10e", delimiter="\t", header=header, comments="")


def save_metrics_csv(cfg: NS2DConfig, grid: Dict[str, np.ndarray]):
    metric_cols = ["MSE", "RMSE", "MAE", "L2", "RelL2", "MaxAbsError", "MeanAbsError"]

    field_rows = []
    all_pred_parts = []
    all_ref_parts = []

    for ti in grid["times"]:
        tag = time_tag(float(ti))
        for fld in ["u", "v", "p", "speed"]:
            pred = grid[f"{fld}_pred_{tag}"]
            ref = grid[f"{fld}_ref_{tag}"]
            m = compute_metrics_np(pred, ref)
            row = {
                "case": cfg.case_name,
                "model": cfg.model_name,
                "time": float(ti),
                "field": fld,
            }
            row.update(m)
            field_rows.append(row)
            all_pred_parts.append(pred.reshape(-1))
            all_ref_parts.append(ref.reshape(-1))

    fieldnames = ["case", "model", "time", "field"] + metric_cols
    write_csv(os.path.join(cfg.out_dir, "metrics", "field_metrics.csv"), field_rows, fieldnames)

    overall_pred = np.concatenate(all_pred_parts, axis=0)
    overall_ref = np.concatenate(all_ref_parts, axis=0)
    overall = compute_metrics_np(overall_pred, overall_ref)
    overall_row = {"case": cfg.case_name, "model": cfg.model_name}
    overall_row.update(overall)
    write_csv(
        os.path.join(cfg.out_dir, "metrics", "overall_metrics.csv"),
        [overall_row],
        ["case", "model"] + metric_cols,
    )
    return overall


def save_plots(cfg: NS2DConfig, grid: Dict[str, np.ndarray], history: Dict[str, List[Any]]):
    x = grid["x"]
    y = grid["y"]
    extent = [x[0], x[-1], y[0], y[-1]]

    def save_map(U, title, fname):
        plt.figure(figsize=(5.5, 4.5))
        plt.imshow(U.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=220)
        plt.close()

    for ti in grid["times"]:
        tag = time_tag(float(ti))
        for fld in ["u", "v", "p", "speed"]:
            save_map(
                grid[f"{fld}_ref_{tag}"],
                f"Reference {fld}, {tag}",
                os.path.join(cfg.out_dir, "figures", "reference", f"{fld}_reference_{tag}.png"),
            )
            save_map(
                grid[f"{fld}_pred_{tag}"],
                f"Prediction {fld}, {tag}",
                os.path.join(cfg.out_dir, "figures", "prediction", f"{fld}_prediction_{tag}.png"),
            )
            save_map(
                np.abs(grid[f"{fld}_err_{tag}"]),
                f"|{fld} error|, {tag}",
                os.path.join(cfg.out_dir, "figures", "error", f"{fld}_error_{tag}.png"),
            )

    # Loss curves
    loss_dir = os.path.join(cfg.out_dir, "figures", "loss")
    ensure_dir(loss_dir)
    curves = [
        ("loss_total_train", "loss_total_val", "Total loss", "loss_total_curve.png"),
        ("loss_pde_train", "loss_pde_val", "PDE loss", "loss_pde_curve.png"),
        ("loss_mom_train", "loss_mom_val", "Momentum loss", "loss_mom_curve.png"),
        ("loss_div_train", "loss_div_val", "Divergence loss", "loss_div_curve.png"),
        ("loss_ic_train", "loss_ic_val", "IC loss", "loss_ic_curve.png"),
        ("loss_bc_train", "loss_bc_val", "BC loss", "loss_bc_curve.png"),
        ("loss_data_train", "loss_data_val", "Data loss", "loss_data_curve.png"),
        ("RelL2_val", None, "Validation RelL2", "rel_l2_curve.png"),
    ]
    for tr_key, va_key, title, fname in curves:
        if tr_key not in history or len(history[tr_key]) == 0:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(history["iter"], history[tr_key], label=tr_key)
        if va_key is not None and va_key in history:
            plt.plot(history["iter"], history[va_key], label=va_key)
        plt.yscale("log")
        plt.xlabel("iter")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(loss_dir, fname), dpi=220)
        plt.close()


def save_grid_npz(cfg: NS2DConfig, grid: Dict[str, np.ndarray]):
    ensure_dir(cfg.out_dir)
    np.savez(os.path.join(cfg.out_dir, "grid.npz"), **grid)


def save_model_info(cfg: NS2DConfig, model: nn.Module):
    ensure_dir(cfg.out_dir)
    path = os.path.join(cfg.out_dir, "model_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Case: {cfg.case_name}\n")
        f.write(f"Model: {cfg.model_name}\n")
        f.write("Architecture: Linear embedding + 2-layer LSTM + MLP head\n")
        f.write(f"Input sequence: x, y, t\n")
        f.write(f"Output fields: u, v, p\n")
        f.write(f"Hidden size: {cfg.hidden}\n")
        f.write(f"LSTM layers: {cfg.lstm_layers}\n")
        f.write(f"Trainable parameters: {count_trainable_parameters(model)}\n")
        f.write(f"Total parameters: {count_total_parameters(model)}\n")
        f.write(f"Parameter size MB: {parameter_size_mb(model):.6f}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Learning rate: {cfg.lr}\n")
        f.write(f"Iterations: {cfg.iters}\n")
        f.write(f"Batch collocation: {cfg.batch_col}\n")
        f.write(f"Batch IC: {cfg.batch_ic}\n")
        f.write(f"Batch BC-x: {cfg.batch_bcx}\n")
        f.write(f"Batch BC-y: {cfg.batch_bcy}\n")
        f.write(f"Seed: {cfg.seed}\n")


def build_threshold_rows(cfg: NS2DConfig, history: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    rows = []
    rel_values = history.get("RelL2_val", [])
    iters = history.get("iter", [])
    times = history.get("time_elapsed_sec", [])
    for th in cfg.rel_l2_thresholds:
        reached = False
        iter_to = -1
        time_to = -1.0
        for it, rel, tsec in zip(iters, rel_values, times):
            if rel <= th:
                reached = True
                iter_to = int(it)
                time_to = float(tsec)
                break
        rows.append({
            "case": cfg.case_name,
            "model": cfg.model_name,
            "metric": "RelL2",
            "threshold": th,
            "reached": reached,
            "iter_to_threshold": iter_to,
            "time_to_threshold_sec": time_to,
        })
    return rows


def save_cost_outputs(
    cfg: NS2DConfig,
    model: nn.Module,
    history: Dict[str, List[Any]],
    iter_times: List[float],
    total_wall_time_sec: float,
    best_iter: int,
    best_rel_l2: float,
    final_rel_l2: float,
):
    cost_dir = os.path.join(cfg.out_dir, "cost")
    ensure_dir(cost_dir)
    hw = get_hardware_info(cfg.device)
    peak_alloc, peak_reserved = peak_gpu_memory(cfg.device)

    mean_time = float(np.mean(iter_times)) if iter_times else 0.0
    median_time = float(np.median(iter_times)) if iter_times else 0.0

    summary = {
        "case": cfg.case_name,
        "model": cfg.model_name,
        "device": hw["device"],
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "parameter_size_MB": parameter_size_mb(model),
        "total_iterations_configured": cfg.iters,
        "total_iterations_completed": history["iter"][-1] if history["iter"] else 0,
        "total_wall_time_sec": total_wall_time_sec,
        "total_wall_time_min": total_wall_time_sec / 60.0,
        "mean_time_per_iter_sec": mean_time,
        "median_time_per_iter_sec": median_time,
        "gpu_name": hw["gpu_name"],
        "gpu_count": hw["gpu_count"],
        "cuda_available": hw["cuda_available"],
        "cuda_version": hw["cuda_version"],
        "torch_version": hw["torch_version"],
        "cpu_name": hw["cpu_name"],
        "gpu_total_memory_MB": hw["gpu_total_memory_MB"],
        "peak_gpu_memory_allocated_MB": peak_alloc,
        "peak_gpu_memory_reserved_MB": peak_reserved,
        "best_iter": best_iter,
        "best_RelL2": best_rel_l2,
        "final_RelL2": final_rel_l2,
    }
    summary_fields = list(summary.keys())
    write_csv(os.path.join(cost_dir, "model_cost_summary.csv"), [summary], summary_fields)

    threshold_rows = build_threshold_rows(cfg, history)
    write_csv(
        os.path.join(cost_dir, "threshold_cost.csv"),
        threshold_rows,
        ["case", "model", "metric", "threshold", "reached", "iter_to_threshold", "time_to_threshold_sec"],
    )


def save_all_outputs(
    cfg: NS2DConfig,
    model: nn.Module,
    grid: Dict[str, np.ndarray],
    history: Dict[str, List[Any]],
    iter_times: List[float],
    total_wall_time_sec: float,
    best_iter: int,
    best_rel_l2: float,
):
    # Directory creation
    for sub in [
        "field_data", "metrics", "cost", "loss",
        "figures/reference", "figures/prediction", "figures/error", "figures/loss",
    ]:
        ensure_dir(os.path.join(cfg.out_dir, sub))

    save_grid_npz(cfg, grid)
    save_model_info(cfg, model)
    write_history_csv(os.path.join(cfg.out_dir, "loss", "train_history.csv"), history)
    save_field_txt_files(cfg, grid)
    overall = save_metrics_csv(cfg, grid)
    save_plots(cfg, grid, history)
    final_rel_l2 = overall["RelL2"]
    save_cost_outputs(cfg, model, history, iter_times, total_wall_time_sec, best_iter, best_rel_l2, final_rel_l2)


# =========================
# 9. Training
# =========================

def make_history() -> Dict[str, List[Any]]:
    keys = [
        "iter",
        "loss_total_train", "loss_total_val",
        "loss_pde_train", "loss_pde_val",
        "loss_mom_train", "loss_mom_val",
        "loss_div_train", "loss_div_val",
        "loss_ic_train", "loss_ic_val",
        "loss_bc_train", "loss_bc_val",
        "loss_data_train", "loss_data_val",
        "loss_data_u_train", "loss_data_u_val",
        "loss_data_v_train", "loss_data_v_val",
        "loss_data_p_train", "loss_data_p_val",
        "loss_data_speed_train", "loss_data_speed_val",
        "MSE_val", "RMSE_val", "MAE_val", "L2_val", "RelL2_val",
        "MaxAbsError_val", "MeanAbsError_val",
        "time_elapsed_sec",
    ]
    return {k: [] for k in keys}


def append_history(history: Dict[str, List[Any]], row: Dict[str, Any]):
    for k in history.keys():
        history[k].append(row.get(k, 0.0))


def train(cfg: NS2DConfig):
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model = BaselineModel(cfg).to(cfg.device, cfg.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    data = build_dataset(cfg)

    save_model_info(cfg, model)

    history = make_history()
    iter_times: List[float] = []
    best_rel_l2 = float("inf")
    best_iter = -1
    best_path = os.path.join(cfg.out_dir, "best_model.pt")
    final_path = os.path.join(cfg.out_dir, "final_model.pt")

    print(f"[INFO] Case: {cfg.case_name} | Model: {cfg.model_name}")
    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"[INFO] Output directory: {os.path.abspath(cfg.out_dir)}")

    t_start = time.time()

    for it in range(1, cfg.iters + 1):
        iter_start = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)

        col_tr = minibatch3(*data["col_tr"], cfg.batch_col, cfg.seed + it)
        ic_tr = minibatch6(*data["ic_tr"], cfg.batch_ic, cfg.seed + 10000 + it)
        bcx_tr = minibatch4(*data["bcx_tr"], cfg.batch_bcx, cfg.seed + 20000 + it)
        bcy_tr = minibatch4(*data["bcy_tr"], cfg.batch_bcy, cfg.seed + 30000 + it)

        loss, parts = batch_losses(cfg, model, col_tr, ic_tr, bcx_tr, bcy_tr)
        if not torch.isfinite(loss):
            print(f"[iter {it}] Non-finite loss detected. Training stopped.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        iter_times.append(time.time() - iter_start)

        if it == 1 or it % cfg.eval_every == 0:
            model.eval()
            loss_total_train = float(loss.detach().item())
            parts_train = {k: float(v.detach().item()) for k, v in parts.items()}

            col_va = minibatch3(*data["col_va"], cfg.val_batch_col, cfg.seed + 40000 + it)
            ic_va = minibatch6(*data["ic_va"], cfg.val_batch_ic, cfg.seed + 50000 + it)
            bcx_va = minibatch4(*data["bcx_va"], cfg.val_batch_bcx, cfg.seed + 60000 + it)
            bcy_va = minibatch4(*data["bcy_va"], cfg.val_batch_bcy, cfg.seed + 70000 + it)

            loss_va_t, parts_va_t = batch_losses(cfg, model, col_va, ic_va, bcx_va, bcy_va)
            loss_total_val = float(loss_va_t.detach().item())
            parts_val = {k: float(v.detach().item()) for k, v in parts_va_t.items()}

            val_metrics = evaluate_val_metrics(cfg, model, data, n=3000)
            elapsed = time.time() - t_start

            row = {
                "iter": int(it),
                "loss_total_train": loss_total_train,
                "loss_total_val": loss_total_val,
                "loss_pde_train": parts_train["pde"],
                "loss_pde_val": parts_val["pde"],
                "loss_mom_train": parts_train["mom"],
                "loss_mom_val": parts_val["mom"],
                "loss_div_train": parts_train["div"],
                "loss_div_val": parts_val["div"],
                "loss_ic_train": parts_train["ic"],
                "loss_ic_val": parts_val["ic"],
                "loss_bc_train": parts_train["bc"],
                "loss_bc_val": parts_val["bc"],
                "loss_data_train": parts_train["data"],
                "loss_data_val": parts_val["data"],
                "loss_data_u_train": parts_train["data_u"],
                "loss_data_u_val": parts_val["data_u"],
                "loss_data_v_train": parts_train["data_v"],
                "loss_data_v_val": parts_val["data_v"],
                "loss_data_p_train": parts_train["data_p"],
                "loss_data_p_val": parts_val["data_p"],
                "loss_data_speed_train": parts_train["data_speed"],
                "loss_data_speed_val": parts_val["data_speed"],
                "MSE_val": val_metrics["MSE"],
                "RMSE_val": val_metrics["RMSE"],
                "MAE_val": val_metrics["MAE"],
                "L2_val": val_metrics["L2"],
                "RelL2_val": val_metrics["RelL2"],
                "MaxAbsError_val": val_metrics["MaxAbsError"],
                "MeanAbsError_val": val_metrics["MeanAbsError"],
                "time_elapsed_sec": elapsed,
            }
            append_history(history, row)

            if val_metrics["RelL2"] < best_rel_l2:
                best_rel_l2 = val_metrics["RelL2"]
                best_iter = int(it)
                torch.save({
                    "iter": best_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_rel_l2": best_rel_l2,
                    "config": cfg.__dict__,
                }, best_path)

            print(
                f"[{it:6d}] "
                f"loss_tr={loss_total_train:.3e} loss_va={loss_total_val:.3e} "
                f"data_va={parts_val['data']:.3e} RMSE={val_metrics['RMSE']:.3e} "
                f"RelL2={val_metrics['RelL2']:.3e} elapsed={elapsed/60:.1f} min"
            )

    total_wall_time_sec = time.time() - t_start

    # Save final checkpoint
    torch.save({
        "iter": history["iter"][-1] if history["iter"] else 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "final_rel_l2_val": history["RelL2_val"][-1] if history["RelL2_val"] else None,
        "config": cfg.__dict__,
    }, final_path)

    # Final grid evaluation and standardized outputs
    model.eval()
    grid = eval_on_grid(cfg, model)
    save_all_outputs(cfg, model, grid, history, iter_times, total_wall_time_sec, best_iter, best_rel_l2)

    overall_path = os.path.join(cfg.out_dir, "metrics", "overall_metrics.csv")
    print(f"[DONE] Total wall time: {total_wall_time_sec/60:.2f} min")
    print(f"[DONE] Best RelL2: {best_rel_l2:.6e} at iter {best_iter}")
    print(f"[DONE] Overall metrics saved to: {overall_path}")
    print(f"[DONE] Outputs saved to: {os.path.abspath(cfg.out_dir)}")
    return model


# =========================
# 10. Main
# =========================

def main():
    cfg = NS2DConfig()
    train(cfg)


if __name__ == "__main__":
    main()
