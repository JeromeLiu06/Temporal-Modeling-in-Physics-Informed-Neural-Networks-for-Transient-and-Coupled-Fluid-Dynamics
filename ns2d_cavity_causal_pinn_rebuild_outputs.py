

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




@dataclass
class CavityConfig:
    case_name: str = "Lid"
    model_name: str = "Causal-PINN"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Physical parameter
    nu: float = 0.01

    # Domain
    x0: float = 0.0
    x1: float = 1.0
    y0: float = 0.0
    y1: float = 1.0

    # Dataset size
    n_col: int = 22000
    n_bc_bottom: int = 2200
    n_bc_top: int = 2200
    n_bc_left: int = 2200
    n_bc_right: int = 2200
    train_ratio: float = 0.7

    # Training
    iters: int = 80000
    lr: float = 1e-4
    batch_col: int = 1024
    batch_bc: int = 256

    # Validation batches
    val_batch_col: int = 512
    val_batch_bc: int = 256

    eval_every: int = 500
    grad_clip: float = 1.0

    # Loss weights
    w_mom: float = 1.0
    w_div: float = 1.0
    w_bc: float = 10.0


    hidden: int = 128
    num_hidden_layers: int = 8
    causal_chunks: int = 10
    causal_epsilon: float = 1.0
    causal_axis: str = "y"

    # Evaluation grid
    eval_nx: int = 120
    eval_ny: int = 120

    # Cost threshold tracking
    rel_l2_thresholds: Tuple[float, float] = (1e-2, 1e-3)

    # Output
    out_dir: str = "./outputs/Lid/Causal-PINN"
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
    if path:
        os.makedirs(path, exist_ok=True)


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


def peak_gpu_memory(device: str) -> Tuple[float, float]:
    if torch.cuda.is_available() and device.startswith("cuda"):
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        return allocated, reserved
    return 0.0, 0.0


# =========================
# 3. Lid-driven cavity MMS functions
# =========================

def fpoly(x: torch.Tensor):
    return x ** 2 - 2.0 * x ** 3 + x ** 4


def fpoly_p(x: torch.Tensor):
    return 2.0 * x - 6.0 * x ** 2 + 4.0 * x ** 3


def fpoly_pp(x: torch.Tensor):
    return 2.0 - 12.0 * x + 12.0 * x ** 2


def fpoly_ppp(x: torch.Tensor):
    return -12.0 + 24.0 * x


def gpoly(y: torch.Tensor):
    return y ** 3 - y ** 2


def gpoly_p(y: torch.Tensor):
    return 3.0 * y ** 2 - 2.0 * y


def gpoly_pp(y: torch.Tensor):
    return 6.0 * y - 2.0


def uvp_mms(x: torch.Tensor, y: torch.Tensor):
    fx = fpoly(x)
    fpx = fpoly_p(x)
    gy = gpoly(y)
    gpy = gpoly_p(y)
    u = fx * gpy
    v = -fpx * gy
    p = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return u, v, p


def forcing_mms(x: torch.Tensor, y: torch.Tensor, nu: float):
    fx = fpoly(x)
    fpx = fpoly_p(x)
    fppx = fpoly_pp(x)
    fpppx = fpoly_ppp(x)
    gy = gpoly(y)
    gpy = gpoly_p(y)
    gppy = gpoly_pp(y)

    u = fx * gpy
    v = -fpx * gy

    u_x = fpx * gpy
    u_y = fx * gppy
    u_xx = fppx * gpy
    u_yy = 6.0 * fx

    v_x = -fppx * gy
    v_y = -fpx * gpy
    v_xx = -fpppx * gy
    v_yy = -fpx * gppy

    p_x = math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
    p_y = math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)

    fu = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    fv = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
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


def build_dataset(cfg: CavityConfig) -> Dict[str, Tuple[torch.Tensor, ...]]:
    # Collocation points
    x_col = sample_uniform(cfg.n_col, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
    y_col = sample_uniform(cfg.n_col, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
    tr, va = split_indices(cfg.n_col, cfg.train_ratio, cfg.seed + 101, cfg.device)
    col_tr = (x_col[tr], y_col[tr])
    col_va = (x_col[va], y_col[va])

    def make_bc(n: int, side_seed: int, kind: str):
        if kind == "bottom":
            x = sample_uniform(n, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
            y = torch.full_like(x, cfg.y0)
        elif kind == "top":
            x = sample_uniform(n, cfg.x0, cfg.x1, cfg.device, cfg.dtype)
            y = torch.full_like(x, cfg.y1)
        elif kind == "left":
            y = sample_uniform(n, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
            x = torch.full_like(y, cfg.x0)
        elif kind == "right":
            y = sample_uniform(n, cfg.y0, cfg.y1, cfg.device, cfg.dtype)
            x = torch.full_like(y, cfg.x1)
        else:
            raise ValueError(f"Unknown boundary kind: {kind}")

        u, v, p = uvp_mms(x, y)
        tr_i, va_i = split_indices(n, cfg.train_ratio, side_seed, cfg.device)
        return (
            (x[tr_i], y[tr_i], u[tr_i].detach(), v[tr_i].detach(), p[tr_i].detach()),
            (x[va_i], y[va_i], u[va_i].detach(), v[va_i].detach(), p[va_i].detach()),
        )

    bc_bottom_tr, bc_bottom_va = make_bc(cfg.n_bc_bottom, cfg.seed + 201, "bottom")
    bc_top_tr, bc_top_va = make_bc(cfg.n_bc_top, cfg.seed + 202, "top")
    bc_left_tr, bc_left_va = make_bc(cfg.n_bc_left, cfg.seed + 203, "left")
    bc_right_tr, bc_right_va = make_bc(cfg.n_bc_right, cfg.seed + 204, "right")

    return {
        "col_tr": col_tr,
        "col_va": col_va,
        "bc_bottom_tr": bc_bottom_tr,
        "bc_bottom_va": bc_bottom_va,
        "bc_top_tr": bc_top_tr,
        "bc_top_va": bc_top_va,
        "bc_left_tr": bc_left_tr,
        "bc_left_va": bc_left_va,
        "bc_right_tr": bc_right_tr,
        "bc_right_va": bc_right_va,
    }


# =========================
# 4. Model
# =========================

class BaselineModel(nn.Module):


    def __init__(self, cfg: CavityConfig):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(2, cfg.hidden))
        layers.append(nn.Tanh())
        for _ in range(cfg.num_hidden_layers - 1):
            layers.append(nn.Linear(cfg.hidden, cfg.hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(cfg.hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


# =========================
# 5. Loss functions
# =========================

def grad1(u: torch.Tensor, x: torch.Tensor):
    return torch.autograd.grad(
        u,
        x,
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


def field_data_losses_from_points(model: BaselineModel, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    u_ref, v_ref, p_ref = uvp_mms(x, y)
    u_pred, v_pred, p_pred = model(x, y)
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


def boundary_loss_from_bundle(model: BaselineModel, bundle: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    x, y, u_ref, v_ref, p_ref = bundle
    u_pred, v_pred, p_pred = model(x, y)
    return (
        torch.mean((u_pred - u_ref) ** 2)
        + torch.mean((v_pred - v_ref) ** 2)
        + torch.mean((p_pred - p_ref) ** 2)
    )


def causal_spatial_weighted_losses(
    point_mom: torch.Tensor,
    point_div: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: CavityConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    coord = y if str(cfg.causal_axis).lower() == "y" else x
    coord_flat = coord.reshape(-1)
    mom_flat = point_mom.reshape(-1)
    div_flat = point_div.reshape(-1)

    low = cfg.y0 if str(cfg.causal_axis).lower() == "y" else cfg.x0
    high = cfg.y1 if str(cfg.causal_axis).lower() == "y" else cfg.x1
    edges = torch.linspace(low, high, cfg.causal_chunks + 1, device=coord.device, dtype=coord.dtype)

    chunk_mom_losses = []
    chunk_div_losses = []
    for i in range(cfg.causal_chunks):
        left = edges[i]
        right = edges[i + 1]
        if i == cfg.causal_chunks - 1:
            mask = (coord_flat >= left) & (coord_flat <= right)
        else:
            mask = (coord_flat >= left) & (coord_flat < right)
        if torch.any(mask):
            chunk_mom_losses.append(torch.mean(mom_flat[mask]))
            chunk_div_losses.append(torch.mean(div_flat[mask]))

    if len(chunk_mom_losses) == 0:
        loss_mom = torch.mean(point_mom)
        loss_div = torch.mean(point_div)
        return loss_mom + loss_div, loss_mom, loss_div

    weighted_mom = []
    weighted_div = []
    cumulative = torch.zeros((), device=coord.device, dtype=coord.dtype)
    for lm, ld in zip(chunk_mom_losses, chunk_div_losses):
        weight = torch.exp(-cfg.causal_epsilon * cumulative.detach())
        weighted_mom.append(weight * lm)
        weighted_div.append(weight * ld)
        cumulative = cumulative + lm + ld

    loss_mom = torch.stack(weighted_mom).mean()
    loss_div = torch.stack(weighted_div).mean()
    return loss_mom + loss_div, loss_mom, loss_div


def batch_losses(
    cfg: CavityConfig,
    model: BaselineModel,
    col,
    bc_bottom,
    bc_top,
    bc_left,
    bc_right,
):
    with torch.enable_grad():
        x_col, y_col = col
        x = x_col.clone().detach().requires_grad_(True)
        y = y_col.clone().detach().requires_grad_(True)

        u, v, p = model(x, y)
        u_x, u_y, lap_u = laplacian_2d(u, x, y)
        v_x, v_y, lap_v = laplacian_2d(v, x, y)
        p_x = grad1(p, x)
        p_y = grad1(p, y)

        fu, fv = forcing_mms(x_col, y_col, cfg.nu)
        r_u = u * u_x + v * u_y + p_x - cfg.nu * lap_u - fu
        r_v = u * v_x + v * v_y + p_y - cfg.nu * lap_v - fv
        r_div = u_x + v_y

        point_mom = r_u ** 2 + r_v ** 2
        point_div = r_div ** 2
        loss_pde, loss_mom, loss_div = causal_spatial_weighted_losses(point_mom, point_div, x, y, cfg)
        loss_ic = torch.zeros((), device=x_col.device, dtype=x_col.dtype)
        loss_bc = (
            boundary_loss_from_bundle(model, bc_bottom)
            + boundary_loss_from_bundle(model, bc_top)
            + boundary_loss_from_bundle(model, bc_left)
            + boundary_loss_from_bundle(model, bc_right)
        )

        data_parts = field_data_losses_from_points(model, x_col, y_col)

        loss = cfg.w_mom * loss_mom + cfg.w_div * loss_div + cfg.w_bc * loss_bc

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

def minibatch2(a: torch.Tensor, b: torch.Tensor, batch: int, seed: int):
    n = a.shape[0]
    g = torch.Generator(device=a.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=a.device)[:min(batch, n)]
    return a[idx], b[idx]


def minibatch5(a, b, c, d, e, batch: int, seed: int):
    n = a.shape[0]
    g = torch.Generator(device=a.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=a.device)[:min(batch, n)]
    return a[idx], b[idx], c[idx], d[idx], e[idx]


# =========================
# 7. Metrics and evaluation
# =========================

def compute_metrics_np(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred).reshape(-1)
    ref = np.asarray(ref).reshape(-1)
    diff = pred - ref
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))
    rel_l2 = float(l2 / (np.linalg.norm(ref) + 1e-12))
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
    l2 = torch.linalg.norm(diff.reshape(-1)).item()
    rel_l2 = l2 / (torch.linalg.norm(ref.reshape(-1)).item() + 1e-12)
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
def evaluate_val_metrics(cfg: CavityConfig, model: BaselineModel, data, n: int = 4000) -> Dict[str, float]:
    x_va, y_va = data["col_va"]
    n = min(n, x_va.shape[0])
    idx = torch.randperm(x_va.shape[0], device=cfg.device)[:n]
    x = x_va[idx]
    y = y_va[idx]
    u_ref, v_ref, p_ref = uvp_mms(x, y)
    sp_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
    u_pred, v_pred, p_pred = model(x, y)
    sp_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)
    ref = torch.cat([u_ref, v_ref, p_ref, sp_ref], dim=1)
    pred = torch.cat([u_pred, v_pred, p_pred, sp_pred], dim=1)
    return compute_metrics_torch(pred, ref)


@torch.no_grad()
def eval_on_grid(cfg: CavityConfig, model: BaselineModel):
    nx, ny = cfg.eval_nx, cfg.eval_ny
    xs = torch.linspace(cfg.x0, cfg.x1, nx, device=cfg.device, dtype=cfg.dtype)
    ys = torch.linspace(cfg.y0, cfg.y1, ny, device=cfg.device, dtype=cfg.dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    X = XX.reshape(-1, 1)
    Y = YY.reshape(-1, 1)

    u_ref, v_ref, p_ref = uvp_mms(X, Y)
    u_list, v_list, p_list = [], [], []
    chunk = 4096
    for i in range(0, X.shape[0], chunk):
        up, vp, pp = model(X[i:i + chunk], Y[i:i + chunk])
        u_list.append(up)
        v_list.append(vp)
        p_list.append(pp)

    u_pred = torch.cat(u_list, 0)
    v_pred = torch.cat(v_list, 0)
    p_pred = torch.cat(p_list, 0)
    sp_ref = torch.sqrt(u_ref ** 2 + v_ref ** 2)
    sp_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)

    result = {
        "x": xs.cpu().numpy(),
        "y": ys.cpu().numpy(),
        "u_ref": u_ref.view(nx, ny).cpu().numpy(),
        "v_ref": v_ref.view(nx, ny).cpu().numpy(),
        "p_ref": p_ref.view(nx, ny).cpu().numpy(),
        "speed_ref": sp_ref.view(nx, ny).cpu().numpy(),
        "u_pred": u_pred.view(nx, ny).cpu().numpy(),
        "v_pred": v_pred.view(nx, ny).cpu().numpy(),
        "p_pred": p_pred.view(nx, ny).cpu().numpy(),
        "speed_pred": sp_pred.view(nx, ny).cpu().numpy(),
    }
    for fld in ["u", "v", "p", "speed"]:
        result[f"{fld}_err"] = result[f"{fld}_pred"] - result[f"{fld}_ref"]
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


def save_field_txt_files(cfg: CavityConfig, grid: Dict[str, np.ndarray]):
    xs = grid["x"]
    ys = grid["y"]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    out_sub = os.path.join(cfg.out_dir, "field_data", "steady")
    ensure_dir(out_sub)
    for fld in ["u", "v", "p", "speed"]:
        pred = grid[f"{fld}_pred"]
        ref = grid[f"{fld}_ref"]
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
        fname = os.path.join(out_sub, f"{fld}_field_data.txt")
        np.savetxt(fname, arr, fmt="%.10e", delimiter="\t", header=header, comments="")


def save_metrics_csv(cfg: CavityConfig, grid: Dict[str, np.ndarray]):
    metric_cols = ["MSE", "RMSE", "MAE", "L2", "RelL2", "MaxAbsError", "MeanAbsError"]

    field_rows = []
    all_pred_parts = []
    all_ref_parts = []

    for fld in ["u", "v", "p", "speed"]:
        pred = grid[f"{fld}_pred"]
        ref = grid[f"{fld}_ref"]
        m = compute_metrics_np(pred, ref)
        row = {
            "case": cfg.case_name,
            "model": cfg.model_name,
            "time": "steady",
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


def save_plots(cfg: CavityConfig, grid: Dict[str, np.ndarray], history: Dict[str, List[Any]]):
    x = grid["x"]
    y = grid["y"]
    extent = [x[0], x[-1], y[0], y[-1]]

    def save_map(U, title, fname):
        plt.figure(figsize=(5.6, 4.6))
        plt.imshow(U.T, origin="lower", extent=extent, aspect="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=220)
        plt.close()

    for fld in ["u", "v", "p", "speed"]:
        save_map(
            grid[f"{fld}_ref"],
            f"Reference {fld}",
            os.path.join(cfg.out_dir, "figures", "reference", f"{fld}_reference.png"),
        )
        save_map(
            grid[f"{fld}_pred"],
            f"Prediction {fld}",
            os.path.join(cfg.out_dir, "figures", "prediction", f"{fld}_prediction.png"),
        )
        save_map(
            np.abs(grid[f"{fld}_err"]),
            f"|{fld} error|",
            os.path.join(cfg.out_dir, "figures", "error", f"{fld}_error.png"),
        )

    # Centerline plots for cavity diagnostics.
    nx, ny = grid["u_ref"].shape
    midx = nx // 2
    midy = ny // 2
    plt.figure(figsize=(7, 4))
    plt.plot(y, grid["u_ref"][midx, :], label="u reference @ x=0.5")
    plt.plot(y, grid["u_pred"][midx, :], "--", label="u prediction @ x=0.5")
    plt.xlabel("y")
    plt.ylabel("u")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "figures", "prediction", "centerline_u.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, grid["v_ref"][:, midy], label="v reference @ y=0.5")
    plt.plot(x, grid["v_pred"][:, midy], "--", label="v prediction @ y=0.5")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "figures", "prediction", "centerline_v.png"), dpi=220)
    plt.close()

    loss_pairs = [
        ("loss_total_train", "loss_total_val", "Total loss", "loss_total_curve.png"),
        ("loss_pde_train", "loss_pde_val", "PDE loss", "loss_pde_curve.png"),
        ("loss_mom_train", "loss_mom_val", "Momentum loss", "loss_mom_curve.png"),
        ("loss_div_train", "loss_div_val", "Divergence loss", "loss_div_curve.png"),
        ("loss_bc_train", "loss_bc_val", "Boundary loss", "loss_bc_curve.png"),
        ("loss_data_train", "loss_data_val", "Data loss", "loss_data_curve.png"),
        ("RelL2_val", None, "Validation RelL2", "rel_l2_curve.png"),
    ]
    for tr, va, title, fname in loss_pairs:
        if len(history.get("iter", [])) == 0 or tr not in history:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(history["iter"], history[tr], label=tr)
        if va is not None and va in history:
            plt.plot(history["iter"], history[va], label=va)
        plt.yscale("log")
        plt.xlabel("iter")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "figures", "loss", fname), dpi=220)
        plt.close()


def save_grid_npz(cfg: CavityConfig, grid: Dict[str, np.ndarray]):
    np.savez(os.path.join(cfg.out_dir, "grid.npz"), **grid)


def save_model_info(cfg: CavityConfig, model: nn.Module):
    ensure_dir(cfg.out_dir)
    path = os.path.join(cfg.out_dir, "model_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Case: {cfg.case_name}\n")
        f.write(f"Model: {cfg.model_name}\n")
        f.write("Architecture: Vanilla fully connected PINN\n")
        f.write("Input dimension: 2 (x, y)\n")
        f.write("Output dimension: 3 (u, v, p)\n")
        f.write(f"Hidden dimension: {cfg.hidden}\n")
        f.write(f"Hidden layers: {cfg.num_hidden_layers}\n")
        f.write("Activation: Tanh\n")
        f.write(f"Trainable parameters: {count_trainable_parameters(model)}\n")
        f.write(f"Total parameters: {count_total_parameters(model)}\n")
        f.write(f"Parameter size MB: {parameter_size_mb(model):.6f}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Learning rate: {cfg.lr}\n")
        f.write(f"Iterations: {cfg.iters}\n")
        f.write(f"Batch col: {cfg.batch_col}\n")
        f.write(f"Batch bc: {cfg.batch_bc}\n")
        f.write(f"Seed: {cfg.seed}\n")


def build_threshold_rows(cfg: CavityConfig, history: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    rows = []
    iters = history.get("iter", [])
    rels = history.get("RelL2_val", [])
    times = history.get("time_elapsed_sec", [])
    for thr in cfg.rel_l2_thresholds:
        reached = False
        iter_to = -1
        time_to = -1.0
        for it, rel, sec in zip(iters, rels, times):
            if rel <= thr:
                reached = True
                iter_to = int(it)
                time_to = float(sec)
                break
        rows.append({
            "case": cfg.case_name,
            "model": cfg.model_name,
            "metric": "RelL2",
            "threshold": thr,
            "reached": reached,
            "iter_to_threshold": iter_to,
            "time_to_threshold_sec": time_to,
        })
    return rows


def save_cost_outputs(
    cfg: CavityConfig,
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
    write_csv(os.path.join(cost_dir, "model_cost_summary.csv"), [summary], list(summary.keys()))

    threshold_rows = build_threshold_rows(cfg, history)
    write_csv(
        os.path.join(cost_dir, "threshold_cost.csv"),
        threshold_rows,
        ["case", "model", "metric", "threshold", "reached", "iter_to_threshold", "time_to_threshold_sec"],
    )


def save_all_outputs(
    cfg: CavityConfig,
    model: nn.Module,
    grid: Dict[str, np.ndarray],
    history: Dict[str, List[Any]],
    iter_times: List[float],
    total_wall_time_sec: float,
    best_iter: int,
    best_rel_l2: float,
):
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


def train(cfg: CavityConfig):
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

        col_tr = minibatch2(*data["col_tr"], cfg.batch_col, cfg.seed + it)
        bc_bottom_tr = minibatch5(*data["bc_bottom_tr"], cfg.batch_bc, cfg.seed + 10000 + it)
        bc_top_tr = minibatch5(*data["bc_top_tr"], cfg.batch_bc, cfg.seed + 20000 + it)
        bc_left_tr = minibatch5(*data["bc_left_tr"], cfg.batch_bc, cfg.seed + 30000 + it)
        bc_right_tr = minibatch5(*data["bc_right_tr"], cfg.batch_bc, cfg.seed + 40000 + it)

        loss, parts = batch_losses(cfg, model, col_tr, bc_bottom_tr, bc_top_tr, bc_left_tr, bc_right_tr)
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

            col_va = minibatch2(*data["col_va"], cfg.val_batch_col, cfg.seed + 50000 + it)
            bc_bottom_va = minibatch5(*data["bc_bottom_va"], cfg.val_batch_bc, cfg.seed + 60000 + it)
            bc_top_va = minibatch5(*data["bc_top_va"], cfg.val_batch_bc, cfg.seed + 70000 + it)
            bc_left_va = minibatch5(*data["bc_left_va"], cfg.val_batch_bc, cfg.seed + 80000 + it)
            bc_right_va = minibatch5(*data["bc_right_va"], cfg.val_batch_bc, cfg.seed + 90000 + it)

            loss_va_t, parts_va_t = batch_losses(
                cfg, model, col_va, bc_bottom_va, bc_top_va, bc_left_va, bc_right_va
            )
            loss_total_val = float(loss_va_t.detach().item())
            parts_val = {k: float(v.detach().item()) for k, v in parts_va_t.items()}

            val_metrics = evaluate_val_metrics(cfg, model, data, n=4000)
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

    torch.save({
        "iter": history["iter"][-1] if history["iter"] else 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "final_rel_l2_val": history["RelL2_val"][-1] if history["RelL2_val"] else None,
        "config": cfg.__dict__,
    }, final_path)

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
    cfg = CavityConfig()
    train(cfg)


if __name__ == "__main__":
    main()
