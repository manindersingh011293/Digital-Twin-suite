"""
STRUCTURAL HEALTH MONITORING: REAL-TIME DIGITAL TWIN INFERENCE ENGINE
File: mthl_digital_twin.py
Backend: Adaptive CUDA (bfloat16) / Host CPU (fp32) Fallback
"""

import os
import time
import math
import threading
import contextlib
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"

try:
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except Exception:
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class BridgeSTTransformer(nn.Module):
    def __init__(
        self, coords: np.ndarray, num_nodes: int = 600, in_channels: int = 3,
        num_tendon_elements: int = 285, time_steps: int = 1024, d_model: int = 64,
        nhead: int = 4, num_encoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.05,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.T = time_steps

        if isinstance(coords, np.ndarray):
            c_min, c_max = coords.min(axis=0), coords.max(axis=0)
            norm_coords = (coords - c_min) / (c_max - c_min + 1e-6)
            self.register_buffer("coords_buf", torch.from_numpy(norm_coords).float())
        else:
            self.register_buffer("coords_buf", coords.float())

        self.coord_encoder = nn.Sequential(nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=time_steps)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.head_tendon = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_tendon_elements),
        )
        self.node_proj = nn.Linear(d_model, d_model)
        self.gcn_w1 = nn.Parameter(torch.empty(d_model, d_model))
        self.gcn_w2 = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.gcn_w1)
        nn.init.xavier_uniform_(self.gcn_w2)
        self.mesh_decoder = nn.Sequential(
            nn.Conv1d(d_model, dim_feedforward, kernel_size=5, padding=2),
            nn.BatchNorm1d(dim_feedforward), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(dim_feedforward, 6, kernel_size=3, padding=1),
        )

    def forward(self, x, adj):
        B = x.shape[0]
        tokens = self.pos_encoder(self.input_proj(x))
        h_t = self.temporal_transformer(tokens)
        out_tendon = self.head_tendon(h_t).permute(0, 2, 1)

        coord_emb = self.coord_encoder(self.coords_buf)
        h_spatial = self.node_proj(h_t).unsqueeze(2) + coord_emb.view(1, 1, self.num_nodes, -1)
        h_flat = h_spatial.reshape(B * self.T, self.num_nodes, -1)

        g1 = F.silu(torch.matmul(adj, torch.matmul(h_flat, self.gcn_w1)))
        g2 = F.silu(torch.matmul(adj, torch.matmul(g1, self.gcn_w2))) + g1

        h_mesh_seq = g2.reshape(B, self.T, self.num_nodes, -1).permute(0, 2, 3, 1).reshape(B * self.num_nodes, -1, self.T)
        out_mesh = self.mesh_decoder(h_mesh_seq).reshape(B, self.num_nodes, 6, self.T)
        return out_mesh, out_tendon

class StreamingRainflowFatigue:
    def __init__(self):
        self.extrema = []
        self.last_val = 0.0
        self.last_slope = 0

    def process_sample(self, val: float):
        cycles = []
        if len(self.extrema) == 0:
            self.extrema.append(val)
            self.last_val = val
            return cycles

        diff = val - self.last_val
        if abs(diff) < 1e-7:
            return cycles

        slope = 1 if diff > 0 else -1
        if slope != self.last_slope and self.last_slope != 0:
            self.extrema.append(self.last_val)
            while len(self.extrema) >= 3:
                s0, s1, s2 = self.extrema[-3], self.extrema[-2], self.extrema[-1]
                r1, r2 = abs(s1 - s0), abs(s2 - s1)
                if r2 >= r1:
                    cycles.append((r1, 0.5 * (s0 + s1), max(s0, s1), min(s0, s1)))
                    self.extrema.pop(-2)
                    self.extrema.pop(-2)
                else:
                    break

        self.last_slope = slope
        self.last_val = val
        return cycles

class ComprehensiveStructuralPostProcessor:
    def __init__(
        self, f_ck: float = 60.0, f_pu: float = 1860.0, A_p: float = 2660.0, f_pe: float = 1120.0,
        sigma_base_soffit: float = -10.5, sigma_base_deck: float = -7.2,
    ):
        self.E_c = 40200.0  # Elastic modulus in MPa (C50/60)
        self.nu = 0.20      # Poisson's ratio
        self.G_c = self.E_c / (2.0 * (1.0 + self.nu))
        
        self.f_ck, self.f_pu, self.A_p, self.f_pe = f_ck, f_pu, A_p, f_pe
        self.sigma_base_soffit = sigma_base_soffit
        self.sigma_base_deck = sigma_base_deck

        # Eurocode 2 Limits
        self.f_ctm = 4.10
        self.f_ctk_95 = 5.30
        self.f_cd = 40.0
        self.f_c_linear = 0.45 * f_ck
        self.f_c_sls = 0.60 * f_ck
        self.tau_web_crack = 0.30 * math.sqrt(f_ck)

        self.f_py = 0.85 * f_pu
        self.f_p_high = 0.75 * f_pu
        self.delta_sigma_Rsk = 120.0
        self.m_tendon = 5.0

        self.rainflow_tendon = StreamingRainflowFatigue()
        self.rainflow_conc_comp = StreamingRainflowFatigue()
        self.D_tendon, self.D_conc_comp, self.total_cycles_counted = 0.0, 0.0, 0

    def reconstruct(self, delta_mesh_phys, delta_tendon_phys):
        sig_xx = delta_mesh_phys[:, :, 0, :]
        sig_yy = delta_mesh_phys[:, :, 1, :]
        tau_xy = delta_mesh_phys[:, :, 2, :]
        
        center_s = 0.5 * (sig_xx + sig_yy)
        rad_s = torch.sqrt(0.25 * ((sig_xx - sig_yy) ** 2) + (tau_xy ** 2) + 1e-8)
        
        sig_1 = center_s + rad_s
        sig_3 = center_s - rad_s
        sig_vm = torch.sqrt(sig_xx ** 2 - sig_xx * sig_yy + sig_yy ** 2 + 3.0 * (tau_xy ** 2) + 1e-8)
        
        # Inverse Hooke's Law for Plane Stress -> microstrain
        eps_xx = (sig_xx - self.nu * sig_yy) / self.E_c * 1e6
        eps_yy = (sig_yy - self.nu * sig_xx) / self.E_c * 1e6
        gamma_xy = tau_xy / self.G_c * 1e6
        
        center_e = 0.5 * (eps_xx + eps_yy)
        rad_e = torch.sqrt(0.25 * ((eps_xx - eps_yy) ** 2) + 0.25 * (gamma_xy ** 2) + 1e-8)
        eps_1 = center_e + rad_e
        eps_3 = center_e - rad_e
        
        return {
            "sig_xx": sig_xx, "sig_yy": sig_yy, "tau_xy": tau_xy,
            "sig_1": sig_1, "sig_3": sig_3, "sig_vm": sig_vm,
            "eps_xx": eps_xx, "eps_yy": eps_yy, "gamma_xy": gamma_xy,
            "eps_1": eps_1, "eps_3": eps_3,
            "delta_sigma_p": delta_tendon_phys / self.A_p,
        }

    def evaluate_failure_and_fatigue(self, cur_s1_soffit, cur_s3_deck, cur_s1_web, cur_pt_force):
        total_s1_soffit = self.sigma_base_soffit + cur_s1_soffit
        total_s3_deck = self.sigma_base_deck + cur_s3_deck
        total_s1_web = cur_s1_web
        total_sigma_p = self.f_pe + (cur_pt_force / self.A_p)

        if total_s1_soffit < 0.0: mode_a_state, mode_a_level = "Precompressed", 0
        elif total_s1_soffit < self.f_ctm: mode_a_state, mode_a_level = "Decompressed", 1
        elif total_s1_soffit < self.f_ctk_95: mode_a_state, mode_a_level = "Cracking", 2
        else: mode_a_state, mode_a_level = "Rupture", 3

        abs_comp = abs(total_s3_deck)
        if abs_comp < self.f_c_linear: mode_b_state, mode_b_level = "Elastic", 0
        elif abs_comp < self.f_c_sls: mode_b_state, mode_b_level = "SLS Microcrack", 1
        elif abs_comp < self.f_cd: mode_b_state, mode_b_level = "Plastic Creep", 2
        else: mode_b_state, mode_b_level = "Crushing", 3

        if total_s1_web < self.tau_web_crack: mode_c_state, mode_c_level = "Uncracked", 0
        else: mode_c_state, mode_c_level = "Shear Crack", 1

        if total_sigma_p < self.f_p_high: mode_d_state, mode_d_level = "Nominal", 0
        elif total_sigma_p < self.f_py: mode_d_state, mode_d_level = "High Cyclic", 1
        elif total_sigma_p < self.f_pu: mode_d_state, mode_d_level = "Yielding", 2
        else: mode_d_state, mode_d_level = "Rupture", 3

        for d_sig, _, _, _ in self.rainflow_tendon.process_sample(total_sigma_p):
            if d_sig > 1.0:
                self.D_tendon += 1.0 / max(1.0, 1e6 * ((self.delta_sigma_Rsk / d_sig) ** self.m_tendon))
                self.total_cycles_counted += 1

        for d_sig, _, s_max, s_min in self.rainflow_conc_comp.process_sample(abs_comp):
            if d_sig > 0.5:
                S_max, S_min = min(0.99, s_max / self.f_ck), max(0.01, s_min / self.f_ck)
                denom = max(0.01, 1.0 - S_min)
                log_Nf = (14.0 / denom) * (1.0 - (S_max / denom))
                self.D_conc_comp += 1.0 / (10.0 ** min(12.0, max(1.0, log_Nf)))

        return {
            "total_s1_soffit": total_s1_soffit, "total_s3_deck": total_s3_deck,
            "total_s1_web": total_s1_web, "total_sigma_p": total_sigma_p,
            "mode_a_state": mode_a_state, "mode_a_level": mode_a_level,
            "mode_b_state": mode_b_state, "mode_b_level": mode_b_level,
            "mode_c_state": mode_c_state, "mode_c_level": mode_c_level,
            "mode_d_state": mode_d_state, "mode_d_level": mode_d_level,
            "D_tendon": self.D_tendon, "D_conc_comp": self.D_conc_comp,
            "total_cycles": self.total_cycles_counted,
        }

class RealTimeAccFileStreamer:
    def __init__(self, file_path="20260106_Acc_100Hz.txt"):
        self.file_path = file_path
        self.file_obj = None
        self.data_start_offset = 0
        self.sps = 100.0
        self.dt = 1.0 / self.sps
        self.sample_idx = 0
        self._find_data_start()

    def _find_data_start(self):
        if not os.path.exists(self.file_path):
            return
        self.file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        line = self.file_obj.readline()
        while line:
            if line.startswith("SPS"):
                try:
                    self.sps = float(line.split("=")[1].strip())
                    self.dt = 1.0 / self.sps
                except Exception:
                    pass
            elif "----------**********----------" in line:
                self.data_start_offset = self.file_obj.tell()
                break
            line = self.file_obj.readline()

    def read_next_sample(self):
        if self.file_obj is None:
            t = self.sample_idx * self.dt
            ay = 0.042 * math.sin(2.0 * math.pi * 2.78 * t) + np.random.normal(0, 0.003)
            ax = 0.006 * math.sin(2.0 * math.pi * 1.85 * t)
            az = 0.004 * math.sin(2.0 * math.pi * 3.40 * t)
            self.sample_idx += 1
            return ax, ay, az, t

        line = self.file_obj.readline()
        if not line:
            self.file_obj.seek(self.data_start_offset)
            self.sample_idx = 0
            line = self.file_obj.readline()
            if not line:
                return 0.0, 0.0, 0.0, 0.0

        tokens = line.replace(",", " ").replace("\t", " ").split()
        if len(tokens) >= 4:
            self.sample_idx += 1
            return float(tokens[2]), float(tokens[1]), float(tokens[3]), float(tokens[0])
        return 0.0, 0.0, 0.0, 0.0

def ensure_edge_package(package_path="mthl_edge_package.pt", model_path="mthl_stgnn_best_model.pt", dataset_path="stgnn_unified.npz"):
    if os.path.exists(package_path):
        return
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    archive = np.load(dataset_path, mmap_mode="r", allow_pickle=True)
    coords, adj = archive["coords"].astype(np.float32), archive["adj_matrix"].astype(np.float32)

    deg = np.sum(adj, axis=1)
    deg_inv = np.power(deg, -0.5, where=deg > 0)
    deg_inv[deg == 0] = 0.0
    A_norm = deg_inv[:, None] * adj * deg_inv[None, :]

    torch.save({
        "model_state_dict": payload["model_state_dict"], "model_config": payload["model_config"],
        "norm_stats": payload["norm_stats"], "span_mask": payload["span_mask"].float(),
        "coords": torch.from_numpy(coords), "A_norm": torch.from_numpy(A_norm),
    }, package_path)

class MTHLDigitalTwin:
    def __init__(self, package_path="mthl_edge_package.pt", telemetry_file="20260106_Acc_100Hz.txt", window_size=1024, stride=25):
        ensure_edge_package(package_path)
        self.window_size, self.stride, self.running, self.playback_speed = window_size, stride, False, 1.0
        self.stop_event = threading.Event()

        pkg = torch.load(package_path, map_location=device, weights_only=False)
        self.coords = pkg["coords"].cpu().numpy()
        self.A_norm = pkg["A_norm"].to(device)
        self.span_mask = pkg["span_mask"].to(device)
        self.norm_stats = {k: v.to(device) for k, v in pkg["norm_stats"].items()}
        self.n_tendon = int(pkg["model_config"].get("num_tendon_elements", 285))

        self.model = BridgeSTTransformer(**pkg["model_config"]).to(device)
        self.model.load_state_dict(pkg["model_state_dict"])
        self.model.eval()

        self.streamer = RealTimeAccFileStreamer(file_path=telemetry_file)
        self.dt = self.streamer.dt
        self.post_processor = ComprehensiveStructuralPostProcessor()
        self.sensor_buffer = deque([np.zeros(3, dtype=np.float32)] * self.window_size, maxlen=self.window_size)
        self.lock = threading.Lock()

        mask_bool = self.span_mask.squeeze().cpu().numpy().astype(bool)
        mid_cands = np.where(mask_bool & (np.abs(self.coords[:, 0] - 30.0) <= 3.5))[0]
        self.soffit_node = int(mid_cands[np.argmin(self.coords[mid_cands, 1])])
        self.deck_node = int(mid_cands[np.argmax(self.coords[mid_cands, 1])])
        web_cands = mid_cands[(self.coords[mid_cands, 1] > 1.2) & (self.coords[mid_cands, 1] < 2.5)]
        self.web_node = int(web_cands[0]) if len(web_cands) > 0 else self.soffit_node

        y_all = self.coords[:, 1]
        self.node_category = np.array(["Web"] * len(self.coords), dtype=object)
        self.node_category[y_all >= 3.20] = "Deck"
        self.node_category[y_all <= 0.40] = "Soffit"
        self.idx_by_cat = {c: np.where(self.node_category == c)[0] for c in ["Deck", "Soffit", "Web"]}

        self.fft_freqs = np.fft.rfftfreq(self.window_size, d=self.dt).astype(np.float32)
        self.fft_window_fn = np.hanning(self.window_size).astype(np.float32)

        self.live_state = {
            "timestamp": time.time(), "state_version": 0, "telemetry_time_s": 0.0, "latency_ms": 0.0,
            "hotspot_node": self.soffit_node, "crit_comp_node": self.deck_node, "crit_vm_node": self.soffit_node,
            "current_accel": np.zeros(3, dtype=np.float32), "accel_window": np.zeros((3, self.window_size), dtype=np.float32),
            "delta_uy_mm": np.zeros(self.window_size, dtype=np.float32),
            "total_s1_history": np.zeros(self.window_size, dtype=np.float32),
            "disp_trajectory": np.zeros((self.coords.shape[0], 3, self.window_size), dtype=np.float32),
            "sxx_trajectory": np.zeros((self.coords.shape[0], self.window_size), dtype=np.float32),
            "s1_trajectory": np.zeros((self.coords.shape[0], self.window_size), dtype=np.float32),
            "s3_trajectory": np.zeros((self.coords.shape[0], self.window_size), dtype=np.float32),
            "vm_trajectory": np.zeros((self.coords.shape[0], self.window_size), dtype=np.float32),
            "e1_trajectory": np.zeros((self.coords.shape[0], self.window_size), dtype=np.float32),
            "e3_trajectory": np.zeros((self.coords.shape[0], self.window_size), dtype=np.float32),
            "pt_trajectory": np.zeros((self.n_tendon, self.window_size), dtype=np.float32),
            "failure_eval": {}, "fft_freqs": self.fft_freqs, "fft_mag_db": np.zeros_like(self.fft_freqs),
            "tendon_force_kn": np.zeros(self.n_tendon, dtype=np.float32), "tendon_stress_mpa": np.zeros(self.n_tendon, dtype=np.float32),
        }

    def push_sample(self, ax, ay, az):
        with self.lock:
            self.sensor_buffer.append(np.array([ax, ay, az], dtype=np.float32))

    def step_inference(self):
        with self.lock:
            window = np.array(self.sensor_buffer, dtype=np.float32)

        t0 = time.perf_counter()
        x_norm = (torch.from_numpy(window).unsqueeze(0).to(device) - self.norm_stats["x_mean"]) / self.norm_stats["x_std"]
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

        with torch.no_grad(), amp_ctx:
            pred_m_norm, pred_t_norm = self.model(x_norm, self.A_norm)

        delta_mesh = pred_m_norm.float() * self.norm_stats["mesh_scale"]
        delta_tendon = pred_t_norm.float() * self.norm_stats["tendon_scale"]

        derived = self.post_processor.reconstruct(delta_mesh, delta_tendon)
        disp_traj = delta_mesh[0, :, 3:6, :].detach().cpu().numpy()
        
        sxx_traj = derived["sig_xx"][0].detach().cpu().numpy()
        s1_traj = derived["sig_1"][0].detach().cpu().numpy()
        s3_traj = derived["sig_3"][0].detach().cpu().numpy()
        vm_traj = derived["sig_vm"][0].detach().cpu().numpy()
        e1_traj = derived["eps_1"][0].detach().cpu().numpy()
        e3_traj = derived["eps_3"][0].detach().cpu().numpy()
        pt_traj = (delta_tendon[0, :, :].detach().cpu().numpy() / 1e3).astype(np.float32)

        fe = self.post_processor.evaluate_failure_and_fatigue(
            float(s1_traj[self.soffit_node, -1]), float(s3_traj[self.deck_node, -1]),
            float(s1_traj[self.web_node, -1]), float(delta_tendon[0, 18, -1].item())
        )

        spec = np.abs(np.fft.rfft(window[:, 1] * self.fft_window_fn)) / (self.window_size * 0.5)
        
        with self.lock:
            self.live_state.update({
                "timestamp": time.time(), "state_version": self.live_state["state_version"] + 1,
                "telemetry_time_s": self.streamer.sample_idx * self.dt, "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "current_accel": window[-1], "accel_window": window.T, "disp_trajectory": disp_traj,
                "sxx_trajectory": sxx_traj, "s1_trajectory": s1_traj, "s3_trajectory": s3_traj, "vm_trajectory": vm_traj,
                "e1_trajectory": e1_traj, "e3_trajectory": e3_traj, "pt_trajectory": pt_traj,
                "delta_uy_mm": delta_mesh[0, self.soffit_node, 4, :].cpu().numpy() * 1e3,
                "delta_sxx_mpa": delta_mesh[0, self.soffit_node, 0, :].cpu().numpy(),
                "total_s1_history": self.post_processor.sigma_base_soffit + s1_traj[self.soffit_node, :],
                "failure_eval": fe, "fft_mag_db": (20.0 * np.log10(spec + 1e-9)).astype(np.float32),
                "tendon_force_kn": pt_traj[:, -1],
                "tendon_stress_mpa": (delta_tendon[0, :, -1] / self.post_processor.A_p).cpu().numpy().astype(np.float32)
            })

    def _feeder_thread_func(self):
        next_t = time.perf_counter()
        while self.running:
            self.push_sample(*self.streamer.read_next_sample()[:3])
            next_t += self.dt / max(0.01, self.playback_speed)
            if self.stop_event.wait(max(0.0, next_t - time.perf_counter())):
                break

    def _worker_thread_func(self):
        while self.running:
            self.step_inference()
            if self.stop_event.wait(max(0.01, (self.stride * self.dt) / max(0.01, self.playback_speed))):
                break

    def start(self):
        self.running = True
        self.stop_event.clear()
        threading.Thread(target=self._feeder_thread_func, daemon=True).start()
        threading.Thread(target=self._worker_thread_func, daemon=True).start()

    def stop(self):
        self.running = False
        self.stop_event.set()