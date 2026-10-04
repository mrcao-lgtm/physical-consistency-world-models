"""
物理一致性度量的世界模型合成数据生成 v5
=====================================

修复R4审稿意见：
1. 加入与精确参考的L2距离度量（解决persistence悖论）
2. 加入谱分析（能谱、涡量谱）
3. 修复t-test类型（Welch's t-test）
4. 添加Cohen's d计算
5. 添加绘图脚本到代码目录
6. 修复requirements.txt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import time
from scipy import stats

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 输出目录（相对路径）
OUT = Path(__file__).parent / 'results_v5'
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 2D Navier-Stokes 流体仿真（涡量-流函数形式）
# ============================================================

class FluidSimulator:
    """2D Navier-Stokes 流体仿真器"""
    
    def __init__(self, nx=32, ny=32, dt=0.01, nu=0.01):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.nu = nu
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny
        self.psi = np.zeros((ny, nx))
        self.omega = np.zeros((ny, nx))
        self.cylinder_x, self.cylinder_y = nx // 4, ny // 2
        self.cylinder_r = nx // 10
        
    def set_initial_condition(self, u_inf=1.0):
        self.psi[:, :] = u_inf * np.arange(self.ny)[:, None] * self.dy
        for i in range(self.ny):
            for j in range(self.nx):
                if (i - self.cylinder_y)**2 + (j - self.cylinder_x)**2 < self.cylinder_r**2:
                    self.psi[i, j] = u_inf * self.cylinder_y * self.dy
        self.omega = self.compute_vorticity()
    
    def compute_velocity(self):
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))
        u[1:-1, :] = (self.psi[2:, :] - self.psi[:-2, :]) / (2 * self.dy)
        v[:, 1:-1] = -(self.psi[:, 2:] - self.psi[:, :-2]) / (2 * self.dx)
        return u, v
    
    def compute_vorticity(self):
        omega = np.zeros((self.ny, self.nx))
        omega[1:-1, 1:-1] = (
            (self.psi[2:, 1:-1] - 2*self.psi[1:-1, 1:-1] + self.psi[:-2, 1:-1]) / self.dy**2 +
            (self.psi[1:-1, 2:] - 2*self.psi[1:-1, 1:-1] + self.psi[1:-1, :-2]) / self.dx**2
        )
        return -omega
    
    def rhs(self, omega):
        psi = self.solve_poisson_fft(omega)
        u = np.zeros_like(omega)
        v = np.zeros_like(omega)
        u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * self.dy)
        v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * self.dx)
        
        adv = np.zeros_like(omega)
        adv[1:-1, 1:-1] = (
            u[1:-1, 1:-1] * (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2*self.dx) +
            v[1:-1, 1:-1] * (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2*self.dy)
        )
        
        diff = np.zeros_like(omega)
        diff[1:-1, 1:-1] = self.nu * (
            (omega[1:-1, 2:] - 2*omega[1:-1, 1:-1] + omega[1:-1, :-2]) / self.dx**2 +
            (omega[2:, 1:-1] - 2*omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / self.dy**2
        )
        
        return -adv + diff
    
    def step_rk4(self):
        k1 = self.rhs(self.omega)
        k2 = self.rhs(self.omega + 0.5 * self.dt * k1)
        k3 = self.rhs(self.omega + 0.5 * self.dt * k2)
        k4 = self.rhs(self.omega + self.dt * k3)
        self.omega += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.omega[0, :] = 0
        self.omega[-1, :] = 0
        self.omega[:, 0] = 0
        self.omega[:, -1] = 0
        self.psi = self.solve_poisson_fft(self.omega)
    
    def solve_poisson_fft(self, omega):
        ny, nx = omega.shape
        omega_hat = np.fft.fft2(omega)
        kx = 2 * np.pi * np.fft.fftfreq(nx, 1.0/nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, 1.0/ny)
        KX, KY = np.meshgrid(kx, ky)
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0
        psi_hat = np.zeros_like(omega_hat)
        psi_hat = omega_hat / k2
        psi_hat[0, 0] = 0
        psi = np.fft.ifft2(psi_hat).real
        return psi
    
    def get_state(self):
        return self.omega.flatten()
    
    def run(self, n_steps=100):
        states = []
        for _ in range(n_steps):
            self.step_rk4()
            states.append(self.get_state())
        return np.array(states, dtype=np.float32)


def generate_fluid_data(n_traj=100, n_steps=50, nx=32, ny=32, dt=0.01, nu=0.01, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n_traj):
        sim = FluidSimulator(nx=nx, ny=ny, dt=dt, nu=nu)
        u_inf = rng.uniform(0.5, 1.5)
        sim.set_initial_condition(u_inf=u_inf)
        states = sim.run(n_steps=n_steps)
        data.append(states)
        if (i + 1) % 10 == 0:
            print(f"  生成轨迹 {i+1}/{n_traj}")
    return np.array(data, dtype=np.float32)


def generate_persistence_baseline(n_traj=100, n_steps=50, nx=32, ny=32, dt=0.01, nu=0.01, seed=42):
    """Persistence baseline: ω_t = ω_0 (no dynamics)"""
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n_traj):
        sim = FluidSimulator(nx=nx, ny=ny, dt=dt, nu=nu)
        u_inf = rng.uniform(0.5, 1.5)
        sim.set_initial_condition(u_inf=u_inf)
        initial_state = sim.get_state()
        states = np.tile(initial_state, (n_steps, 1))
        data.append(states)
    return np.array(data, dtype=np.float32)


# ============================================================
# 2. 世界模型 (VAE + LSTM)
# ============================================================

class WorldModel(nn.Module):
    def __init__(self, state_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_decode = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.state_dim)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        z = z.reshape(batch, seq_len, self.latent_dim)
        
        h0 = torch.zeros(1, batch, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, batch, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(z, (h0, c0))
        z_pred = self.fc_decode(lstm_out)
        
        z_pred_flat = z_pred.reshape(-1, self.latent_dim)
        x_recon = self.decode(z_pred_flat)
        x_recon = x_recon.reshape(batch, seq_len, self.state_dim)
        
        return x_recon, mu, logvar
    
    def generate(self, n_samples, seq_len, device='cpu', seed=42):
        torch.manual_seed(seed)
        z = torch.randn(n_samples, seq_len, self.latent_dim).to(device)
        h0 = torch.zeros(1, n_samples, self.hidden_dim).to(device)
        c0 = torch.zeros(1, n_samples, self.hidden_dim).to(device)
        lstm_out, _ = self.lstm(z, (h0, c0))
        z_gen = self.fc_decode(lstm_out)
        z_gen_flat = z_gen.reshape(-1, self.latent_dim)
        x_gen = self.decode(z_gen_flat)
        x_gen = x_gen.reshape(n_samples, seq_len, self.state_dim)
        return x_gen


# ============================================================
# 3. 物理一致性度量（含L2距离和谱分析）
# ============================================================

def compute_physical_consistency(states, reference_states=None, nx=32, ny=32):
    """计算物理一致性度量（含L2距离和谱分析）"""
    n_traj, n_steps, _ = states.shape
    omega = states.reshape(n_traj, n_steps, ny, nx)
    
    # 计算流函数
    psi = np.zeros_like(omega)
    for t in range(n_traj):
        for step in range(n_steps):
            psi[t, step] = solve_poisson_fft(omega[t, step])
    
    # 计算速度场
    u = np.zeros_like(omega)
    v = np.zeros_like(omega)
    u[:, :, 1:-1, :] = (psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2 * (1.0/ny))
    v[:, :, :, 1:-1] = -(psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2 * (1.0/nx))
    
    # 动能
    kinetic_energy = 0.5 * np.sum(u**2 + v**2, axis=(2, 3)) * (1.0/nx) * (1.0/ny)
    
    # 涡量拟能
    enstrophy = 0.5 * np.sum(omega**2, axis=(2, 3)) * (1.0/nx) * (1.0/ny)
    
    # 质量守恒（速度散度）
    div = np.zeros_like(omega)
    div[:, :, 1:-1, 1:-1] = (
        (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / (2 * (1.0/nx)) +
        (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2 * (1.0/ny))
    )
    divergence = np.abs(div).mean(axis=(2, 3))
    
    # 能量漂移
    ke_drift = np.abs(kinetic_energy - kinetic_energy[:, 0:1]).mean(axis=1)
    en_drift = np.abs(enstrophy - enstrophy[:, 0:1]).mean(axis=1)
    
    # 动态性度量
    if n_steps > 1:
        traj_change = np.abs(np.diff(omega, axis=1)).mean(axis=(1, 2, 3))
    else:
        traj_change = np.zeros(n_traj)
    
    # 初始距离
    init_distance = np.abs(omega - omega[:, 0:1]).mean(axis=(1, 2, 3))
    
    # L2距离（与精确参考）
    if reference_states is not None:
        ref_omega = reference_states.reshape(n_traj, n_steps, ny, nx)
        l2_distance = np.sqrt(np.mean((omega - ref_omega)**2, axis=(1, 2, 3)))
    else:
        l2_distance = np.zeros(n_traj)
    
    # 谱分析：能谱
    ke_spectrum = compute_energy_spectrum(u, v, nx, ny)
    en_spectrum = compute_enstrophy_spectrum(omega, nx, ny)
    
    return {
        'kinetic_energy': kinetic_energy.mean(axis=1),
        'enstrophy': enstrophy.mean(axis=1),
        'divergence': divergence.mean(axis=1),
        'ke_drift': ke_drift,
        'en_drift': en_drift,
        'traj_change': traj_change,
        'init_distance': init_distance,
        'l2_distance': l2_distance,
        'ke_spectrum': ke_spectrum,
        'en_spectrum': en_spectrum,
    }


def compute_energy_spectrum(u, v, nx, ny):
    """计算能谱 E(k)"""
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    E_k = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
    
    # 径向平均
    kx = np.fft.fftfreq(nx, 1.0/nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, 1.0/ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)
    
    # 分bin
    k_bins = np.linspace(0, k_mag.max(), 10)
    E_spectrum = np.zeros((u.shape[0], u.shape[1], len(k_bins)-1))
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        E_spectrum[:, :, i] = E_k[:, :, mask].mean(axis=2)
    
    return E_spectrum


def compute_enstrophy_spectrum(omega, nx, ny):
    """计算涡量谱"""
    omega_hat = np.fft.fft2(omega)
    Omega_k = 0.5 * np.abs(omega_hat)**2
    
    kx = np.fft.fftfreq(nx, 1.0/nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, 1.0/ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)
    
    k_bins = np.linspace(0, k_mag.max(), 10)
    Omega_spectrum = np.zeros((omega.shape[0], omega.shape[1], len(k_bins)-1))
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        Omega_spectrum[:, :, i] = Omega_k[:, :, mask].mean(axis=2)
    
    return Omega_spectrum


def solve_poisson_fft(omega):
    ny, nx = omega.shape
    omega_hat = np.fft.fft2(omega)
    kx = 2 * np.pi * np.fft.fftfreq(nx, 1.0/nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, 1.0/ny)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0
    psi_hat = np.zeros_like(omega_hat)
    psi_hat = omega_hat / k2
    psi_hat[0, 0] = 0
    psi = np.fft.ifft2(psi_hat).real
    return psi


# ============================================================
# 4. 训练世界模型
# ============================================================

def train_world_model(model, data, epochs=50, batch_size=32, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_samples = data.shape[0]
    
    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch = torch.from_numpy(data[batch_idx]).float().to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            
            recon_loss = ((x_recon - batch)**2).mean()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
            
            loss = recon_loss + 0.001 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.6f}")
    
    return model


# ============================================================
# 5. 主实验
# ============================================================

def main():
    print("=" * 60)
    print("物理一致性度量的世界模型合成数据生成 v5")
    print("修复: L2距离度量 + 谱分析 + Welch's t-test + Cohen's d")
    print("=" * 60)
    
    nx, ny = 32, 32
    n_traj = 100
    n_steps = 50
    dt = 0.01
    nu = 0.01
    
    # 1. 生成RK4 ground truth数据 (dt=0.01)
    print(f"\n[1] 生成RK4 ground truth数据 (dt={dt})...")
    data_rk4 = generate_fluid_data(n_traj=n_traj, n_steps=n_steps, nx=nx, ny=ny, dt=dt, nu=nu, seed=SEED)
    print(f"  RK4数据形状: {data_rk4.shape}")
    np.save(OUT / 'data_rk4.npy', data_rk4)
    
    # 2. 生成更细时间步RK4 (dt=0.001) 作为精确参考
    print(f"\n[2] 生成更细时间步RK4 (dt=0.001)...")
    data_rk4_fine = generate_fluid_data(n_traj=n_traj, n_steps=n_steps, nx=nx, ny=ny, dt=0.001, nu=nu, seed=SEED)
    print(f"  细RK4数据形状: {data_rk4_fine.shape}")
    np.save(OUT / 'data_rk4_fine.npy', data_rk4_fine)
    
    # 3. 生成persistence baseline
    print(f"\n[3] 生成persistence baseline...")
    data_persistence = generate_persistence_baseline(n_traj=n_traj, n_steps=n_steps, nx=nx, ny=ny, dt=dt, nu=nu, seed=SEED)
    print(f"  Persistence数据形状: {data_persistence.shape}")
    np.save(OUT / 'data_persistence.npy', data_persistence)
    
    # 4. 训练世界模型
    print(f"\n[4] 训练世界模型...")
    state_dim = nx * ny
    model = WorldModel(state_dim=state_dim, latent_dim=64, hidden_dim=256)
    model = model.to(device)
    
    data_mean = data_rk4.mean(axis=(0, 1), keepdims=True)
    data_std = data_rk4.std(axis=(0, 1), keepdims=True) + 1e-8
    data_norm = (data_rk4 - data_mean) / data_std
    
    model = train_world_model(model, data_norm, epochs=30, batch_size=32)
    
    # 5. 生成合成数据
    print(f"\n[5] 生成合成数据...")
    n_gen = 100
    synthetic_data_norm = model.generate(n_gen, n_steps, device=device)
    synthetic_data_norm = synthetic_data_norm.detach().cpu().numpy()
    synthetic_data = synthetic_data_norm * data_std + data_mean
    print(f"  合成数据形状: {synthetic_data.shape}")
    np.save(OUT / 'synthetic_data.npy', synthetic_data)
    
    # 6. 计算物理一致性度量（含L2距离和谱分析）
    print(f"\n[6] 计算物理一致性度量...")
    
    rk4_metrics = compute_physical_consistency(data_rk4, reference_states=data_rk4_fine, nx=nx, ny=ny)
    rk4_fine_metrics = compute_physical_consistency(data_rk4_fine, nx=nx, ny=ny)
    persistence_metrics = compute_physical_consistency(data_persistence, reference_states=data_rk4_fine, nx=nx, ny=ny)
    syn_metrics = compute_physical_consistency(synthetic_data, reference_states=data_rk4_fine, nx=nx, ny=ny)
    
    def sem(x):
        return x.std() / np.sqrt(len(x))
    
    print(f"\n  RK4 (dt=0.01):")
    print(f"    动能漂移: {rk4_metrics['ke_drift'].mean():.6f} ± {sem(rk4_metrics['ke_drift']):.6f}")
    print(f"    涡量拟能漂移: {rk4_metrics['en_drift'].mean():.6f} ± {sem(rk4_metrics['en_drift']):.6f}")
    print(f"    速度散度: {rk4_metrics['divergence'].mean():.6e} ± {sem(rk4_metrics['divergence']):.6e}")
    print(f"    轨迹变化率: {rk4_metrics['traj_change'].mean():.6f} ± {sem(rk4_metrics['traj_change']):.6f}")
    print(f"    初始距离: {rk4_metrics['init_distance'].mean():.6f} ± {sem(rk4_metrics['init_distance']):.6f}")
    print(f"    L2距离: {rk4_metrics['l2_distance'].mean():.6f} ± {sem(rk4_metrics['l2_distance']):.6f}")
    
    print(f"\n  RK4 (dt=0.001):")
    print(f"    动能漂移: {rk4_fine_metrics['ke_drift'].mean():.6f} ± {sem(rk4_fine_metrics['ke_drift']):.6f}")
    print(f"    涡量拟能漂移: {rk4_fine_metrics['en_drift'].mean():.6f} ± {sem(rk4_fine_metrics['en_drift']):.6f}")
    print(f"    速度散度: {rk4_fine_metrics['divergence'].mean():.6e} ± {sem(rk4_fine_metrics['divergence']):.6e}")
    print(f"    轨迹变化率: {rk4_fine_metrics['traj_change'].mean():.6f} ± {sem(rk4_fine_metrics['traj_change']):.6f}")
    print(f"    初始距离: {rk4_fine_metrics['init_distance'].mean():.6f} ± {sem(rk4_fine_metrics['init_distance']):.6f}")
    
    print(f"\n  Persistence:")
    print(f"    动能漂移: {persistence_metrics['ke_drift'].mean():.6f} ± {sem(persistence_metrics['ke_drift']):.6f}")
    print(f"    涡量拟能漂移: {persistence_metrics['en_drift'].mean():.6f} ± {sem(persistence_metrics['en_drift']):.6f}")
    print(f"    速度散度: {persistence_metrics['divergence'].mean():.6e} ± {sem(persistence_metrics['divergence']):.6e}")
    print(f"    轨迹变化率: {persistence_metrics['traj_change'].mean():.6f} ± {sem(persistence_metrics['traj_change']):.6f}")
    print(f"    初始距离: {persistence_metrics['init_distance'].mean():.6f} ± {sem(persistence_metrics['init_distance']):.6f}")
    print(f"    L2距离: {persistence_metrics['l2_distance'].mean():.6f} ± {sem(persistence_metrics['l2_distance']):.6f}")
    
    print(f"\n  Synthetic:")
    print(f"    动能漂移: {syn_metrics['ke_drift'].mean():.6f} ± {sem(syn_metrics['ke_drift']):.6f}")
    print(f"    涡量拟能漂移: {syn_metrics['en_drift'].mean():.6f} ± {sem(syn_metrics['en_drift']):.6f}")
    print(f"    速度散度: {syn_metrics['divergence'].mean():.6e} ± {sem(syn_metrics['divergence']):.6e}")
    print(f"    轨迹变化率: {syn_metrics['traj_change'].mean():.6f} ± {sem(syn_metrics['traj_change']):.6f}")
    print(f"    初始距离: {syn_metrics['init_distance'].mean():.6f} ± {sem(syn_metrics['init_distance']):.6f}")
    print(f"    L2距离: {syn_metrics['l2_distance'].mean():.6f} ± {sem(syn_metrics['l2_distance']):.6f}")
    
    # 7. 统计检验（Welch's t-test + Cohen's d）
    print(f"\n[7] 统计检验...")
    
    # Welch's t-test
    t_ke, p_ke = stats.ttest_ind(syn_metrics['ke_drift'], rk4_fine_metrics['ke_drift'], equal_var=False)
    t_en, p_en = stats.ttest_ind(syn_metrics['en_drift'], rk4_fine_metrics['en_drift'], equal_var=False)
    t_l2, p_l2 = stats.ttest_ind(syn_metrics['l2_distance'], rk4_fine_metrics['l2_distance'], equal_var=False)
    
    # Cohen's d
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / dof)
        return (x.mean() - y.mean()) / pooled_std
    
    d_ke = cohens_d(syn_metrics['ke_drift'], rk4_fine_metrics['ke_drift'])
    d_en = cohens_d(syn_metrics['en_drift'], rk4_fine_metrics['en_drift'])
    d_l2 = cohens_d(syn_metrics['l2_distance'], rk4_fine_metrics['l2_distance'])
    
    print(f"  合成 vs RK4-fine 动能漂移: t={t_ke:.2f}, p={p_ke:.2e}, d={d_ke:.2f}")
    print(f"  合成 vs RK4-fine 涡量拟能漂移: t={t_en:.2f}, p={p_en:.2e}, d={d_en:.2f}")
    print(f"  合成 vs RK4-fine L2距离: t={t_l2:.2f}, p={p_l2:.2e}, d={d_l2:.2f}")
    
    # L2距离双样本检验（合成 vs RK4 dt=0.01）
    t_l2_two, p_l2_two = stats.ttest_ind(syn_metrics['l2_distance'], rk4_metrics['l2_distance'], equal_var=False)
    d_l2_two = cohens_d(syn_metrics['l2_distance'], rk4_metrics['l2_distance'])
    print(f"  合成 vs RK4-0.01 L2距离: t={t_l2_two:.2f}, p={p_l2_two:.2e}, d={d_l2_two:.2f}")
    
    # 正态性检验
    from scipy.stats import shapiro, normaltest, skew, kurtosis
    sw_ke = shapiro(syn_metrics['ke_drift'])
    sw_en = shapiro(syn_metrics['en_drift'])
    sw_l2 = shapiro(syn_metrics['l2_distance'])
    dp_ke = normaltest(syn_metrics['ke_drift'])
    dp_en = normaltest(syn_metrics['en_drift'])
    dp_l2 = normaltest(syn_metrics['l2_distance'])
    skew_ke = skew(syn_metrics['ke_drift'])
    skew_en = skew(syn_metrics['en_drift'])
    skew_l2 = skew(syn_metrics['l2_distance'])
    kurt_ke = kurtosis(syn_metrics['ke_drift'])
    kurt_en = kurtosis(syn_metrics['en_drift'])
    kurt_l2 = kurtosis(syn_metrics['l2_distance'])
    
    print(f"\n  正态性检验:")
    print(f"    D_KE: Shapiro W={sw_ke.statistic:.3f}, p={sw_ke.pvalue:.3f}; D'Agostino K2={dp_ke.statistic:.3f}, p={dp_ke.pvalue:.3f}")
    print(f"    D_EN: Shapiro W={sw_en.statistic:.3f}, p={sw_en.pvalue:.3f}; D'Agostino K2={dp_en.statistic:.3f}, p={dp_en.pvalue:.3f}")
    print(f"    D_L2: Shapiro W={sw_l2.statistic:.3f}, p={sw_l2.pvalue:.3f}; D'Agostino K2={dp_l2.statistic:.3f}, p={dp_l2.pvalue:.3f}")
    print(f"    偏度: KE={skew_ke:.3f}, EN={skew_en:.3f}, L2={skew_l2:.3f}")
    print(f"    峰度: KE={kurt_ke:.3f}, EN={kurt_en:.3f}, L2={kurt_l2:.3f}")
    
    # 非参数检验（Mann-Whitney U）
    from scipy.stats import mannwhitneyu
    u_ke, p_mwu_ke = mannwhitneyu(syn_metrics['ke_drift'], rk4_fine_metrics['ke_drift'], alternative='two-sided')
    u_en, p_mwu_en = mannwhitneyu(syn_metrics['en_drift'], rk4_fine_metrics['en_drift'], alternative='two-sided')
    u_l2, p_mwu_l2 = mannwhitneyu(syn_metrics['l2_distance'], rk4_metrics['l2_distance'], alternative='two-sided')
    print(f"\n  非参数检验 (Mann-Whitney U):")
    print(f"    D_KE: U={u_ke:.0f}, p={p_mwu_ke:.2e}")
    print(f"    D_EN: U={u_en:.0f}, p={p_mwu_en:.2e}")
    print(f"    D_L2: U={u_l2:.0f}, p={p_mwu_l2:.2e}")
    
    # Bootstrap CI
    rng = np.random.default_rng(SEED)
    def bootstrap_ci(data, n_boot=10000):
        boot_means = []
        for _ in range(n_boot):
            boot_sample = rng.choice(data, size=len(data), replace=True)
            boot_means.append(np.mean(boot_sample))
        return np.percentile(boot_means, [2.5, 97.5])
    
    ci_ke = bootstrap_ci(syn_metrics['ke_drift'])
    ci_en = bootstrap_ci(syn_metrics['en_drift'])
    ci_l2 = bootstrap_ci(syn_metrics['l2_distance'])
    
    print(f"\n  Bootstrap 95% CI:")
    print(f"    D_KE: [{ci_ke[0]:.6f}, {ci_ke[1]:.6f}]")
    print(f"    D_EN: [{ci_en[0]:.6f}, {ci_en[1]:.6f}]")
    print(f"    D_L2: [{ci_l2[0]:.6f}, {ci_l2[1]:.6f}]")
    
    # Jackknife
    def jackknife_sem(data):
        n = len(data)
        jack_means = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            jack_means.append(np.mean(jack_sample))
        return np.std(jack_means, ddof=1)
    
    jk_ke = jackknife_sem(syn_metrics['ke_drift'])
    jk_en = jackknife_sem(syn_metrics['en_drift'])
    jk_l2 = jackknife_sem(syn_metrics['l2_distance'])
    
    print(f"\n  Jackknife SEM:")
    print(f"    D_KE: {jk_ke:.6f}")
    print(f"    D_EN: {jk_en:.6f}")
    print(f"    D_L2: {jk_l2:.6f}")
    
    # 8. 保存结果
    results = {
        'rk4': {
            'ke_drift_mean': float(rk4_metrics['ke_drift'].mean()),
            'ke_drift_sem': float(sem(rk4_metrics['ke_drift'])),
            'en_drift_mean': float(rk4_metrics['en_drift'].mean()),
            'en_drift_sem': float(sem(rk4_metrics['en_drift'])),
            'divergence_mean': float(rk4_metrics['divergence'].mean()),
            'divergence_sem': float(sem(rk4_metrics['divergence'])),
            'traj_change_mean': float(rk4_metrics['traj_change'].mean()),
            'traj_change_sem': float(sem(rk4_metrics['traj_change'])),
            'init_distance_mean': float(rk4_metrics['init_distance'].mean()),
            'init_distance_sem': float(sem(rk4_metrics['init_distance'])),
            'l2_distance_mean': float(rk4_metrics['l2_distance'].mean()),
            'l2_distance_sem': float(sem(rk4_metrics['l2_distance'])),
        },
        'rk4_fine': {
            'ke_drift_mean': float(rk4_fine_metrics['ke_drift'].mean()),
            'ke_drift_sem': float(sem(rk4_fine_metrics['ke_drift'])),
            'en_drift_mean': float(rk4_fine_metrics['en_drift'].mean()),
            'en_drift_sem': float(sem(rk4_fine_metrics['en_drift'])),
            'divergence_mean': float(rk4_fine_metrics['divergence'].mean()),
            'divergence_sem': float(sem(rk4_fine_metrics['divergence'])),
            'traj_change_mean': float(rk4_fine_metrics['traj_change'].mean()),
            'traj_change_sem': float(sem(rk4_fine_metrics['traj_change'])),
            'init_distance_mean': float(rk4_fine_metrics['init_distance'].mean()),
            'init_distance_sem': float(sem(rk4_fine_metrics['init_distance'])),
        },
        'persistence': {
            'ke_drift_mean': float(persistence_metrics['ke_drift'].mean()),
            'ke_drift_sem': float(sem(persistence_metrics['ke_drift'])),
            'en_drift_mean': float(persistence_metrics['en_drift'].mean()),
            'en_drift_sem': float(sem(persistence_metrics['en_drift'])),
            'divergence_mean': float(persistence_metrics['divergence'].mean()),
            'divergence_sem': float(sem(persistence_metrics['divergence'])),
            'traj_change_mean': float(persistence_metrics['traj_change'].mean()),
            'traj_change_sem': float(sem(persistence_metrics['traj_change'])),
            'init_distance_mean': float(persistence_metrics['init_distance'].mean()),
            'init_distance_sem': float(sem(persistence_metrics['init_distance'])),
            'l2_distance_mean': float(persistence_metrics['l2_distance'].mean()),
            'l2_distance_sem': float(sem(persistence_metrics['l2_distance'])),
        },
        'synthetic': {
            'ke_drift_mean': float(syn_metrics['ke_drift'].mean()),
            'ke_drift_sem': float(sem(syn_metrics['ke_drift'])),
            'en_drift_mean': float(syn_metrics['en_drift'].mean()),
            'en_drift_sem': float(sem(syn_metrics['en_drift'])),
            'divergence_mean': float(syn_metrics['divergence'].mean()),
            'divergence_sem': float(sem(syn_metrics['divergence'])),
            'traj_change_mean': float(syn_metrics['traj_change'].mean()),
            'traj_change_sem': float(sem(syn_metrics['traj_change'])),
            'init_distance_mean': float(syn_metrics['init_distance'].mean()),
            'init_distance_sem': float(sem(syn_metrics['init_distance'])),
            'l2_distance_mean': float(syn_metrics['l2_distance'].mean()),
            'l2_distance_sem': float(sem(syn_metrics['l2_distance'])),
        },
        'statistics': {
            'welch_ttest_ke': {'t': float(t_ke), 'p': float(p_ke), 'cohens_d': float(d_ke)},
            'welch_ttest_en': {'t': float(t_en), 'p': float(p_en), 'cohens_d': float(d_en)},
            'welch_ttest_l2': {'t': float(t_l2), 'p': float(p_l2), 'cohens_d': float(d_l2)},
            'welch_ttest_l2_two_sample': {'t': float(t_l2_two), 'p': float(p_l2_two), 'cohens_d': float(d_l2_two)},
            'shapiro_wilk': {
                'ke': {'W': float(sw_ke.statistic), 'p': float(sw_ke.pvalue)},
                'en': {'W': float(sw_en.statistic), 'p': float(sw_en.pvalue)},
                'l2': {'W': float(sw_l2.statistic), 'p': float(sw_l2.pvalue)},
            },
            'dagostino_pearson': {
                'ke': {'K2': float(dp_ke.statistic), 'p': float(dp_ke.pvalue)},
                'en': {'K2': float(dp_en.statistic), 'p': float(dp_en.pvalue)},
                'l2': {'K2': float(dp_l2.statistic), 'p': float(dp_l2.pvalue)},
            },
            'skewness': {'ke': float(skew_ke), 'en': float(skew_en), 'l2': float(skew_l2)},
            'excess_kurtosis': {'ke': float(kurt_ke), 'en': float(kurt_en), 'l2': float(kurt_l2)},
            'mann_whitney_u': {
                'ke': {'U': float(u_ke), 'p': float(p_mwu_ke)},
                'en': {'U': float(u_en), 'p': float(p_mwu_en)},
                'l2': {'U': float(u_l2), 'p': float(p_mwu_l2)},
            },
            'bootstrap_ci_95': {
                'ke': [float(ci_ke[0]), float(ci_ke[1])],
                'en': [float(ci_en[0]), float(ci_en[1])],
                'l2': [float(ci_l2[0]), float(ci_l2[1])],
            },
            'jackknife_sem': {'ke': float(jk_ke), 'en': float(jk_en), 'l2': float(jk_l2)},
        },
    }
    
    with open(OUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"结果保存到: {OUT / 'results.json'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
