"""
绘图脚本：生成论文所有图表
==========================
运行：python3.12 generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUT = Path(__file__).parent / 'results_v5'

# 加载数据
data_rk4 = np.load(OUT / 'data_rk4.npy')
data_rk4_fine = np.load(OUT / 'data_rk4_fine.npy')
data_persistence = np.load(OUT / 'data_persistence.npy')
syn_data = np.load(OUT / 'synthetic_data.npy')

nx, ny = 32, 32

# 图1: 涡量场可视化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
step = 25
omega_rk4 = data_rk4[0, step].reshape(ny, nx)
omega_rk4_fine = data_rk4_fine[0, step].reshape(ny, nx)
omega_persistence = data_persistence[0, step].reshape(ny, nx)
omega_syn = syn_data[0, step].reshape(ny, nx)
vmax = max(abs(omega_rk4).max(), abs(omega_rk4_fine).max(), abs(omega_syn).max())

axes[0].imshow(omega_rk4, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
axes[0].set_title('RK4 (dt=0.01)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

axes[1].imshow(omega_rk4_fine, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
axes[1].set_title('RK4 (dt=0.001)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

axes[2].imshow(omega_persistence, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
axes[2].set_title('Persistence')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')

axes[3].imshow(omega_syn, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
axes[3].set_title('Synthetic (World Model)')
axes[3].set_xlabel('x')
axes[3].set_ylabel('y')

plt.tight_layout()
plt.savefig(OUT / 'fig1_vorticity.png', dpi=150, bbox_inches='tight')
plt.close()
print('图1保存: fig1_vorticity.png')

# 图2: 能量时间序列
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

def compute_energy_timeseries(states, nx=32, ny=32):
    n_traj, n_steps, _ = states.shape
    omega = states.reshape(n_traj, n_steps, ny, nx)
    psi = np.zeros_like(omega)
    for t in range(n_traj):
        for step in range(n_steps):
            psi[t, step] = solve_poisson_fft(omega[t, step])
    u = np.zeros_like(omega)
    v = np.zeros_like(omega)
    u[:, :, 1:-1, :] = (psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2 * (1.0/ny))
    v[:, :, :, 1:-1] = -(psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2 * (1.0/nx))
    ke = 0.5 * np.sum(u**2 + v**2, axis=(2, 3)) * (1.0/nx) * (1.0/ny)
    en = 0.5 * np.sum(omega**2, axis=(2, 3)) * (1.0/nx) * (1.0/ny)
    return ke, en

ke_rk4, en_rk4 = compute_energy_timeseries(data_rk4)
ke_rk4_fine, en_rk4_fine = compute_energy_timeseries(data_rk4_fine)
ke_persistence, en_persistence = compute_energy_timeseries(data_persistence)
ke_syn, en_syn = compute_energy_timeseries(syn_data)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(ke_rk4.mean(axis=0), label='RK4 (dt=0.01)', linewidth=2)
axes[0].plot(ke_rk4_fine.mean(axis=0), label='RK4 (dt=0.001)', linewidth=2)
axes[0].plot(ke_persistence.mean(axis=0), label='Persistence', linewidth=2, linestyle='--')
axes[0].plot(ke_syn.mean(axis=0), label='Synthetic', linewidth=2)
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Kinetic Energy')
axes[0].set_title('Kinetic Energy Time Series')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(en_rk4.mean(axis=0), label='RK4 (dt=0.01)', linewidth=2)
axes[1].plot(en_rk4_fine.mean(axis=0), label='RK4 (dt=0.001)', linewidth=2)
axes[1].plot(en_persistence.mean(axis=0), label='Persistence', linewidth=2, linestyle='--')
axes[1].plot(en_syn.mean(axis=0), label='Synthetic', linewidth=2)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Enstrophy')
axes[1].set_title('Enstrophy Time Series')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig2_energy_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print('图2保存: fig2_energy_timeseries.png')

# 图3: 物理一致性度量对比
with open(OUT / 'results.json') as f:
    results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = ['ke_drift', 'en_drift', 'divergence']
labels = ['Kinetic Energy Drift', 'Enstrophy Drift', 'Velocity Divergence']
colors = ['#2196F3', '#FF9800', '#4CAF50']

for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
    rk4_val = results['rk4'][f'{metric}_mean']
    rk4_err = results['rk4'][f'{metric}_sem']
    rk4_fine_val = results['rk4_fine'][f'{metric}_mean']
    rk4_fine_err = results['rk4_fine'][f'{metric}_sem']
    persistence_val = results['persistence'][f'{metric}_mean']
    persistence_err = results['persistence'][f'{metric}_sem']
    syn_val = results['synthetic'][f'{metric}_mean']
    syn_err = results['synthetic'][f'{metric}_sem']
    
    x = [0, 1, 2, 3]
    vals = [rk4_val, rk4_fine_val, persistence_val, syn_val]
    errs = [rk4_err, rk4_fine_err, persistence_err, syn_err]
    
    axes[i].bar(x, vals, yerr=errs, capsize=5, color=color, alpha=0.7)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(['RK4\n(0.01)', 'RK4\n(0.001)', 'Persist', 'Synth'])
    axes[i].set_title(label)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT / 'fig3_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('图3保存: fig3_metrics_comparison.png')

# 图4: 动态性度量 + L2距离
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = ['traj_change', 'init_distance', 'l2_distance']
labels = ['Trajectory Change Rate', 'Initial Distance', 'L2 Distance']

for i, (metric, label) in enumerate(zip(metrics, labels)):
    rk4_val = results['rk4'][f'{metric}_mean']
    rk4_err = results['rk4'][f'{metric}_sem']
    if metric == 'l2_distance':
        rk4_fine_val = 0
        rk4_fine_err = 0
    else:
        rk4_fine_val = results['rk4_fine'][f'{metric}_mean']
        rk4_fine_err = results['rk4_fine'][f'{metric}_sem']
    persistence_val = results['persistence'][f'{metric}_mean']
    persistence_err = results['persistence'][f'{metric}_sem']
    syn_val = results['synthetic'][f'{metric}_mean']
    syn_err = results['synthetic'][f'{metric}_sem']
    
    x = [0, 1, 2, 3]
    vals = [rk4_val, rk4_fine_val, persistence_val, syn_val]
    errs = [rk4_err, rk4_fine_err, persistence_err, syn_err]
    
    axes[i].bar(x, vals, yerr=errs, capsize=5, color='#9C27B0', alpha=0.7)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(['RK4\n(0.01)', 'RK4\n(0.001)', 'Persist', 'Synth'])
    axes[i].set_title(label)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT / 'fig4_dynamical_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print('图4保存: fig4_dynamical_metrics.png')
print('所有图表生成完成！')
