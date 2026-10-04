#!/usr/bin/env python3.12
"""
Spectral filter analysis for the physical-consistency study.

Measures the wavenumber-resolved transfer function between RK4 training data
and VAE-LSTM synthetic data, fits the two-component model

    T^2(k) = f + A * exp(-(k/kc)^2),

and performs the falsifiable test: applying the measured filter to RK4 and
comparing the resulting conservation metrics against the synthetic values.

Outputs results_v5/spectral_analysis.json
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

import experiment_v5 as E

NX = NY = 32
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v5")


def load(name):
    return np.load(os.path.join(RES, name))


def vorticity_power_spectrum(omega):
    """Shell-averaged vorticity power spectrum.

    omega: (n_traj, n_steps, ny, nx). Returns array indexed by integer |k|.
    """
    oh = np.fft.fft2(omega, axes=(2, 3))
    power = np.abs(oh) ** 2 / (NX * NY) ** 2
    kx = np.fft.fftfreq(NX, d=1.0 / NX)
    ky = np.fft.fftfreq(NY, d=1.0 / NY)
    KX, KY = np.meshgrid(kx, ky)
    kbin = np.round(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    spec = np.zeros(NX // 2 + 1)
    for k in range(NX // 2 + 1):
        m = (kbin == k)
        if m.sum() > 0:
            spec[k] = power[:, :, m].sum(axis=2).mean()
    return spec


def apply_filter(data, f, A, kc):
    """Apply the measured transfer function to a (n_traj, n_steps, N) array."""
    n_traj, n_steps, _ = data.shape
    om = data.reshape(n_traj, n_steps, NY, NX)
    # Match the 2*pi convention used by experiment_v5.solve_poisson_fft
    kx = 2 * np.pi * np.fft.fftfreq(NX, 1.0 / NX)
    ky = 2 * np.pi * np.fft.fftfreq(NY, 1.0 / NY)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    R = f + A * np.exp(-(K / kc) ** 2)
    om_f = np.real(np.fft.ifft2(np.fft.fft2(om, axes=(2, 3)) * R, axes=(2, 3)))
    return om_f.reshape(n_traj, n_steps, -1)


def model(k, f, A, kc):
    return f + A * np.exp(-(k / kc) ** 2)


def metrics(states, ref=None):
    r = E.compute_physical_consistency(states, ref)
    return {
        "ke_drift": float(r["ke_drift"].mean()),
        "en_drift": float(r["en_drift"].mean()),
        "l2_distance": float(r["l2_distance"].mean()) if ref is not None else None,
    }


def phase_randomize(data, seed):
    """Prescribe the amplitude spectrum of ``data`` with random Hermitian phases.

    Hermitian symmetry is automatic because the phases are taken from the FFT of
    a real random field. The amplitude spectrum is preserved exactly (up to
    floating-point round-off).
    """
    n_traj, n_steps, _ = data.shape
    om = data.reshape(n_traj, n_steps, NY, NX)
    amp = np.abs(np.fft.fft2(om, axes=(2, 3)))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_traj, n_steps, NY, NX))
    phase = np.angle(np.fft.fft2(noise, axes=(2, 3)))
    out = np.fft.ifft2(amp * np.exp(1j * phase), axes=(2, 3))
    return out.real.reshape(n_traj, n_steps, -1)


def phase_invariance_test(rk4, rk4f, n_seeds=8):
    """Verify Proposition: D_EN is invariant under phase randomization."""
    base = metrics(rk4)
    ke_perm, en_perm, l2_perm = [], [], []
    for sd in range(n_seeds):
        p = metrics(phase_randomize(rk4, sd))
        ke_perm.append(p["ke_drift"])
        en_perm.append(p["en_drift"])
        q = metrics(phase_randomize(rk4, sd), phase_randomize(rk4f, sd))
        l2_perm.append(q["l2_distance"])
    ke_perm = np.array(ke_perm)
    en_perm = np.array(en_perm)
    l2_perm = np.array(l2_perm)

    # amplitude preservation check
    om = rk4.reshape(-1, 50, NY, NX)
    pr = phase_randomize(rk4, 0).reshape(-1, 50, NY, NX)
    dev = np.max(np.abs(np.abs(np.fft.fft2(pr, axes=(2, 3)))
                        - np.abs(np.fft.fft2(om, axes=(2, 3)))))

    return {
        "amplitude_max_deviation": float(dev),
        "ke_unmodified": base["ke_drift"],
        "ke_permuted_mean": float(ke_perm.mean()),
        "ke_permuted_std": float(ke_perm.std()),
        "ke_relative_deviation": float(abs(ke_perm.mean() - base["ke_drift"]) / base["ke_drift"]),
        "en_unmodified": base["en_drift"],
        "en_permuted_mean": float(en_perm.mean()),
        "en_permuted_std": float(en_perm.std()),
        "en_relative_deviation": float(abs(en_perm.mean() - base["en_drift"]) / base["en_drift"]),
        "l2_between_independent_realizations": float(l2_perm.mean()),
        "l2_realization_std": float(l2_perm.std()),
    }


def main():
    syn = load("synthetic_data.npy")
    rk4 = load("data_rk4.npy")
    rk4f = load("data_rk4_fine.npy")

    om_syn = syn.reshape(-1, 50, NY, NX)
    om_rk4 = rk4.reshape(-1, 50, NY, NX)

    S_syn = vorticity_power_spectrum(om_syn)
    S_rk4 = vorticity_power_spectrum(om_rk4)

    k = np.arange(1, 17)
    T2 = S_syn[1:17] / S_rk4[1:17]

    popt, pcov = curve_fit(model, k, T2, p0=[0.04, 0.85, 4.0], maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    pred = model(k, *popt)
    r2 = 1 - np.sum((T2 - pred) ** 2) / np.sum((T2 - T2.mean()) ** 2)

    f, A, kc = (float(popt[0]), float(popt[1]), float(popt[2]))

    # Falsifiable test
    base_rk4 = metrics(rk4)
    base_syn = metrics(syn, rk4f)
    filt = metrics(apply_filter(rk4, f, A, kc), apply_filter(rk4f, f, A, kc))

    # Phase-invariance test (Proposition 1)
    phase = phase_invariance_test(rk4, rk4f)

    out = {
        "transfer_function": {
            "k": k.tolist(),
            "T2_measured": T2.tolist(),
            "T2_fit": pred.tolist(),
            "params": {"f": f, "A": A, "kc": kc},
            "errors": {"f": float(perr[0]), "A": float(perr[1]), "kc": float(perr[2])},
            "r_squared": float(r2),
            "cutoff_grid_points": float(NX / kc),
        },
        "falsifiable_test": {
            "rk4_unfiltered": base_rk4,
            "rk4_filtered": filt,
            "synthetic": base_syn,
            "ke_reduction_filtered": base_rk4["ke_drift"] / filt["ke_drift"],
            "ke_reduction_synthetic": base_rk4["ke_drift"] / base_syn["ke_drift"],
            "en_reduction_filtered": base_rk4["en_drift"] / filt["en_drift"],
            "en_reduction_synthetic": base_rk4["en_drift"] / base_syn["en_drift"],
            "l2_filter_prediction_error_pct": abs(filt["l2_distance"] - base_syn["l2_distance"])
            / base_syn["l2_distance"] * 100,
        },
        "phase_invariance_test": phase,
    }

    path = os.path.join(RES, "spectral_analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"Transfer function: f={f:.4f}, A={A:.4f}, kc={kc:.4f}, R^2={r2:.6f}")
    print(f"  cutoff scale: {NX/kc:.2f} grid points")
    print()
    print("Falsifiable test (apply measured filter to RK4):")
    print(f"  D_KE: RK4={base_rk4['ke_drift']:.3e} -> filtered={filt['ke_drift']:.3e} "
          f"({base_rk4['ke_drift']/filt['ke_drift']:.0f}x); synthetic={base_syn['ke_drift']:.3e}")
    print(f"  D_EN: RK4={base_rk4['en_drift']:.3e} -> filtered={filt['en_drift']:.3e} "
          f"({base_rk4['en_drift']/filt['en_drift']:.0f}x); synthetic={base_syn['en_drift']:.3e}")
    print(f"  D_L2: filtered={filt['l2_distance']:.4f} vs synthetic={base_syn['l2_distance']:.4f} "
          f"(error {out['falsifiable_test']['l2_filter_prediction_error_pct']:.1f}%)")
    print()
    print("Phase-invariance test (Proposition 1):")
    print(f"  amplitude spectrum preserved to {phase['amplitude_max_deviation']:.2e}")
    print(f"  D_EN relative deviation = {phase['en_relative_deviation']:.2e}  (invariant)")
    print(f"  D_KE relative deviation = {phase['ke_relative_deviation']:.2e}")
    print(f"  D_L2 between independent realizations = {phase['l2_between_independent_realizations']:.4f}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
