"""
Drude models for copper, silver, and gold (DC-anchored).
=========================================================

Computes the Drude free-electron parameters for Cu, Ag, Au from two measured
inputs per metal (mass density rho_m and DC resistivity rho) plus standard
constants. Nothing is fitted; the carrier density n is computed from rho_m
(not taken from a table).

    n            = Z * rho_m * N_A / M                     carrier density [m^-3]
    sigma_dc     = 1 / rho                                 DC conductivity [S/m]
    omega_p      = sqrt(n e^2 / (eps0 m_e))                plasma frequency [rad/s]
    tau          = sigma_dc / (eps0 omega_p^2)             relaxation time [s]
    gamma        = 1 / tau                                 damping rate [rad/s]
    sigma(omega) = sigma_dc / (1 - i omega tau)            AC conductivity [S/m]
    eps(omega)   = eps_inf - omega_p^2/(omega^2 + i gamma omega)

Inputs from the CRC Handbook (rho, rho_m, M). Equations: standard Drude
free-electron model (Ashcroft & Mermin Ch. 1; originally P. Drude, 1900).
"""

import math
import cmath

# Physical constants (CODATA 2018)
E_CHARGE = 1.602176634e-19
M_E      = 9.1093837015e-31
EPS0     = 8.8541878128e-12
HBAR     = 1.054571817e-34
MU0      = 1.25663706212e-6
N_A      = 6.02214076e23

# Measured / standard inputs (CRC Handbook)
RHO_DC = {"Cu": 1.678e-8, "Ag": 1.587e-8, "Au": 2.44e-8}     # ohm*m
RHO_M  = {"Cu": 8960.0,   "Ag": 10490.0,  "Au": 19300.0}     # kg/m^3
MOLAR  = {"Cu": 0.063546, "Ag": 0.1078682,"Au": 0.1969665}   # kg/mol
Z_VAL  = {"Cu": 1,        "Ag": 1,        "Au": 1}

# Carrier density computed from mass density (NOT taken from a table)
N_DENSITY     = {m: Z_VAL[m] * RHO_M[m] * N_A / MOLAR[m] for m in RHO_DC}
SIGMA_DC_MEAS = {m: 1.0 / RHO_DC[m] for m in RHO_DC}


def ev_from_rads(omega):
    return HBAR * omega / E_CHARGE


def _build_models():
    metals = {}
    for m in ("Cu", "Ag", "Au"):
        n = N_DENSITY[m]
        sigma_dc = SIGMA_DC_MEAS[m]
        wp = math.sqrt(n * E_CHARGE**2 / (EPS0 * M_E))
        tau = sigma_dc / (EPS0 * wp**2)
        gamma = 1.0 / tau
        metals[m] = {"n": n, "sigma_dc": sigma_dc, "omega_p": wp,
                     "tau": tau, "gamma": gamma, "eps_inf": 1.0}
    return metals


METALS = _build_models()


def drude_sigma(omega, metal):
    p = METALS[metal]
    return p["sigma_dc"] / (1.0 - 1j * omega * p["tau"])


def drude_epsilon(omega, metal):
    p = METALS[metal]
    return p["eps_inf"] - p["omega_p"]**2 / (omega**2 + 1j * p["gamma"] * omega)


def skin_depth(freq_hz, metal):
    omega = 2.0 * math.pi * freq_hz
    return math.sqrt(2.0 / (omega * MU0 * drude_sigma(omega, metal).real))


def surface_resistance(freq_hz, metal):
    """Surface resistance R_s = Re(sqrt(i w mu0 / sigma(omega))), full Drude sigma.
    Matches the website Table 4 (e.g. Cu 82.03 mohm at 100 GHz). Its low-frequency
    limit is sqrt(pi f mu0 / sigma_dc)."""
    omega = 2.0 * math.pi * freq_hz
    return cmath.sqrt(1j * omega * MU0 / drude_sigma(omega, metal)).real


def _report():
    print("=" * 74)
    print("DC-ANCHORED DRUDE PARAMETERS (n computed from mass density)")
    print("=" * 74)
    print(f"{'metal':5} {'n [m^-3]':>12} {'sigma_dc':>11} {'omega_p':>12} {'eV':>6} {'gamma':>11} {'tau[fs]':>8}")
    for m in ("Cu", "Ag", "Au"):
        p = METALS[m]
        print(f"{m:5} {p['n']:12.4e} {p['sigma_dc']:11.4e} {p['omega_p']:12.4e} "
              f"{ev_from_rads(p['omega_p']):6.2f} {p['gamma']:11.4e} {p['tau']*1e15:8.2f}")
    print("\nAUTOMATED CHECKS")
    ok = True
    for m in ("Cu", "Ag", "Au"):
        p = METALS[m]
        rel = abs(EPS0 * p["omega_p"]**2 * p["tau"] - p["sigma_dc"]) / p["sigma_dc"]
        cond = rel < 1e-9; ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {m}: eps0*wp^2*tau == 1/rho (rel err {rel:.1e})")
    d = skin_depth(100e9, "Cu") * 1e6
    cond = abs(d - 0.206) < 0.01; ok &= cond
    print(f"  [{'PASS' if cond else 'FAIL'}] Cu skin depth at 100 GHz = {d:.3f} um (~0.206)")
    for m, want in (("Cu", 82.03), ("Ag", 80.11), ("Au", 98.91)):
        rs = surface_resistance(100e9, m) * 1e3
        cond = abs(rs - want) < 0.1; ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {m} R_s at 100 GHz = {rs:.2f} mohm (~{want}, full sigma -> Table 4)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    _report()
