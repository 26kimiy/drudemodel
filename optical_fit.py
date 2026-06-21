"""
Optical-fit Drude parameters for Cu, Ag, Au.
========================================================================
We fit the Drude dielectric function

    eps(w) = eps_inf - wp^2 / (w^2 + i*gamma*w)

to measured optical constants (n, k), restricted to the free-electron
(below-interband) window, and compare the result to:
  - the DC-anchored free-electron values (drude-dc page), and
  - Ordal et al.'s published Drude fit.

Data: complex refractive index (n, k) vs wavelength, from refractiveindex.info
  Cu, Au : Ordal et al. 1987 (far-IR), n,k.
  Ag     : Johnson & Christy 1972 (visible/near-IR), n,k  [silver has no
           Ordal far-IR set in the database, so this is an independent source].

eps from (n,k):  eps = (n + i k)^2,  i.e. eps1 = n^2 - k^2, eps2 = 2 n k.
omega from wavelength:  w = 2*pi*c / lambda.
"""

import math
import numpy as np
from scipy.optimize import least_squares

# ---- constants (CODATA 2018) ----
C    = 2.99792458e8
HBAR = 1.054571817e-34
EV   = 1.602176634e-19
EPS0 = 8.8541878128e-12

CM1_TO_RADS = 2.0 * math.pi * C * 100.0   # 1 cm^-1 -> rad/s

# Ordal et al. 1985 published Drude fit, in wavenumbers [cm^-1]
ORDAL_CM1 = {"Cu": {"wp": 59600.0, "g": 73.2},
             "Ag": {"wp": 72700.0, "g": 145.2},
             "Au": {"wp": 72800.0, "g": 215.0}}

# DC-anchored free-electron values (rad/s) from the drude-dc page
DC = {"Cu": {"wp": 1.6439e16, "g": 4.015e13},
      "Ag": {"wp": 1.3652e16, "g": 2.619e13},
      "Au": {"wp": 1.3704e16, "g": 4.057e13}}
SIGMA_DC_MEAS = {"Cu": 5.9595e7, "Ag": 6.3012e7, "Au": 4.0984e7}

SOURCE = {"Cu": "Ordal et al. 1987", "Au": "Ordal et al. 1987",
          "Ag": "Johnson and Christy 1972"}

# below-interband fit window: keep photon energy <= EMAX_eV
# interband onsets ~ Cu 2.1 eV, Au 2.4 eV, Ag 3.9 eV
EMAX_eV = {"Cu": 1.0, "Au": 1.0, "Ag": 2.0}

# ---- measured n,k data:  "wl_um,n,k;wl_um,n,k;..." ----
DATA = {
"Au": "100,225,319;125,263,353;154,302,390;200,356,444;286,447,534;0.667,0.219,3.91;0.714,0.177,4.38;0.769,0.179,4.88;0.833,0.188,5.44;0.909,0.205,6.06;1,0.229,6.79;1.05,0.243,7.2;1.11,0.258,7.65;1.18,0.276,8.15;1.25,0.296,8.71;1.33,0.319,9.35;1.43,0.348,10.1;1.54,0.384,10.9;1.67,0.43,11.9;1.82,0.492,13;2,0.581,14.3;2.11,0.64,15.1;2.22,0.71,16;2.35,0.793,17;2.5,0.89,18;2.67,1,19.2;2.86,1.14,20.6;3.08,1.31,22.2;3.33,1.52,24.1;3.64,1.81,26.3;4,2.2,28.9;4.44,2.74,32.1;5,3.5,36;5.71,4.54,41;6.67,6.03,47.4;8,8.29,56.2;10,12.1,69.2;11.1,14.7,76.3;12.5,18.2,84.7;14.3,23,94.9;16.7,29.8,108;20,39.9,124;22.2,46.8,133;25,54.8,144;28.6,64.3,158;33.3,76.4,175;40,93.5,197;44.4,105,210;50,119,225;57.1,137,243;66.7,160,263;80,188,288",
"Cu": "0.517,1.16,2.64;0.67,0.399,3.97;1,0.538,6.53;1.11,0.57,7.32;1.25,0.62,8.31;1.43,0.699,9.53;1.67,0.816,11.2;2,0.879,13.4;2.11,0.942,14.2;2.22,1,14.9;2.35,1.09,15.8;2.5,1.18,16.8;2.67,1.31,17.9;2.86,1.46,19.1;3.08,1.62,20.6;3.33,1.77,22.3;3.64,1.98,24.2;4,2.25,26.5;4.44,2.58,28.8;5,3.26,33;5.26,3.45,34.5;5.56,3.65,36.3;5.88,3.84,38.3;6.25,4.16,40.7;6.67,4.57,43.3;7.14,5.1,46.2;7.69,5.74,49.5;8.33,6.43,53.2;9.09,7.21,57.7;10,8.31,63;10.5,9,65.9;11.1,9.77,69.1;11.8,10.5,72.6;12.5,11.4,76.5;13.3,12.2,80.8;14.3,13.2,85.6;15.4,13.8,91.1;16.7,14.7,97.7;18.2,15.4,105;20,16.2,115;20.8,16.8,120;21.7,17,124;22.7,17.7,130;23.8,18.4,136;25,19.1,142;26.3,19.5,150;27.8,21,158;29.4,22.1,167;31.3,23.4,178;33.3,25.7,190;35.7,28.5,203;38.5,31.9,219;41.7,36.2,238;45.5,40.9,258;50,50,284;55.6,61.2,313",
"Ag": "0.1879,1.07,1.212;0.1916,1.1,1.232;0.1953,1.12,1.255;0.1993,1.14,1.277;0.2033,1.15,1.296;0.2073,1.18,1.312;0.2119,1.2,1.325;0.2164,1.22,1.336;0.2214,1.25,1.342;0.2262,1.26,1.344;0.2313,1.28,1.357;0.2371,1.28,1.367;0.2426,1.3,1.378;0.249,1.31,1.389;0.2551,1.33,1.393;0.2616,1.35,1.387;0.2689,1.38,1.372;0.2761,1.41,1.331;0.2844,1.41,1.264;0.2924,1.39,1.161;0.3009,1.34,0.964;0.3107,1.13,0.616;0.3204,0.81,0.392;0.3315,0.17,0.829;0.3425,0.14,1.142;0.3542,0.1,1.419;0.3679,0.07,1.657;0.3815,0.05,1.864;0.3974,0.05,2.07;0.4133,0.05,2.275;0.4305,0.04,2.462;0.4509,0.04,2.657;0.4714,0.05,2.869;0.4959,0.05,3.093;0.5209,0.05,3.324;0.5486,0.06,3.586;0.5821,0.05,3.858;0.6168,0.06,4.152;0.6595,0.05,4.483;0.7045,0.04,4.838;0.756,0.03,5.242;0.8211,0.04,5.727;0.892,0.04,6.312;0.984,0.04,6.992;1.088,0.04,7.795;1.216,0.09,8.828;1.393,0.13,10.1;1.61,0.15,11.85;1.937,0.24,14.08",
}


def parse(s):
    out = []
    for tok in s.split(";"):
        wl, n, k = tok.split(",")
        out.append((float(wl), float(n), float(k)))
    return out


def fit_metal(m, fit_eps_inf=True):
    pts = parse(DATA[m])
    wl = np.array([p[0] for p in pts]) * 1e-6        # m
    n  = np.array([p[1] for p in pts])
    k  = np.array([p[2] for p in pts])
    w  = 2.0 * math.pi * C / wl                       # rad/s
    E  = HBAR * w / EV                                # eV
    mask = E <= EMAX_eV[m]
    w, n, k = w[mask], n[mask], k[mask]
    eps = (n + 1j * k) ** 2

    def resid(p):
        if fit_eps_inf:
            einf, wp, g = p
        else:
            wp, g = p
            einf = 1.0
        model = einf - wp * wp / (w * w + 1j * g * w)
        r = (model - eps) / np.abs(eps)              # relative residuals
        return np.concatenate([r.real, r.imag])

    if fit_eps_inf:
        p0 = [1.0, DC[m]["wp"], DC[m]["g"]]
        lo = [1.0, 1e15, 1e12]
        hi = [30.0, 5e16, 5e14]
    else:
        p0 = [DC[m]["wp"], DC[m]["g"]]
        lo = [1e15, 1e12]
        hi = [5e16, 5e14]

    sol = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14)
    if fit_eps_inf:
        einf, wp, g = sol.x
    else:
        wp, g = sol.x
        einf = 1.0
    rms = math.sqrt(np.mean(sol.fun ** 2))
    return dict(eps_inf=einf, wp=wp, gamma=g, npts=mask.sum(), rms=rms,
                Emin=E[mask].min(), Emax=E[mask].max())


def eV(w):
    return HBAR * w / EV


if __name__ == "__main__":
    print("OPTICAL-FIT DRUDE PARAMETERS (our least-squares fit)")
    print("=" * 78)
    for m in ("Cu", "Ag", "Au"):
        f = fit_metal(m, fit_eps_inf=True)
        wp_o = ORDAL_CM1[m]["wp"] * CM1_TO_RADS
        g_o  = ORDAL_CM1[m]["g"]  * CM1_TO_RADS
        sig_fit = EPS0 * f["wp"] ** 2 / f["gamma"]
        print(f"\n{m}   data: {SOURCE[m]}   fit window {f['Emin']:.2f}-{f['Emax']:.2f} eV, "
              f"{f['npts']} pts, rms {f['rms']*100:.2f}%")
        print(f"   {'':14}{'wp (eV)':>10}{'gamma (meV)':>13}{'eps_inf':>9}")
        print(f"   our optical  {eV(f['wp']):>10.2f}{eV(f['gamma'])*1e3:>13.1f}{f['eps_inf']:>9.2f}")
        print(f"   Ordal 1985   {eV(wp_o):>10.2f}{eV(g_o)*1e3:>13.1f}{1.0:>9.2f}")
        print(f"   DC-anchored  {eV(DC[m]['wp']):>10.2f}{eV(DC[m]['g'])*1e3:>13.1f}{1.0:>9.2f}")
        print(f"   implied sigma_dc(optical) = {sig_fit:.3e} S/m   (measured {SIGMA_DC_MEAS[m]:.3e})")

    print("\n" + "=" * 78)
    print("CHECKS")
    ok = True
    # our optical wp should land near Ordal's published wp (same/similar data)
    for m in ("Cu", "Au"):
        f = fit_metal(m, fit_eps_inf=True)
        wp_o = ORDAL_CM1[m]["wp"] * CM1_TO_RADS
        rel = abs(f["wp"] - wp_o) / wp_o
        c = rel < 0.15; ok &= c
        print(f"  [{'PASS' if c else 'FAIL'}] {m} optical wp within 15% of Ordal ({rel*100:.1f}%)")
    # fits should be good (low rms)
    for m in ("Cu", "Ag", "Au"):
        f = fit_metal(m, fit_eps_inf=True)
        c = f["rms"] < 0.20; ok &= c
        print(f"  [{'PASS' if c else 'FAIL'}] {m} fit rms < 20% ({f['rms']*100:.1f}%)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
