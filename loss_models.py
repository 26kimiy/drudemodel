"""
TEM-line loss for coax, microstrip, and CPW, fed by the DC-anchored Drude
model of Cu, Ag, Au.

Chain:  Drude sigma(omega) = sigma_dc / (1 - i*omega*tau)
        ->  R_s = Re( sqrt(i*omega*mu0 / sigma(omega)) )   surface resistance
        ->  per-line attenuation.
Conductor loss alpha_c and dielectric loss alpha_d are computed separately;
total = alpha_c + alpha_d. Results converted to dB/m (x 8.686 from Np/m).

We use the FULL complex Drude conductivity sigma(omega), not just its DC value.
At and below 100 GHz the two agree to about 1% (the metal is still essentially
in its DC regime); using sigma(omega) keeps the curves exact up to 1 THz, where
the gap reaches ~8%.

Formula sources:
  R_s, coax alpha_c, dielectric alpha_d : D. M. Pozar, Microwave Engineering.
  Microstrip eps_eff, Z0 (Hammerstad-Jensen); microstrip alpha_c ~ R_s/(Z0 W).
  CPW conformal mapping (Z0, eps_eff) and Ghione conductor loss : qucs tech notes
     (node86), after Gupta and Ghione-Naldi.
  sigma_dc and tau : DC-anchored Drude model (CRC resistivities + densities).
"""

import math
import cmath

# constants
MU0  = 1.25663706212e-6
EPS0 = 8.8541878128e-12
C    = 2.99792458e8
ETA0 = math.sqrt(MU0/EPS0)           # 376.730 ohm
NP2DB = 8.685889638                  # Np/m -> dB/m

# DC conductivity and relaxation time from the DC-anchored Drude model
SIGMA = {"Cu": 5.9595e7, "Ag": 6.3012e7, "Au": 4.0984e7}     # S/m
TAU   = {"Cu": 2.491e-14, "Ag": 3.818e-14, "Au": 2.465e-14}  # s


def sigma_ac(f, metal):
    """Full Drude conductivity sigma(omega) = sigma_dc / (1 - i*omega*tau)."""
    w = 2.0 * math.pi * f
    return SIGMA[metal] / complex(1.0, -w * TAU[metal])


def Rs(f, metal):
    """Surface resistance R_s = Re(Z_s), Z_s = sqrt(i*omega*mu0 / sigma(omega))."""
    w = 2.0 * math.pi * f
    return cmath.sqrt(1j * w * MU0 / sigma_ac(f, metal)).real


def Rs_dc(f, metal):
    """DC-limit surface resistance sqrt(pi f mu0 / sigma_dc), for comparison."""
    return math.sqrt(math.pi * f * MU0 / SIGMA[metal])


# ----- elliptic integral helpers (AGM) -----
def agm(a, b):
    for _ in range(60):
        a, b = (a + b) / 2.0, math.sqrt(a * b)
        if abs(a - b) < 1e-16 * a:
            break
    return a

def K(k):
    """Complete elliptic integral of the first kind, modulus k."""
    return math.pi / (2.0 * agm(1.0, math.sqrt(1.0 - k * k)))

def Kp(k):
    """Complementary: K(sqrt(1-k^2))."""
    return K(math.sqrt(1.0 - k * k))


# =================== COAX ===================
class Coax:
    def __init__(self, a, b, eps_r, tand):
        self.a, self.b, self.eps_r, self.tand = a, b, eps_r, tand
        self.eta = ETA0 / math.sqrt(eps_r)
        self.Z0 = self.eta / (2 * math.pi) * math.log(b / a)
    def alpha_c(self, f, metal):                       # Np/m
        return Rs(f, metal) * (1/self.a + 1/self.b) / (2 * self.eta * math.log(self.b/self.a))
    def alpha_d(self, f):                              # Np/m
        return math.pi * f * math.sqrt(self.eps_r) / C * self.tand


# =================== MICROSTRIP (Hammerstad-Jensen) ===================
class Microstrip:
    def __init__(self, W, h, eps_r, tand):
        self.W, self.h, self.eps_r, self.tand = W, h, eps_r, tand
        u = W / h
        a_u = 1 + math.log((u**4 + (u/52)**2)/(u**4 + 0.432))/49 + math.log(1 + (u/18.1)**3)/18.7
        b_er = 0.564 * ((eps_r - 0.9)/(eps_r + 3))**0.053
        self.eps_eff = (eps_r + 1)/2 + (eps_r - 1)/2 * (1 + 10/u)**(-a_u*b_er)
        f_u = 6 + (2*math.pi - 6)*math.exp(-(30.666/u)**0.7528)
        self.Z0 = ETA0/(2*math.pi*math.sqrt(self.eps_eff)) * math.log(f_u/u + math.sqrt(1 + (2/u)**2))
    def alpha_c(self, f, metal):                       # Np/m  (Pozar R_s/(Z0 W) form; approximate, W/h ~ 1 here)
        return Rs(f, metal) / (self.Z0 * self.W)
    def alpha_d(self, f):                              # Np/m  (Pozar)
        k0 = 2*math.pi*f/C
        er, ee = self.eps_r, self.eps_eff
        return k0 * er * (ee - 1) / (2*math.sqrt(ee)*(er - 1)) * self.tand


# =================== CPW (conformal mapping + Ghione) ===================
class CPW:
    def __init__(self, W, s, eps_r, tand, t):
        self.W, self.s, self.eps_r, self.tand, self.t = W, s, eps_r, tand, t
        self.k1 = W / (W + 2*s)
        self.eps_eff = (eps_r + 1)/2.0
        self.Z0 = 30*math.pi/math.sqrt(self.eps_eff) * Kp(self.k1)/K(self.k1)
    def alpha_c(self, f, metal):                       # Np/m (Ghione-Naldi)
        k1, t = self.k1, self.t
        a = self.W/2.0; b = self.W/2.0 + self.s
        rs = Rs(f, metal)
        pref = rs*math.sqrt(self.eps_eff)/(480*math.pi*K(k1)*Kp(k1)*(1 - k1*k1))
        ta = (1/a)*(math.pi + math.log(8*math.pi*a*(1 - k1)/(t*(1 + k1))))
        tb = (1/b)*(math.pi + math.log(8*math.pi*b*(1 - k1)/(t*(1 + k1))))
        return pref*(ta + tb)
    def alpha_d(self, f):                              # Np/m
        k0 = 2*math.pi*f/C
        er, ee = self.eps_r, self.eps_eff
        return k0 * er * (ee - 1) / (2*math.sqrt(ee)*(er - 1)) * self.tand


# ----- solve 50-ohm designs -----
def solve_microstrip_W(h, eps_r, Z_target=50.0):
    lo, hi = 0.05*h, 12*h
    for _ in range(80):
        mid = (lo+hi)/2
        z = Microstrip(mid, h, eps_r, 0).Z0
        if z > Z_target: lo = mid       # wider strip -> lower Z
        else: hi = mid
    return (lo+hi)/2

def solve_cpw_gap(W, eps_r, Z_target=50.0):
    lo, hi = 0.02*W, 20*W
    for _ in range(80):
        mid = (lo+hi)/2
        z = CPW(W, mid, eps_r, 0, 3e-6).Z0
        if z < Z_target: lo = mid       # wider gap -> higher Z
        else: hi = mid
    return (lo+hi)/2


if __name__ == "__main__":
    # ---- standard 50-ohm designs ----
    coax = Coax(a=0.5e-3, b=1.673e-3, eps_r=2.1, tand=2e-4)          # PTFE-filled
    h_ms = 254e-6                                                    # 10-mil alumina
    W_ms = solve_microstrip_W(h_ms, 9.9)
    ms = Microstrip(W=W_ms, h=h_ms, eps_r=9.9, tand=1e-4)
    W_cp = 50e-6
    s_cp = solve_cpw_gap(W_cp, 9.9)
    cpw = CPW(W=W_cp, s=s_cp, eps_r=9.9, tand=1e-4, t=3e-6)
    lines = [("Coax", coax), ("Microstrip", ms), ("CPW", cpw)]

    print("=== 50-ohm designs (target 50) ===")
    print(f"Coax  a=0.50 mm  b={coax.b*1e3:.3f} mm  PTFE er=2.1   -> Z0 = {coax.Z0:.2f} ohm")
    print(f"MS    W={W_ms*1e6:.1f} um  h=254 um  alumina er=9.9    -> Z0 = {ms.Z0:.2f} ohm, eps_eff={ms.eps_eff:.2f}")
    print(f"CPW   W=50 um  s={s_cp*1e6:.1f} um  alumina er=9.9     -> Z0 = {cpw.Z0:.2f} ohm, eps_eff={cpw.eps_eff:.2f}")

    f = 100e9
    print("\n=== 100 GHz snapshot, total loss (dB/m) ===")
    for name, ln in lines:
        tot = {m: (ln.alpha_c(f, m) + ln.alpha_d(f)) * NP2DB for m in ("Cu", "Ag", "Au")}
        ac = ln.alpha_c(f, "Cu") * NP2DB
        ad = ln.alpha_d(f) * NP2DB
        print(f"  {name:11} Cu {tot['Cu']:7.2f}  Ag {tot['Ag']:7.2f}  Au {tot['Au']:7.2f}"
              f"   (Cu conductor {ac:6.2f} / dielectric {ad:5.2f})")

    print("\n=== Table 4: surface resistance at 100 GHz, full sigma(w) (DC-limit in parens) ===")
    for m in ("Cu", "Ag", "Au"):
        print(f"  {m}: Rs = {Rs(f, m)*1e3:.2f} mohm   (DC-limit {Rs_dc(f, m)*1e3:.2f} mohm)")

    print("\n=== full sigma(w) vs DC limit, R_s ratio ===")
    for ff in (1e9, 1e11, 3e11, 1e12):
        print(f"  {ff:8.0e} Hz : Cu {Rs(ff,'Cu')/Rs_dc(ff,'Cu'):.4f}  "
              f"Ag {Rs(ff,'Ag')/Rs_dc(ff,'Ag'):.4f}  Au {Rs(ff,'Au')/Rs_dc(ff,'Au'):.4f}")

    # ---- checks ----
    print("\nCHECKS")
    ok = True
    for nm, ln in [("coax", coax), ("microstrip", ms), ("cpw", cpw)]:
        cond = abs(ln.Z0 - 50) < 0.6; ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {nm} Z0 within 0.6 ohm of 50 ({ln.Z0:.2f})")
    # low-frequency limit: at 1 GHz the full R_s matches the sigma_dc value
    lo = Rs(1e9, "Cu") / Rs_dc(1e9, "Cu")
    cond = abs(lo - 1) < 5e-4; ok &= cond
    print(f"  [{'PASS' if cond else 'FAIL'}] R_s -> DC limit at 1 GHz (ratio {lo:.5f})")
    # high-frequency lift: at 1 THz the full R_s sits a few percent above the DC value
    hi = Rs(1e12, "Cu") / Rs_dc(1e12, "Cu")
    cond = 1.05 < hi < 1.12; ok &= cond
    print(f"  [{'PASS' if cond else 'FAIL'}] R_s lifts ~8% above DC at 1 THz (ratio {hi:.4f})")
    # R_s increases monotonically with frequency
    mono = all(Rs(10**k, "Cu") < Rs(10**(k+1), "Cu") for k in range(9, 12))
    ok &= mono
    print(f"  [{'PASS' if mono else 'FAIL'}] R_s increases monotonically with f")
    # silver lowest conductor loss, gold highest (tracks sigma_dc)
    a_cu = coax.alpha_c(f, "Cu"); a_ag = coax.alpha_c(f, "Ag"); a_au = coax.alpha_c(f, "Au")
    cond = a_ag < a_cu < a_au; ok &= cond
    print(f"  [{'PASS' if cond else 'FAIL'}] Ag < Cu < Au conductor loss "
          f"({a_ag*NP2DB:.3f} < {a_cu*NP2DB:.3f} < {a_au*NP2DB:.3f})")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
