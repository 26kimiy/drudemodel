# 100 GHz with metal: Drude parameters and TEM-line loss

Why a metal interconnect struggles at 100 GHz, worked from first principles: from the Drude free-electron model of copper, silver, and gold to the loss of a 50 ohm transmission line.

Prepared by Kimi Yashar, made using Claude, for Ali Khalatpour (Piris Labs).

## View it

Open `drude-dc.html` in any browser, or turn on GitHub Pages (Settings -> Pages -> Deploy from a branch -> `main` -> `/root`). It then serves at:

- `https://26kimiy.github.io/drudemodel/` (lands on the Drude parameters page)
- `https://26kimiy.github.io/drudemodel/tem-line-loss.html`

No build step and no dependencies; the loss page loads Chart.js from a CDN at runtime.

## The two pages

- **drude-dc.html** - the DC-anchored Drude parameters for Cu, Ag, Au. Every number (carrier density n, DC conductivity, plasma frequency, relaxation time, damping frequency) is computed from CRC Handbook inputs and CODATA constants; nothing is fitted. It ends at the surface resistance R_s.
- **tem-line-loss.html** - turns R_s into the conductor and dielectric loss of a 50 ohm coax, microstrip, and coplanar waveguide, with an interactive loss-vs-frequency graph (1 GHz to 1 THz), cross-section diagrams, and a 100 GHz snapshot.

## The chain

measured rho, density, molar mass  ->  Drude sigma(omega) = sigma_dc / (1 - i omega tau)  ->  R_s = Re sqrt(i omega mu0 / sigma(omega))  ->  per-line attenuation (dB/m).

The curves use the full complex Drude sigma(omega). At 100 GHz it is within about 1 percent of the DC value; using sigma(omega) keeps the curves exact up to 1 THz.

## Backing code (optional)

- **loss_models.py** - coax, microstrip (Hammerstad-Jensen), and CPW (Ghione-Naldi) loss, fed by the full Drude sigma(omega). Run `python3 loss_models.py` to reproduce every number on the site; it prints self-checks.
- **drude_models.py** - the Drude parameter derivation with automated checks. Run `python3 drude_models.py`.

## Sources

CRC Handbook of Chemistry and Physics (resistivity, density, atomic weight); N. W. Ashcroft and N. D. Mermin, *Solid State Physics* (Drude free-electron model); D. M. Pozar, *Microwave Engineering* (surface resistance, coax and microstrip loss); Qucs technical reference after Gupta and Ghione-Naldi (microstrip Hammerstad-Jensen, CPW conformal mapping).

Companion explainer on why 100 GHz is hard: https://26kimiy.github.io/restraints-of-100ghz/
