"""
Module to calculate geothermal profiles for Earth-like planets.

Implements a 1D thermal conduction model for the lithosphere/crust based on
Hasterok & Chapman (2011) and related thermal geophysics literature.

Main features:
- Steady-state 1D thermal conduction equation
- Radiogenic heat production with exponential decay (Hasterok model)
- Thermal conductivity via Voigt-Reuss-Hill (VRH) averaging
- Temperature-pressure-density coupling using BurnMan
- Depth-dependent local gravity
- Automatic BurnMan temperature-limit detection
- Temporal evolution models of the geothermal gradient

Temperature-limit detection:
-----------------------------
calculate_geotherm() includes automatic detection of temperature limits
(T_max_safe, default 2150 K). When the calculated temperature exceeds this
limit, the calculation stops and returns a partial profile with only the
successfully computed layers. This prevents BurnMan failures in high-
temperature regimes (typical for ancient epochs with high heat flow).

Temporal evolution:
-------------------
Functions for modelling how the geothermal gradient varies over geological time:
- q_s_turcotte()               : Temporal heat flux model (Turcotte & Schubert 2014)
- A_surface_temporal()         : Temporal radiogenic production of the crust
- A_mantle_temporal()          : Temporal radiogenic production of the mantle
- calculate_geotherm_evolution(): Calculates T(z) profiles for multiple epochs

Author : Santiago Orjuela
Date   : October 2025
Last updated: November 24, 2025
Based on: Hasterok & Chapman (2011), Turcotte & Schubert (2014)
"""

import math
import numpy as np
import pandas as pd
from astropy import constants
from burnman import minerals, Composite

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
G  = constants.G.value        # Gravitational constant
Me = constants.M_earth.value  # Earth mass (kg)
Re = constants.R_earth.value  # Earth radius (m)
km = 1000.0                   # Meters per kilometer

# =============================================================================
# GLOBAL CACHE
# =============================================================================
_SINGLE_COMPOSITE_CACHE = {}  # Cache for single-mineral Composites (performance optimization)


# =============================================================================
# MODAL LAYER COMPOSITIONS (vol %)
# =============================================================================

# Based on petrological studies of continental crust: Hasterok & Chapman (2011)
COMPOSITION_DEFAULT = {
    # UPPER CRUST (Felsic - Granite/Tonalite type)
    # Rich in quartz and feldspars, low-density minerals
    "upper": {
        "Quartz":     27,
        "Orthoclase": 15,
        "Albite":     32,
        "Anorthite":   8,
        "Phlogopite":  5,
        "Hornblende": 13
        # Sum: 100% (felsic minerals + micas + amphiboles)
    },

    # MIDDLE CRUST (Intermediate - Diorite type)
    # Transition between felsic and mafic
    # TTG (Tonalite-Trondhjemite-Granodiorite) composition
    "middle": {
        "Quartz":     15,
        "Orthoclase":  5,
        "Albite":     35,
        "Anorthite":  20,
        "Hornblende": 20
        # Sum: 95% (as in Hasterok & Chapman 2011, Table 1)
        # Normalized automatically
    },

    # LOWER CRUST (Mafic - Gabbro/Granulite type)
    # Rich in high-density mafic minerals
    "lower": {
        "Quartz":      2,
        "Orthoclase": 10,
        "Albite":     10,
        "Anorthite":  18,
        "Hornblende": 47,
        "Diopside":    1,
        "Hedenbergite": 1,
        "Enstatite":   1,
        "Ferrosillite": 1
    },

    # MANTLE (Lithospheric Upper Mantle - Peridotite)
    "mantle": {
        "Diopside":    5.47,
        "Hedenbergite": 0.53,
        "Enstatite":  18.47,
        "Ferrosillite": 1.53,
        "Forsterite": 63.65,
        "Fayalite":    3.35,
        "Pyrope":      5.58,
        "Almandine":   1.42
    }
}


# =============================================================================
# THERMAL CONDUCTIVITY PARAMETERS BY MINERAL
# =============================================================================

# Based on Hofmeister (1999), Stackhouse (2015), Hasterok & Chapman (2011)
MINERAL_PARAMS = {
    "Quartz":      {"lambda0": 6.5, "n": 0.5, "KT": 60.0,  "KTp": 4.0, "lambdaRmax": 0.5, "TR": 1400.0, "omega": 300.0},
    "Orthoclase":  {"lambda0": 2.5, "n": 0.5, "KT": 60.0,  "KTp": 4.0, "lambdaRmax": 0.2, "TR": 1400.0, "omega": 300.0},
    "Albite":      {"lambda0": 3.0, "n": 0.5, "KT": 60.0,  "KTp": 4.0, "lambdaRmax": 0.3, "TR": 1400.0, "omega": 300.0},
    "Anorthite":   {"lambda0": 3.5, "n": 0.5, "KT": 70.0,  "KTp": 4.0, "lambdaRmax": 0.4, "TR": 1400.0, "omega": 300.0},
    "Phlogopite":  {"lambda0": 1.7, "n": 0.4, "KT": 40.0,  "KTp": 4.0, "lambdaRmax": 0.1, "TR": 1300.0, "omega": 300.0},
    "Hornblende":  {"lambda0": 2.2, "n": 0.5, "KT": 60.0,  "KTp": 4.0, "lambdaRmax": 0.5, "TR": 1350.0, "omega": 300.0},
    "Diopside":    {"lambda0": 3.8, "n": 0.5, "KT": 105.0, "KTp": 4.3, "lambdaRmax": 1.0, "TR": 1400.0, "omega": 300.0},
    "Hedenbergite":{"lambda0": 3.0, "n": 0.5, "KT": 100.0, "KTp": 4.0, "lambdaRmax": 0.8, "TR": 1350.0, "omega": 300.0},
    "Enstatite":   {"lambda0": 4.2, "n": 0.5, "KT": 107.0, "KTp": 4.5, "lambdaRmax": 1.5, "TR": 1300.0, "omega": 300.0},
    "Ferrosillite":{"lambda0": 3.5, "n": 0.5, "KT": 100.0, "KTp": 4.0, "lambdaRmax": 1.0, "TR": 1300.0, "omega": 300.0},
    "Forsterite":  {"lambda0": 5.5, "n": 0.6, "KT": 128.0, "KTp": 4.2, "lambdaRmax": 3.0, "TR": 1200.0, "omega": 300.0},
    "Fayalite":    {"lambda0": 4.5, "n": 0.5, "KT": 130.0, "KTp": 4.0, "lambdaRmax": 1.5, "TR": 1200.0, "omega": 300.0},
    "Pyrope":      {"lambda0": 4.0, "n": 0.6, "KT": 171.0, "KTp": 4.3, "lambdaRmax": 2.0, "TR": 1400.0, "omega": 300.0},
    "Almandine":   {"lambda0": 3.8, "n": 0.5, "KT": 175.0, "KTp": 4.5, "lambdaRmax": 1.5, "TR": 1400.0, "omega": 300.0}
}


# =============================================================================
# SCALING FUNCTIONS
# =============================================================================

def mass_to_radius(M_planet, M_ref=Me, R_ref=Re):
    """
    Mass-radius scaling relation for rocky Earth-like planets.
    Valid up to ~5-6 M⊕.

    Parameters
    ----------
    M_planet : float
        Planet mass (kg).
    M_ref : float
        Reference mass (kg). Default: M_Earth.
    R_ref : float
        Reference radius (m). Default: R_Earth.

    Returns
    -------
    float
        Planet radius (m).
    """
    gamma = 0.27
    return R_ref * (M_planet / M_ref)**gamma


def surface_pressure(M_planet, M_ref=Me, P0_ref=1.01325e5):
    """
    Scale the atmospheric surface pressure for Earth-like planets.

    p0 ∝ M^0.92  (Earth-type atmosphere)

    Parameters
    ----------
    M_planet : float
        Planet mass (kg).
    M_ref : float
        Reference mass (kg). Default: M_Earth.
    P0_ref : float
        Reference surface pressure (Pa). Default: 1 atm.

    Returns
    -------
    float
        Surface pressure (Pa).
    """
    return P0_ref * (M_planet / M_ref)**0.92


def scale_hr(M_planet, R_planet, M_ref=Me, R_ref=Re, hr_ref=10e3):
    """
    Scale the radiogenic scale height h_r.

    Parameters
    ----------
    M_planet : float
        Planet mass (kg).
    R_planet : float
        Planet radius (m).
    M_ref : float
        Reference mass (kg). Default: M_Earth.
    R_ref : float
        Reference radius (m). Default: R_Earth.
    hr_ref : float
        Earth radiogenic scale height (m). Default: 10 km.

    Returns
    -------
    float
        Radiogenic scale height (m).
    """
    h_r = hr_ref * ((R_planet / R_ref)**2 / (M_planet / M_ref))
    return h_r


# =============================================================================
# BURNMAN MINERAL OBJECTS
# =============================================================================

def get_mineral_objects():
    """
    Create a dictionary with BurnMan mineral objects.

    Returns
    -------
    dict
        Dictionary {mineral_name: BurnMan_object}.
    """
    return {
        "Quartz":      minerals.SLB_2011.qtz(),
        "Albite":      minerals.SLB_2011.albite(),
        "Anorthite":   minerals.SLB_2011.anorthite(),
        "Diopside":    minerals.SLB_2011.diopside(),
        "Hedenbergite":minerals.SLB_2011.hedenbergite(),
        "Enstatite":   minerals.SLB_2011.enstatite(),
        "Ferrosillite":minerals.SLB_2011.ferrosilite(),
        "Forsterite":  minerals.SLB_2011.forsterite(),
        "Fayalite":    minerals.SLB_2011.fayalite(),
        "Pyrope":      minerals.SLB_2011.pyrope(),
        "Almandine":   minerals.SLB_2011.almandine(),
        "Phlogopite":  minerals.JH_2015.phl(),
        "Orthoclase":  minerals.HP_2011_ds62.hol(),
        "Hornblende":  minerals.SLB_2011.mg_tschermaks()
    }


# =============================================================================
# THERMAL CONDUCTIVITY FUNCTIONS
# =============================================================================

def lambda_lattice(mineral, T, P=0.0):
    """
    Lattice (phononic) thermal conductivity as a function of T and P.

    λ_lattice = λ₀ (298/T)ⁿ (1 + K'ₜ/Kₜ · P[GPa])

    Parameters
    ----------
    mineral : str
        Mineral name.
    T : float
        Temperature (K).
    P : float
        Pressure (Pa).

    Returns
    -------
    float
        Lattice conductivity (W/m·K).
    """
    p = MINERAL_PARAMS[mineral]
    P_GPa = P / 1e9
    KT  = p["KT"]
    KTp = p["KTp"]
    lam0 = p["lambda0"]
    n    = p["n"]
    return lam0 * (298.0 / T)**n * (1.0 + (KTp / KT) * P_GPa)


def lambda_radiative(mineral, T):
    """
    Radiative (photonic) thermal conductivity as a function of T.

    λ_rad = 0.5 λ_R,max [1 + erf((T - T_R)/ω)]

    Parameters
    ----------
    mineral : str
        Mineral name.
    T : float
        Temperature (K).

    Returns
    -------
    float
        Radiative conductivity (W/m·K).
    """
    p = MINERAL_PARAMS[mineral]
    lamRmax = p["lambdaRmax"]

    if lamRmax == 0.0:
        return 0.0

    TR    = p["TR"]
    omega = p["omega"]
    return 0.5 * lamRmax * (1.0 + math.erf((T - TR) / omega))


def lambda_effective_VRH(comp_dict, T, P=0.0):
    """
    Effective thermal conductivity using Voigt-Reuss-Hill averaging.
    Appropriate for polycrystalline aggregates (Hasterok & Chapman 2011).

    λ_VRH = 0.5 (λ_Voigt + λ_Reuss)
    where:
      λ_Voigt = Σ fᵢ λᵢ          (weighted sum, upper bound)
      λ_Reuss = (Σ fᵢ/λᵢ)⁻¹     (harmonic mean, lower bound)

    Parameters
    ----------
    comp_dict : dict
        Dictionary {mineral: fraction} normalized to sum 1.0.
    T : float or array-like
        Temperature (K).
    P : float
        Pressure (Pa).

    Returns
    -------
    float or np.ndarray
        Effective thermal conductivity (W/m·K).
    """
    minerals_list = list(comp_dict.keys())
    fracs = np.array([comp_dict[m] for m in minerals_list], dtype=float)
    scalar_input = np.isscalar(T)
    T_arr = np.atleast_1d(T).astype(float)
    out = np.zeros_like(T_arr, dtype=float)

    for idx, Ti in enumerate(T_arr):
        lambda_voigt     = 0.0
        lambda_reuss_inv = 0.0
        for j, mname in enumerate(minerals_list):
            lam_lat = lambda_lattice(mname, Ti, P)
            lam_rad = lambda_radiative(mname, Ti)
            lam  = max(lam_lat + lam_rad, 1e-6)
            frac = fracs[j]
            lambda_voigt     += frac * lam
            lambda_reuss_inv += frac / lam
        lambda_reuss = 1.0 / lambda_reuss_inv
        out[idx] = 0.5 * (lambda_voigt + lambda_reuss)

    return out[0] if scalar_input else out


# =============================================================================
# COMPOSITION AND STRUCTURE FUNCTIONS
# =============================================================================

def blend_modal(compo_upper, compo_lower, w):
    """
    Linear blend between two modal compositions (values may be % or fractions).

    Parameters
    ----------
    compo_upper : dict
        Modal composition of the upper end-member.
    compo_lower : dict
        Modal composition of the lower end-member.
    w : float
        Weight towards compo_lower, in [0, 1].

    Returns
    -------
    dict
        Unnormalized blended composition. Use normalize_modal_dict if needed.
    """
    keys = set(compo_upper.keys()) | set(compo_lower.keys())
    out = {}
    for k in keys:
        out[k] = compo_upper.get(k, 0.0) * (1.0 - w) + compo_lower.get(k, 0.0) * w
    return out


def get_composition_at_depth(z, composition=None, boundaries=None, width=5000.0):
    """
    Return the modal composition dictionary for a given depth.

    Parameters
    ----------
    z : float
        Depth (m).
    composition : dict, optional
        Dictionary with keys 'upper', 'middle', 'lower', 'mantle'.
        Default: COMPOSITION_DEFAULT.
    boundaries : list, optional
        [d1, d2, d3] layer boundary depths (m). Default: [16e3, 23e3, 39e3] (Earth).
    width : float, optional
        Transition zone half-width (m). Default: 5000.0.

    Returns
    -------
    dict
        Modal composition dictionary {mineral: fraction}.
    """
    if composition is None:
        composition = COMPOSITION_DEFAULT
    if boundaries is None:
        boundaries = [16e3, 23e3, 39e3]
    d1, d2, d3 = boundaries

    if z < d1 - width:
        return composition["upper"]
    if d1 - width <= z < d1 + width:
        w = (z - (d1 - width)) / (2 * width)
        return normalize_modal_dict(blend_modal(composition["upper"], composition["middle"], np.clip(w, 0, 1)))
    if d1 + width <= z < d2 - width:
        return composition["middle"]
    if d2 - width <= z < d2 + width:
        w = (z - (d2 - width)) / (2 * width)
        return normalize_modal_dict(blend_modal(composition["middle"], composition["lower"], np.clip(w, 0, 1)))
    if d2 + width <= z < d3 - width:
        return composition["lower"]
    if d3 - width <= z < d3 + width:
        w = (z - (d3 - width)) / (2 * width)
        return normalize_modal_dict(blend_modal(composition["lower"], composition["mantle"], np.clip(w, 0, 1)))
    return composition["mantle"]


def normalize_modal_dict(modal_dict):
    """
    Normalize a modal dictionary so that fractions sum to 1.0.

    Parameters
    ----------
    modal_dict : dict
        Dictionary {mineral: value}.

    Returns
    -------
    dict
        Normalized dictionary {mineral: fraction}.

    Raises
    ------
    ValueError
        If the sum of modal fractions is <= 0.
    """
    comp = {m: v for m, v in modal_dict.items() if v > 0}
    s = sum(comp.values())

    if s <= 0:
        raise ValueError("Sum of modal fractions <= 0")

    for m in comp:
        comp[m] = comp[m] / s

    return comp


def modal_to_mass_fractions(modal_dict, mineral_objects, P=1e5, T=298.0):
    """
    Convert modal (volumetric) fractions to mass fractions.

    Parameters
    ----------
    modal_dict : dict
        Modal fractions {mineral: vol_fraction} (sum = 1.0).
    mineral_objects : dict
        BurnMan objects {mineral_name: mineral_obj}.
    P : float
        Pressure (Pa).
    T : float
        Temperature (K).

    Returns
    -------
    dict
        Mass fractions {mineral: mass_fraction}.
    """
    mass_props = {}
    for name, vol_frac in modal_dict.items():
        if name not in mineral_objects:
            raise ValueError(f"Mineral {name} not found in mineral_objects.")
        # Use cache for Composite([mineral], [1.0])
        if name not in _SINGLE_COMPOSITE_CACHE:
            _SINGLE_COMPOSITE_CACHE[name] = Composite([mineral_objects[name]], [1.0])
        single = _SINGLE_COMPOSITE_CACHE[name]
        single.set_state(P, T)
        rho = float(single.density)
        mass_props[name] = vol_frac * rho
    total = sum(mass_props.values())
    if total <= 0:
        raise ValueError("Sum of mass properties = 0")
    mass_fracs = {name: mp / total for name, mp in mass_props.items()}
    return mass_fracs


def make_composite_from_modal(modal_dict, mineral_objects, P=1e5, T=298.0):
    """
    Create a BurnMan Composite from a modal composition.

    Pipeline: modal → mass fractions → mole fractions → Composite

    Parameters
    ----------
    modal_dict : dict
        Modal composition (may be unnormalized).
    mineral_objects : dict
        BurnMan mineral objects.
    P : float
        Pressure (Pa).
    T : float
        Temperature (K).

    Returns
    -------
    burnman.Composite
        Composite object ready for set_state().
    """
    # Normalize
    modal_norm = normalize_modal_dict(modal_dict)

    # Modal → mass
    mass_fracs = modal_to_mass_fractions(modal_norm, mineral_objects, P=P, T=T)

    # Mass → molar
    minerals_list = []
    mols = []

    for name, mass_frac in mass_fracs.items():
        m = mineral_objects[name]
        molar_mass = getattr(m, "molar_mass", None)

        if molar_mass is None:
            raise AttributeError(f"Mineral {name} has no 'molar_mass' attribute")

        mols.append(mass_frac / molar_mass)
        minerals_list.append(m)

    mols = np.array(mols)
    mole_fracs = (mols / mols.sum()).tolist()

    comp = Composite(minerals_list, mole_fracs)
    return comp


def scale_layer_boundaries(R_planet, ref_boundaries=[16e3, 23e3, 39e3],
                            R_ref=Re, max_fraction=0.5):
    """
    Scale layer boundary depths according to planet radius.

    Parameters
    ----------
    R_planet : float
        Planet radius (m).
    ref_boundaries : list
        Reference depths [d1, d2, d3] (m). Default: [16e3, 23e3, 39e3].
    R_ref : float
        Reference radius (m). Default: R_Earth.
    max_fraction : float
        Maximum allowed fraction of planet radius for boundaries.

    Returns
    -------
    list
        [d1_scaled, d2_scaled, d3_scaled] (m).

    Raises
    ------
    ValueError
        If R_planet <= 0 or ref_boundaries does not have 3 elements.
    """
    if R_planet <= 0:
        raise ValueError("R_planet must be > 0")

    if len(ref_boundaries) != 3:
        raise ValueError("ref_boundaries must have exactly 3 elements")

    scale = float(R_planet) / float(R_ref)
    max_depth = float(R_planet) * float(max_fraction)

    scaled = []
    for b in ref_boundaries:
        d = float(b) * scale
        d = max(0.0, min(d, max_depth))
        scaled.append(d)

    return sorted(scaled)


# =============================================================================
# RADIOGENIC HEAT PRODUCTION MODEL
# =============================================================================

def radiogenic_heat_profile(z, model='exponential', A_surface=2.5e-6, h_r=10e3,
                             A_upper=1.0e-6, A_lower=0.4e-6, A_mantle=0.02e-6,
                             boundaries=None):
    """
    Calculate radiogenic heat production at depth z.

    Two models available:

    1. 'exponential' (default, Lachenbruch 1970, Turcotte & Schubert):
       A(z) = A_surface · exp(-z/h_r)

       Simplified model assuming exponential decay with depth.
       Widely used in geodynamics; physically motivated by chemical
       differentiation during crustal formation.

    2. 'layered' (Hasterok & Chapman 2011):
       A(z) = A_upper  (0 < z < d1)
              A_lower  (d1 < z < d3)  [includes middle + lower crust]
              A_mantle (z > d3)

       Constant-layer model based on direct granulite and xenolith
       measurements. More realistic for chemically stratified crust.

    Parameters
    ----------
    z : float
        Depth from surface (m).
    model : str
        'exponential' or 'layered'.
    A_surface : float
        Surface production (W/m³). Default: 2.5 μW/m³.
    h_r : float
        Characteristic depth (m). Default: 10 km.
    A_upper : float
        Upper crust heat generation (W/m³). Default: 1.0 μW/m³.
    A_lower : float
        Lower crust heat generation (W/m³). Default: 0.4 μW/m³.
    A_mantle : float
        Mantle heat generation (W/m³). Default: 0.02 μW/m³.
    boundaries : list, optional
        [d1, d2, d3] boundary depths (m). Default: [16e3, 23e3, 39e3].

    Returns
    -------
    float
        Radiogenic production A(z) in W/m³.

    References
    ----------
    - Lachenbruch (1970): Exponential model for granitic batholiths.
    - Hasterok & Chapman (2011): Layered model based on petrology.
    - Turcotte & Schubert (2014): Geodynamics, chapter 4.
    """
    if z < 0:
        raise ValueError("Depth z must be >= 0")

    if model == 'exponential':
        return A_surface * np.exp(-z / h_r)

    elif model == 'layered':
        if boundaries is None:
            boundaries = [16e3, 23e3, 39e3]
        d1, d2, d3 = boundaries

        if z < d1:
            return A_upper
        elif z < d3:
            return A_lower
        else:
            return A_mantle

    else:
        raise ValueError(f"Model '{model}' not recognized. Use 'exponential' or 'layered'.")


def get_porosity(P_Pa, phi_surface=0.15, Pc_MPa=200.0, c=4.0):
    """
    Calculate fractional porosity as a function of lithostatic pressure.

    Parameters
    ----------
    P_Pa : float
        Pressure (Pa).
    phi_surface : float
        Surface porosity fraction. Default: 0.15.
    Pc_MPa : float
        Characteristic compaction pressure (MPa). Default: 200.
    c : float
        Compaction exponent. Default: 4.0.

    Returns
    -------
    float
        Fractional porosity in [0, 1].
    """
    P_MPa = P_Pa / 1e6
    phi = phi_surface * np.exp(-c * P_MPa / Pc_MPa)
    return phi


# =============================================================================
# MAIN FUNCTION: CALCULATE GEOTHERMAL PROFILE
# =============================================================================

def A0(q_surface, F=0.60, D=10.0 * km):
    """
    Estimate the surface radiogenic heat production from the surface heat flux.

    Formula: A_0 = [(1 - F) * q_s] / D

    Parameters
    ----------
    q_surface : float
        Surface heat flux (W/m²).
    F : float
        Basal fraction. Default: 0.60.
    D : float
        Thickness of the enriched upper crust (m). Default: 10 km.

    Returns
    -------
    float
        Radiogenic heat production A_0 (W/m³).
    """
    return ((1 - F) * q_surface) / D


def calculate_geotherm(rocks, q_s, z_max, dz, R_planet, M_total,
                       composition=None, boundaries=None,
                       P_top=1e5, T_top=288.0, rho_top=2800.0, g_top=None,
                       radiogenic_model='exponential',
                       A_surface=2.5e-6, h_r=10e3, phi=0.15,
                       A_upper=1.0e-6, A_lower=0.4e-6, A_mantle=0.02e-6,
                       T_max_safe=2150.0,
                       max_iter_T=60, max_iter_P=60,
                       tol_T=1e-3, tol_P=1e-3,
                       DEBUG=False):
    """
    Calculate the geothermal profile T(z), P(z), ρ(z), q(z) using the bootstrap method.

    Solves the 1D steady-state thermal conduction equation:
        d/dz[λ(z,T,P) dT/dz] + A(z) = 0

    with iterative T-P-ρ coupling using BurnMan for thermodynamic properties
    consistent with the crustal mineralogy.

    Parameters
    ----------
    rocks : dict
        BurnMan Composites per layer:
        {'upper': Composite, 'middle': Composite, 'lower': Composite, 'mantle': Composite}
    q_s : float
        Surface heat flux (W/m²).
    z_max : float
        Maximum integration depth (m).
    dz : float
        Depth step (m).
    R_planet : float
        Planet radius (m).
    M_total : float
        Total planet mass (kg).
    composition : dict, optional
        Modal compositions per layer.
    boundaries : list, optional
        [d1, d2, d3] layer boundary depths (m).
    P_top : float
        Surface pressure (Pa). Default: 1e5.
    T_top : float
        Surface temperature (K). Default: 288.
    rho_top : float
        Surface density (kg/m³). Default: 2800.
    g_top : float, optional
        Surface gravity (m/s²).
    radiogenic_model : str
        'exponential' or 'layered'. Default: 'exponential'.
    A_surface : float
        Surface radiogenic production (W/m³). Default: 2.5e-6.
    h_r : float
        Characteristic radiogenic depth (m). Default: 10e3.
    A_upper, A_lower, A_mantle : float
        Layer heat generation (W/m³) for the layered model.
    T_max_safe : float
        Maximum safe temperature (K) before stopping. Default: 2150.
    max_iter_T, max_iter_P : int
        Maximum iterations for T and P convergence loops.
    tol_T, tol_P : float
        Convergence tolerances.
    DEBUG : bool
        If True, print detailed iteration information.

    Returns
    -------
    pd.DataFrame
        Columns:
          - depth_m      : depth (m)
          - T_K          : temperature (K)
          - P_Pa         : pressure (Pa)
          - rho_kg_m3    : density (kg/m³)
          - q_W_m2       : heat flux (W/m²)
          - A_W_m3       : radiogenic production (W/m³)
          - lambda_W_mK  : thermal conductivity (W/m·K)
          - g_m_s2       : local gravity (m/s²)
          - layer        : layer name ('upper', 'middle', 'lower', 'mantle')

    References
    ----------
    - Chapman (1986): Bootstrap method.
    - Hasterok & Chapman (2011): Continental lithosphere geotherms.
    - Turcotte & Schubert (2014): Geodynamics.
    """
    # =========================================================================
    # INITIALIZATION AND VALIDATION
    # =========================================================================

    if composition is None:
        composition = COMPOSITION_DEFAULT

    if boundaries is None:
        boundaries = scale_layer_boundaries(R_planet)

    d1, d2, d3 = boundaries

    # Build depth grid
    z = np.arange(0.0, z_max + dz, dz)
    nz = len(z)

    if DEBUG:
        print("=" * 80)
        print("GEOTHERMAL PROFILE CALCULATION")
        print("=" * 80)
        print(f"Parameters:")
        print(f"  Surface heat flux: q_s = {q_s*1000:.2f} mW/m²")
        print(f"  Maximum depth: z_max = {z_max/1000:.1f} km")
        print(f"  Integration step: dz = {dz:.1f} m")
        print(f"  Number of layers: {nz-1}")
        print(f"  Radiogenic model: {radiogenic_model}")
        if radiogenic_model == 'exponential':
            print(f"    A_surface = {A_surface*1e6:.2f} μW/m³, h_r = {h_r/1000:.1f} km")
        print(f"  Boundaries: {d1/1000:.1f}, {d2/1000:.1f}, {d3/1000:.1f} km")
        print("=" * 80 + "\n")

    # =========================================================================
    # RESULT ARRAYS
    # =========================================================================

    T_array      = np.zeros(nz)  # Temperature (K)
    P_array      = np.zeros(nz)  # Pressure (Pa)
    rho_array    = np.zeros(nz)  # Density (kg/m³)
    q_array      = np.zeros(nz)  # Heat flux (W/m²)
    g_array      = np.zeros(nz)  # Local gravity (m/s²)
    lambda_array = np.zeros(nz)  # Effective thermal conductivity (W/m·K)
    phi_array    = np.zeros(nz)  # Fractional porosity
    layer_names  = []            # Layer name per level

    # Initial conditions
    T_array[0]   = T_top
    P_array[0]   = P_top
    q_array[0]   = q_s
    phi_array[0] = get_porosity(P_top, phi_surface=phi)

    rho_array[0] = rho_top * (1 - phi_array[0])

    if g_top is None:
        g_array[0] = G * M_total / (R_planet**2)
    else:
        g_array[0] = g_top

    comp_modal_0 = normalize_modal_dict(
        get_composition_at_depth(0.0, composition, boundaries)
    )

    lambda_fluid = 0.6  # W/m·K, typical value for pore fluids
    lambda_array[0] = (lambda_effective_VRH(comp_modal_0, T_top, P_top) ** (1 - phi_array[0])) * (lambda_fluid ** phi_array[0])

    layer_names.append('upper')

    # =========================================================================
    # RADIOGENIC PRODUCTION PROFILE
    # =========================================================================

    A_array = np.zeros(nz)
    for i, zi in enumerate(z):
        if radiogenic_model == 'exponential':
            A_array[i] = radiogenic_heat_profile(zi, model='exponential',
                                                  A_surface=A_surface, h_r=h_r)
        elif radiogenic_model == 'layered':
            A_array[i] = radiogenic_heat_profile(zi, model='layered',
                                                  A_upper=A_upper, A_lower=A_lower,
                                                  A_mantle=A_mantle, boundaries=boundaries)
        else:
            raise ValueError(f"Radiogenic model '{radiogenic_model}' not recognized")

    # =========================================================================
    # CONTROL VARIABLES
    # =========================================================================

    mass_above      = 0.0   # Accumulated mass above current layer (kg)
    stopped_early   = False # Early-stop flag
    last_valid_index = 0    # Last successfully computed index

    # =========================================================================
    # MAIN LOOP: DOWNWARD INTEGRATION
    # =========================================================================

    for i in range(nz - 1):
        # Extract current layer values
        Ti    = float(T_array[i])
        Pi    = float(P_array[i])
        rhoi  = float(rho_array[i])
        qi    = float(q_array[i])
        gi    = float(g_array[i])
        zi    = float(z[i])
        Ai    = float(A_array[i])
        phii  = float(phi_array[i])

        # =====================================================================
        # TEMPERATURE LIMIT CHECK
        # =====================================================================

        if Ti > T_max_safe:
            if DEBUG:
                print(f"\n⚠  TEMPERATURE LIMIT REACHED")
                print(f"   z = {zi/1000:.2f} km, T = {Ti:.1f} K > {T_max_safe:.1f} K")
                print(f"   BurnMan may fail at higher T.")
                print(f"   Returning partial profile with {i} layers.\n")
            stopped_early    = True
            last_valid_index = i
            break

        # =====================================================================
        # LAYER GEOMETRY
        # =====================================================================

        # Depth at layer midpoint
        z_mid = zi + dz / 2.0
        r_mid = R_planet - z_mid

        if r_mid <= 0:
            raise RuntimeError(f"Negative radius at z = {zi/1000:.2f} km")

        # =====================================================================
        # SELECT COMPOSITE AND COMPOSITION BY DEPTH
        # =====================================================================

        if z_mid < d1:
            burn_comp  = rocks['upper']
            layer_name = 'upper'
        elif z_mid < d2:
            burn_comp  = rocks['middle']
            layer_name = 'middle'
        elif z_mid < d3:
            burn_comp  = rocks['lower']
            layer_name = 'lower'
        else:
            burn_comp  = rocks['mantle']
            layer_name = 'mantle'

        # Get normalized modal composition
        comp_modal_raw = get_composition_at_depth(z_mid, composition, boundaries)
        comp_modal     = normalize_modal_dict(comp_modal_raw)

        # =====================================================================
        # INITIAL ESTIMATES
        # =====================================================================

        # Initial pressure (hydrostatic with constant gravity)
        P_mid = Pi + rhoi * gi * dz

        # Initial thermal conductivity
        lambda_init = (lambda_effective_VRH(comp_modal, max(Ti, 298.0), P_mid)
                       ** (1.0 - phii)) * (lambda_fluid ** phii)
        if not np.isfinite(lambda_init) or lambda_init <= 0:
            raise RuntimeError(f"Invalid initial λ at z = {zi/1000:.2f} km")

        # Initial temperature (bootstrap equation):
        # T_{i+1} = T_i + (q_i/λ) Δz - (A_i/2λ) Δz²
        A_bulk = Ai * (1 - phii)  # Only the solid fraction produces heat
        T_next = Ti + (qi / lambda_init) * dz - (A_bulk / (2.0 * lambda_init)) * dz * dz

        # =====================================================================
        # COUPLED ITERATION: T-P-ρ
        # =====================================================================

        converged_T    = False
        rho_mid        = rhoi
        g_mid          = gi
        burnman_failed = False

        for it_T in range(max_iter_T):
            # Average layer temperature
            T_mid = 0.5 * (Ti + T_next)

            # -----------------------------------------------------------------
            # PREVENTIVE CHECK: Temperature exceeds limit
            # -----------------------------------------------------------------
            if T_next > T_max_safe:
                if DEBUG:
                    print(f"\n⚠  Computed T ({T_next:.1f} K) > limit at z = {zi/1000:.2f} km")
                    print(f"   Stopping calculation.\n")
                stopped_early    = True
                last_valid_index = i
                burnman_failed   = True
                break

            # -----------------------------------------------------------------
            # INNER ITERATION: P-ρ COUPLING
            # -----------------------------------------------------------------
            P_mid_local = P_mid

            for it_P in range(max_iter_P):
                # Calculate thermodynamic properties with BurnMan
                try:
                    burn_comp.set_state(P_mid_local, T_mid)
                    rho_grain  = float(burn_comp.density)
                    phi_actual = get_porosity(P_mid_local, phi_surface=phi)
                    rho_mid    = rho_grain * (1 - phi_actual)

                except Exception as e:
                    if DEBUG:
                        print(f"\n⚠  BurnMan failed at z = {zi/1000:.2f} km")
                        print(f"   T_mid = {T_mid:.1f} K, P = {P_mid_local/1e9:.3f} GPa")
                        print(f"   Error: {type(e).__name__}")
                        print(f"   Returning partial profile.\n")
                    stopped_early    = True
                    last_valid_index = i
                    burnman_failed   = True
                    break

                # Spherical shell mass
                shell_volume = 4.0 * np.pi * (r_mid**2) * dz
                shell_mass   = rho_mid * shell_volume

                # Enclosed mass at radius r_mid
                M_enclosed = M_total - mass_above - 0.5 * shell_mass
                M_enclosed = max(M_enclosed, M_total * 1e-12)

                # Local gravity at r_mid
                g_mid = G * M_enclosed / (r_mid**2)

                # Updated pressure (hydrostatic integration)
                P_mid_new = Pi + rho_mid * g_mid * dz

                # Check pressure convergence
                if abs(P_mid_new - P_mid_local) < tol_P:
                    P_mid_local = P_mid_new
                    break

                # Relaxation for stability
                P_mid_local = 0.5 * P_mid_local + 0.5 * P_mid_new

            # If BurnMan failed, exit temperature loop
            if burnman_failed:
                break

            # -----------------------------------------------------------------
            # UPDATE THERMAL CONDUCTIVITY with T_next and P_mid_local
            # -----------------------------------------------------------------

            lambda_grain = lambda_effective_VRH(comp_modal, max(T_next, 298.0), P_mid_local)
            lambda_eff   = (lambda_grain ** (1.0 - phi_actual)) * (lambda_fluid ** phi_actual)

            if not np.isfinite(lambda_eff) or lambda_eff <= 0:
                raise RuntimeError(f"Invalid λ at z = {zi/1000:.2f} km (iter {it_T})")

            # -----------------------------------------------------------------
            # RECALCULATE TEMPERATURE with updated λ
            # Bootstrap equation: T_{i+1} = T_i + (q_i/λ) Δz - (A_i/2λ) Δz²
            # -----------------------------------------------------------------
            T_new = Ti + (qi / lambda_eff) * dz - (A_bulk / (2.0 * lambda_eff)) * dz * dz

            # -----------------------------------------------------------------
            # DEBUG OUTPUT
            # -----------------------------------------------------------------
            if DEBUG and (i < 6 or i % 500 == 0):
                print(f"[Layer {i:4d}] z={zi/1000:6.2f} km | it_T={it_T:2d} | "
                      f"T={T_new:7.2f} K | P={P_mid_local/1e9:6.3f} GPa | "
                      f"ρ={rho_mid:6.1f} kg/m³ | λ={lambda_eff:5.3f} W/m·K")

            # -----------------------------------------------------------------
            # CHECK TEMPERATURE CONVERGENCE
            # -----------------------------------------------------------------
            if abs(T_new - T_next) < tol_T:
                T_next  = T_new
                P_mid   = P_mid_local
                converged_T = True
                break

            # Relaxation for next iteration
            T_next = 0.5 * T_next + 0.5 * T_new
            P_mid  = P_mid_local

        # =====================================================================
        # CHECK BURNMAN FAILURE
        # =====================================================================

        if burnman_failed:
            break

        # Warning if not converged
        if not converged_T and DEBUG:
            print(f"⚠  [WARNING] Layer {i} did not converge in T after {max_iter_T} iterations")

        # =====================================================================
        # UPDATE ACCUMULATED MASS
        # =====================================================================

        mass_above += shell_mass

        # =====================================================================
        # UPDATE HEAT FLUX
        # Heat decreases due to integrated radiogenic production:
        # q_{i+1} = q_i - A_i Δz
        # =====================================================================
        q_next = qi - A_bulk * dz

        # =====================================================================
        # SAVE LAYER RESULTS
        # =====================================================================

        T_array[i+1]      = T_next
        P_array[i+1]      = P_mid
        rho_array[i+1]    = rho_mid
        q_array[i+1]      = q_next
        g_array[i+1]      = g_mid
        lambda_array[i+1] = lambda_eff
        layer_names.append(layer_name)
        phi_array[i+1]    = phi_actual
        last_valid_index  = i + 1

    # =========================================================================
    # POST-PROCESSING
    # =========================================================================

    # Truncate arrays if stopped early
    if stopped_early:
        z            = z[:last_valid_index+1]
        T_array      = T_array[:last_valid_index+1]
        P_array      = P_array[:last_valid_index+1]
        rho_array    = rho_array[:last_valid_index+1]
        q_array      = q_array[:last_valid_index+1]
        g_array      = g_array[:last_valid_index+1]
        A_array      = A_array[:last_valid_index+1]
        lambda_array = lambda_array[:last_valid_index+1]
        phi_array    = phi_array[:last_valid_index+1]
        layer_names  = layer_names[:last_valid_index+1]

    # =========================================================================
    # CREATE RESULTS DATAFRAME
    # =========================================================================

    df = pd.DataFrame({
        'depth_m':     z,
        'T_K':         T_array,
        'P_Pa':        P_array,
        'rho_kg_m3':   rho_array,
        'q_W_m2':      q_array,
        'A_W_m3':      A_array,
        'g_m_s2':      g_array,
        'lambda_W_mK': lambda_array,
        'phi':         phi_array,
        'layer':       layer_names
    })

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    if DEBUG:
        print("\n" + "=" * 80)
        print("PROFILE SUMMARY")
        print("=" * 80)
        print(f"Final depth:       {z[-1]/1000:.2f} km")
        print(f"Final temperature: {T_array[-1]:.2f} K ({T_array[-1]-273.15:.2f} °C)")
        print(f"Final pressure:    {P_array[-1]/1e9:.3f} GPa")
        print(f"Final density:     {rho_array[-1]:.1f} kg/m³")
        print(f"Final heat flux:   {q_array[-1]*1000:.2f} mW/m²")
        if stopped_early:
            print(f"\n⚠  CALCULATION STOPPED EARLY")
            print(f"   Partial profile returned ({len(z)} of {nz} layers)")
        else:
            print(f"\n✅  CALCULATION COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    return df


# =============================================================================
# AUXILIARY FUNCTION: PREPARE ROCKS DICT
# =============================================================================

def prepare_rocks_dict(composition=None, mineral_objects=None, P=1e5, T=288.0):
    """
    Prepare the dictionary of BurnMan Composites for each layer.

    Parameters
    ----------
    composition : dict, optional
        Modal compositions per layer. Default: COMPOSITION_DEFAULT.
    mineral_objects : dict, optional
        BurnMan mineral objects. If None, created with get_mineral_objects().
    P : float
        Reference pressure (Pa).
    T : float
        Reference temperature (K).

    Returns
    -------
    dict
        {'upper': Composite, 'middle': Composite, 'lower': Composite, 'mantle': Composite}
    """
    if composition is None:
        composition = COMPOSITION_DEFAULT

    if mineral_objects is None:
        mineral_objects = get_mineral_objects()

    rocks = {}
    for key in composition.keys():
        rocks[key] = make_composite_from_modal(
            composition[key], mineral_objects, P=P, T=T
        )

    return rocks


# =============================================================================
# TEMPORAL EVOLUTION MODELS
# =============================================================================

def q_s_turcotte(t_Ga, q0=65e-3, tau=2.0):
    """
    Temporal heat flux model from Turcotte & Schubert (2014).

    q_s(t) = q₀ · exp(t/τ)

    Simple exponential model based on average radioactive decay.

    Parameters
    ----------
    t_Ga : float or array-like
        Time before present (Ga). Positive values point to the past.
        e.g.: t=0 (present), t=1 (1 Ga ago), t=4.5 (Earth formation).
    q0 : float
        Current surface heat flux (W/m²). Default: 65e-3 (65 mW/m²).
    tau : float
        Characteristic time scale (Ga). Default: 2.0 Ga
        (effective decay time of radioactive elements).

    Returns
    -------
    float or np.ndarray
        Surface heat flux (W/m²).

    References
    ----------
    Turcotte, D. L., & Schubert, G. (2014). *Geodynamics* (3rd ed.).
    Cambridge University Press.
    """
    return q0 * np.exp(t_Ga / tau)


def A_surface_temporal(t_Ga, A_present=2.5e-6, tau=2.0, t_present=4.5):
    """
    Surface radiogenic heat production as a function of time,
    valid for past, present, and future.

    Convention:
      - t_Ga     : time since planet formation (Ga)
      - t_present: current age of the planet (Ga)
      - A_present: current (present-day) radiogenic production

    Definition:
        Δt = t_Ga - t_present
        A(t) = A_present · exp(-Δt / τ)

    So that:
        t_Ga < t_present  → past   → A > A_present
        t_Ga = t_present  → present→ A = A_present
        t_Ga > t_present  → future → A < A_present

    Parameters
    ----------
    t_Ga : float or array-like
        Time since planet formation (Ga).
    A_present : float
        Current surface radiogenic production (W/m³). Default: 2.5e-6.
    tau : float
        Effective decay time scale (Ga). Default: 2.0 (K-40 + U dominated).
    t_present : float
        Current age of the planet (Ga). Default: 4.5.

    Returns
    -------
    float or np.ndarray
        Surface radiogenic production (W/m³).

    Notes
    -----
    - Compatible with grids extending beyond the present (future epochs).
    - Directly referenced to observable present-day values.
    - Consistent with Turcotte & Schubert (2014).
    """
    dt = t_Ga - t_present
    return A_present * np.exp(-dt / tau)


def A_mantle_temporal(t_Ga, A_present=0.015e-6, tau=3.0, t_present=4.5):
    """
    Mantle radiogenic heat production as a function of time,
    valid for past, present, and future.

    Convention:
      - t_Ga     : time since planet formation (Ga)
      - t_present: current age of the planet (Ga)
      - A_present: current (present-day) mantle radiogenic production

    Definition:
        Δt = t_Ga - t_present
        A(t) = A_present · exp(-Δt / τ)

    Parameters
    ----------
    t_Ga : float or array-like
        Time since planet formation (Ga).
    A_present : float
        Current mantle radiogenic production (W/m³). Default: 0.015 μW/m³.
    tau : float
        Effective decay time scale (Ga). Default: 3.0 (U-238 + Th-232 dominated).
    t_present : float
        Current age of the planet (Ga). Default: 4.5.

    Returns
    -------
    float or np.ndarray
        Mantle radiogenic production (W/m³).

    Notes
    -----
    - Correctly captures higher production in the Archean and future decay.
    - Larger τ than for the crust due to lower K-40 contribution.
    - Consistent with Hasterok & Chapman (2011), Korenaga (2008), and
      Turcotte & Schubert (2014).
    """
    dt = t_Ga - t_present
    return A_present * np.exp(-dt / tau)


def calculate_geotherm_evolution(rocks, composition,
                                 R_planet=Re, M_total=Me,
                                 z_max=100e3, dz=100.0,
                                 boundaries=None,
                                 T_top=288.0, h_r=10e3,
                                 q0=65e-3, tau=2.0,
                                 t_Ga=np.linspace(0.001, 10, 200), qss=None,
                                 T_max_safe=2150.0,
                                 As_type='temporal'):
    """
    Calculate the temporal evolution of the geothermal gradient.

    Uses the Turcotte & Schubert (2014) model for temporal heat flux and
    computes geothermal profiles for multiple epochs.

    Parameters
    ----------
    rocks : dict
        BurnMan Composites {'upper', 'middle', 'lower', 'mantle'}.
    composition : dict
        Modal compositions per layer.
    R_planet : float
        Planet radius (m). Default: R_Earth.
    M_total : float
        Planet mass (kg). Default: M_Earth.
    z_max : float
        Maximum depth (m). Default: 100 km.
    dz : float
        Depth step (m). Default: 100 m.
    boundaries : list, optional
        [d1, d2, d3] layer boundaries (m).
    T_top : float
        Surface temperature (K). Default: 288.
    h_r : float
        Radiogenic characteristic depth (m). Default: 10 km.
    q0 : float
        Current heat flux (W/m²). Default: 65e-3.
    tau : float
        Time scale (Ga). Default: 2.0.
    t_Ga : array-like
        Times to calculate (Ga).
    qss : array-like, optional
        Prescribed heat flux array (bypasses Turcotte model).
    T_max_safe : float
        Maximum safe temperature (K). Default: 2150.
    As_type : str
        'temporal' or 'Hasterok'. Selects the surface radiogenic model.

    Returns
    -------
    dict
        Keys:
          - 't_Ga'     : array of times (Ga)
          - 'q_s'      : array of heat fluxes (W/m²)
          - 'profiles' : list of DataFrames with T(z) profiles
          - 'gradients': array of surface gradients (K/km)

    Notes
    -----
    For each time t:
      1. Calculate q_s(t) with the Turcotte model (or use qss directly).
      2. Calculate A_surface(t).
      3. Calculate T(z) profile with calculate_geotherm().
      4. Extract surface gradient (first 1 km).

    If a profile stops due to BurnMan limits (T > T_max_safe), a partial
    profile is returned for that epoch.
    """
    if boundaries is None:
        boundaries = scale_layer_boundaries(R_planet)

    if qss is None:
        # Calculate heat fluxes with the Turcotte model
        q_s = q_s_turcotte(t_Ga, q0=q0, tau=tau)
    else:
        q_s = qss

    profiles  = []
    gradients = []

    print("=" * 80)
    print(f"COMPUTING TEMPORAL EVOLUTION OF THE GEOTHERMAL GRADIENT")
    print(f"Model: Turcotte & Schubert (2014)")
    print(f"Number of time steps: {len(t_Ga)}")
    print(f"R_planet = {R_planet/1e6:.3f} x 10^6 m")
    print(f"M_total  = {M_total/1e24:.3f} x 10^24 kg")
    print("=" * 80)

    for i, (t, q) in enumerate(zip(t_Ga, q_s)):
        # Radiogenic production for this epoch
        if As_type == 'temporal':
            A_surf = A_surface_temporal(t)

        if As_type == 'Hasterok':
            A_surf = A0(q)

        A_mant = A_mantle_temporal(t)

        # Calculate geothermal profile
        df = calculate_geotherm(
            rocks=rocks,
            q_s=q,
            z_max=z_max,
            dz=dz,
            R_planet=R_planet,
            M_total=M_total,
            composition=composition,
            boundaries=boundaries,
            T_top=T_top,
            A_surface=A_surf,
            A_mantle=A_mant,
            h_r=h_r,
            T_max_safe=T_max_safe,
            DEBUG=False
        )

        profiles.append(df)

        # Calculate surface gradient (first 1 km)
        idx_1km = np.argmin(np.abs(df['depth_m']/1000 - 1.0))
        dT      = df['T_K'].iloc[idx_1km] - df['T_K'].iloc[0]
        dz_km   = df['depth_m'].iloc[idx_1km] / 1000
        gradient = dT / dz_km  # K/km
        gradients.append(gradient)

        print(f"  t = {t:6.3f} Ga | q_s = {q*1000:6.1f} mW/m² | dT/dz = {gradient:5.1f} K/km")

    print("=" * 80)
    print()

    return {
        't_Ga':      t_Ga,
        'q_s':       q_s,
        'profiles':  profiles,
        'gradients': np.array(gradients)
    }


# =============================================================================
# SUMMARY OF AVAILABLE FUNCTIONS
# =============================================================================
"""
MAIN FUNCTIONS:
===============

Profile calculation:
--------------------
calculate_geotherm()              - Calculate T(z), P(z), ρ(z) profile for one epoch
                                    WITH automatic BurnMan temperature-limit detection

Temporal evolution:
-------------------
calculate_geotherm_evolution()    - Evolution of the geothermal gradient over time
q_s_turcotte()                    - Temporal heat flux model
A_surface_temporal()              - Temporal radiogenic production of the crust
A_mantle_temporal()               - Temporal radiogenic production of the mantle

Data preparation:
-----------------
get_mineral_objects()             - Get BurnMan mineral objects
prepare_rocks_dict()              - Create layer Composites from modal compositions
make_composite_from_modal()       - Convert modal composition → Composite

Utilities:
----------
normalize_modal_dict()            - Normalize modal composition to sum = 1.0
get_composition_at_depth()        - Get composition at a given depth
scale_layer_boundaries()          - Scale layer boundaries by planet radius
radiogenic_heat_profile()         - A(z) exponential profile (Hasterok 2011)

Thermal conductivity:
---------------------
lambda_lattice()                  - Phononic conductivity λ(T, P)
lambda_radiative()                - Photonic conductivity λ(T)
lambda_effective_VRH()            - Effective conductivity (Voigt-Reuss-Hill)

Conversions:
------------
modal_to_mass_fractions()         - Modal (vol) → mass fractions
"""


if __name__ == "__main__":
    print("=" * 70)
    print("USAGE EXAMPLE: geotherm_calculator.py")
    print("=" * 70)

    # Prepare minerals and composites
    print("\n1. Preparing mineral compositions...")
    mineral_objects = get_mineral_objects()
    rocks = prepare_rocks_dict(mineral_objects=mineral_objects)
    print(f"   Composites created: {list(rocks.keys())}")

    # Calculate geothermal profile for Earth (present)
    print("\n2. Calculating Earth geothermal profile (present)...")
    df_geotherm = calculate_geotherm(
        rocks=rocks,
        q_s=65e-3,
        z_max=300e3,
        dz=100.0,
        R_planet=Re,
        M_total=Me,
        boundaries=[16e3, 23e3, 39e3],
        T_top=288.0,
        P_top=1e5,
        A_surface=2.5e-6,
        h_r=10e3,
        A_mantle=0.015e-6,
        T_max_safe=2150.0,
        DEBUG=False
    )

    print(f"   Profile computed: {len(df_geotherm)} points")
    print(f"   T range: {df_geotherm['T_K'].min():.1f} - {df_geotherm['T_K'].max():.1f} K")
    print(f"   P range: {df_geotherm['P_Pa'].min()/1e9:.3f} - {df_geotherm['P_Pa'].max()/1e9:.3f} GPa")
    print(f"   ρ range: {df_geotherm['rho_kg_m3'].min():.1f} - {df_geotherm['rho_kg_m3'].max():.1f} kg/m³")

    # Temporal evolution (0–1 Ga to avoid BurnMan issues)
    print("\n3. Computing temporal evolution of the geothermal gradient (0–1 Ga)...")
    results = calculate_geotherm_evolution(
        rocks=rocks,
        composition=COMPOSITION_DEFAULT,
        R_planet=Re,
        M_total=Me,
        z_max=100e3,
        dz=100.0,
        boundaries=[16e3, 23e3, 39e3],
        T_top=288.0,
        h_r=10e3,
        q0=65e-3,
        tau=2.0,
        T_max_safe=2150.0,
        t_Ga=np.linspace(0.001, 1.0, 5)
    )

    print("\n   Temporal evolution results:")
    print(f"   Time steps computed: {len(results['t_Ga'])}")
    print(f"   Heat flux range: {results['q_s'][0]*1000:.1f} - {results['q_s'][-1]*1000:.1f} mW/m²")
    print(f"   Gradient range:  {results['gradients'][0]:.1f} - {results['gradients'][-1]:.1f} K/km")

    print("\n" + "=" * 70)
    print("✅  geotherm_calculator.py functional")
    print("   - Automatic BurnMan temperature-limit detection")
    print("   - Partial profiles returned when T > T_max_safe")
    print("   - Temporal evolution models included")
    print("=" * 70)
