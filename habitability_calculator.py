"""
Module to calculate subsurface aquability and habitability zones.

Implements thermodynamic and biological analyses to identify habitable regions
in planetary subsurfaces, based on geothermal profiles.

Main features:
- Planetary equilibrium temperature (radiative balance)
- Circumstellar habitable zone boundaries (Kopparapu 2013)
- Water phase analysis using IAPWS standards (R14-08, IAPWS-95)
- Identification of aquability zones (thermodynamically stable liquid water)
- Identification of habitable zones (extremophile biological limits)
- P-T, P-depth, T-depth visualizations with marked zones

Scientific references:
----------------------
Biological limits:
- Maximum temperature (394 K / 121°C): Kashefi & Lovley (2003), Science 301(5635), 934
  Organism: Geogemma barossii (hyperthermophilic archaea)
- Maximum pressure (110 MPa / 1100 bar):
  * Yayanos et al. (1981), PNAS 78(9), 5212-5215 (Shewanella benthica)
  * Bartlett (2002), Biochimica et Biophysica Acta 1595(1-2), 367-381
  * Oger & Jebbar (2010), Research in Microbiology 161(10), 799-809

Water thermodynamics:
- IAPWS R14-08 (2011): Melting curves for ices Ih, III, V, VI, VII
- IAPWS-95: Equation of state for water (boiling)
- IAPWS-08: Seawater properties (salinity)

Circumstellar habitable zone:
- Kopparapu et al. (2013), ApJ 765, 131
- Kopparapu et al. (2014), ApJ 787, L29
- Kopparapu et al. (2016), ApJ 819, 84

Subsurface life:
- Onstott et al. (2006), Geomicrobiology Journal 23(6), 369-414

Author : Santiago Orjuela
Date   : December 2025
Last updated: May 7, 2026
Based on: IAPWS standards, Kashefi & Lovley (2003), Kopparapu et al. (2013)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geotherm_calculator as gc
from scipy.optimize import brentq
from astropy import constants as const

# Import IAPWS libraries for water properties
try:
    from iapws import IAPWS95
    from iapws.iapws08 import _Tf, _Tb
    from iapws._iapws import _Melting_Pressure
    IAPWS_AVAILABLE = True
except ImportError:
    IAPWS_AVAILABLE = False
    print("WARNING: IAPWS library not available. Water phase calculations will be limited.")

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann constant (W m^-2 K^-4)
AU  = 1.495978707e11        # Astronomical Unit (m)
Me  = const.M_earth.value   # Earth mass (kg)
Re  = const.R_earth.value   # Earth radius (m)

# BIOLOGICAL LIMITS
T_BIO_MIN_DEFAULT = 253.15  # K (−20°C)  - Bains et al. (2015)
T_BIO_MAX_DEFAULT = 423.15  # K ( 150°C) - Bains et al. (2015)
P_BIO_MAX_DEFAULT = 200.0e6 # Pa (200 MPa, 2000 bar) - Shewanella benthica (Yayanos et al. 1981)

# WATER PHASE CONSTANTS
T_TRIPLE = 273.16    # K  - Triple point temperature
P_TRIPLE = 611.657   # Pa - Triple point pressure
T_CRIT   = 647.096   # K  - Critical point temperature
P_CRIT   = 22.064e6  # Pa - Critical point pressure


# =============================================================================
# EQUILIBRIUM TEMPERATURE
# =============================================================================

def T_eq(distance_AU, L_star=1.0, albedo=0.30, tau=0.21):
    """
    Calculate planetary equilibrium temperature using radiative balance.

    The equilibrium temperature is obtained by equating the energy absorbed
    from the star with the energy emitted by the planet (Stefan-Boltzmann law).

    Parameters
    ----------
    distance_AU : float
        Orbital distance [AU].
    L_star : float
        Stellar luminosity normalized to the Sun [L_☉]. Default: 1.0 (Sun).
    albedo : float
        Planetary albedo (fraction of reflected light) [dimensionless].
        Default: 0.30 (Earth-like).
    tau : float
        Greenhouse effect factor [dimensionless]. Default: 0.21.
        tau = 0 → no atmosphere; tau > 0 → with greenhouse effect.

    Returns
    -------
    float
        Equilibrium temperature [K].

    Notes
    -----
    Formula: T_eq = 331.0 * [(L_star * (1 - albedo)) / ((1 + tau) * d^2)]^0.25

    The constant 331.0 K comes from T_☉ = 5778 K and R_☉ = 6.96e8 m.
    For Earth at 1 AU: T_eq ≈ 255 K (no atmosphere), 288 K (with atmosphere).

    References
    ----------
    - Kopparapu et al. (2013), ApJ 765, 131
    - Kasting et al. (1993), Icarus 101, 108
    """
    return 331.0 * ((L_star * (1 - albedo)) / ((1 + tau) * distance_AU**2))**0.25


# =============================================================================
# HABITABLE ZONE (KOPPARAPU ET AL. 2013)
# =============================================================================

# Coefficients from Kopparapu et al. (2013), Table 3.
# For each boundary: S_eff = S_eff_sun + a*dT + b*dT^2 + c*dT^3 + d*dT^4
# where dT = T_eff - 5780 K
HZ_LIMITS_KOPPARAPU_2013 = {
    # OPTIMISTIC limits
    'Recent Venus': {
        'S_eff_sun': 1.7763,
        'a': 1.4335e-4,
        'b': 3.3954e-9,
        'c': -7.6364e-12,
        'd': -1.1950e-15,
        'type': 'optimistic',
        'boundary': 'inner'
    },
    'Runaway Greenhouse': {
        'S_eff_sun': 1.0385,
        'a': 1.2456e-4,
        'b': 1.4612e-8,
        'c': -7.6345e-12,
        'd': -1.7511e-15,
        'type': 'conservative',
        'boundary': 'inner'
    },
    'Moist Greenhouse': {
        'S_eff_sun': 1.0146,
        'a': 8.1884e-5,
        'b': 1.9394e-9,
        'c': -4.3618e-12,
        'd': -6.8260e-16,
        'type': 'conservative',
        'boundary': 'inner'
    },
    'Maximum Greenhouse': {
        'S_eff_sun': 0.3507,
        'a': 5.9578e-5,
        'b': 1.6707e-9,
        'c': -3.0058e-12,
        'd': -5.1925e-16,
        'type': 'conservative',
        'boundary': 'outer'
    },
    'Early Mars': {
        'S_eff_sun': 0.3207,
        'a': 5.4471e-5,
        'b': 1.5275e-9,
        'c': -2.1709e-12,
        'd': -3.8282e-16,
        'type': 'optimistic',
        'boundary': 'outer'
    }
}


def hz_distance(T_eff, L_star, limit_name):
    """
    Calculate the orbital distance of a habitable zone boundary
    according to Kopparapu et al. (2013).

    Parameters
    ----------
    T_eff : float
        Stellar effective temperature [K].
    L_star : float
        Stellar luminosity [W] (can be normalized to L_☉ or absolute).
    limit_name : str
        HZ boundary name. Valid keys in HZ_LIMITS_KOPPARAPU_2013:
          - 'Recent Venus'        : inner boundary (optimistic)
          - 'Runaway Greenhouse'  : inner boundary (conservative)
          - 'Moist Greenhouse'    : inner boundary (most conservative)
          - 'Maximum Greenhouse'  : outer boundary (conservative)
          - 'Early Mars'          : outer boundary (optimistic)

    Returns
    -------
    float
        Orbital distance of the boundary [AU].

    Notes
    -----
    Formula:
        S_eff = S_eff_sun + a*dT + b*dT^2 + c*dT^3 + d*dT^4
        where dT = T_eff - 5780 K
        d_AU = sqrt(L_star / L_sun / S_eff)

    HZ boundaries for the Sun:
        - Conservative: 0.99 – 1.67 AU (Moist GH – Maximum GH)
        - Optimistic:   0.75 – 1.77 AU (Recent Venus – Early Mars)

    Valid for F, G, K, M stars (2600 K < T_eff < 7200 K).
    Assumes Earth-like planet (1 M⊕, N2-H2O-CO2 atmosphere).

    References
    ----------
    - Kopparapu et al. (2013), ApJ 765, 131
    - Kopparapu et al. (2014), ApJ 787, L29
    - Kopparapu et al. (2016), ApJ 819, 84

    Examples
    --------
    >>> T_eff_sun = 5780.0  # K
    >>> L_sun = 3.828e26    # W
    >>> d_in  = hz_distance(T_eff_sun, L_sun, 'Moist Greenhouse')
    >>> d_out = hz_distance(T_eff_sun, L_sun, 'Maximum Greenhouse')
    >>> print(f"Conservative HZ: {d_in:.3f} – {d_out:.3f} AU")
    Conservative HZ: 0.984 – 1.669 AU
    """
    L_sun    = const.L_sun.value
    L_ratio  = L_star / L_sun

    if limit_name not in HZ_LIMITS_KOPPARAPU_2013:
        raise ValueError(f"Invalid limit_name: {limit_name}. "
                         f"Valid options: {list(HZ_LIMITS_KOPPARAPU_2013.keys())}")

    params = HZ_LIMITS_KOPPARAPU_2013[limit_name]
    dT = T_eff - 5780.0

    S_eff = (params['S_eff_sun'] +
             params['a'] * dT +
             params['b'] * dT**2 +
             params['c'] * dT**3 +
             params['d'] * dT**4)

    d_AU = np.sqrt(L_ratio / S_eff)
    return d_AU


def get_hz_boundaries(T_eff, L_star, conservative=True):
    """
    Get the inner and outer habitable zone boundaries.

    Parameters
    ----------
    T_eff : float
        Stellar effective temperature [K].
    L_star : float
        Stellar luminosity [W].
    conservative : bool
        If True, use conservative boundaries (Moist GH – Maximum GH).
        If False, use optimistic boundaries (Recent Venus – Early Mars).
        Default: True.

    Returns
    -------
    tuple of (float, float)
        (d_inner, d_outer) in AU.

    Notes
    -----
    Conservative boundaries are more restrictive but more reliable.
    Optimistic boundaries expand the HZ but with greater uncertainty.
    """
    if conservative:
        inner_limit = 'Moist Greenhouse'
        outer_limit = 'Maximum Greenhouse'
    else:
        inner_limit = 'Recent Venus'
        outer_limit = 'Early Mars'

    d_inner = hz_distance(T_eff, L_star, inner_limit)
    d_outer = hz_distance(T_eff, L_star, outer_limit)

    return d_inner, d_outer


# =============================================================================
# ICE POLYMORPH SELECTION
# =============================================================================

def get_stable_ice(P_MPa):
    """
    Determine the stable ice polymorph at a given pressure.

    Parameters
    ----------
    P_MPa : float
        Pressure [MPa].

    Returns
    -------
    str
        Stable ice type ('Ih', 'III', 'V', 'VI', or 'VII').

    Notes
    -----
    Stability ranges:
      - Ih : 0 – 209.9 MPa   (ordinary hexagonal ice)
      - III: 209.9 – 350.1 MPa (tetragonal ice)
      - V  : 350.1 – 632.4 MPa (monoclinic ice)
      - VI : 632.4 – 2216 MPa  (tetragonal ice)
      - VII: > 2216 MPa         (high-pressure cubic ice)

    On Earth, ice Ih is the only one found naturally at the surface.
    Ices III–VII exist in the mantles of icy moons (Europa, Enceladus).
    Boundaries are approximate and slightly temperature-dependent.

    References
    ----------
    IAPWS R14-08 (2011): "Revised Release on the Pressure along the
    Melting and Sublimation Curves of Ordinary Water Substance".
    """
    if P_MPa < 209.9:
        return "Ih"
    elif P_MPa < 350.1:
        return "III"
    elif P_MPa < 632.4:
        return "V"
    elif P_MPa < 2216:
        return "VI"
    else:
        return "VII"


# =============================================================================
# WATER PHASE BOUNDARIES
# =============================================================================

def _Melting_Pressure_inverse(P_MPa, ice='Ih'):
    """
    Invert _Melting_Pressure(T) to obtain T(P) for a given ice type.

    Uses the brentq root-finding method to solve:
        _Melting_Pressure(T) = P_MPa

    Parameters
    ----------
    P_MPa : float
        Pressure [MPa].
    ice : str
        Ice type ('Ih', 'III', 'V', 'VI', 'VII'). Default: 'Ih'.

    Returns
    -------
    float
        Melting temperature [K], or np.nan if the root cannot be found.

    Notes
    -----
    Search ranges are specific to each polymorph.
    Returns np.nan if the pressure is outside the stability range.
    Requires IAPWS to be installed.
    """
    if not IAPWS_AVAILABLE:
        print("ERROR: IAPWS library required for melting pressure inversion")
        return np.nan

    # Temperature search ranges for each ice polymorph
    T_ranges = {
        'Ih':  (251.165, 273.16),
        'III': (251.166, 256.164),
        'V':   (256.165, 273.31),
        'VI':  (273.32,  355),
        'VII': (355.1,   715),
    }

    T_min, T_max = T_ranges.get(ice, (200, 800))

    try:
        return brentq(lambda T: _Melting_Pressure(T, ice=ice) - P_MPa,
                      T_min, T_max, xtol=1e-6)
    except Exception:
        return np.nan


def T_melting_IAPWS(P, salinity=0.0, ice_type='Ih'):
    """
    Calculate ice melting temperature using IAPWS standards.

    For pure water (salinity=0), uses IAPWS R14-08 (2011) melting curves
    for different ice polymorphs. For saline water, uses IAPWS-08 Seawater.

    Parameters
    ----------
    P : float or array-like
        Pressure [Pa].
    salinity : float
        Salinity [kg_salt/kg_water]. Default: 0.0 (pure water).
    ice_type : str or None
        Forced ice type ('Ih', 'III', 'V', 'VI', 'VII').
        If None, automatically selected based on pressure. Default: 'Ih'.

    Returns
    -------
    float or np.ndarray
        Melting temperature [K].

    Notes
    -----
    For P < 209.9 MPa (crust/lithosphere), ice Ih is dominant.
    Salinity reduces the melting temperature (cryoscopic depression).

    References
    ----------
    - IAPWS R14-08 (2011): Melting curves.
    - IAPWS-08: Seawater properties.
    """
    if not IAPWS_AVAILABLE:
        print("ERROR: IAPWS library required for melting temperature calculation")
        return np.nan

    P_MPa  = np.atleast_1d(P) / 1e6
    T_melt = np.zeros_like(P_MPa)

    for i, p in enumerate(P_MPa):
        # Select ice polymorph
        if ice_type is None:
            ice = get_stable_ice(p)
        else:
            ice = ice_type

        # Calculate melting temperature
        if salinity == 0:
            try:
                T_melt[i] = _Melting_Pressure_inverse(p, ice=ice)
            except Exception:
                T_melt[i] = np.nan
        else:
            try:
                T_melt[i] = _Tf(p, salinity)
            except Exception:
                # Fallback to pure water
                T_melt[i] = _Melting_Pressure_inverse(p, ice=ice)

    return T_melt[0] if np.isscalar(P) else T_melt


def T_boiling_IAPWS(P, salinity=0.0):
    """
    Calculate saturation (boiling) temperature using IAPWS-95.

    For pure water, uses the liquid-vapor saturation curve from IAPWS-95.
    For saline water, uses IAPWS-08 Seawater (ebullioscopic elevation).

    Parameters
    ----------
    P : float or array-like
        Pressure [Pa].
    salinity : float
        Salinity [kg_salt/kg_water]. Default: 0.0 (pure water).

    Returns
    -------
    float or np.ndarray
        Saturation temperature [K].

    Notes
    -----
    Valid from triple point (611.657 Pa) to critical point (22.064 MPa).
    Above the critical point, returns T_crit (supercritical water).
    Below the triple point, returns np.nan (sublimation regime).
    Salinity increases the boiling temperature.

    References
    ----------
    - IAPWS-95: "Revised Release on the IAPWS Formulation 1995..." (2016).
    - IAPWS-08: Release on the IAPWS Formulation 2008 for Seawater.
    """
    if not IAPWS_AVAILABLE:
        print("ERROR: IAPWS library required for boiling temperature calculation")
        return np.nan

    P_array = np.atleast_1d(P)
    T_sat   = np.zeros_like(P_array)

    if salinity > 0:
        # Use IAPWS-08 for saline water
        for i, p in enumerate(P_array):
            try:
                T_sat[i] = _Tb(p / 1e6, salinity)  # _Tb expects MPa
            except Exception:
                # Fallback: pure water
                if P_TRIPLE <= p <= P_CRIT:
                    T_sat[i] = IAPWS95(P=p/1e6, x=0).T  # Saturation line
                elif p > P_CRIT:
                    T_sat[i] = T_CRIT   # Supercritical
                else:
                    T_sat[i] = np.nan
    else:
        # Pure water: use IAPWS95 directly
        for i, p in enumerate(P_array):
            if P_TRIPLE <= p <= P_CRIT:
                try:
                    T_sat[i] = IAPWS95(P=p/1e6, x=0).T  # x=0 = saturated liquid
                except Exception:
                    T_sat[i] = np.nan
            elif p > P_CRIT:
                T_sat[i] = T_CRIT  # Above critical point
            else:
                T_sat[i] = np.nan

    return T_sat[0] if np.isscalar(P) else T_sat


# =============================================================================
# HABITABILITY ZONE IDENTIFICATION
# =============================================================================

def find_liquid_zone(df_geotherm, salinity=0.0, phi_min=0, information=True):
    """
    Identify the aquability zone (liquid water) in a geothermal profile.

    The aquability zone is the region where water is thermodynamically stable
    in liquid phase:
        T_fusion(P) < T < T_boiling(P) < T_crit

    Parameters
    ----------
    df_geotherm : pd.DataFrame
        DataFrame with geothermal profile.
        Required columns: 'depth_m', 'T_K', 'P_Pa', 'phi'.
    salinity : float
        Salinity [kg_salt/kg_water]. Default: 0.0 (pure water).
    phi_min : float
        Minimum porosity for a point to be included. Default: 0.
    information : bool
        If True, print information about the found zone. Default: True.

    Returns
    -------
    dict
        Keys:
          - 'liquid_zone': depths of the liquid zone [m] (or None)
          - 'indices'    : DataFrame indices of the zone
          - 'T_liquid'   : temperatures in liquid zone [K]
          - 'P_liquid'   : pressures in liquid zone [Pa]
          - 'T_melt'     : melting temperatures for entire profile [K]
          - 'T_boil'     : boiling temperatures for entire profile [K]
        All values are None if no liquid zone is found.

    Notes
    -----
    The aquability zone is necessary but not sufficient for habitability.
    Uses IAPWS standards for all phase curves.
    """
    z   = df_geotherm['depth_m'].values
    T   = df_geotherm['T_K'].values
    P   = df_geotherm['P_Pa'].values
    phi = df_geotherm['phi'].values

    # Calculate phase boundaries
    T_melt = T_melting_IAPWS(P, salinity=salinity)
    T_boil = T_boiling_IAPWS(P, salinity=salinity)

    # Identify liquid region
    T_crit_local = 647.096  # K
    liquid_mask = (T > T_melt) & (T < T_boil) & (T < T_crit_local) & ~np.isnan(T_melt) & (phi > phi_min)

    indices = np.where(liquid_mask)[0]
    if len(indices) == 0:
        if information:
            print("No liquid water zone found.")
        return {
            'liquid_zone': None,
            'indices':     None,
            'T_liquid':    None,
            'P_liquid':    None,
            'T_melt':      T_melt,
            'T_boil':      T_boil,
        }

    return {
        'liquid_zone': z[indices],
        'indices':     indices,
        'T_liquid':    T[indices],
        'P_liquid':    P[indices],
        'T_melt':      T_melt,
        'T_boil':      T_boil,
    }


def find_habitable_zone(liquid_indices, df_geotherm,
                        T_bio_min=T_BIO_MIN_DEFAULT,
                        T_bio_max=T_BIO_MAX_DEFAULT,
                        P_bio_max=P_BIO_MAX_DEFAULT,
                        phi_min=1e-3,
                        information=True):
    """
    Identify the habitable zone within the aquability zone.

    The habitable zone is the subset of the aquability zone that also
    satisfies the biological limits of known extremophiles:
        T_bio_min < T < T_bio_max   (thermal limits)
        P < P_bio_max               (pressure limit)
        phi >= phi_min              (minimum porosity)

    Parameters
    ----------
    liquid_indices : array-like or None
        Indices of the liquid zone (from find_liquid_zone).
    df_geotherm : pd.DataFrame
        DataFrame with geothermal profile.
        Required columns: 'depth_m', 'T_K', 'P_Pa', 'phi'.
    T_bio_min : float
        Minimum biological temperature [K]. Default: 253.15 K (−20°C).
    T_bio_max : float
        Maximum biological temperature [K]. Default: 394.0 K (121°C, Geogemma barossii).
    P_bio_max : float
        Maximum biological pressure [Pa]. Default: 110e6 Pa (Shewanella benthica).
    phi_min : float
        Minimum porosity for habitability. Default: 1e-3.
    information : bool
        If True, print information about the found zone. Default: True.

    Returns
    -------
    dict
        Keys:
          - 'habitable_zone': depths of the habitable zone [m] (or None)
          - 'indices'       : DataFrame indices of the zone
          - 'T_habitable'   : temperatures in habitable zone [K]
          - 'P_habitable'   : pressures in habitable zone [Pa]
          - 'phi_habitable' : porosity values in habitable zone
        All values are None if no habitable zone is found.

    Notes
    -----
    Thermal limit based on Kashefi & Lovley (2003): 394 K.
    Pressure limit based on Yayanos et al. (1981): 110 MPa.
    The habitable zone is always a subset of the aquability zone.

    References
    ----------
    - Kashefi & Lovley (2003), Science 301(5635), 934
    - Yayanos et al. (1981), PNAS 78(9), 5212-5215
    - Bartlett (2002), Biochimica et Biophysica Acta 1595(1-2), 367-381
    - Oger & Jebbar (2010), Research in Microbiology 161(10), 799-809
    """
    z   = df_geotherm['depth_m'].values
    T   = df_geotherm['T_K'].values
    P   = df_geotherm['P_Pa'].values
    phi = df_geotherm['phi'].values

    if liquid_indices is None or len(liquid_indices) == 0:
        if information:
            print("No habitable zone found because there is no liquid zone.")
        return {
            'habitable_zone': None,
            'T_habitable':    None,
            'P_habitable':    None,
            'indices':        None,
        }

    # Apply biological limits
    bio_mask = (T > T_bio_min) & (T < T_bio_max) & (P < P_bio_max) & (phi >= phi_min)

    # Intersect with liquid zone
    hab_mask = np.zeros_like(bio_mask, dtype=bool)
    hab_mask[liquid_indices] = bio_mask[liquid_indices]

    indices = np.where(hab_mask)[0]

    if len(indices) == 0:
        if information:
            print("No habitable zone found.")
        return {
            'habitable_zone': None,
            'T_habitable':    None,
            'P_habitable':    None,
            'indices':        None,
        }

    return {
        'habitable_zone': z[indices],
        'T_habitable':    T[indices],
        'P_habitable':    P[indices],
        'indices':        indices,
    }


# =============================================================================
# ZONE SUMMARY AND REPORTING
# =============================================================================

def liq_hab_zone_data(liquid_zone_data, habitable_zone_data):
    """
    Print a detailed summary of the aquability and habitability zones.

    Parameters
    ----------
    liquid_zone_data : dict
        Dictionary returned by find_liquid_zone().
    habitable_zone_data : dict
        Dictionary returned by find_habitable_zone().

    Notes
    -----
    Temperatures are shown in K and °C.
    Pressures are shown in MPa.
    Depths and thicknesses are shown in km.
    """
    if liquid_zone_data['liquid_zone'] is not None:
        liq_top = liquid_zone_data['liquid_zone'][0]
        liq_bot = liquid_zone_data['liquid_zone'][-1]

        # Aquability zone summary
        print(f"\n Aquability Layer:")
        print(f"   Depth:     {liq_top/1000:.2f} – {liq_bot/1000:.2f} km")
        print(f"   Temperature: {liquid_zone_data['T_liquid'][0]:.2f} – {liquid_zone_data['T_liquid'][-1]:.2f} K")
        print(f"   Pressure:    {liquid_zone_data['P_liquid'][0]/1e6:.2f} – {liquid_zone_data['P_liquid'][-1]/1e6:.2f} MPa")
        print(f"   Thickness:   {(liq_bot - liq_top)/1000:.3f} km")

        # Habitable zone summary
        if habitable_zone_data['habitable_zone'] is not None:
            bio_top = habitable_zone_data['habitable_zone'][0]
            bio_bot = habitable_zone_data['habitable_zone'][-1]

            print(f"\n Habitable Layer:")
            print(f"   Depth:     {bio_top/1000:.2f} – {bio_bot/1000:.2f} km")
            print(f"   Temperature: {habitable_zone_data['T_habitable'][0]:.2f} – {habitable_zone_data['T_habitable'][-1]:.2f} K "
                  f"({habitable_zone_data['T_habitable'][0]-273.15:.1f} – {habitable_zone_data['T_habitable'][-1]-273.15:.1f}°C)")
            print(f"   Pressure:    {habitable_zone_data['P_habitable'][0]/1e6:.2f} – {habitable_zone_data['P_habitable'][-1]/1e6:.2f} MPa")
            print(f"   Thickness:   {(bio_bot - bio_top)/1000:.3f} km")

            frac_habitable = ((bio_bot - bio_top) / (liq_bot - liq_top)) * 100
            print(f"\n📊 Habitable Layer = {frac_habitable:.1f}% of the Aquability Layer")
        else:
            print(f"\n HABITABLE LAYER: Not found")

    else:
        print("No Aquability Layer found.")

    print("\n" + "=" * 80)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_habitability_zones(df_geotherm, liquid_zone_data, habitable_zone_data,
                             figsize=(25, 10), save_path=None):
    """
    Create a three-panel visualization of habitability zones.

    Generates three plots:
      1. P-T Diagram: phase curves, liquid zone, habitable zone, geothermal profile.
      2. P-depth Profile: pressure vs depth with marked zones.
      3. T-depth Profile: temperature vs depth with marked zones.

    Parameters
    ----------
    df_geotherm : pd.DataFrame
        DataFrame with geothermal profile.
    liquid_zone_data : dict
        Dictionary returned by find_liquid_zone().
    habitable_zone_data : dict
        Dictionary returned by find_habitable_zone().
    figsize : tuple
        Figure size (width, height) in inches. Default: (25, 10).
    save_path : str or None
        Path to save the figure. If None, only displays it. Default: None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Notes
    -----
    Zones are color-coded: blue (aquability), green (habitable).
    Biological limits shown as dashed lines.
    IAPWS phase curves shown in the P-T diagram.
    """
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # P-T Diagram
    ax2 = fig.add_subplot(gs[0, 1])  # P-depth
    ax3 = fig.add_subplot(gs[0, 2])  # T-depth

    # Extract data
    z = df_geotherm['depth_km'].values
    T = df_geotherm['T_K'].values
    P = df_geotherm['P_Pa'].values / 1e6  # Convert to MPa

    # Zone data
    if liquid_zone_data['liquid_zone'] is not None:
        z_water = liquid_zone_data['liquid_zone'] / 1000  # km
        T_water = liquid_zone_data['T_liquid']             # K
        P_water = liquid_zone_data['P_liquid'] / 1e6      # MPa
        T_melt  = liquid_zone_data['T_melt']
        T_boil  = liquid_zone_data['T_boil']
        T_melt_water = T_melt[:len(P_water)]
        T_boil_water = T_boil[:len(P_water)]
    else:
        for ax in (ax1, ax2, ax3):
            ax.text(0.5, 0.5, 'No liquid water zone found',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig

    if habitable_zone_data['habitable_zone'] is not None:
        z_hab = habitable_zone_data['habitable_zone'] / 1000  # km
        T_hab = habitable_zone_data['T_habitable']             # K
        P_hab = habitable_zone_data['P_habitable'] / 1e6      # MPa

    # Biological limits
    T_bio_max = T_BIO_MAX_DEFAULT          # K (150°C)
    P_bio_max = P_BIO_MAX_DEFAULT / 1e6    # MPa

    # =============================================================================
    # PANEL 1: P-T DIAGRAM
    # =============================================================================

    # Liquid water region
    mask_valid_melt = ~np.isnan(T_melt_water)
    P_melt_valid    = P_water[mask_valid_melt]
    T_melt_valid    = T_melt_water[mask_valid_melt]

    T_fill = np.concatenate([T_melt_valid, T_boil_water[::-1]])
    P_fill = np.concatenate([P_melt_valid, P_water[::-1]])
    ax1.fill(T_fill, P_fill, color='lightskyblue', alpha=0.4,
             label='Liquid Water', zorder=1)

    # Habitable region
    if habitable_zone_data['habitable_zone'] is not None:
        T_hab_clipped     = T_hab[T_hab <= T_bio_max]
        P_hab_clipped     = P_hab[:len(T_hab_clipped)]
        T_melt_hab_clipped = T_melt_valid[:len(T_hab_clipped)]
        T_fill_hab = np.concatenate([T_melt_hab_clipped, T_hab_clipped[::-1]])
        P_fill_hab = np.concatenate([P_hab_clipped, P_hab_clipped[::-1]])
        ax1.fill(T_fill_hab, P_fill_hab, color='lightgreen', alpha=0.5,
                 label='Habitable Zone', zorder=2)

    # Phase curves
    ax1.plot(T_melt_valid, P_melt_valid, 'b-', linewidth=2.5, label='Fusion (IAPWS)',  zorder=3)
    ax1.plot(T_boil_water, P_water,       'r-', linewidth=2.5, label='Boiling (IAPWS)', zorder=3)

    # Geothermal profile
    ax1.plot(T, P, 'k-', linewidth=3, label='Geothermal Profile', zorder=4)

    # Biological limits
    ax1.axvline(T_bio_max, color='orange', linestyle='--', linewidth=2.5,
                label=f'$T_{{bio}}$ = {T_bio_max:.0f} K\n(121°C)', zorder=5)
    ax1.axhline(P_bio_max, color='purple', linestyle='--', linewidth=2.5,
                label=f'$P_{{bio}}$ = {P_bio_max:.0f} MPa\n(1100 bar)', zorder=5)

    ax1.set_xlabel('Temperature [K]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pressure [MPa]',  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax1.set_xlim(min(T) * 0.9, max(T) * 1.1)
    ax1.set_ylim(0, max(P_water) * 1.1)

    # =============================================================================
    # PANEL 2: P-depth PROFILE
    # =============================================================================

    ax2.axhspan(z_water[0], z_water[-1], color='lightskyblue', alpha=0.4,
                label='Liquid Water', zorder=1)

    if habitable_zone_data['habitable_zone'] is not None:
        ax2.axhspan(z_hab[0], z_hab[-1], color='lightgreen', alpha=0.5,
                    label='Habitable Zone', zorder=2)

    ax2.plot(P, z, 'purple', linewidth=3, label='P(z)', zorder=3)

    if habitable_zone_data['habitable_zone'] is not None:
        ax2.axhline(z_hab[0],  color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax2.axhline(z_hab[-1], color='red',   linestyle=':', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Pressure [MPa]', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Depth [km]',     fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax2.set_xlim(0, P_bio_max)
    ax2.set_ylim(z_water[-1] * 1.1, 0)

    # =============================================================================
    # PANEL 3: T-depth PROFILE
    # =============================================================================

    ax3.axhspan(z_water[0], z_water[-1], color='lightskyblue', alpha=0.4,
                label='Liquid Water', zorder=1)

    if habitable_zone_data['habitable_zone'] is not None:
        ax3.axhspan(z_hab[0], z_hab[-1], color='lightgreen', alpha=0.5,
                    label='Habitable Zone', zorder=2)

    ax3.plot(T, z, 'red', linewidth=3, label='T(z) Profile', zorder=3)

    if habitable_zone_data['habitable_zone'] is not None:
        ax3.axhline(z_hab[0],  color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax3.axhline(z_hab[-1], color='red',   linestyle=':', linewidth=2, alpha=0.7)

    ax3.set_xlabel('Temperature [K]', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Depth [km]',      fontsize=13, fontweight='bold')
    ax3.invert_yaxis()
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax3.set_xlim(min(T), T_bio_max)
    ax3.set_ylim(z_water[-1] * 1.1, 0)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def analyze_habitability(df_geotherm, salinity=0.0,
                          T_bio_max=T_BIO_MAX_DEFAULT,
                          P_bio_max=P_BIO_MAX_DEFAULT,
                          plot=True, verbose=True):
    """
    Convenience function that executes a complete habitability analysis.

    Runs in sequence:
      1. find_liquid_zone()         – identify aquability zone
      2. find_habitable_zone()      – identify habitable zone
      3. liq_hab_zone_data()        – print summary
      4. plot_habitability_zones()  – create visualization (optional)

    Parameters
    ----------
    df_geotherm : pd.DataFrame
        DataFrame with geothermal profile.
    salinity : float
        Salinity [kg_salt/kg_water]. Default: 0.0.
    T_bio_max : float
        Maximum biological temperature [K]. Default: 394.0 K.
    P_bio_max : float
        Maximum biological pressure [Pa]. Default: 110e6 Pa.
    plot : bool
        If True, generate plots. Default: True.
    verbose : bool
        If True, print detailed information. Default: True.

    Returns
    -------
    tuple
        (liquid_zone_data, habitable_zone_data)
    """
    # Find liquid zone
    liquid_zone_data = find_liquid_zone(df_geotherm, salinity=salinity,
                                        information=verbose)

    # Find habitable zone
    habitable_zone_data = find_habitable_zone(
        liquid_zone_data['indices'],
        df_geotherm,
        T_bio_max=T_bio_max,
        P_bio_max=P_bio_max,
        information=verbose
    )

    # Print summary
    if verbose:
        liq_hab_zone_data(liquid_zone_data, habitable_zone_data)

    # Plot if requested
    if plot and liquid_zone_data['liquid_zone'] is not None:
        plot_habitability_zones(df_geotherm, liquid_zone_data, habitable_zone_data)

    return liquid_zone_data, habitable_zone_data


def Volumen3D(distance, rocks, R_planet, M_planet, qs,
              A_surface=2.5e-6, h_r=10e3):
    """
    Calculate the liquid water and habitable zone volumes for a planet
    given a 1D geothermal profile.

    Parameters
    ----------
    distance : float
        Orbital distance from the star [AU].
    rocks : dict
        BurnMan Composites per layer.
    R_planet : float
        Planet radius [m].
    M_planet : float
        Planet mass [kg].
    qs : float
        Surface heat flux [W/m²].
    A_surface : float
        Surface radiogenic heat production [W/m³]. Default: 2.5e-6.
    h_r : float
        Radiogenic scale height [m]. Default: 10 km.

    Returns
    -------
    dict
        Keys:
          - 'liquid_volume_km3'  : liquid water volume [km³]
          - 'habitable_volume_km3': habitable zone volume [km³]
    """
    T_surf = T_eq(distance)

    df_geotherm = gc.calculate_geotherm(
        rocks=rocks,
        q_s=qs,
        z_max=20e3,
        dz=100.0,
        R_planet=R_planet,
        M_total=M_planet,
        boundaries=gc.scale_layer_boundaries(R_planet=R_planet),
        T_top=T_surf,
        A_surface=A_surface,
        h_r=h_r
    )

    liquid_zone_data = find_liquid_zone(df_geotherm, salinity=0.0, information=False)

    if liquid_zone_data['liquid_zone'] is not None:
        habitable_zone_data = find_habitable_zone(
            liquid_zone_data['indices'], df_geotherm, information=False
        )

        liq_top       = liquid_zone_data['liquid_zone'][0]
        liq_bot       = liquid_zone_data['liquid_zone'][-1]
        liq_thickness = liq_bot - liq_top

        volume_liquid = (4/3) * np.pi * (Re**3 - (Re - liq_thickness)**3)

        if habitable_zone_data is not None and habitable_zone_data['habitable_zone'] is not None:
            bio_top       = habitable_zone_data['habitable_zone'][0]
            bio_bot       = habitable_zone_data['habitable_zone'][-1]
            bio_thickness = bio_bot - bio_top
            volume_habitable = (4/3) * np.pi * (Re**3 - (Re - bio_thickness)**3)
        else:
            bio_thickness    = 0.0
            volume_habitable = 0.0

    else:
        liq_thickness    = 0.0
        bio_thickness    = 0.0
        volume_liquid    = 0.0
        volume_habitable = 0.0

    return {
        'liquid_volume_km3':   volume_liquid    / 1e9,  # km³
        'habitable_volume_km3': volume_habitable / 1e9,  # km³
    }
