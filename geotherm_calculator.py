"""
Módulo para calcular perfiles geotérmicos de planetas tipo Tierra

Este módulo implementa un modelo de conducción térmica 1D para la litosfera/corteza
basado en Hasterok & Chapman (2011) y otros trabajos sobre geofísica térmica.

Características principales:
- Ecuación de conducción térmica en estado estacionario
- Producción radiactiva con decaimiento exponencial (Hasterok model)
- Conductividad térmica con promedio Voigt-Reuss-Hill (VRH)
- Acoplamiento presión-temperatura-densidad usando BurnMan
- Gravedad local variable con la profundidad
- Detección automática de límites de temperatura de BurnMan
- Modelos de evolución temporal del gradiente geotérmico

Detección de Límites:
---------------------
La función calculate_geotherm() ahora incluye detección automática de límites
de temperatura (T_max_safe, default 2150 K). Cuando la temperatura calculada
excede este límite, el cálculo se detiene y retorna un perfil parcial con
solo las capas calculadas exitosamente. Esto previene fallos de BurnMan en
regímenes de alta temperatura (típicamente para épocas antiguas con alto
flujo de calor).

Evolución Temporal:
------------------
Nuevas funciones para modelar cómo varía el gradiente geotérmico a lo largo
del tiempo geológico:
- q_s_turcotte(): Modelo de flujo de calor temporal (Turcotte & Schubert 2014)
- A_surface_temporal(): Producción radiactiva temporal de la corteza
- A_mantle_temporal(): Producción radiactiva temporal del manto
- calculate_geotherm_evolution(): Calcula perfiles T(z) para múltiples épocas



Autor: Santiago Orjuela
Fecha: Octubre 2025
Última actualización: Noviembre 24, 2025
Basado en: Hasterok & Chapman (2011), Turcotte & Schubert (2014)
"""

import math
import numpy as np
import pandas as pd
from astropy import constants
from burnman import minerals, Composite

# =============================================================================
# PHYSICAL CONSTANTS / CONSTANTES FÍSICAS
# =============================================================================
G = constants.G.value  # Gravitational constant / Constante gravitacional
Me = constants.M_earth.value  # Earth mass (kg) / Masa de la Tierra (kg)
Re = constants.R_earth.value  # Earth radius (m) / Radio de la Tierra (m)
km = 1000.0  # meters per kilometer / metros por kilómetro

# =============================================================================
# GLOBAL CACHE / CACHÉ GLOBAL
# =============================================================================
_SINGLE_COMPOSITE_CACHE = {}  # Cache for single-mineral Composites (performance optimization)


# =============================================================================
# MODAL LAYER COMPOSITIONS (vol %) / COMPOSICIONES MODALES DE CAPAS (% volumétrico)
# =============================================================================

# Based on petrological studies of continental crust Hasterok & Chapman (2011)
# Basadas en estudios petrológicos de corteza continental Hasterok & Chapman (2011)
COMPOSITION_DEFAULT = {
    # UPPER CRUST (Corteza Superior Félsica - tipo Granito/Tonalita)
    # Rica en cuarzo y feldespatos, minerales de baja densidad
    "upper": {
        "Quartz": 27,           
        "Orthoclase": 15,       
        "Albite": 32,           
        "Anorthite": 8,         
        "Phlogopite": 5,        
        "Hornblende": 13        
        # Suma: 100% (minerales félsicos + micas + anfíboles)
    },
    
    # MIDDLE CRUST (Corteza Media Intermedia - tipo Diorita)
    # Transición entre félsica y máfica
    # Composición TTG (Tonalite-Trondhjemite-Granodiorite)
    "middle": {
        "Quartz": 15,           
        "Orthoclase": 5,        
        "Albite": 35,          
        "Anorthite": 20,        
        "Hornblende": 20       
        # Suma: 95% (como en Hasterok & Chapman 2011, Table 1)
        # Se normalizará automáticamente
    },
    
    # LOWER CRUST (Corteza Inferior Máfica - tipo Gabro/Granulita)
    # Rica en minerales máficos de alta densidad
    "lower": {
        "Quartz": 2,            
        "Orthoclase": 10,
        "Albite": 10,
        "Anorthite": 18,
        "Hornblende": 47,
        "Diopside": 1,
        "Hedenbergite": 1,
        "Enstatite": 1,
        "Ferrosillite": 1
    },
    
    # MANTLE (Manto Superior - Peridotita Litosférica)
    "mantle": {
        "Diopside": 5.47,
        "Hedenbergite": 0.53,
        "Enstatite": 18.47,
        "Ferrosillite": 1.53,
        "Forsterite": 63.65,
        "Fayalite": 3.35,
        "Pyrope": 5.58,
        "Almandine": 1.42
    }
    
}



# =============================================================================
# THERMAL CONDUCTIVITY PARAMETERS BY MINERAL / PARÁMETROS DE CONDUCTIVIDAD TÉRMICA POR MINERAL
# =============================================================================

# Based on Hofmeister (1999), Stackhouse (2015), Hasterok & Chapman (2011)
# Basados en Hofmeister (1999), Stackhouse (2015), Hasterok & Chapman (2011)
MINERAL_PARAMS = {
    "Quartz": {"lambda0": 6.5, "n": 0.5, "KT": 60.0, "KTp": 4.0,
               "lambdaRmax": 0.5, "TR": 1400.0, "omega": 300.0},
    "Orthoclase": {"lambda0": 2.5, "n": 0.5, "KT": 60.0, "KTp": 4.0,
                   "lambdaRmax": 0.2, "TR": 1400.0, "omega": 300.0},
    "Albite": {"lambda0": 3.0, "n": 0.5, "KT": 60.0, "KTp": 4.0,
               "lambdaRmax": 0.3, "TR": 1400.0, "omega": 300.0},
    "Anorthite": {"lambda0": 3.5, "n": 0.5, "KT": 70.0, "KTp": 4.0,
                  "lambdaRmax": 0.4, "TR": 1400.0, "omega": 300.0},
    "Phlogopite": {"lambda0": 1.7, "n": 0.4, "KT": 40.0, "KTp": 4.0,
                   "lambdaRmax": 0.1, "TR": 1300.0, "omega": 300.0},
    "Hornblende": {"lambda0": 2.2, "n": 0.5, "KT": 60.0, "KTp": 4.0,
                   "lambdaRmax": 0.5, "TR": 1350.0, "omega": 300.0},
    "Diopside": {"lambda0": 3.8, "n": 0.5, "KT": 105.0, "KTp": 4.3,
                 "lambdaRmax": 1.0, "TR": 1400.0, "omega": 300.0},
    "Hedenbergite": {"lambda0": 3.0, "n": 0.5, "KT": 100.0, "KTp": 4.0,
                     "lambdaRmax": 0.8, "TR": 1350.0, "omega": 300.0},
    "Enstatite": {"lambda0": 4.2, "n": 0.5, "KT": 107.0, "KTp": 4.5,
                  "lambdaRmax": 1.5, "TR": 1300.0, "omega": 300.0},
    "Ferrosillite": {"lambda0": 3.5, "n": 0.5, "KT": 100.0, "KTp": 4.0,
                     "lambdaRmax": 1.0, "TR": 1300.0, "omega": 300.0},
    "Forsterite": {"lambda0": 5.5, "n": 0.6, "KT": 128.0, "KTp": 4.2,
                   "lambdaRmax": 3.0, "TR": 1200.0, "omega": 300.0},
    "Fayalite": {"lambda0": 4.5, "n": 0.5, "KT": 130.0, "KTp": 4.0,
                 "lambdaRmax": 1.5, "TR": 1200.0, "omega": 300.0},
    "Pyrope": {"lambda0": 4.0, "n": 0.6, "KT": 171.0, "KTp": 4.3,
               "lambdaRmax": 2.0, "TR": 1400.0, "omega": 300.0},
    "Almandine": {"lambda0": 3.8, "n": 0.5, "KT": 175.0, "KTp": 4.5,
                  "lambdaRmax": 1.5, "TR": 1400.0, "omega": 300.0}
}

# =============================================================================
# Funciones de Escalado
# =============================================================================
def mass_to_radius(M_planet, 
                   M_ref=Me, 
                   R_ref=Re):
    """
    Masa-radio para planetas rocosos tipo Tierra.
    Válido hasta ~5-6 M⊕
    """
    gamma = 0.27
    return R_ref * (M_planet / M_ref)**gamma

def surface_pressure(M_planet,
                     M_ref=Me,
                     P0_ref=1.01325e5):
    """
    Escala la presión atmosférica superficial para planetas tipo Tierra.
    
    p0 ∝ M^0.92  (atmósfera tipo Tierra)
    """
    return P0_ref * (M_planet / M_ref)**0.92

def scale_hr(M_planet,
             R_planet,
             M_ref=Me,
             R_ref=Re,
             hr_ref=10e3):
    """
    Escalado del scale height radiogénico h_r.

    Parameters
    ----------
    M_planet : float
        Masa del planeta en masas terrestres (M/M_earth)
    R_planet : float
        Radio del planeta en radios terrestres (R/R_earth)
    hr_ref : float
        Scale height terrestre en km (default = 10 km)

    Returns
    -------
    h_r : float
        Scale height radiogénico en km
    """

    h_r = hr_ref * ((R_planet/R_ref)**2 / (M_planet/M_ref))
    return h_r


# =============================================================================
# BURNMAN MINERAL OBJECTS / OBJETOS MINERALES DE BURNMAN
# =============================================================================

def get_mineral_objects():
    """
    Español:
    Crea diccionario con objetos de minerales de BurnMan.
    
    Retorna:
        dict: Diccionario {nombre_mineral: objeto_BurnMan}
    
    English:
    Create dictionary with BurnMan mineral objects.
    
    Returns:
        dict: Dictionary {mineral_name: BurnMan_object}
    """
    return {
        "Quartz": minerals.SLB_2011.qtz(),
        "Albite": minerals.SLB_2011.albite(),
        "Anorthite": minerals.SLB_2011.anorthite(),
        "Diopside": minerals.SLB_2011.diopside(),
        "Hedenbergite": minerals.SLB_2011.hedenbergite(),
        "Enstatite": minerals.SLB_2011.enstatite(),
        "Ferrosillite": minerals.SLB_2011.ferrosilite(),
        "Forsterite": minerals.SLB_2011.forsterite(),
        "Fayalite": minerals.SLB_2011.fayalite(),
        "Pyrope": minerals.SLB_2011.pyrope(),
        "Almandine": minerals.SLB_2011.almandine(),
        "Phlogopite": minerals.JH_2015.phl(),
        "Orthoclase": minerals.HP_2011_ds62.hol(),
        "Hornblende": minerals.SLB_2011.mg_tschermaks()
    }


# =============================================================================
# THERMAL CONDUCTIVITY FUNCTIONS / FUNCIONES DE CONDUCTIVIDAD TÉRMICA
# =============================================================================

def lambda_lattice(mineral, T, P=0.0):
    """
    Español:
    Conductividad térmica lattice (fonónica) dependiente de T y P.
    
    λ_lattice = λ₀ (298/T)ⁿ (1 + K'ₜ/Kₜ · P[GPa])
    
    Parámetros:
        mineral (str): Nombre del mineral
        T (float): Temperatura (K)
        P (float): Presión (Pa)
        
    Retorna:
        float: Conductividad lattice (W/m·K)
    
    English:
    Lattice (phononic) thermal conductivity dependent on T and P.
    
    λ_lattice = λ₀ (298/T)ⁿ (1 + K'ₜ/Kₜ · P[GPa])
    
    Parameters:
        mineral (str): Mineral name
        T (float): Temperature (K)
        P (float): Pressure (Pa)
        
    Returns:
        float: Lattice conductivity (W/m·K)
    """
    p = MINERAL_PARAMS[mineral]
    P_GPa = P / 1e9
    KT = p["KT"]
    KTp = p["KTp"]
    lam0 = p["lambda0"]
    n = p["n"]
    
    return lam0 * (298.0 / T)**n * (1.0 + (KTp / KT) * P_GPa)


def lambda_radiative(mineral, T):
    """
    Español:
    Conductividad térmica radiativa (fotónica) dependiente de T.
    
    λ_rad = 0.5 λ_R,max [1 + erf((T - T_R)/ω)]
    
    Parámetros:
        mineral (str): Nombre del mineral
        T (float): Temperatura (K)
        
    Retorna:
        float: Conductividad radiativa (W/m·K)
    
    English:
    Radiative (photonic) thermal conductivity dependent on T.
    
    λ_rad = 0.5 λ_R,max [1 + erf((T - T_R)/ω)]
    
    Parameters:
        mineral (str): Mineral name
        T (float): Temperature (K)
        
    Returns:
        float: Radiative conductivity (W/m·K)
    """
    p = MINERAL_PARAMS[mineral]
    lamRmax = p["lambdaRmax"]
    
    if lamRmax == 0.0:
        return 0.0
    
    TR = p["TR"]
    omega = p["omega"]
    
    return 0.5 * lamRmax * (1.0 + math.erf((T - TR) / omega))


def lambda_effective_VRH(comp_dict, T, P=0.0):
    """
    Español:
    Conductividad térmica efectiva usando promedio Voigt-Reuss-Hill.
    Apropiado para agregados policristalinos (Hasterok & Chapman 2011).
    
    λ_VRH = 0.5 (λ_Voigt + λ_Reuss)
    donde:
    - λ_Voigt = Σ fᵢ λᵢ (suma ponderada, límite superior)
    - λ_Reuss = (Σ fᵢ/λᵢ)⁻¹ (media armónica, límite inferior)
    
    Parámetros:
        comp_dict (dict): Diccionario {mineral: fracción} normalizado a suma 1.0
        T (float): Temperatura (K)
        P (float): Presión (Pa)
        
    Retorna:
        float: Conductividad térmica efectiva (W/m·K)
    
    English:
    Effective thermal conductivity using Voigt-Reuss-Hill averaging.
    Appropriate for polycrystalline aggregates (Hasterok & Chapman 2011).
    
    λ_VRH = 0.5 (λ_Voigt + λ_Reuss)
    where:
    - λ_Voigt = Σ fᵢ λᵢ (weighted sum, upper bound)
    - λ_Reuss = (Σ fᵢ/λᵢ)⁻¹ (harmonic mean, lower bound)
    
    Parameters:
        comp_dict (dict): Dictionary {mineral: fraction} normalized to sum 1.0
        T (float): Temperature (K)
        P (float): Pressure (Pa)
        
    Returns:
        float: Effective thermal conductivity (W/m·K)
    """
    # Precompute mineral list and fractions
    minerals_list = list(comp_dict.keys())
    fracs = np.array([comp_dict[m] for m in minerals_list], dtype=float)
    # Ensure T is array-like
    scalar_input = np.isscalar(T)
    T_arr = np.atleast_1d(T).astype(float)
    out = np.zeros_like(T_arr, dtype=float)

    # For each temperature evaluate lam_i and compute VRH
    for idx, Ti in enumerate(T_arr):
        lambda_voigt = 0.0
        lambda_reuss_inv = 0.0
        for j, mname in enumerate(minerals_list):
            lam_lat = lambda_lattice(mname, Ti, P)
            lam_rad = lambda_radiative(mname, Ti)
            lam = lam_lat + lam_rad
            lam = max(lam, 1e-6)
            frac = fracs[j]
            lambda_voigt += frac * lam
            lambda_reuss_inv += frac / lam
        lambda_reuss = 1.0 / lambda_reuss_inv
        out[idx] = 0.5 * (lambda_voigt + lambda_reuss)

    return out[0] if scalar_input else out


# =============================================================================
# COMPOSITION AND STRUCTURE FUNCTIONS / FUNCIONES DE COMPOSICIÓN Y ESTRUCTURA
# =============================================================================

def blend_modal(compo_upper, compo_lower, w):
    """
    Mezcla lineal entre dos composiciones modales (valores pueden ser % o fracciones).
    w in [0,1] es el peso hacia compo_lower.
    Retorna dict no normalizado (usa normalize_modal_dict si hace falta).
    """
    keys = set(compo_upper.keys()) | set(compo_lower.keys())
    out = {}
    for k in keys:
        out[k] = compo_upper.get(k, 0.0) * (1.0 - w) + compo_lower.get(k, 0.0) * w
    return out


def get_composition_at_depth(z, composition=None, boundaries=None, width=5000.0):
    """
    Español:
    Devuelve el diccionario de composición modal para una profundidad dada.
    
    Parámetros:
        z (float): Profundidad (m)
        composition (dict, optional): Diccionario con claves 'upper', 'middle', 'lower', 'mantle'
        boundaries (list, optional): Lista [d1, d2, d3] con fronteras entre capas (m)
                                     Default: [16e3, 23e3, 39e3] (Tierra)
        width (float, optional): Ancho de la zona de transición (m), default: 5000.0
        
    Retorna:
        dict: Diccionario de composición modal {mineral: fracción}
    
    English:
    Returns the modal composition dictionary for a given depth.
    
    Parameters:
        z (float): Depth (m)
        composition (dict, optional): Dictionary with keys 'upper', 'middle', 'lower', 'mantle'
        boundaries (list, optional): List [d1, d2, d3] with layer boundaries (m)
                                     Default: [16e3, 23e3, 39e3] (Earth)
        width (float, optional): Transition zone width (m), default: 5000.0
        
    Returns:
        dict: Modal composition dictionary {mineral: fraction}
    """
    if composition is None:
        composition = COMPOSITION_DEFAULT
    if boundaries is None:
        boundaries = [16e3, 23e3, 39e3]
    d1, d2, d3 = boundaries
    # below d1 - upper
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
    Español:
    Normaliza un diccionario modal a fracciones que sumen 1.0.
    
    Parámetros:
        modal_dict (dict): Diccionario {mineral: valor}
        
    Retorna:
        dict: Diccionario normalizado {mineral: fracción}
    
    English:
    Normalize a modal dictionary to fractions that sum to 1.0.
    
    Parameters:
        modal_dict (dict): Dictionary {mineral: value}
        
    Returns:
        dict: Normalized dictionary {mineral: fraction}
    """
    comp = {m: v for m, v in modal_dict.items() if v > 0}
    s = sum(comp.values())
    
    if s <= 0:
        raise ValueError("Suma de composiciones modal <= 0")
    
    for m in comp:
        comp[m] = comp[m] / s
    
    return comp


def modal_to_mass_fractions(modal_dict, mineral_objects, P=1e5, T=298.0):
    """
    Español:
    Convierte fracciones modales (volumétricas) a fracciones de masa.
    
    Parámetros:
        modal_dict (dict): Fracciones modales {mineral: fracción_vol} (suma = 1.0)
        mineral_objects (dict): Objetos BurnMan {mineral_name: mineral_obj}
        P (float): Presión (Pa)
        T (float): Temperatura (K)
        
    Retorna:
        dict: Fracciones de masa {mineral: fracción_masa}
    
    English:
    Convert modal (volumetric) fractions to mass fractions.
    
    Parameters:
        modal_dict (dict): Modal fractions {mineral: vol_fraction} (sum = 1.0)
        mineral_objects (dict): BurnMan objects {mineral_name: mineral_obj}
        P (float): Pressure (Pa)
        T (float): Temperature (K)
        
    Returns:
        dict: Mass fractions {mineral: mass_fraction}
    """
    mass_props = {}
    for name, vol_frac in modal_dict.items():
        if name not in mineral_objects:
            raise ValueError(f"Mineral {name} no encontrado en mineral_objects.")
        # cache para Composite([mineral], [1.0])
        if name not in _SINGLE_COMPOSITE_CACHE:
            _SINGLE_COMPOSITE_CACHE[name] = Composite([mineral_objects[name]], [1.0])
        single = _SINGLE_COMPOSITE_CACHE[name]
        # actualizar estado (set_state es barato en comparación a recrear Composite)
        single.set_state(P, T)
        rho = float(single.density)
        mass_props[name] = vol_frac * rho
    total = sum(mass_props.values())
    if total <= 0:
        raise ValueError("Suma de propiedades de masa = 0")
    mass_fracs = {name: mp / total for name, mp in mass_props.items()}
    return mass_fracs


def make_composite_from_modal(modal_dict, mineral_objects, P=1e5, T=298.0):
    """
    Español:
    Crea un Composite de BurnMan desde composición modal.
    
    Pipeline: modal → mass fracs → mole fracs → Composite
    
    Parámetros:
        modal_dict (dict): Composición modal (puede no estar normalizada)
        mineral_objects (dict): Objetos BurnMan
        P (float): Presión (Pa)
        T (float): Temperatura (K)
        
    Retorna:
        burnman.Composite: Objeto Composite listo para set_state()
    
    English:
    Create a BurnMan Composite from modal composition.
    
    Pipeline: modal → mass fracs → mole fracs → Composite
    
    Parameters:
        modal_dict (dict): Modal composition (may not be normalized)
        mineral_objects (dict): BurnMan objects
        P (float): Pressure (Pa)
        T (float): Temperature (K)
        
    Returns:
        burnman.Composite: Composite object ready for set_state()
    """
    # Normalizar
    modal_norm = normalize_modal_dict(modal_dict)
    
    # Modal → masa
    mass_fracs = modal_to_mass_fractions(modal_norm, mineral_objects, P=P, T=T)
    
    # Masa → molar
    minerals_list = []
    mols = []
    
    for name, mass_frac in mass_fracs.items():
        m = mineral_objects[name]
        molar_mass = getattr(m, "molar_mass", None)
        
        if molar_mass is None:
            raise AttributeError(f"Mineral {name} no tiene 'molar_mass'")
        
        mols.append(mass_frac / molar_mass)
        minerals_list.append(m)
    
    mols = np.array(mols)
    mole_fracs = (mols / mols.sum()).tolist()
    
    comp = Composite(minerals_list, mole_fracs)
    return comp


def scale_layer_boundaries(R_planet, ref_boundaries=[16e3, 23e3, 39e3],
                           R_ref=Re, max_fraction=0.5):
    """
    Español:
    Escala las profundidades de fronteras de capas según el radio del planeta.
    
    Parámetros:
        R_planet (float): Radio del planeta (m)
        ref_boundaries (list): Profundidades de referencia [d1, d2, d3] (m)
        R_ref (float): Radio de referencia (m), default = R_Earth
        max_fraction (float): Fracción máxima del radio permitida para fronteras
        
    Retorna:
        list: Lista [d1_scaled, d2_scaled, d3_scaled] (m)
    
    English:
    Scale layer boundary depths according to planet radius.
    
    Parameters:
        R_planet (float): Planet radius (m)
        ref_boundaries (list): Reference depths [d1, d2, d3] (m)
        R_ref (float): Reference radius (m), default = R_Earth
        max_fraction (float): Maximum fraction of radius allowed for boundaries
        
    Returns:
        list: List [d1_scaled, d2_scaled, d3_scaled] (m)
    """
    if R_planet <= 0:
        raise ValueError("R_planet debe ser > 0")
    
    if len(ref_boundaries) != 3:
        raise ValueError("ref_boundaries debe tener 3 elementos")
    
    scale = float(R_planet) / float(R_ref)
    max_depth = float(R_planet) * float(max_fraction)
    
    scaled = []
    for b in ref_boundaries:
        d = float(b) * scale
        d = max(0.0, min(d, max_depth))
        scaled.append(d)
    
    return sorted(scaled)


# =============================================================================
# RADIOGENIC HEAT PRODUCTION MODEL / MODELO DE PRODUCCIÓN RADIACTIVA
# =============================================================================

def radiogenic_heat_profile(z, model='exponential', A_surface=2.5e-6, h_r=10e3,
                           A_upper=1.0e-6, A_lower=0.4e-6, A_mantle=0.02e-6,
                           boundaries=None):
    """
    Español:
    Calcula la producción radiogénica de calor a una profundidad z.
    
    Dos modelos disponibles:
    
    1. 'exponential' (default, Lachenbruch 1970, Turcotte & Schubert):
       A(z) = A_surface · exp(-z/h_r)
       
       Modelo simplificado que asume decaimiento exponencial con profundidad.
       Ampliamente usado en geodinámica. Físicamente motivado por diferenciación
       química durante formación cortical.
       
    2. 'layered' (Hasterok & Chapman 2011):
       A(z) = A_upper  (0 < z < d1)
              A_lower  (d1 < z < d3)  [incluye middle+lower crust]
              A_mantle (z > d3)
       
       Modelo constante por capas basado en mediciones directas de granulitas
       y xenolitos. Más realista para corteza estratificada químicamente.
    
    Parámetros:
        z (float): Profundidad desde superficie (m)
        model (str): 'exponential' o 'layered'
        
        # Para modelo exponencial:
        A_surface (float): Producción superficial (W/m³), default: 2.5 μW/m³
        h_r (float): Profundidad característica (m), default: 10 km
        
        # Para modelo por capas:
        A_upper (float): HG corteza superior (W/m³), default: 1.0 μW/m³
        A_lower (float): HG corteza inferior (W/m³), default: 0.4 μW/m³
        A_mantle (float): HG manto (W/m³), default: 0.02 μW/m³
        boundaries (list): [d1, d2, d3] fronteras (m), default: [16e3, 23e3, 39e3]
        
    Retorna:
        float: Producción radiogénica A(z) en W/m³
    
    Referencias:
        - Lachenbruch (1970): Modelo exponencial para batholiths graníticos
        - Hasterok & Chapman (2011): Modelo por capas basado en petología
        - Turcotte & Schubert (2014): Geodynamics, capítulo 4
    
    English:
    Calculate radiogenic heat production at depth z.
    
    Two models available:
    
    1. 'exponential' (default, Lachenbruch 1970, Turcotte & Schubert):
       A(z) = A_surface · exp(-z/h_r)
       
       Simplified model assuming exponential decay with depth.
       Widely used in geodynamics. Physically motivated by chemical
       differentiation during crustal formation.
       
    2. 'layered' (Hasterok & Chapman 2011):
       A(z) = A_upper  (0 < z < d1)
              A_lower  (d1 < z < d3)  [includes middle+lower crust]
              A_mantle (z > d3)
       
       Constant-layer model based on direct granulite and xenolith
       measurements. More realistic for chemically stratified crust.
    
    Parameters:
        z (float): Depth from surface (m)
        model (str): 'exponential' or 'layered'
        
        # For exponential model:
        A_surface (float): Surface production (W/m³), default: 2.5 μW/m³
        h_r (float): Characteristic depth (m), default: 10 km
        
        # For layered model:
        A_upper (float): Upper crust HG (W/m³), default: 1.0 μW/m³
        A_lower (float): Lower crust HG (W/m³), default: 0.4 μW/m³
        A_mantle (float): Mantle HG (W/m³), default: 0.02 μW/m³
        boundaries (list): [d1, d2, d3] boundaries (m), default: [16e3, 23e3, 39e3]
        
    Returns:
        float: Radiogenic production A(z) in W/m³
    
    References:
        - Lachenbruch (1970): Exponential model for granitic batholiths
        - Hasterok & Chapman (2011): Layered model based on petrology
        - Turcotte & Schubert (2014): Geodynamics, chapter 4
    """
    if z < 0:
        raise ValueError("Profundidad z debe ser >= 0")
    
    if model == 'exponential':
        # Modelo exponencial: A(z) = A₀ exp(-z/h_r)
        return A_surface * np.exp(-z / h_r)
    
    elif model == 'layered':
        # Modelo por capas constantes
        if boundaries is None:
            boundaries = [16e3, 23e3, 39e3]
        
        d1, d2, d3 = boundaries
        
        if z < d1:
            return A_upper
        elif z < d3:  # Incluye middle (d1-d2) y lower (d2-d3)
            return A_lower
        else:
            return A_mantle
    
    else:
        raise ValueError(f"Modelo '{model}' no reconocido. Use 'exponential' o 'layered'")



def get_porosity(P_Pa, phi_surface=0.15, Pc_MPa=200.0, c=4.0):
    """
    Calcula la porosidad fraccional dependiente de la presión litoestática.
    """
    P_MPa = P_Pa / 1e6

    phi = phi_surface * np.exp(-c * P_MPa / Pc_MPa)

    return phi

# =============================================================================
# MAIN FUNCTION: CALCULATE GEOTHERMAL PROFILE / FUNCIÓN PRINCIPAL: CALCULAR PERFIL GEOTÉRMICO 
# =============================================================================

def A0(q_surface, F = 0.60, D = 10.0*km):
    """
    Calcula el área superficial necesaria para disipar un flujo de calor dado.
    
    Parámetros:
    - q_surface: Flujo de calor en la superficie (W/m²)
    - F: Fracción basal
    - D: Espesor de la corteza superior enriquecida [m] (ej. 10000 m = 10 km)
    
    Retorna:
    - A0 en W/m^3
    """
    # Fórmula: A_0 = [(1 - F) * q_s] / D
    A0 = ((1 - F) * q_surface) / D
    
    return A0

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
    Español:
    Calcula perfil geotérmico T(z), P(z), ρ(z), q(z) usando bootstrap method.
    
    Resuelve ecuación de conducción térmica 1D en estado estacionario:
        d/dz[λ(z,T,P) dT/dz] + A(z) = 0
    
    con acoplamiento iterativo T-P-ρ usando BurnMan para propiedades 
    termodinámicas consistentes con la mineralogía.
    
    Parámetros:
        rocks (dict): Composites de BurnMan por capa
                     {'upper': Composite, 'middle': Composite, 
                      'lower': Composite, 'mantle': Composite}
        q_s (float): Flujo de calor superficial (W/m²)
        z_max (float): Profundidad máxima de inte        R_planet (float): Radio del planeta (m)
        M_total (float): Masa total del planeta (kg)
        
        composition (dict, optional): Composiciones modales por capa
        boundaries (list, optional): [d1, d2, d3] fronteras entre capas (m)
        
        P_top (float): Presión superficial (Pa). Default: 1e5
        T_top (float): Temperatura superficial (K). Default: 288
        rho_top (float): Densidad superficial (kg/m³). Default: 2800
        g_top (float, optional): Gravedad superficial (m/s²)
        
        radiogenic_model (str): 'exponential' o 'layered'. Default: 'exponential'
        
        # Para modelo exponencial:
        A_surface (float): Producción superficial (W/m³). Default: 2.5e-6
        h_r (float): Profundidad característica (m). Default: 10e3
        
        # Para modelo por capas:
        A_upper, A_lower, A_mantle (float): HG por capa (W/m³)
        
        T_max_safe (float): Temperatura máxima segura (K). Default: 2150
        max_iter_T, max_iter_P (int): Iteraciones máximas
        tol_T, tol_P (float): Tolerancias de convergencia
        DEBUG (bool): Imprimir información detallada
        
    Retorna:
        pd.DataFrame con columnas:
            - depth_m: profundidad (m)
            - T_K: temperatura (K)
            - P_Pa: presión (Pa)
            - rho_kg_m3: densidad (kg/m³)
            - q_W_m2: flujo de calor (W/m²)
            - A_W_m3: producción radiogénica (W/m³)
            - lambda_W_mK: conductividad térmica (W/m·K)
            - g_m_s2: gravedad local (m/s²)
            - layer: nombre de la capa ('upper', 'middle', 'lower', 'mantle')
    
    English:
    Calculate geothermal profile T(z), P(z), ρ(z), q(z) using bootstrap method.
    
    Solves 1D steady-state thermal conduction equation:
        d/dz[λ(z,T,P) dT/dz] + A(z) = 0
    
    with iterative T-P-ρ coupling using BurnMan for thermodynamic
    properties consistent with mineralogy.
    
    Referencias / References:
        - Chapman (1986): Bootstrap method
        - Hasterok & Chapman (2011): Continental lithosphere geotherms
        - Turcotte & Schubert (2014): Geodynamics
    """
    # =========================================================================
    # INICIALIZACIÓN Y VALIDACIÓN / INITIALIZATION AND VALIDATION
    # =========================================================================
    
    if composition is None:
        composition = COMPOSITION_DEFAULT
    
    if boundaries is None:
        boundaries = scale_layer_boundaries(R_planet)
    
    d1, d2, d3 = boundaries
    
    # Crear malla de profundidades
    z = np.arange(0.0, z_max + dz, dz)
    nz = len(z)
    
    if DEBUG:
        print("=" * 80)
        print("CÁLCULO DE PERFIL GEOTÉRMICO")
        print("=" * 80)
        print(f"Parámetros:")
        print(f"  Flujo superficial: q_s = {q_s*1000:.2f} mW/m²")
        print(f"  Profundidad máxima: z_max = {z_max/1000:.1f} km")
        print(f"  Paso de integración: dz = {dz:.1f} m")
        print(f"  Número de capas: {nz-1}")
        print(f"  Modelo radiogénico: {radiogenic_model}")
        if radiogenic_model == 'exponential':
            print(f"    A_surface = {A_surface*1e6:.2f} μW/m³, h_r = {h_r/1000:.1f} km")
        print(f"  Fronteras: {d1/1000:.1f}, {d2/1000:.1f}, {d3/1000:.1f} km")
        print("=" * 80 + "\n")
    
    # =========================================================================
    # ARRAYS DE RESULTADOS / RESULT ARRAYS
    # =========================================================================
    
    T_array = np.zeros(nz)      # Temperatura (K)
    P_array = np.zeros(nz)      # Presión (Pa)
    rho_array = np.zeros(nz)    # Densidad (kg/m³)
    q_array = np.zeros(nz)      # Flujo de calor (W/m²)
    g_array = np.zeros(nz)      # Gravedad local (m/s²)
    lambda_array = np.zeros(nz)   # Conductividad térmica efectiva (W/m·K)
    phi_array = np.zeros(nz)    # Porosidad fraccional (0-1)
    layer_names = []            # Nombre de capa por nivel
    
    # Condiciones iniciales
    T_array[0] = T_top
    P_array[0] = P_top
    q_array[0] = q_s
    phi_array[0] = get_porosity(P_top, phi_surface=phi)
    
    rho_array[0] = rho_top * (1 - phi_array[0])
    
    if g_top is None:
        g_array[0] = G * M_total / (R_planet**2)
    else:
        g_array[0] = g_top
    
    comp_modal_0 = normalize_modal_dict(
        get_composition_at_depth(0.0, composition, boundaries)
    )
    
    lambda_fluid = 0.6  # W/m·K, valor típico para fluidos en poros
    lambda_array[0] = (lambda_effective_VRH(comp_modal_0, T_top, P_top) ** (1 - phi_array[0])) * (lambda_fluid ** phi_array[0])
    
    layer_names.append('upper')
    
    # =========================================================================
    # PERFIL DE PRODUCCIÓN RADIOGÉNICA / RADIOGENIC PRODUCTION PROFILE
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
            raise ValueError(f"Modelo radiogénico '{radiogenic_model}' no reconocido")
    
    # =========================================================================
    # VARIABLES DE CONTROL / CONTROL VARIABLES
    # =========================================================================
    
    mass_above = 0.0           # Masa acumulada sobre capa actual (kg)
    stopped_early = False      # Flag de detención anticipada
    last_valid_index = 0       # Último índice válido calculado
    
    # =========================================================================
    # BUCLE PRINCIPAL: INTEGRACIÓN HACIA ABAJO / MAIN LOOP: DOWNWARD INTEGRATION
    # =========================================================================
    
    for i in range(nz - 1):
        # Extraer valores de la capa actual
        Ti = float(T_array[i])
        Pi = float(P_array[i])
        rhoi = float(rho_array[i])
        qi = float(q_array[i])
        gi = float(g_array[i])
        zi = float(z[i])
        Ai = float(A_array[i])
        phii = float(phi_array[i])
        
        # =====================================================================
        # VERIFICACIÓN DE LÍMITES DE TEMPERATURA
        # =====================================================================
        
        if Ti > T_max_safe:
            if DEBUG:
                print(f"\n⚠️  LÍMITE DE TEMPERATURA ALCANZADO")
                print(f"   z = {zi/1000:.2f} km, T = {Ti:.1f} K > {T_max_safe:.1f} K")
                print(f"   BurnMan puede fallar a T más altas.")
                print(f"   Retornando perfil parcial con {i} capas.\n")
            stopped_early = True
            last_valid_index = i
            break
        
        # =====================================================================
        # GEOMETRÍA DE LA CAPA
        # =====================================================================
        
        # Profundidad en el centro de la capa
        z_mid = zi + dz / 2.0
        r_mid = R_planet - z_mid
        
        if r_mid <= 0:
            raise RuntimeError(f"Radio negativo en z = {zi/1000:.2f} km")
        
        # =====================================================================
        # SELECCIONAR COMPOSITE Y COMPOSICIÓN SEGÚN PROFUNDIDAD
        # =====================================================================
        
        if z_mid < d1:
            burn_comp = rocks['upper']
            layer_name = 'upper'
        elif z_mid < d2:
            burn_comp = rocks['middle']
            layer_name = 'middle'
        elif z_mid < d3:
            burn_comp = rocks['lower']
            layer_name = 'lower'
        else:
            burn_comp = rocks['mantle']
            layer_name = 'mantle'
        
        # Obtener composición modal normalizada
        comp_modal_raw = get_composition_at_depth(z_mid, composition, boundaries)
        comp_modal = normalize_modal_dict(comp_modal_raw)
        
        # =====================================================================
        # ESTIMACIONES INICIALES
        # =====================================================================
        
        # Presión inicial (hidrostática con gravedad constante)
        P_mid = Pi + rhoi * gi * dz
        
        # Conductividad térmica inicial
        lambda_init = (lambda_effective_VRH(comp_modal, max(Ti, 298.0), P_mid)
                    ** (1.0 - phii)) * (lambda_fluid ** phii)        
        if not np.isfinite(lambda_init) or lambda_init <= 0:
            raise RuntimeError(f"λ inicial inválida en z = {zi/1000:.2f} km")
        
        # Temperatura inicial (usando ecuación de bootstrap):
        # T_{i+1} = T_i + (q_i/λ) Δz - (A_i/2λ) Δz²
        A_bulk = Ai * (1 - phii)  # Solo la parte sólida produce calor
        T_next = Ti + (qi / lambda_init) * dz - (A_bulk / (2.0 * lambda_init)) * dz * dz
        
        # =====================================================================
        # ITERACIÓN ACOPLADA: T-P-ρ
        # =====================================================================
        
        converged_T = False
        rho_mid = rhoi
        g_mid = gi
        burnman_failed = False
        
        for it_T in range(max_iter_T):
            # Temperatura promedio en la capa
            T_mid = 0.5 * (Ti + T_next)
            
            # -----------------------------------------------------------------
            # VERIFICACIÓN PREVENTIVA: Temperatura excede límite
            # -----------------------------------------------------------------
            if T_next > T_max_safe:
                if DEBUG:
                    print(f"\n⚠️  T calculada ({T_next:.1f} K) > límite en z = {zi/1000:.2f} km")
                    print(f"   Deteniendo cálculo.\n")
                stopped_early = True
                last_valid_index = i
                burnman_failed = True
                break
            
            # -----------------------------------------------------------------
            # ITERACIÓN INTERNA: P-ρ COUPLING
            # -----------------------------------------------------------------
            P_mid_local = P_mid
            
            for it_P in range(max_iter_P):
                # Calcular propiedades termodinámicas con BurnMan
                try:
                    burn_comp.set_state(P_mid_local, T_mid)
                    rho_grain = float(burn_comp.density)
                    phi_actual = get_porosity(P_mid_local, phi_surface=phi)
                    rho_mid = rho_grain * (1 - phi_actual) 
                    
                except Exception as e:
                    if DEBUG:
                        print(f"\n⚠️  BurnMan falló en z = {zi/1000:.2f} km")
                        print(f"   T_mid = {T_mid:.1f} K, P = {P_mid_local/1e9:.3f} GPa")
                        print(f"   Error: {type(e).__name__}")
                        print(f"   Retornando perfil parcial.\n")
                    stopped_early = True
                    last_valid_index = i
                    burnman_failed = True
                    break
                
                # Masa de la capa esférica (shell)
                shell_volume = 4.0 * np.pi * (r_mid**2) * dz
                shell_mass = rho_mid * shell_volume
                
                # Masa encerrada al radio r_mid
                M_enclosed = M_total - mass_above - 0.5 * shell_mass
                M_enclosed = max(M_enclosed, M_total * 1e-12)
                
                # Gravedad local en r_mid
                g_mid = G * M_enclosed / (r_mid**2)
                
                # Presión actualizada (integración hidrostática)
                P_mid_new = Pi + rho_mid * g_mid * dz
                
                # Verificar convergencia en presión
                if abs(P_mid_new - P_mid_local) < tol_P:
                    P_mid_local = P_mid_new
                    break
                
                # Relajación para estabilidad
                P_mid_local = 0.5 * P_mid_local + 0.5 * P_mid_new
            
            # Si BurnMan falló, salir del loop de temperatura
            if burnman_failed:
                break
            
            # -----------------------------------------------------------------
            # ACTUALIZAR CONDUCTIVIDAD TÉRMICA con T_next y P_mid_local
            # -----------------------------------------------------------------
            
            lambda_grain = lambda_effective_VRH(comp_modal, max(T_next, 298.0), P_mid_local)
            
            lambda_eff = (lambda_grain ** (1.0 - phi_actual)) * (lambda_fluid ** phi_actual)
            
            if not np.isfinite(lambda_eff) or lambda_eff <= 0:
                raise RuntimeError(f"λ inválida en z = {zi/1000:.2f} km (iter {it_T})")
            
            # -----------------------------------------------------------------
            # RECALCULAR TEMPERATURA con λ actualizada
            # -----------------------------------------------------------------
            # Bootstrap equation:
            # T_{i+1} = T_i + (q_i/λ) Δz - (A_i/2λ) Δz²
            T_new = Ti + (qi / lambda_eff) * dz - (A_bulk / (2.0 * lambda_eff)) * dz * dz
            
            # -----------------------------------------------------------------
            # DEBUG OUTPUT
            # -----------------------------------------------------------------
            if DEBUG and (i < 6 or i % 500 == 0):
                print(f"[Capa {i:4d}] z={zi/1000:6.2f} km | it_T={it_T:2d} | "
                      f"T={T_new:7.2f} K | P={P_mid_local/1e9:6.3f} GPa | "
                      f"ρ={rho_mid:6.1f} kg/m³ | λ={lambda_eff:5.3f} W/m·K")
            
            # -----------------------------------------------------------------
            # VERIFICAR CONVERGENCIA EN TEMPERATURA
            # -----------------------------------------------------------------
            if abs(T_new - T_next) < tol_T:
                T_next = T_new
                P_mid = P_mid_local
                converged_T = True
                break
            
            # Relajación para siguiente iteración
            T_next = 0.5 * T_next + 0.5 * T_new
            P_mid = P_mid_local
        
        # =====================================================================
        # VERIFICAR FALLO DE BURNMAN
        # =====================================================================
        
        if burnman_failed:
            break
        
        # Advertencia si no convergió
        if not converged_T and DEBUG:
            print(f"⚠️  [WARNING] Capa {i} no convergió en T tras {max_iter_T} iteraciones")
        
        # =====================================================================
        # ACTUALIZAR MASA ACUMULADA
        # =====================================================================
        
        mass_above += shell_mass
        
        # =====================================================================
        # ACTUALIZAR FLUJO DE CALOR
        # =====================================================================
        # El flujo disminuye por producción radiactiva integrada:
        # q_{i+1} = q_i - A_i Δz
        q_next = qi - A_bulk * dz
        
        # =====================================================================
        # GUARDAR RESULTADOS DE LA CAPA
        # =====================================================================
        
        T_array[i+1] = T_next
        P_array[i+1] = P_mid
        rho_array[i+1] = rho_mid
        q_array[i+1] = q_next
        g_array[i+1] = g_mid
        lambda_array[i+1] = lambda_eff
        layer_names.append(layer_name)
        phi_array[i+1] = phi_actual
        last_valid_index = i + 1
    
    # =========================================================================
    # POST-PROCESAMIENTO / POST-PROCESSING
    # =========================================================================
    
    # Si se detuvo anticipadamente, truncar arrays
    if stopped_early:
        z = z[:last_valid_index+1]
        T_array = T_array[:last_valid_index+1]
        P_array = P_array[:last_valid_index+1]
        rho_array = rho_array[:last_valid_index+1]
        q_array = q_array[:last_valid_index+1]
        g_array = g_array[:last_valid_index+1]
        A_array = A_array[:last_valid_index+1]
        lambda_array = lambda_array[:last_valid_index+1]
        phi_array = phi_array[:last_valid_index+1]
        layer_names = layer_names[:last_valid_index+1]
    
    # =========================================================================
    # CREAR DATAFRAME DE RESULTADOS / CREATE RESULTS DATAFRAME
    # =========================================================================
    
    df = pd.DataFrame({
        'depth_m': z,
        'T_K': T_array,
        'P_Pa': P_array,
        'rho_kg_m3': rho_array,
        'q_W_m2': q_array,
        'A_W_m3': A_array,
        'g_m_s2': g_array,
        'lambda_W_mK': lambda_array,
        'phi': phi_array,
        'layer': layer_names
    })
    
    # =========================================================================
    # RESUMEN FINAL / FINAL SUMMARY
    # =========================================================================
    
    if DEBUG:
        print("\n" + "=" * 80)
        print("RESUMEN DEL PERFIL CALCULADO")
        print("=" * 80)
        print(f"Profundidad final: {z[-1]/1000:.2f} km")
        print(f"Temperatura final: {T_array[-1]:.2f} K ({T_array[-1]-273.15:.2f} °C)")
        print(f"Presión final: {P_array[-1]/1e9:.3f} GPa")
        print(f"Densidad final: {rho_array[-1]:.1f} kg/m³")
        print(f"Flujo de calor final: {q_array[-1]*1000:.2f} mW/m²")
        if stopped_early:
            print(f"\n⚠️  CÁLCULO DETENIDO ANTICIPADAMENTE")
            print(f"   Perfil parcial retornado ({len(z)} de {nz} capas)")
        else:
            print(f"\n✅  CÁLCULO COMPLETADO EXITOSAMENTE")
        print("=" * 80 + "\n")
    
    return df


# =============================================================================
# AUXILIARY FUNCTION: PREPARE ROCKS DICT / FUNCIÓN AUXILIAR: PREPARAR ROCKS DICT
# =============================================================================

def prepare_rocks_dict(composition=None, mineral_objects=None, P=1e5, T=288.0):
    """
    Español:
    Prepara el diccionario de Composites de BurnMan para cada capa.
    
    Parámetros:
        composition (dict, optional): Composiciones modales. Default: COMPOSITION_DEFAULT
        mineral_objects (dict, optional): Objetos BurnMan. Si None, se crean con get_mineral_objects()
        P (float): Presión de referencia (Pa)
        T (float): Temperatura de referencia (K)
        
    Retorna:
        dict: {'upper': Composite, 'middle': Composite, 'lower': Composite, 'mantle': Composite}
    
    English:
    Prepare the dictionary of BurnMan Composites for each layer.
    
    Parameters:
        composition (dict, optional): Modal compositions. Default: COMPOSITION_DEFAULT
        mineral_objects (dict, optional): BurnMan objects. If None, created with get_mineral_objects()
        P (float): Reference pressure (Pa)
        T (float): Reference temperature (K)
        
    Returns:
        dict: {'upper': Composite, 'middle': Composite, 'lower': Composite, 'mantle': Composite}
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
# TEMPORAL EVOLUTION MODELS / MODELOS DE EVOLUCIÓN TEMPORAL
# =============================================================================

def q_s_turcotte(t_Ga, q0=65e-3, tau=2.0):
    """
    Español:
    Modelo de flujo de calor temporal de Turcotte & Schubert (2014).
    
    q_s(t) = q₀ · exp(t/τ)
    
    Modelo exponencial simple basado en el decaimiento radiactivo promedio.
    
    Parámetros:
        t_Ga (float or array): Tiempo antes del presente (Ga). Positivo hacia el pasado.
                              Ej: t=0 (presente), t=1 (hace 1 Ga), t=4.5 (formación de la Tierra)
        q0 (float): Flujo de calor superficial actual (W/m²). Default: 65e-3 (65 mW/m²)
        tau (float): Escala de tiempo característica (Ga). Default: 2.0 Ga
                    (tiempo de decaimiento efectivo de elementos radiactivos)
        
    Retorna:
        float or array: Flujo de calor superficial (W/m²)
    
    English:
    Temporal heat flux model from Turcotte & Schubert (2014).
    
    q_s(t) = q₀ · exp(t/τ)
    
    Simple exponential model based on average radioactive decay.
    
    Parameters:
        t_Ga (float or array): Time before present (Ga). Positive towards past.
                              Ex: t=0 (present), t=1 (1 Ga ago), t=4.5 (Earth formation)
        q0 (float): Current surface heat flux (W/m²). Default: 65e-3 (65 mW/m²)
        tau (float): Characteristic time scale (Ga). Default: 2.0 Ga
                    (effective decay time of radioactive elements)
        
    Returns:
        float or array: Surface heat flux (W/m²)
        
    References:
        Turcotte, D. L., & Schubert, G. (2014). Geodynamics (3rd ed.).
        Cambridge University Press.
    """
    return q0 * np.exp(t_Ga / tau)


def A_surface_temporal(t_Ga, A_present=2.5e-6,
                       tau=2.0, t_present=4.5):
    """
    Español:
    Producción radiactiva superficial en función del tiempo,
    válida para pasado, presente y futuro.

    Convención usada (única y consistente):
    - t_Ga = tiempo desde la formación del planeta (Ga)
    - t_present = edad actual del planeta (Ga)
    - A_present = producción radiogénica ACTUAL (presente)

    Definición:
        Δt = t_Ga - t_present

        A(t) = A_present · exp( -Δt / τ )

    De modo que:
        t_Ga < t_present  → pasado  → A > A_present
        t_Ga = t_present  → presente → A = A_present
        t_Ga > t_present  → futuro   → A < A_present

    Parámetros:
        t_Ga (float or array): Tiempo desde la formación (Ga)
        A_present (float): Producción radiogénica superficial actual (W/m³)
                           Default: 2.5e-6 W/m³
        tau (float): Escala temporal efectiva de decaimiento (Ga)
                     Default: 2.0 Ga (dominancia K-40 + U)
        t_present (float): Edad actual del planeta (Ga)
                           Default: 4.5 Ga

    Retorna:
        float or array: Producción radiogénica superficial (W/m³)

    Notas:
        - Compatible con grids que van más allá del presente (futuro).
        - Referenciado directamente a valores observables actuales.
        - En línea con Turcotte & Schubert (2014).
    """
    dt = t_Ga - t_present
    return A_present * np.exp(-dt / tau)

def A_mantle_temporal(t_Ga, A_present=0.015e-6,
                      tau=3.0, t_present=4.5):
    """
    Español:
    Producción radiactiva del manto en función del tiempo,
    válida para pasado, presente y futuro.

    Convención usada:
    - t_Ga = tiempo desde la formación del planeta (Ga)
    - t_present = edad actual del planeta (Ga)
    - A_present = producción radiogénica ACTUAL del manto

    Definición:
        Δt = t_Ga - t_present

        A(t) = A_present · exp( -Δt / τ )

    Parámetros:
        t_Ga (float or array): Tiempo desde la formación (Ga)
        A_present (float): Producción radiogénica del manto actual (W/m³)
                           Default: 0.015 μW/m³
        tau (float): Escala temporal efectiva de decaimiento (Ga)
                     Default: 3.0 Ga (U-238 + Th-232 dominantes)
        t_present (float): Edad actual del planeta (Ga)
                           Default: 4.5 Ga

    Retorna:
        float or array: Producción radiogénica del manto (W/m³)

    Notas:
        - Captura correctamente mayor producción en el Arcaico
          y decaimiento futuro.
        - τ mayor que en la corteza por menor contribución de K-40.
        - Consistente con Hasterok & Chapman (2011),
          Korenaga (2008) y Turcotte & Schubert (2014).
    """
    dt = t_Ga - t_present
    return A_present * np.exp(-dt / tau)


def calculate_geotherm_evolution(rocks, composition,
                                 R_planet=Re, M_total=Me,
                                 z_max=100e3, dz=100.0,
                                 boundaries=None,
                                 T_top=288.0, h_r=10e3,
                                 q0=65e-3, tau=2.0, t_Ga=np.linspace(0.001, 10, 200), qss=None,
                                 T_max_safe=2150.0,
                                 As_type = 'temporal'):
    """
    Español:
    Calcula la evolución temporal del gradiente geotérmico.
    
    Usa el modelo de Turcotte & Schubert (2014) para el flujo de calor
    temporal y calcula perfiles geotérmicos para múltiples épocas.
    
    English:
    Calculate the temporal evolution of the geothermal gradient.
    
    Uses the Turcotte & Schubert (2014) model for temporal heat flux
    and calculates geothermal profiles for multiple epochs.
    
    Parámetros / Parameters:
        rocks (dict): Composites de BurnMan / BurnMan Composites
                     {'upper', 'middle', 'lower', 'mantle'}
        composition (dict): Composiciones modales por capa / Modal compositions by layer
        R_planet (float): Radio del planeta (m) / Planet radius (m). Default: R_Earth
        M_total (float): Masa del planeta (kg) / Planet mass (kg). Default: M_Earth
        z_max (float): Profundidad máxima (m) / Maximum depth (m). Default: 100 km
        dz (float): Paso de profundidad (m) / Depth step (m). Default: 100 m
        boundaries (list, optional): [d1, d2, d3] fronteras de capas (m) / layer boundaries (m)
        T_top (float): Temperatura superficial (K) / Surface temperature (K). Default: 288
        h_r (float): Profundidad característica radiactiva (m) / Radiogenic characteristic depth (m)
                    Default: 10 km
        q0 (float): Flujo de calor actual (W/m²) / Current heat flux (W/m²)
                   Default: 65e-3 (65 mW/m²)
        tau (float): Escala de tiempo (Ga) / Time scale (Ga). Default: 2.0
        t_Ga (array): Tiempos a calcular (Ga) / Times to calculate (Ga)
        T_max_safe (float): Temperatura máxima segura (K) / Maximum safe temperature (K)
                           Default: 2150
        
    Retorna / Returns:
        dict: Diccionario con / Dictionary with:
            - 't_Ga': array de tiempos (Ga) / array of times (Ga)
            - 'q_s': array de flujos de calor (W/m²) / array of heat fluxes (W/m²)
            - 'profiles': lista de DataFrames con perfiles T(z) / list of DataFrames with T(z) profiles
            - 'gradients': array de gradientes superficiales (K/km) / array of surface gradients (K/km)
        
    Notas / Notes:
        Para cada tiempo t / For each time t:
        1. Calcula q_s(t) con modelo de Turcotte / Calculate q_s(t) with Turcotte model
        2. Calcula A_surface(t) / Calculate A_surface(t)
        3. Calcula perfil T(z) con calculate_geotherm() / Calculate T(z) profile with calculate_geotherm()
        4. Extrae gradiente superficial (primeros 1 km) / Extract surface gradient (first 1 km)
        
        Si un perfil se detiene por límites de BurnMan (T > T_max_safe),
        retorna perfil parcial para ese tiempo.
        
        If a profile stops due to BurnMan limits (T > T_max_safe),
        returns partial profile for that time.
    """
    if boundaries is None:
        boundaries = scale_layer_boundaries(R_planet)

    if qss is None:    
        # Calcular flujos de calor con modelo de Turcotte
        q_s = q_s_turcotte(t_Ga, q0=q0, tau=tau)

    else:
        q_s = qss
    
    # Calcular perfiles para cada tiempo
    profiles = []
    gradients = []
    
    print("=" * 80)
    print(f"CALCULANDO EVOLUCIÓN TEMPORAL DEL GRADIENTE GEOTÉRMICO")
    print(f"Modelo: Turcotte & Schubert (2014)")
    print(f"Número de tiempos: {len(t_Ga)}")
    print(f"R_planet = {R_planet/1e6:.3f} x 10^6 m")
    print(f"M_total = {M_total/1e24:.3f} x 10^24 kg")
    print("=" * 80)
    
    for i, (t, q) in enumerate(zip(t_Ga, q_s)):
        # Producción radiactiva para este tiempo
        
        if As_type == 'temporal':
            A_surf = A_surface_temporal(t)
        
        if As_type == 'Hasterok':
            A_surf = A0(q)
        
        A_mant = A_mantle_temporal(t)
        
        # Calcular perfil geotérmico
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
        
        # Calcular gradiente superficial (primeros 1 km)
        idx_1km = np.argmin(np.abs(df['depth_m']/1000 - 1.0))
        dT = df['T_K'].iloc[idx_1km] - df['T_K'].iloc[0]
        dz_km = df['depth_m'].iloc[idx_1km] / 1000
        gradient = dT / dz_km  # K/km
        gradients.append(gradient)
        
        print(f"  t = {t:6.3f} Ga | q_s = {q*1000:6.1f} mW/m² | dT/dz = {gradient:5.1f} K/km")
    
    print("=" * 80)
    print()
    
    return {
        't_Ga': t_Ga,
        'q_s': q_s,
        'profiles': profiles,
        'gradients': np.array(gradients)
    }


# =============================================================================
# USAGE EXAMPLE / EJEMPLO DE USO
# =============================================================================

# =============================================================================
# SUMMARY OF AVAILABLE FUNCTIONS / RESUMEN DE FUNCIONES DISPONIBLES
# =============================================================================
"""
FUNCIONES PRINCIPALES:
=====================

Cálculo de Perfiles:
-------------------
calculate_geotherm()              - Calcula perfil T(z), P(z), ρ(z) para un tiempo
                                   CON detección automática de límites de BurnMan

Evolución Temporal:
------------------
calculate_geotherm_evolution()    - Evolución del gradiente geotérmico en el tiempo
q_s_turcotte()                    - Modelo de flujo de calor temporal
A_surface_temporal()              - Producción radiactiva temporal de la corteza
A_mantle_temporal()               - Producción radiactiva temporal del manto

Preparación de Datos:
--------------------
get_mineral_objects()             - Obtiene objetos minerales de BurnMan
prepare_rocks_dict()              - Crea Composites por capa desde composiciones modales
make_composite_from_modal()       - Convierte composición modal → Composite

Utilidades:
----------
normalize_modal_dict()            - Normaliza composición modal a suma = 1.0
get_composition_at_depth()        - Obtiene composición según profundidad
scale_layer_boundaries()          - Escala fronteras de capas según radio del planeta
radiogenic_heat_profile()         - Perfil A(z) exponencial (Hasterok 2011)

Conductividad Térmica:
---------------------
lambda_lattice()                  - Conductividad fonónica λ(T, P)
lambda_radiative()                - Conductividad fotónica λ(T)
lambda_effective_VRH()            - Conductividad efectiva (Voigt-Reuss-Hill)

Conversiones:
------------
modal_to_mass_fractions()         - Modal (vol) → fracciones de masa
"""


if __name__ == "__main__":
    print("=" * 70)
    print("USAGE EXAMPLE / EJEMPLO DE USO: geotherm_calculator.py")
    print("=" * 70)
    
    # Prepare minerals and composites / Preparar minerales y composites
    print("\n1. Preparing mineral compositions / Preparando composiciones minerales...")
    mineral_objects = get_mineral_objects()
    rocks = prepare_rocks_dict(mineral_objects=mineral_objects)
    print(f"   ✓ Composites created / Composites creados: {list(rocks.keys())}")
    
    # Calculate geothermal profile for Earth / Calcular perfil geotérmico para la Tierra
    print("\n2. Calculating Earth geothermal profile (present) / Calculando perfil geotérmico de la Tierra (presente)...")
    df_geotherm = calculate_geotherm(
        rocks=rocks,
        q_s=65e-3,           # 65 mW/m²
        z_max=300e3,         # 300 km
        dz=100.0,            # 100 m
        R_planet=Re,
        M_total=Me,
        boundaries=[16e3, 23e3, 39e3],
        T_top=288.0,
        P_top=1e5,
        A_surface=2.5e-6,    # 2.5 μW/m³
        h_r=10e3,            # 10 km
        A_mantle=0.015e-6,   # 0.015 μW/m³
        T_max_safe=2150.0,   # Límite de seguridad de BurnMan
        DEBUG=False
    )
    
    print(f"   ✓ Perfil calculado: {len(df_geotherm)} puntos")
    print(f"   Rango de T: {df_geotherm['T_K'].min():.1f} - {df_geotherm['T_K'].max():.1f} K")
    print(f"   Rango de P: {df_geotherm['P_Pa'].min()/1e9:.3f} - {df_geotherm['P_Pa'].max()/1e9:.3f} GPa")
    print(f"   Rango de ρ: {df_geotherm['rho_kg_m3'].min():.1f} - {df_geotherm['rho_kg_m3'].max():.1f} kg/m³")
    
    # Temperatura al Moho
    idx_moho = np.argmin(np.abs(df_geotherm['depth_km'].values - 39.0))
    T_moho = df_geotherm['T_K'].iloc[idx_moho]
    print(f"\n3. Temperatura al Moho (39 km): {T_moho:.1f} K ({T_moho-273:.1f} °C)")
    
    # Gradiente superficial
    depths_10km = df_geotherm['depth_km'].values <= 10.0
    T_10km = df_geotherm[depths_10km]['T_K'].iloc[-1]
    grad_surf = (T_10km - df_geotherm['T_K'].iloc[0]) / 10.0
    print(f"\n4. Gradiente superficial (0-10 km): {grad_surf:.1f} K/km")
    print(f"   Esperado continental: 25-30 K/km")
    
    # LAB (Litosfera-Astenosfera Boundary)
    print("\n5. Buscando LAB (Límite Litosfera-Astenosfera)...")
    for T_lab in [1350, 1450, 1573]:
        idx = np.where(df_geotherm['T_K'].values >= T_lab)[0]
        if len(idx) > 0:
            lab_depth = df_geotherm['depth_km'].iloc[idx[0]]
            print(f"   LAB a T={T_lab}K: {lab_depth:.1f} km")
    
    print("\n" + "=" * 70)
    print("EJEMPLO ADICIONAL: Evolución temporal del gradiente geotérmico")
    print("=" * 70)
    
    # Calcular evolución temporal (0-1 Ga para evitar problemas con BurnMan)
    print("\n6. Calculando evolución temporal del gradiente (0-1 Ga)...")
    
    # Nota: Usar composition desde COMPOSITION_DEFAULT
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
        n_times=5,
        t_range=(0.001, 1.0)
    )
    
    print("\n   Resultados de evolución temporal:")
    print(f"   Tiempos calculados: {len(results['t_Ga'])}")
    print(f"   Flujo de calor: {results['q_s'][0]*1000:.1f} - {results['q_s'][-1]*1000:.1f} mW/m²")
    print(f"   Gradiente: {results['gradients'][0]:.1f} - {results['gradients'][-1]:.1f} K/km")
    
    print("\n" + "=" * 70)
    print("✅ Módulo geotherm_calculator.py funcional")
    print("   - Detección automática de límites de BurnMan")
    print("   - Perfiles parciales cuando T > T_max_safe")
    print("   - Modelos de evolución temporal incluidos")
    print("=" * 70)
