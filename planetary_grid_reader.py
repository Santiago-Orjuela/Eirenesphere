"""
Module to extract information from the Planetary Grid (PlanetaryGrid).

This module contains functions to:
  - Read STRUC.dat and TEVOL.dat files from the planetary grid
  - Calculate planetary properties (gravity, heat flux, etc.)
  - Process complete planet models according to CMF and IMF
"""

import re
import os
import numpy as np
import pandas as pd
from astropy import constants

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
G = constants.G.value          # Gravitational constant
Me = constants.M_earth.value   # Earth mass (kg)
Re = constants.R_earth.value   # Earth radius (m)


# =============================================================================
# FILE READING FUNCTIONS
# =============================================================================

def parse_norm(header_lines):
    """
    Extract the #norm={...} dictionary if it exists in the header.

    Parameters
    ----------
    header_lines : list of str
        Header lines from the file.

    Returns
    -------
    dict
        Dictionary with normalization values, or {} if not found.
    """
    for L in header_lines:
        if L.strip().startswith('#norm='):
            txt = L.strip()[len("#norm="):]
            try:
                norm = eval(txt)
                return norm
            except Exception as e:
                print(f"Could not evaluate #norm: {e}")
    return {}


def read_struc_dat(path):
    """
    Read a STRUC.dat file from the planetary grid.

    Parameters
    ----------
    path : str
        Path to the STRUC.dat file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ur, r, mr, rho, P, g, phi, T, composition.
        Additional attributes:
          - df.attrs['header']: header lines
          - df.attrs['norm']: normalization dictionary
          - df.attrs['composition']: layer mass fractions
    """
    header = []
    data_lines = []
    layers = {}

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header.append(line.rstrip("\n"))
                # Extract layer information
                if line.startswith('#layer'):
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[1]
                        frac = float(parts[2])
                        layers[name] = frac
            elif line.strip() != '':
                data_lines.append(line)

    # Load numerical data
    data = np.loadtxt(data_lines)
    cols = ['ur', 'r', 'mr', 'rho', 'P', 'g', 'phi', 'T', 'composition']
    df = pd.DataFrame(data, columns=cols)

    # Add metadata as attributes
    df.attrs['header'] = header
    df.attrs['norm'] = parse_norm(header)
    df.attrs['composition'] = layers

    return df


def read_tevol_dat(path):
    """
    Read a TEVOL.dat file from the planetary grid.

    Parameters
    ----------
    path : str
        Path to the TEVOL.dat file.

    Returns
    -------
    pd.DataFrame
        DataFrame with thermal evolution data.
        Typical columns: t, Qconv, Ri, R*, Qc, Qm, Qr, Tcmb, Tl, Tup, RiFlag, Bs
    """
    # Read header line
    header_line = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_line = line.lstrip("\n").strip()
                break

    if header_line is None:
        raise Exception(f"No header line starting with '#' found in {path}")

    # Process column names (remove units in [])
    cols = [re.sub(r'\[.*?\]', '', c).strip()
            for c in re.split(r'\s+', header_line)
            if c.strip() != '']

    # Try reading with pandas
    try:
        df = pd.read_csv(path, comment='#', sep=r'\s+', header=None,
                         names=cols, engine='python')
        return df
    except Exception as e:
        # Fallback: use numpy
        data = np.genfromtxt(path, comments='#', invalid_raise=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(cols):
            raise ValueError(
                f"Failed to read: data with shape {data.shape} but "
                f"{len(cols)} columns expected. Error: {e}"
            )
        df = pd.DataFrame(data, columns=cols)
        return df


# =============================================================================
# PROPERTY CALCULATION FUNCTIONS
# =============================================================================

def gravity_profile(path_struc):
    """
    Calculate the gravity profile g(r) for a planet.

    Parameters
    ----------
    path_struc : str
        Path to the STRUC.dat file.

    Returns
    -------
    np.ndarray
        Array with gravity values (m/s²) at each radial point.
    """
    data_struc = read_struc_dat(path_struc)
    rs = np.array(data_struc.r)
    mrs = np.array(data_struc.mr)

    # Convert to SI units
    Rps = Re * rs   # m
    Mps = Me * mrs  # kg

    gs = np.zeros_like(rs)
    nz = rs > 0
    gs[nz] = G * Mps[nz] / Rps[nz]**2

    return gs


def get_radius(path_struc):
    """
    Get the planet's radius.

    Parameters
    ----------
    path_struc : str
        Path to the STRUC.dat file.

    Returns
    -------
    float or None
        Radius in R_Earth units, or None if it fails.
    """
    data_struc = read_struc_dat(path_struc)
    norm = data_struc.attrs['norm']

    # Try to get from norm dictionary
    if 'R' in norm:
        try:
            return norm['R']
        except Exception:
            pass

    # Fallback: last value of column r
    try:
        r = data_struc.r.iloc[-1]
        return r
    except Exception:
        return None


def get_mass(path_struc):
    """
    Get the planet's mass.

    Parameters
    ----------
    path_struc : str
        Path to the STRUC.dat file.

    Returns
    -------
    float or None
        Mass in M_Earth units, or None if it fails.
    """
    data_struc = read_struc_dat(path_struc)
    norm = data_struc.attrs['norm']

    # Try to get from norm dictionary
    if 'M' in norm:
        try:
            return norm['M']
        except Exception:
            pass

    # Fallback: last value of column mr
    try:
        m = data_struc.mr.iloc[-1]
        return m
    except Exception:
        return None


def get_surface_heat_flux(path_struc, path_tevol):
    """
    Calculate the surface heat flux q [W/m²].

    q = Q_m / (4π R²)

    Parameters
    ----------
    path_struc : str
        Path to the STRUC.dat file.
    path_tevol : str
        Path to the TEVOL.dat file.

    Returns
    -------
    float
        Surface heat flux in W/m².
    """
    data_struc = read_struc_dat(path_struc)
    data_tevol = read_tevol_dat(path_tevol)

    # Planet radius in meters
    Rps = np.array(data_struc.r) * Re
    R_planet = Rps[-1]

    # Mantle heat flux (last temporal column)
    Qs = np.array(data_tevol.Qm)

    # Surface area
    A = 4 * np.pi * R_planet**2

    # Surface heat flux
    qs = Qs[-1] / A

    return qs


def get_CMF_IMF(modelname):
    """
    Extract CMF and IMF values from a model name.

    Parameters
    ----------
    modelname : str
        Model name (e.g.: 'CMF_0.30-IMF_0.10').

    Returns
    -------
    tuple of (float, float) or (None, None)
        (CMF, IMF), or (None, None) if the pattern does not match.
    """
    model = re.match(r'^CMF_([0-9.+-eE]+)-IMF_([0-9.+-eE]+)$', modelname)
    if model:
        try:
            return float(model.group(1)), float(model.group(2))
        except Exception:
            return None, None
    return None, None


# =============================================================================
# MAIN FUNCTION: PROCESS COMPLETE MODEL
# =============================================================================

def process_planet_model(model_folder):
    """
    Process all files from a planetary model folder (CMF_X-IMF_Y).

    Parameters
    ----------
    model_folder : str
        Path to model folder (e.g.: 'PlanetaryGrid/CMF_0.30-IMF_0.00').

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per model mass. Columns:
          - Mp        : planet mass [M_Earth]
          - Rp        : planet radius [R_Earth]
          - P_surf    : surface pressure [Pa]
          - rho_surf  : surface density [kg/m³]
          - T_surf    : surface temperature [K]
          - g_surf    : surface gravity [m/s²]
          - q_surf    : surface heat flux [W/m²]

        DataFrame attributes:
          - df.attrs['cmf']: Core Mass Fraction
          - df.attrs['imf']: Ice Mass Fraction
          - df.attrs['mmf']: Mantle Mass Fraction (1 - CMF - IMF)
    """
    model_folder = os.path.abspath(model_folder)
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"Folder not found: {model_folder}")

    # Extract CMF and IMF from folder name
    cmf, imf = get_CMF_IMF(os.path.basename(model_folder))

    # List files
    files = sorted(os.listdir(model_folder))
    STRUC_files = [f for f in files if f.endswith('STRUC.dat')]
    TEVOL_files = [f for f in files if f.endswith('TEVOL.dat')]

    if len(TEVOL_files) == 0:
        raise FileNotFoundError(f"No TEVOL.dat files found in {model_folder}")

    rows = []
    _rx_M = re.compile(r'^M([0-9.+-eE]+)-')

    for tfile in TEVOL_files:
        TEVOL_path = os.path.join(model_folder, tfile)

        # Extract mass from filename
        base = os.path.basename(tfile)
        m_match = re.search(_rx_M, base)
        if not m_match:
            continue
        m = float(m_match.group(1))

        # Find corresponding STRUC file
        matching_struc = None
        for sfile in STRUC_files:
            if sfile.startswith(f'M{m:0.2f}-') or sfile.startswith(f'M{m}-'):
                matching_struc = sfile
                break

        if matching_struc is None:
            print(f"Warning: STRUC.dat not found for M={m} in {model_folder}")
            continue

        STRUC_path = os.path.join(model_folder, matching_struc)

        # Read data
        data_struc = read_struc_dat(STRUC_path)
        data_tevol = read_tevol_dat(TEVOL_path)

        # Extract properties
        Mp = get_mass(STRUC_path)
        Rp = get_radius(STRUC_path)

        try:
            P_surf   = np.array(data_struc['P'])[-1]
            rho_surf = np.array(data_struc['rho'])[-1]
            T_surf   = np.array(data_struc['T'])[-1]
            g_surf   = np.array(data_struc['g'])[-1]
        except Exception:
            P_surf = rho_surf = T_surf = g_surf = None

        try:
            qs = get_surface_heat_flux(STRUC_path, TEVOL_path)
        except Exception:
            qs = None

        # Add row
        rows.append({
            'Mp':       Mp,
            'Rp':       Rp,
            'P_surf':   P_surf,
            'rho_surf': rho_surf,
            'T_surf':   T_surf,
            'g_surf':   g_surf,
            'q_surf':   qs,
        })

    # Create DataFrame
    df_out = pd.DataFrame(rows)
    df_out.attrs['cmf'] = cmf
    df_out.attrs['imf'] = imf
    df_out.attrs['mmf'] = 1 - cmf - imf if (cmf is not None and imf is not None) else None

    return df_out


# =============================================================================
# AUXILIARY FUNCTION: PROCESS MULTIPLE MODELS
# =============================================================================

def process_all_models(planetary_grid_path, imf_filter=None):
    """
    Process all models in the PlanetaryGrid directory.

    Parameters
    ----------
    planetary_grid_path : str
        Path to the main directory containing CMF_X-IMF_Y sub-folders.
    imf_filter : float, optional
        If specified, only process models with this IMF value.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all models, including 'CMF' and 'IMF' columns.
    """
    folders = [f for f in sorted(os.listdir(planetary_grid_path))
               if f.startswith("CMF_")]

    all_data = []

    for folder in folders:
        cmf, imf = get_CMF_IMF(folder)

        # Filter by IMF if specified
        if imf_filter is not None and imf != imf_filter:
            continue

        folder_path = os.path.join(planetary_grid_path, folder)
        print(f"Processing: {folder_path}")

        try:
            df_model = process_planet_model(folder_path)
            df_model['CMF'] = cmf
            df_model['IMF'] = imf
            all_data.append(df_model)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No data found")
        return pd.DataFrame()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("USAGE EXAMPLE: planetary_grid_reader.py")
    print("=" * 70)

    planetary_grid_path = "PlanetaryGrid"

    # Example 1: Read a specific STRUC.dat file
    print("\n1. Reading a specific STRUC.dat file...")
    path_struc = os.path.join(planetary_grid_path, "CMF_0.30-IMF_0.00", "M1.00-STRUC.dat")
    if os.path.exists(path_struc):
        data = read_struc_dat(path_struc)
        print(f"   Columns: {list(data.columns)}")
        print(f"   Radius: {get_radius(path_struc):.3f} R_Earth")
        print(f"   Mass:   {get_mass(path_struc):.3f} M_Earth")
    else:
        print(f"   File not found: {path_struc}")

    # Example 2: Process a complete model
    print("\n2. Processing complete model CMF_0.30-IMF_0.00...")
    model_folder = os.path.join(planetary_grid_path, "CMF_0.30-IMF_0.00")
    if os.path.exists(model_folder):
        df_model = process_planet_model(model_folder)
        print(f"   Planets processed: {len(df_model)}")
        print(f"   CMF: {df_model.attrs['cmf']}")
        print(f"   IMF: {df_model.attrs['imf']}")
        print("\n   First 3 planets:")
        print(df_model.head(3))

    # Example 3: Process all models with IMF=0.00
    print("\n3. Processing all models with IMF=0.00...")
    if os.path.exists(planetary_grid_path):
        df_all = process_all_models(planetary_grid_path, imf_filter=0.00)
        if not df_all.empty:
            print(f"   Total planets: {len(df_all)}")
            print(f"   Mass range: {df_all['Mp'].min():.2f} - {df_all['Mp'].max():.2f} M_Earth")
            print(f"   q_surf range: {df_all['q_surf'].min()*1000:.1f} - {df_all['q_surf'].max()*1000:.1f} mW/m²")

    print("\n" + "=" * 70)
