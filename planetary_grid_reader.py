"""
Español:

Módulo para extraer información del Grid de Planetas (PlanetaryGrid)
Este módulo contiene funciones para:
  - Leer archivos STRUC.dat y TEVOL.dat del grid de planetas
  - Calcular propiedades planetarias (gravedad, flujo de calor, etc.)
  - Procesar modelos completos de planetas según CMF e IMF

English:

    Module to extract information from the Planetary Grid (PlanetaryGrid)
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
# PHYSICAL CONSTANTS / CONSTANTES FÍSICAS
# =============================================================================
G = constants.G.value  # Gravitational constant / Constante gravitacional
Me = constants.M_earth.value  # Earth mass (kg) / Masa de la Tierra (kg)
Re = constants.R_earth.value  # Earth radius (m) / Radio de la Tierra (m)


# =============================================================================
# FILE READING FUNCTIONS / FUNCIONES DE LECTURA DE ARCHIVOS
# =============================================================================

def parse_norm(header_lines):
    """
    Español:
    Extrae el diccionario #norm={...} si existe en el header.
    
    Parámetros:
        header_lines (list of str): Líneas de encabezado del archivo
    
    Retorna:
        dict: Diccionario con valores de normalización o {} si no existe
    
    English:
    Extract the #norm={...} dictionary if it exists in the header.
    
    Parameters:
        header_lines (list of str): Header lines from the file
        
    Returns:
        dict: Dictionary with normalization values or {} if it doesn't exist
    """
    for L in header_lines:
        if L.strip().startswith('#norm='):
            txt = L.strip()[len("#norm="):]
            try:
                norm = eval(txt)
                return norm
            except Exception as e:
                print(f"No pude evaluar #norm: {e}")
    return {}


def read_struc_dat(path):
    """
    Español:
    Lee archivo STRUC.dat del grid planetario.
    
    Parámetros:
        path (str): Ruta al archivo STRUC.dat
        
    Retorna:
        pd.DataFrame: DataFrame con columnas: ur, r, mr, rho, P, g, phi, T, composition
        Atributos adicionales:
        - df.attrs['header']: líneas de encabezado
        - df.attrs['norm']: diccionario de normalización
        - df.attrs['composition']: fracciones de capas
    
    English:
    Read STRUC.dat file from the planetary grid.
    
    Parameters:
        path (str): Path to STRUC.dat file
        
    Returns:
        pd.DataFrame: DataFrame with columns: ur, r, mr, rho, P, g, phi, T, composition
        Additional attributes:
        - df.attrs['header']: header lines
        - df.attrs['norm']: normalization dictionary
        - df.attrs['composition']: layer fractions
    """
    header = []
    data_lines = []
    layers = {}
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header.append(line.rstrip("\n"))
                # Extract layer information / Extraer información de capas
                if line.startswith('#layer'):
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[1]
                        frac = float(parts[2])
                        layers[name] = frac
            elif line.strip() != '':
                data_lines.append(line)
    
    # Load numerical data / Cargar datos numéricos
    data = np.loadtxt(data_lines)
    cols = ['ur', 'r', 'mr', 'rho', 'P', 'g', 'phi', 'T', 'composition']
    df = pd.DataFrame(data, columns=cols)
    
    # Add metadata as attributes / Agregar metadatos como atributos
    df.attrs['header'] = header
    df.attrs['norm'] = parse_norm(header)
    df.attrs['composition'] = layers
    
    return df


def read_tevol_dat(path):
    """
    Español:
    Lee archivo TEVOL.dat del grid planetario.
    
    Parámetros:
        path (str): Ruta al archivo TEVOL.dat
        
    Retorna:
        pd.DataFrame: DataFrame con datos de evolución térmica
        Columnas típicas: t, Qconv, Ri, R*, Qc, Qm, Qr, Tcmb, Tl, Tup, RiFlag, Bs
    
    English:
    Read TEVOL.dat file from the planetary grid.
    
    Parameters:
        path (str): Path to TEVOL.dat file
        
    Returns:
        pd.DataFrame: DataFrame with thermal evolution data
        Typical columns: t, Qconv, Ri, R*, Qc, Qm, Qr, Tcmb, Tl, Tup, RiFlag, Bs
    """
    # Read header line / Leer línea de encabezado
    header_line = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_line = line.lstrip("\n").strip()
                break
    
    if header_line is None:
        raise Exception(f"No se encontró línea header que empiece con '#' en {path}")
    
    # Process column names (remove units in []) / Procesar nombres de columnas (remover unidades entre [])
    cols = [re.sub(r'\[.*?\]', '', c).strip() 
            for c in re.split(r'\s+', header_line) 
            if c.strip() != '']
    
    # Try reading with pandas / Intentar leer con pandas
    try:
        df = pd.read_csv(path, comment='#', sep='\s+', header=None, 
                        names=cols, engine='python')
        return df
    except Exception as e:
        # Fallback: use numpy / usar numpy
        data = np.genfromtxt(path, comments='#', invalid_raise=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(cols):
            raise ValueError(
                f"Lectura fallida: datos con forma {data.shape} pero "
                f"{len(cols)} columnas esperadas. Error: {e}"
            )
        df = pd.DataFrame(data, columns=cols)
        return df


# =============================================================================
# PROPERTY CALCULATION FUNCTIONS / FUNCIONES DE CÁLCULO DE PROPIEDADES
# =============================================================================

def gravity_profile(path_struc):
    """
    Español:
    Calcula el perfil de gravedad g(r) para un planeta.
    
    Parámetros:
        path_struc (str): Ruta al archivo STRUC.dat
        
    Retorna:
        np.ndarray: Array con valores de gravedad (m/s²) en cada punto radial
    
    English:
    Calculate the gravity profile g(r) for a planet.
    
    Parameters:
        path_struc (str): Path to STRUC.dat file
        
    Returns:
        np.ndarray: Array with gravity values (m/s²) at each radial point
    """
    data_struc = read_struc_dat(path_struc)
    rs = np.array(data_struc.r)
    mrs = np.array(data_struc.mr)
    
    # Convert to SI units / Convertir a unidades SI
    Rps = Re * rs  # m
    Mps = Me * mrs  # kg
    
    gs = np.zeros_like(rs)
    nz = rs > 0
    gs[nz] = G * Mps[nz] / Rps[nz]**2
    
    return gs


def get_radius(path_struc):
    """
    Español:
    Obtiene el radio del planeta.
    
    Parámetros:
        path_struc (str): Ruta al archivo STRUC.dat
        
    Retorna:
        float or None: Radio en unidades de R_Earth, o None si falla
    
    English:
    Get the planet's radius.
    
    Parameters:
        path_struc (str): Path to STRUC.dat file
        
    Returns:
        float or None: Radius in R_Earth units, or None if it fails
    """
    data_struc = read_struc_dat(path_struc)
    norm = data_struc.attrs['norm']
    
    # Try to get from norm dictionary / Intentar obtener del diccionario norm
    if 'R' in norm:
        try:
            return norm['R']
        except Exception:
            pass
    
    # Fallback: last value of column r / último valor de la columna r
    try:
        r = data_struc.r.iloc[-1]
        return r
    except Exception:
        return None


def get_mass(path_struc):
    """
    Español:
    Obtiene la masa del planeta.
    
    Parámetros:
        path_struc (str): Ruta al archivo STRUC.dat
        
    Retorna:
        float or None: Masa en unidades de M_Earth, o None si falla
    
    English:
    Get the planet's mass.
    
    Parameters:
        path_struc (str): Path to STRUC.dat file
        
    Returns:
        float or None: Mass in M_Earth units, or None if it fails
    """
    data_struc = read_struc_dat(path_struc)
    norm = data_struc.attrs['norm']
    
    # Try to get from norm dictionary / Intentar obtener del diccionario norm
    if 'M' in norm:
        try:
            return norm['M']
        except Exception:
            pass
    
    # Fallback: last value of column mr / último valor de la columna mr
    try:
        m = data_struc.mr.iloc[-1]
        return m
    except Exception:
        return None

def get_surface_heat_flux(path_struc, path_tevol):
    """
    Español:
    Calcula el flujo de calor superficial q [W/m²].
    
    q = Q_m / (4π R²)
    
    Parámetros:
        path_struc (str): Ruta al archivo STRUC.dat
        path_tevol (str): Ruta al archivo TEVOL.dat
        
    Retorna:
        float: Flujo de calor superficial en W/m²
    
    English:
    Calculate the surface heat flux q [W/m²].
    
    q = Q_m / (4π R²)
    
    Parameters:
        path_struc (str): Path to STRUC.dat file
        path_tevol (str): Path to TEVOL.dat file
        
    Returns:
        float: Surface heat flux in W/m²
    """
    data_struc = read_struc_dat(path_struc)
    data_tevol = read_tevol_dat(path_tevol)
    
    # Planet radius in meters / Radio del planeta en metros
    Rps = np.array(data_struc.r) * Re
    R_planet = Rps[-1]
    
    # Mantle heat flux (last temporal column) / Flujo de calor del manto (última columna temporal)
    Qs = np.array(data_tevol.Qm)
    
    
    # Surface area / Área superficial
    A = 4 * np.pi * R_planet**2
    
    # Surface heat flux / Flujo de calor superficial
    qs = Qs[-1] / A
    
    return qs



def get_CMF_IMF(modelname):
    """
    Español:
    Extrae los valores de CMF e IMF del nombre de un modelo.
    
    Parámetros:
        modelname (str): Nombre del modelo (ej: 'CMF_0.30-IMF_0.10')
        
    Retorna:
        tuple of (float, float) or (None, None): (CMF, IMF) o (None, None) si no coincide el patrón
    
    English:
    Extract CMF and IMF values from a model name.
    
    Parameters:
        modelname (str): Model name (e.g.: 'CMF_0.30-IMF_0.10')
        
    Returns:
        tuple of (float, float) or (None, None): (CMF, IMF) or (None, None) if pattern doesn't match
    """
    model = re.match(r'^CMF_([0-9.+-eE]+)-IMF_([0-9.+-eE]+)$', modelname)
    if model:
        try:
            return float(model.group(1)), float(model.group(2))
        except Exception:
            return None, None
    return None, None


# =============================================================================
# MAIN FUNCTION: PROCESS COMPLETE MODEL / FUNCIÓN PRINCIPAL: PROCESAR MODELO COMPLETO
# =============================================================================

def process_planet_model(model_folder):
    """
    Español:
    Procesa todos los archivos de un modelo planetario (carpeta CMF_X-IMF_Y).
    
    Parámetros:
        model_folder (str): Ruta a la carpeta del modelo (ej: 'PlanetaryGrid/CMF_0.30-IMF_0.10')
        
    Retorna:
        pd.DataFrame: DataFrame con una fila por cada masa del modelo, columnas:
        - Mp: masa del planeta [M_Earth]
        - Rp: radio del planeta [R_Earth]
        - P_surf: presión superficial [Pa]
        - rho_surf: densidad superficial [kg/m³]
        - T_surf: temperatura superficial [K]
        - g_surf: gravedad superficial [m/s²]
        - q_surf: flujo de calor superficial [W/m²]
        
        Atributos del DataFrame:
        - df.attrs['cmf']: Core Mass Fraction
        - df.attrs['imf']: Ice Mass Fraction
        - df.attrs['mmf']: Mantle Mass Fraction (1 - CMF - IMF)
    
    English:
    Process all files from a planetary model (CMF_X-IMF_Y folder).
    
    Parameters:
        model_folder (str): Path to model folder (e.g.: 'PlanetaryGrid/CMF_0.30-IMF_0.10')
        
    Returns:
        pd.DataFrame: DataFrame with one row per model mass, columns:
        - Mp: planet mass [M_Earth]
        - Rp: planet radius [R_Earth]
        - P_surf: surface pressure [Pa]
        - rho_surf: surface density [kg/m³]
        - T_surf: surface temperature [K]
        - g_surf: surface gravity [m/s²]
        - q_surf: surface heat flux [W/m²]
        
        DataFrame attributes:
        - df.attrs['cmf']: Core Mass Fraction
        - df.attrs['imf']: Ice Mass Fraction
        - df.attrs['mmf']: Mantle Mass Fraction (1 - CMF - IMF)
    """
    model_folder = os.path.abspath(model_folder)
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"No existe carpeta: {model_folder}")
    
    # Extract CMF and IMF from folder name / Extraer CMF e IMF del nombre de la carpeta
    cmf, imf = get_CMF_IMF(os.path.basename(model_folder))
    
    # List files / Listar archivos
    files = sorted(os.listdir(model_folder))
    STRUC_files = [f for f in files if f.endswith('STRUC.dat')]
    TEVOL_files = [f for f in files if f.endswith('TEVOL.dat')]
    
    if len(TEVOL_files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos TEVOL.dat en {model_folder}")
    
    rows = []
    _rx_M = re.compile(r'^M([0-9.+-eE]+)-')
    
    for tfile in TEVOL_files:
        TEVOL_path = os.path.join(model_folder, tfile)
        
        # Extract mass from filename / Extraer masa del nombre del archivo
        base = os.path.basename(tfile)
        m_match = re.search(_rx_M, base)
        if not m_match:
            continue
        m = float(m_match.group(1))
        
        # Find corresponding STRUC file / Buscar archivo STRUC correspondiente
        matching_struc = None
        for sfile in STRUC_files:
            if sfile.startswith(f'M{m:0.2f}-') or sfile.startswith(f'M{m}-'):
                matching_struc = sfile
                break
        
        if matching_struc is None:
            print(f"Warning: No se encontró archivo STRUC.dat para M={m} en {model_folder}")
            continue
        
        STRUC_path = os.path.join(model_folder, matching_struc)
        
        # Read data / Leer datos
        data_struc = read_struc_dat(STRUC_path)
        data_tevol = read_tevol_dat(TEVOL_path)
        
        # Extract properties / Extraer propiedades
        Mp = get_mass(STRUC_path)
        Rp = get_radius(STRUC_path)
        
        
        try:
            P_surf = np.array(data_struc['P'])[-1]
            rho_surf = np.array(data_struc['rho'])[-1]
            T_surf = np.array(data_struc['T'])[-1]
            g_surf = np.array(data_struc['g'])[-1]
        except:
            P_surf = None
            rho_surf = None
            T_surf = None
            g_surf = None
        
        try:
            qs = get_surface_heat_flux(STRUC_path, TEVOL_path)
        except:
            qs = None
        
        # Add row / Agregar fila
        rows.append({
            'Mp': Mp,
            'Rp': Rp,
            'P_surf': P_surf,
            'rho_surf': rho_surf,
            'T_surf': T_surf,
            'g_surf': g_surf,
            'q_surf': qs,
        })
    
    # Create DataFrame / Crear DataFrame
    df_out = pd.DataFrame(rows)
    df_out.attrs['cmf'] = cmf
    df_out.attrs['imf'] = imf
    df_out.attrs['mmf'] = 1 - cmf - imf if (cmf is not None and imf is not None) else None
    
    return df_out


# =============================================================================
# AUXILIARY FUNCTION: PROCESS MULTIPLE MODELS / FUNCIÓN AUXILIAR: PROCESAR MÚLTIPLES MODELOS
# =============================================================================

def process_all_models(planetary_grid_path, imf_filter=None):
    """
    Español:
    Procesa todos los modelos en el directorio PlanetaryGrid.
    
    Parámetros:
        planetary_grid_path (str): Ruta al directorio principal con carpetas CMF_X-IMF_Y
        imf_filter (float, optional): Si se especifica, solo procesa modelos con este valor de IMF
        
    Retorna:
        pd.DataFrame: DataFrame combinado con todos los modelos, incluyendo columna 'CMF'
    
    English:
    Process all models in the PlanetaryGrid directory.
    
    Parameters:
        planetary_grid_path (str): Path to main directory with CMF_X-IMF_Y folders
        imf_filter (float, optional): If specified, only process models with this IMF value
        
    Returns:
        pd.DataFrame: Combined DataFrame with all models, including 'CMF' column
    """
    folders = [f for f in sorted(os.listdir(planetary_grid_path)) 
               if f.startswith("CMF_")]
    
    all_data = []
    
    for folder in folders:
        cmf, imf = get_CMF_IMF(folder)
        
        # Filter by IMF if specified / Filtrar por IMF si se especificó
        if imf_filter is not None and imf != imf_filter:
            continue
        
        folder_path = os.path.join(planetary_grid_path, folder)
        print(f"Procesando: {folder_path}")
        
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
        print("No se encontraron datos")
        return pd.DataFrame()


# =============================================================================
# USAGE EXAMPLE / EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO DE USO: planetary_grid_reader.py")
    print("=" * 70)
    
    # Path to planetary grid / Ruta al grid de planetas
    planetary_grid_path = "PlanetaryGrid"
    
    # Example 1: Read a specific file / Ejemplo 1: Leer un archivo específico
    print("\n1. Leyendo archivo STRUC.dat específico...")
    path_struc = os.path.join(planetary_grid_path, "CMF_0.30-IMF_0.00", "M1.00-STRUC.dat")
    if os.path.exists(path_struc):
        data = read_struc_dat(path_struc)
        print(f"   Columnas: {list(data.columns)}")
        print(f"   Radio: {get_radius(path_struc):.3f} R_Earth")
        print(f"   Masa: {get_mass(path_struc):.3f} M_Earth")
    else:
        print(f"   Archivo no encontrado: {path_struc}")
    
    # Example 2: Process a complete model / Ejemplo 2: Procesar un modelo completo
    print("\n2. Processing complete model CMF_0.30-IMF_0.00... / Procesando modelo completo CMF_0.30-IMF_0.00...")
    model_folder = os.path.join(planetary_grid_path, "CMF_0.30-IMF_0.00")
    if os.path.exists(model_folder):
        df_model = process_planet_model(model_folder)
        print(f"   Planetas procesados: {len(df_model)}")
        print(f"   CMF: {df_model.attrs['cmf']}")
        print(f"   IMF: {df_model.attrs['imf']}")
        print("\n   Primeros 3 planetas:")
        print(df_model.head(3))
    
    # Example 3: Process all models with IMF=0.00 / Ejemplo 3: Procesar todos los modelos con IMF=0.00
    print("\n3. Processing all models with IMF=0.00... / Procesando todos los modelos con IMF=0.00...")
    if os.path.exists(planetary_grid_path):
        df_all = process_all_models(planetary_grid_path, imf_filter=0.00)
        if not df_all.empty:
            print(f"   Total de planetas: {len(df_all)}")
            print(f"   Rango de masas: {df_all['Mp'].min():.2f} - {df_all['Mp'].max():.2f} M_Earth")
            print(f"   Rango de qs: {df_all['qs'].min()*1000:.1f} - {df_all['qs'].max()*1000:.1f} mW/m²")
    
    print("\n" + "=" * 70)
