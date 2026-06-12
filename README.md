# Inside the Eirenesphere: The Interplay of Porosity, Heat Flux and Mineralogy in Exoplanetary Aquable and Habitable Layers

**Authors:** 
Santiago Andres Orjuela Montealegre
Jorge Ivan Zuluaga Callejas

Repository of the study of the **habitability and subsurface Aquability** of Earth-like planets and super-Earths, integrating models of internal structure, thermal evolution and physical criteria for the stability of liquid water.

## Scripts

### `planetary_grid_reader.py`

**Description:**
Python module designed to read, process, and organize planetary interior models generated from the **Planetary Grid** simulations. It provides utilities for extracting structural and thermal information from model outputs and preparing them for subsequent habitability analyses.

**Main Features:**

* Reading structural and thermal model files:

  * `STRUC.dat`: radial internal structure of the planet.
  * `TEVOL.dat`: thermal evolution history.
* Automatic extraction of metadata from file headers:

  * Normalization parameters (`#norm={...}`).
  * Layer mass fractions (core, mantle, ice).
* Computation of planetary properties:

  * Gravity profile ( g(r) ).
  * Planetary mass and radius.
  * Surface heat flux.
* Systematic processing of planetary models organized by:

  * **CMF** (Core Mass Fraction).
  * **IMF** (Ice Mass Fraction).

**Key Functions:**

* `read_struc_dat(path)`
* `read_tevol_dat(path)`
* `get_mass(path)`
* `get_radius(path)`
* `get_surface_heat_flux(path_struc, path_tevol)`
* `process_planet_model(model_folder)`
* `process_all_models(planetary_grid_path, imf_filter=None)`

---

### `geotherm_calculator.py`

**Description:**
Python module for computing **1D planetary geotherms**, incorporating radiogenic heat production, effective thermal conductivity, and boundary conditions appropriate for rocky planets.

**Main Features:**

* Construction of radial geotherms based on:

  * Internal layer distribution.
  * Internal heat production.
  * Thermal properties of planetary materials.
* Computation of temperature profiles and geothermal gradients.
* Evaluation of the internal thermal state and its relationship to:

  * Subsurface habitability.
  * Stability of geological layers and mineral phases.

---

### `habitability_calculator.py`

**Description:**
Core module for computing **subsurface aquability and habitability indices** in rocky planets from their internal structural and thermal properties.

This module integrates information derived from:

* Planetary Grid structural models.
* Planetary geotherms.
* Physical constraints governing the stability of liquid water in planetary interiors.

**Main Features:**

* Calculation of dimensionless indices of:

  * **Subsurface aquability**.
  * **Subsurface habitability**.
* Identification of internal regions compatible with:

  * Pressure and temperature conditions suitable for liquid water.
  * Long-term persistence of favorable environmental conditions.
* Scaling of habitability metrics as a function of:

  * Planetary mass.
  * Internal composition (CMF, IMF).
  * Planetary thermal state.

## Notebooks
### `Thermal_Structure.ipynb`

This notebook develops the calculation of planetary geotherms and geothermal profiles for Earth-like rocky planets across a range of planetary masses. It also implements the parametric mineralogical framework used throughout the HAB-3D model to investigate how lithospheric composition influences subsurface aquable and habitable environments.

**Main topics covered:**
- Computation of conductive planetary geotherms.
- Scaling of geothermal gradients with planetary mass.
- Internal heat production and surface heat flux parameterization.
- Implementation of the parametric lithospheric mineralogy model.
- Estimation of effective thermophysical properties:
  - Thermal conductivity.
  - Density.
  - Heat capacity.
- Analysis of the impact of:
  - Felsic fraction.
  - Fe/Mg ratio.
  - Hydration state.
- Evaluation of how mineralogical diversity modifies:
  - Aquable layer thickness.
  - Habitable layer thickness.
  - Depth of subsurface liquid-water reservoirs.

**Outputs:**
- Temperature–depth profiles.
- Geothermal gradients.
- Mineralogical sensitivity analyses.
- Figures presented in the thermal structure and mineralogy sections of the paper.

### `Analysis_Layers.ipynb`

This notebook analyzes the formation, extent, and evolution of subsurface **aquable** and **habitable** layers in rocky planets. Using the thermal structure models developed previously, it evaluates how stellar irradiation and internal heat flow jointly control the distribution of liquid water and biologically viable environments within the lithosphere.

**Main topics covered:**

* Identification of subsurface aquable layers where liquid water is thermodynamically stable.
* Identification of habitable layers constrained by biological temperature and pressure limits.
* Calculation of:

  * Layer thickness.
  * Upper and lower depth boundaries.
  * Habitable fraction of the aquable reservoir.
* Parametric exploration of:

  * Orbital distance.
  * Surface heat flux.
  * Planetary thermal evolution.
* Comparison with the classical circumstellar habitable zone framework.
* Incorporation of the habitable zone limits proposed by Kopparapu et al.
* Analysis of habitability beyond the classical stellar habitable zone through geothermal heating.
* Temporal evolution of aquability and habitability for Earth-like planets.

**Outputs:**

* Aquable and habitable layer thickness profiles.
* Orbital distance versus heat flux contour maps.
* Habitability regime maps.
* Comparisons between subsurface habitability and the classical Kopparapu habitable zone.
* Figures presented in the aquability, habitability, and parametric sensitivity sections of the paper.

### `Geothermal_Evolution.ipynb`

This notebook investigates the long-term evolution of subsurface aquability and habitability by coupling the HAB-3D framework with thermal evolution models of rocky planets. Surface heat fluxes are extracted from a planetary thermal evolution grid and used as time-dependent boundary conditions for the geothermal model.

The thermal evolution data employed in this notebook are derived from the planetary evolution models presented by **Zuluaga et al. (2013)**, allowing the assessment of how the gradual cooling of a planet affects the extent of its subsurface habitable environments over geological timescales.

**Main topics covered:**

* Extraction of time-dependent surface heat fluxes from thermal evolution models.
* Reconstruction of planetary geothermal profiles through time.
* Evolution of aquable and habitable layer boundaries.
* Quantification of changes in:

  * Layer thickness.
  * Upper and lower depth limits.
  * Habitable fraction of the aquable reservoir.
* Comparison of subsurface habitability under different stellar irradiation conditions.
* Analysis of Earth-like planets placed at:

  * Earth's orbital distance (1 AU).
  * Mars' orbital distance (~1.52 AU).
* Investigation of the interplay between planetary cooling and stellar insolation.

**Outputs:**

* Time-dependent geothermal profiles.
* Evolutionary tracks of aquable and habitable layer thickness.
* Changes in habitable fraction over planetary history.
* Comparisons between Earth-analog and Mars-analog orbital configurations.
* Figures presented in the thermal evolution section of the paper.

**Reference:**
Zuluaga, J. I., Bustamante, S., Cuartas, P. A., & Hoyos, J. H. (2013). *The influence of thermal evolution in the magnetic protection of terrestrial planets*. The Astrophysical Journal, 770, 23.

### `EVI_Index.ipynb`

**Description:**  
This notebook develops and applies the **Eirenesphere Volume Index (EVI)**, the main volumetric metric introduced in this work to quantify the astrobiological potential of rocky planets.

The analysis extends beyond one-dimensional habitability layers by estimating the total volume of subsurface environments compatible with liquid water and biological constraints.

**Main analyses:**
- Calculation of:
  - Aquable layer volumes.
  - Eirenesphere volumes.
- Conversion of habitable volumes into **Terrestrial Ocean (TO)** units for intuitive comparison.
- Computation of the **EVI** across a wide range of planetary masses.
- Exploration of the dependence of EVI on:
  - Planetary mass.
  - Surface heat flux.
  - Internal structure.
- Identification of planetary regimes that maximize subsurface habitable volume.
