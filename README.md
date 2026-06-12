# Inside the Eirenesphere: The Interplay of Porosity, Heat Flux and Mineralogy in Exoplanetary Aquable and Habitable Layers

**Authors:** [Santiago A. Orjuela](mailto:santiagoa.orjuela@udea.edu.co) and [Jorge I. Zuluaga](mailto:jorge.zuluaga@udea.edu.co)

**Affiliation:** [SEAP/FACom](https://www.udea.edu.co), Instituto de Física – FCEN, Universidad de Antioquia, Calle 70 No. 52-21, Medellín, Colombia.

This repository contains the models, scripts, and notebooks underlying a self-consistent geophysical framework that quantifies the three-dimensional subsurface habitable volume — the *eirenesphere* — of rocky exoplanets. The framework couples internal radial structure, a parametric mineralogical model, radiogenic heat production, and pressure-dependent crustal porosity to distinguish between *subsurface aquability* (thermodynamic liquid-water stability) and *subsurface habitability* (conditions within extremophile biological limits). It introduces the **Eirenesphere Volumetric Index (EVI)** to compare planetary habitability across a wide range of masses, heat fluxes, and orbital distances, and demonstrates that crustal mineralogy and secular cooling are first-order controls on the extent of deep biospheres.

## Citing this work

If you use this code or results from this paper, please cite:

> Orjuela, S. A. & Zuluaga, J. I. (2026). *Inside the Eirenesphere: The Interplay of Porosity, Heat Flux and Mineralogy in Exoplanetary Aquable Layers and Eirenespheres*. Submitted to Astrobiology.

```bibtex
@misc{OrjuelaZuluaga2026,
  author        = {Orjuela, Santiago A. and Zuluaga, Jorge I.},
  title         = {Inside the Eirenesphere: The Interplay of Porosity, Heat Flux
                   and Mineralogy in Exoplanetary Aquable Layers and Eirenespheres},
  year          = {2026},
  eprint        = {0000.00000},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.EP},
  url           = {https://arxiv.org/abs/0000.00000}
}
```

---

## Before running anything

### System requirements

> **Python 3.9 – 3.12 required.**
> This project depends on `burnman`, which in turn requires `numba 0.59`. That version of `numba` **is not compatible with Python 3.13 or later**. Python 3.12 is recommended.

### Steps

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Eirenesphere
   ```

2. Create a virtual environment with Python 3.12:

   ```bash
   python3.12 -m venv .venv
   ```

3. Activate the virtual environment:

   ```bash
   # macOS / Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Register the kernel in Jupyter (optional, if using VS Code or an external JupyterLab):

   ```bash
   python -m ipykernel install --user --name eirenesphere --display-name "Python (Eirenesphere)"
   ```

6. Launch the notebooks:

   ```bash
   jupyter lab
   ```

---

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

---

### `Layers_Analysis.ipynb`

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

---

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

---

### `EVI_Index.ipynb`

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

---

## Modules

- **`planetary_grid_reader.py`** — Reads, processes, and organizes planetary interior models from the Planetary Grid simulations. Extracts structural and thermal information (`STRUC.dat`, `TEVOL.dat`), computes gravity profiles, mass, radius, and surface heat flux, and processes model grids organized by CMF and IMF.

  *Main features:*
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
  * Systematic processing of planetary models organized by **CMF** (Core Mass Fraction) and **IMF** (Ice Mass Fraction).

  *Key functions:* `read_struc_dat(path)`, `read_tevol_dat(path)`, `get_mass(path)`, `get_radius(path)`, `get_surface_heat_flux(path_struc, path_tevol)`, `process_planet_model(model_folder)`, `process_all_models(planetary_grid_path, imf_filter=None)`.

- **`geotherm_calculator.py`** — Computes 1D planetary geotherms incorporating radiogenic heat production, effective thermal conductivity, and boundary conditions appropriate for rocky planets. Returns radial temperature profiles and geothermal gradients as a function of internal composition and heat flux.

  *Main features:*
  * Construction of radial geotherms based on:
    * Internal layer distribution.
    * Internal heat production.
    * Thermal properties of planetary materials.
  * Computation of temperature profiles and geothermal gradients.
  * Evaluation of the internal thermal state and its relationship to:
    * Subsurface habitability.
    * Stability of geological layers and mineral phases.

- **`habitability_calculator.py`** — Core module for computing subsurface aquability and habitability indices. Integrates Planetary Grid structural models, geotherms, and physical constraints on liquid water stability to produce dimensionless habitability metrics as a function of planetary mass, composition (CMF, IMF), and thermal state.

  *Main features:*
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

---

## Paper figures

The table below maps each figure in the paper to the notebook where it is generated.

| Figure | Description | Notebook |
|---|---|---|
| Fig. 1 — `phase_water_diagram` | Phase diagram of pure water in P–T space; shaded region marks liquid-water stability relevant to planetary interiors. | `Layers_Analysis.ipynb` |
| Fig. 2 — `qss` | Geotherms for an Earth-like planet varying surface heat flow from 20 to 200 mW m⁻². | `Thermal_Strucutre.ipynb` |
| Fig. 3 — `mass_profiles` | Internal T, P, and porosity profiles for planets from 0.1 to 10 M⊕ with fixed terrestrial composition. | `Thermal_Strucutre.ipynb` |
| Fig. 4 — `Layer_thickness` | Aquable layer and eirenesphere thickness and upper depth vs. orbital distance for an Earth analog. | `Layers_Analysis.ipynb` |
| Fig. 5 — `Mass_thickness` | Aquable layer and eirenesphere thickness vs. planetary mass at fixed heat flow (65 mW m⁻²). | `Thermal_Strucutre.ipynb` |
| Fig. 6 — `geological_time` | Temporal evolution of aquable layer and eirenesphere thickness at 1 au and 1.5 au. | `Geothermal_evolution.ipynb` |
| Fig. 7 — `minero` | Eirenesphere thickness as a function of felsic fraction and Fe/Mg ratio for dry and hydrated crusts. | `Thermal_Strucutre.ipynb` |
| Fig. 8 — `Contours` | Parametric sensitivity of subsurface habitability vs. orbital distance and surface heat flux (thickness, fraction, regime). | `Layers_Analysis.ipynb` |
| Fig. 9 — `regime_grid` | Map of subsurface habitability regimes (Frozen World, Sterile Aquable, Porosity-Limited, Habitable Subsurface). | `Layers_Analysis.ipynb` |
| Fig. 10 — `volume_mass` | Aquable and habitable volumes vs. planetary mass at 1 au and 65 mW m⁻². | `EVI_Index.ipynb` |
| Fig. 11 — `EVI` | EVI heatmap as a function of planetary mass and surface heat flux, in Terrestrial Ocean units. | `EVI_Index.ipynb` |
| Fig. A1 — `Teq_transmitancia` | Equilibrium surface temperature vs. orbital distance for a solar-type star (appendix). | `Layers_Analysis.ipynb` |
| Fig. A2 — `Teq_diagram` | Schematic of the 1-layer atmospheric radiative balance model (appendix). | `Layers_Analysis.ipynb` |

---

## AI assistance disclosure

Portions of the code review, inline documentation, and debugging in this repository were assisted by AI language models, specifically **Google Gemini Pro** and **Anthropic Claude Sonnet 4.6**.

The human authors assert that all scientific ideas, the overall project conception, the package and notebook architecture, the design of the numerical and scientific experiments, and the entirety of their interpretation and conclusions are original contributions of the human authors. AI tools were used exclusively as coding assistants — analogous to a spell-checker or a compiler — and bear no intellectual authorship over the scientific content of this work. In addition, AI models assisted with the translation of text from Spanish (the native language of the human authors) into English, and with English spelling and grammar review.
