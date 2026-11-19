# Public packages
import numpy as np
from dataclasses import dataclass
import time
from multiprocessing.shared_memory import SharedMemory
import pandas as pd

# subcomponents packages
import cmf
from openalea.rhizosoil.mimics_cn_grid import MIMICS_CN
from openalea.soiltemp.model import PyCampbell


# Utility packages
from openalea.metafspm.component_factory import *
from openalea.metafspm.component import Model, declare


debug = False

@dataclass
class SoilModel(Model):
    """
    Empty doc
    """

    # --- @note INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM CARBON MODEL
    hexose_exudation_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    phloem_hexose_exudation_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    hexose_uptake_from_soil_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    phloem_hexose_uptake_from_soil_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    mucilage_secretion_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of equivalent hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    cells_release_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of equivalent hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    
    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")

    # FROM NITROGEN MODEL
    mineralN_uptake: float = declare(default=0., unit="gN.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_uptake: float = declare(default=0., unit="gC.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_roots: float =  declare(default=0., unit="gN.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_roots: float =  declare(default=0., unit="gC.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_xylem: float =  declare(default=0., unit="gN.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_xylem: float =  declare(default=0., unit="gC.s-1", unit_comment="of amino_acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    
    # FROM WATER MODEL
    water_uptake: float =  declare(default=0., unit="m3.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_water", state_variable_type="extensive", edit_by="user")
    
    # FROM METEO
    water_irrigation: float =  declare(default=10/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    water_evaporation: float =  declare(default=5/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    water_drainage: float =  declare(default=5/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    voxel_mineral_N_fertilization: float =  declare(default=0., unit="g.s-1", unit_comment="of nitrogen", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="meteo", state_variable_type="extensive", edit_by="user")
    mineral_N_fertilization_rate: float =  declare(default=0., unit="g.s-1", unit_comment="of nitrogen", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="meteo", state_variable_type="extensive", edit_by="user")

    # --- @note STATE VARIABLES INITIALIZATION ---
    # Temperature
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                                 value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_temperature", state_variable_type="intensive", edit_by="user")
    rain_intensity: float = declare(default=0., unit="mm/h", unit_comment="", description="hourly rain intensity",
                                                 value_comment="", references="", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    # NOTE (leaching fix): before this variable existed, CMF had rain as its only surface flux and no evaporation at all,
    # so every mm of rain ended up percolating through the profile and pushed nitrate out of the root zone as a piston
    # (e.g. 55 mm of rain after fertilization displaced the fertilizer pulse by ~13 cm at theta ~0.43).
    # Declared as a state variable like rain_intensity so that an hourly input-table column with the same name can override it
    # (set soil_evaporation_from_meteo to False in that case). Otherwise it is updated every day from the ETPP column of the daily meteo file.
    soil_evaporation_intensity: float = declare(default=1. / 24, unit="mm/h", unit_comment="of water", description="Potential evaporation from the soil surface, withdrawn from the top CMF layer and scaled by its relative wetness. Solutes are left behind in the soil.",
                                                 value_comment="Fallback value only (1 mm/day). By default replaced daily by the potential evapotranspiration of the daily meteo file, which is 0.2 to 0.6 mm/day in December-February at Grignon.", references="", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # C related
    POC: float = declare(default=2.e-3, unit="adim", unit_comment="gC per g of dry soil", description="Particulate Organic Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    MAOC: float = declare(default=8.e-3, unit="adim", unit_comment="gC per g of dry soil", description="Mineral Associated Organic Carbon in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    DOC: float = declare(default=2e-7, unit="adim", unit_comment="gC per g of dry soil", description="Dissolved Organic Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_C: float = declare(default=0.2e-3, unit="adim", unit_comment="gC per g of dry soil", description="microbial Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    CO2: float = declare(default=0, unit="adim", unit_comment="gC per g of dry soil", description="Carbon dioxyde massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_hexose_soil: float = declare(default=2.4e-3, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    content_hexose_soil: float = declare(default=2.4e-3, unit="mol.g-1", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_mucilage_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_cells_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    # N related
    PON: float = declare(default=0.1e-3, unit="adim", unit_comment="gN per g of dry soil", description="Particulate Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    MAON: float = declare(default=0.8e-3, unit="adim", unit_comment="gN per g of dry soil", description="Mineral-Associated Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    DON: float = declare(default=2e-8, unit="adim", unit_comment="gN per g of dry soil", description="Dissolved Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_N: float = declare(default=0.03e-3, unit="adim", unit_comment="gN per g of dry soil", description="microbial N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Nm_fungus: float = declare(default=0., unit="adim", unit_comment="gN per g of dry soil", description="mycorrhiza N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    dissolved_mineral_N: float = declare(default=20e-6, unit="adim", unit_comment="gN per g of dry soil", description="dissolved mineral N massic concentration in soil",
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    C_mineralN_soil: float = declare(default=2.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_amino_acids_soil: float = declare(default=8.2e-3, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
    # All solutes
    Cv_solutes_soil: float = declare(default=32.2 / 10, unit="mol.m-3", unit_comment="mol of  all dissolved mollecules in the soil solution", description="All dissolved mollecules concentration", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # Water related
    water_potential_soil: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="Mean soil water potential", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    soil_moisture: float = declare(default=0.3, unit="adim", unit_comment="g.g-1", description="Volumetric proportion of water per volume of soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    water_volume: float = declare(default=0.25e-6, unit="m3", unit_comment="", description="Volume of the water in the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    # NOTE (uptake availability fix): the plant computes its mineral N uptake from the soil concentration of the previous step,
    # and may therefore request more N than what remains in the voxel within one time step. The excess used to be silently
    # cancelled by clamping the soil pool to 0, i.e. N given to the plant was created out of nothing. That unmet part is now kept as a
    # debt of the voxel, repaid from the next mineral N inputs (mineralization, fertilization, transport), so that N is conserved.
    mineral_N_diffusion: float = declare(default=0., unit="gN.s-1", unit_comment="of mineral nitrogen", description="Net mineral N gained by the voxel through diffusion from its neighbors (negative when it loses N to them)",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mineral_N_leaching: float = declare(default=0., unit="gN.s-1", unit_comment="of mineral nitrogen", description="Mineral N leaving the soil profile with free drainage water, attributed to the bottom voxel of each column (0 elsewhere)",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mineralN_uptake_debt: float = declare(default=0., unit="gN", unit_comment="of mineral nitrogen", description="Mineral N already given to roots but not yet available in the voxel, to be withdrawn from the next mineral N inputs of this voxel",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    actual_water_uptake: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="Water uptake actually withdrawn from the voxel, after clamping the requested water_uptake to the water locally available in that voxel this timestep, to keep the CMF Richards solver from being asked to extract more than physically exists",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    
    # Structure related
    voxel_volume: float = declare(default=1e-6, unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    bulk_density: float = declare(default=1.42, unit="g.mL", unit_comment="", description="Volumic density of the dry soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    dry_soil_mass: float = declare(default=1.42, unit="g", unit_comment="", description="dry weight of the considered voxel element", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    clay_percentage: float = declare(default=12.3, unit="%", unit_comment="", description="clay percentage of voxel", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.8, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    sand_percentage: float = declare(default=14.9, unit="%", unit_comment="", description="clay percentage of voxel", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.8, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    silt_percentage: float = declare(default=72.8, unit="%", unit_comment="", description="clay percentage of voxel", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.8, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
    # --- @note RATES INITIALIZATION ---
    # In-voxel rates
    mineral_N_net_mineralization: float = declare(default=0., unit=".s-1", unit_comment="gN per g of soil per second", description="mineral N uptake by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    microbial_respiration: float = declare(default=0., unit="gC.g-1.s-1", unit_comment="gC per g of soil per second", description="CO2 respiration by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    N_mineralization: float = declare(default=0., unit="gN.g-1.s-1", unit_comment="gN per g of soil per second", description="SOM net N mineralization by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    microbial_N_uptake: float = declare(default=0., unit="gN.g-1.s-1", unit_comment="gN per g of soil per second", description="N acquisisition by micro organisms competing with root N uptake", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mucilage_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of mucilage degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    cells_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of root cells degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")


    # --- @note PARAMETERS ---

    # C related
    
    ratio_C_per_amino_acid: float = declare(default=5, unit="adim", unit_comment="number of carbon per molecule of amino acid", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CN_ratio_amino_acids: float = declare(default=5/1.4, unit="adim", unit_comment="", description="CN ratio of amino acids (6 C and 1.4 N on average)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Water-related parameters
    water_volumic_mass: float = declare(default=1e6, unit="g.m-3", unit_comment="", description="Constant water volumic mass", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="extensive", edit_by="user")
    g_acceleration: float = declare(default=9.806, unit="m.s-2", unit_comment="", description="gravitationnal acceleration constant", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="extensive", edit_by="user")
    saturated_hydraulic_conductivity: float = declare(default=0.2653, unit="adim", unit_comment="m.day-1", description="staturated hydraulic conductivity parameter", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_R: float = declare(default=0.0602, unit="adim", unit_comment="m3.m-3", description="Soil retention moisture", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_S: float = declare(default=0.4509, unit="adim", unit_comment="m3.m-3", description="Soil saturation moisture", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_alpha: float = declare(default=0.0048, unit="cm-1", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_n: float = declare(default=1.6914, unit="cm-3", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    C_solutes_background: float = declare(default=0, unit="mol.m-3", unit_comment="", description="Background non C and non N solutes concentration in soil",
                                        value_comment="Raw estimation to align with inorganic N range for now", references="TODO", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    no_flux_bottom_boundary: bool = declare(default=False, unit="adim", unit_comment="", description="If True, the bottom of the soil profile is a no-flux (impermeable) boundary, as in a pot experiment. If False (default), the bottom layer drains freely under gravity (unit hydraulic gradient), as for a field soil without shallow water table.",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    soil_water_spinup: bool = declare(default=True, unit="adim", unit_comment="", description="If True (and free drainage), the initial moisture profile is obtained by running the water balance of one soil column (rain, evaporation, free drainage) with the daily meteo from the spin-up starting date to the simulation starting date, starting from the soil_moisture initialization. This replaces the initial soil_moisture of the scenario.",
                                        value_comment="Avoids starting from a uniform moisture that the free drainage boundary would quickly drain: the profile starts in the state left by the previous months (autumn recharge for a December start).", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cmf_parallel_threads: int = declare(default=1, unit="adim", unit_comment="", description="Number of OpenMP threads used by the CMF solvers of this process",
                                        value_comment="CMF parallelizes every flux evaluation with OpenMP and uses all cores by default. For a grid of ~1000 layers, and even more for the single column of the water spin-up, the synchronization overhead dominates: with one other simulation running, the spin-up column was 250 times slower with all 14 cores than with 1 thread, and the scene grid 4.5 times slower. With several simulations launched in parallel, the cores are oversubscribed and the spin-up seemed to never end.", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mineral_N_diffusion_between_voxels: bool = declare(default=True, unit="adim", unit_comment="", description="If True, dissolved mineral N diffuses between neighboring voxels after each CMF transport step (CMF only transports solutes by advection with water fluxes)",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_input_rate: float = declare(default=1., unit="mgC.cm-3.y-1", unit_comment="", description="Annual residue C input to MIMICS in the plough layer, also used for its steady-state initialization",
                                        value_comment="About 300 gC/m2/y over 30 cm", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_input_plough_depth: float = declare(default=0.3, unit="m", unit_comment="", description="Depth over which residue inputs are constant",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_input_decay_length: float = declare(default=0.2, unit="m", unit_comment="", description="e-folding length of residue inputs below the plough layer",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_total_CN: float = declare(default=80., unit="adim", unit_comment="", description="C:N ratio of residues, used with the lignin fraction to split them into metabolic and structural litter (Daycent)",
                                        value_comment="As this split is C-weighted, the effective C:N of N inputs is about 32", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_metabolic_CN: float = declare(default=15., unit="adim", unit_comment="", description="C:N ratio of metabolic litter",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_lignin_fraction: float = declare(default=0.2, unit="adim", unit_comment="", description="Lignin mass fraction of residues",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    litter_C_fraction: float = declare(default=0.45, unit="adim", unit_comment="", description="C mass fraction of residues",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    DOM_dissolved_fraction: float = declare(default=5e-3, unit="adim", unit_comment="", description="Fraction of the MIMICS available SOM pool (SOM_3) in soil solution, i.e. the voxel DOC and DON exposed to transport and root uptake",
                                        value_comment="Gives about 15 mgC/L of DOC and 1.5 mgN/L of DON at MIMICS steady state, within field soil solution ranges", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    nitrate_diffusion_coefficient_water: float = declare(default=1.902e-9, unit="m2.s-1", unit_comment="", description="Molecular diffusion coefficient of nitrate in free water at 25 °C, corrected for temperature with the Stokes-Einstein relation",
                                        value_comment="", references="CRC Handbook of Chemistry and Physics, ionic diffusion coefficients in water at infinite dilution", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    soil_evaporation_from_meteo: bool = declare(default=True, unit="adim", unit_comment="", description="If True, soil_evaporation_intensity is set every day from the potential evapotranspiration (ETPP column) of the daily meteo file. Set to False when providing soil_evaporation_intensity through an hourly input table.",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    max_uptake_fraction_of_available_water: float = declare(default=0.75, unit="adim", unit_comment="", description="Upper bound on water_uptake actually handed to the CMF solver, expressed as a fraction of the water available above theta_R in the voxel this timestep. Prevents a prescribed uptake flux from driving a voxel's water storage negative and crashing the Richards solver.",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # W patch initialization parameters
    water_moisture_patch: float = declare(default=0.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in a located patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_depth_water_moisture: float = declare(default=0., unit="m", unit_comment="", description="Depth of a nitrate patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_uniform_width_water_moisture: float = declare(default=2*0.1, unit="m", unit_comment="", description="Width of the zone of the patch with uniform concentration of nitrate", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_transition_water_moisture: float = declare(default=1e-3, unit="m", unit_comment="", description="Variance of the normal law smooting the boundary transition of a nitrate patch with the background concentration", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")


    # N patch initialization parameters
    
    dissolved_mineral_N_patch: float = declare(default=20e-6, unit="mol.g-1", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in a located patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_depth_mineralN: float = declare(default=10e-2, unit="m", unit_comment="", description="Depth of a nitrate patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_uniform_width_mineralN: float = declare(default=4e-2, unit="m", unit_comment="", description="Width of the zone of the patch with uniform concentration of nitrate", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_transition_mineralN: float = declare(default=1e-3, unit="m", unit_comment="", description="Variance of the normal law smooting the boundary transition of a nitrate patch with the background concentration", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Temperature

    mucilage_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    cells_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    mucilage_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of mucilage in soil", 
                                        value_comment="", references="We assume that the maximum degradation rate for mucilage is equivalent to the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_mucilage_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil mucilage degradation ", 
                                        value_comment="", references="We assume that Km for mucilage degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10 / 2, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of root cells at the soil/root interface", 
                                        value_comment="", references="We assume that the maximum degradation rate for cells is equivalent to the half of the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_cells_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil cells degradation", 
                                        value_comment="", references="We assume that Km for cells degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    
    def __init__(self, time_step, scene_xrange=1., scene_yrange=1., soil_depth=1., voxel_side_length=0.03, **scenario):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.time_step = time_step
        self.simulation_time_hours = 0
        self.run_mimics = True  # TODO pass as a parameter, and maybe rename
        self.initiate_voxel_soil(scene_xrange, scene_yrange, soil_depth, voxel_length=voxel_side_length, voxel_height=voxel_side_length)
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.voxels, compartment="soil")
        self.voxel_neighbor = {}


    # SERVICE FUNCTIONS

    # Just ressource for now
    def initiate_voxel_soil(self, scene_xrange=1., scene_yrange=1., soil_depth=1., 
                            voxel_length=3e-2, voxel_height=3e-2):
        """
        Note : not tested for now, just computed to support discussions.
        """
        self.voxels = {}

        
        self.planting_depth = 5e-2

        voxel_width = voxel_length
        voxel_volume = voxel_height * voxel_width * voxel_width

        self.delta_z = voxel_height
        self.voxels_Z_section_area = voxel_width * voxel_width
        
        self.voxel_number_x = int(scene_xrange / voxel_width) + 1
        actual_voxel_width = scene_xrange / self.voxel_number_x
        self.scene_xrange = scene_xrange

        self.voxel_number_y = int(scene_yrange / voxel_width) + 1
        actual_voxel_length = scene_yrange / self.voxel_number_y
        self.scene_yrange = scene_yrange

        voxel_volume = voxel_height * actual_voxel_width * actual_voxel_length

        scene_zrange = soil_depth
        self.voxel_number_z = int(scene_zrange / voxel_height) + 1
        self.scene_zrange = scene_zrange

        # Uncentered, positive grid
        y, z, x = np.indices((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels["x1"] = x * actual_voxel_width
        self.voxels["x2"] = self.voxels["x1"] + actual_voxel_width
        self.voxels["y1"] = y * actual_voxel_length
        self.voxels["y2"] = self.voxels["y1"] + actual_voxel_length
        self.voxels["z1"] = z * voxel_height
        self.voxels["z2"] = self.voxels["z1"] + voxel_height

        self.voxel_dx = actual_voxel_width
        self.voxel_dy = actual_voxel_length
        self.voxel_dz = voxel_height

        self.voxel_grid_to_self("voxel_volume", voxel_volume)

        for name in self.state_variables + self.inputs:
            if name != "voxel_volume":
                self.voxel_grid_to_self(name, init_value=getattr(self, name))

        # Set an heterogeneity uppon the mean background
        # Nitrogen
        self.add_patch_repartition_to_soil(property_name="dissolved_mineral_N", patch_value=self.dissolved_mineral_N_patch, 
                                           z_loc=self.patch_depth_mineralN, 
                                           z_width=self.patch_uniform_width_mineralN, 
                                           z_dev=self.patch_transition_mineralN)
        # Water
        self.add_patch_repartition_to_soil(property_name="soil_moisture", patch_value=self.water_moisture_patch, 
                                           z_loc=self.patch_depth_water_moisture, 
                                           z_width=self.patch_uniform_width_water_moisture, 
                                           z_dev=self.patch_transition_water_moisture)

        # TODO: Add utility for layered stratification, for example for clay percentage
        
        # Initialize volumic concentrations
        self.voxels["dry_soil_mass"] = 1e6 * self.voxels["voxel_volume"] * self.voxels["bulk_density"]
        self.voxels["water_volume"] = self.voxels["voxel_volume"] * self.voxels["soil_moisture"]
        self.voxels["C_mineralN_soil"] = self.voxels["dissolved_mineral_N"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 14
        self.voxels["C_amino_acids_soil"] = self.voxels["DON"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 14
        self.voxels["C_hexose_soil"] = self.voxels["DOC"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 6 / 12
        self.voxels["Cv_solutes_soil"] = self.voxels["C_mineralN_soil"] # Until we are sure of proper initialization and balance of these different concentrations

        # Daily meteo and dates shared by the soil water spin-up, the daily evaporation and the Campbell temperature model
        self.daily_meteo_df = pd.read_csv('inputs/daily_grignon_meteo.csv', sep=",")
        self.daily_meteo_df["DATE"] = pd.to_datetime({"year": self.daily_meteo_df["AN"], "month": self.daily_meteo_df["MOIS"], "day": self.daily_meteo_df["JOUR"]}, errors="coerce")
        self.spinup_starting_date = pd.to_datetime("1998-01-01")
        self.simulation_initial_date = pd.to_datetime("1998-12-17")

        # Initiate the transport model
        # NOTE (leaching fix): CMF cells were built with the nominal voxel_length (3 cm) while voxels are resized to fit the scene
        # (2.5 cm here). CMF layers were then 1.44 times larger than the voxels they represent, so a prescribed root water uptake (m3.s-1)
        # dried them 1.44 times too slowly. The actual voxel dimensions are now used.
        self.initiate_cmf(nx=self.voxel_number_x, ny=self.voxel_number_y, nz=self.voxel_number_z,
                          dx=self.voxel_dx, dy=self.voxel_dy, dz=self.voxel_dz)

        if self.run_mimics:
            # NOTE (litter profile fix): the steady state was solved for other inputs (0.7 mgC/cm3/y at C:N 15) than those applied at
            # each step (1 mgC/cm3/y at C:N 80, uniform over the whole profile), so pools drifted from the start. Both now use the same
            # residue inputs, decreasing below the plough layer.
            self._litter_inputs = self.litter_inputs_profile()
            li = self._litter_inputs
            self.mimics = MIMICS_CN(time_step_in_hours=self.time_step / 3600,
                                    clay_percentage=self.voxels["clay_percentage"],
                                    litter_inputs=li["structural"], litter_CN=li["structural_CN"],
                                    rhizodeposits_inputs=li["metabolic"], rhizodeposits_CN=li["metabolic_CN"])
            # NOTE (DOM fix): the voxel DOC and DON are only the dissolved fraction of the MIMICS available pool. Overwriting that pool
            # with the scenario values (0 or 2e-7 g/g, 2500 times below steady state) starved microbes and cut mineralization by about 5.
            conv = 1e6 * self.voxels["voxel_volume"] / self.voxels["dry_soil_mass"] / 1e3 # mg/cm3 to g/g
            self.voxels["DOC"] = self.DOM_dissolved_fraction * self.mimics.SOC_available * conv
            self.voxels["DON"] = self.DOM_dissolved_fraction * self.mimics.SON_available * conv
            self._DOC_written, self._DON_written = self.voxels["DOC"].copy(), self.voxels["DON"].copy()
            self.mimics_cn_states()

        self.initiate_campbell()


    def wetness_from_theta(self, theta):
        """
        cmf's RetentionCurve expects normalized wetness (effective saturation Se = (theta - theta_r) / (phi - theta_r)),
        not the absolute volumetric water content theta itself.
        """
        return (theta - self.theta_R) / (self.theta_S - self.theta_R)

    def initiate_cmf(self, nx, ny, nz, dx, dy, dz):
        # Limit CMF's OpenMP threads before any solver is built (spin-up column included), see cmf_parallel_threads.
        # NOTE: this was why, with several simulations launched in parallel, the spin-up never ended from the second one on.
        cmf.set_parallel_threads(int(self.cmf_parallel_threads))
        # 1. Create a project with transported solutes
        self.cmf_accounted_solutes = ["DOC", "DON", "dissolved_mineral_N"] # Manual
        # self.cmf_accounted_solutes = ["dissolved_mineral_N"] # Manual

        # Specific sting formating to create project in CMF (space separator between solutes)
        solute_string = ''
        for k in range(len(self.cmf_accounted_solutes)):
            solute_string += self.cmf_accounted_solutes[k]
            if k < len(self.cmf_accounted_solutes) - 1:
                solute_string += ' '

        self.cmf_project = cmf.project(solute_string)
        # Retrieve the nitrate solute by name: a positional unpacking only worked while dissolved_mineral_N was the single transported solute
        nitrate = self.cmf_project.solutes[self.cmf_accounted_solutes.index("dissolved_mineral_N")]

        # Retention curve with soil parameters
        self.r_curve=cmf.VanGenuchtenMualem(Ksat=self.saturated_hydraulic_conductivity, # m.day-1
                                        theta_r=self.theta_R,
                                        phi=self.theta_S, # theta_s
                                        alpha=self.water_alpha, 
                                        n=self.water_n) # Example for silt-loam

        # 2. Build a cubic voxel grid
        self.cmf_id_grid = np.arange(nx * ny).reshape((nx, ny))
        symetry = True # TODO : add as parameter
        real_3D = True # TODO : add as parameter
        solve_tolerance = 1e-9 * 1e3

        self.cmf_cells = {}
        cell_id = 0
        # Like in STICS, we create a "cell", i.e. a surface element with several layers
        for ix in range(nx):
            for iy in range(ny):
                cell = self.cmf_project.NewCell(x=ix*dx, y=iy*dy, z=0, area=dx*dy)
                for iz in range(nz):
                    depth = (iz+1)*dz
                    cell.add_layer(depth, self.r_curve)
                cell.install_connection(cmf.Richards)
                # We store cells in a dictionnary to be able to connect them when we want to simulate a real 3D grid
                self.cmf_cells[self.cmf_id_grid[ix, iy]] = cell
                cell_id += 1

        # 3. Create connections between cells
        for ix in range(nx):
            for iy in range(ny):
                cell = self.cmf_cells[self.cmf_id_grid[ix, iy]]
                # Row neighbor (x+1 on same column iy)
                if ix < nx-1:
                    cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[ix + 1, iy]], dx)
                else:
                    if symetry:
                        # If scene is symetrical the last element in row is neighbor to first element in row
                        cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[0, iy]], dx)
                
                # Column neighbor (y+1)
                if iy < ny-1:
                    cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[ix, iy + 1]], dy)
                else:
                    if symetry:
                        # If scene is symetrical the last element in column is neighbor to first element in column
                        cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[ix, 0]], dy)

        # (3bis OPTIONAL) If the scene is real 3D, we install connections with lateral richards flux and advection 
        if real_3D:
            cmf.connect_cells_with_flux(self.cmf_project, cmf.Richards_lateral)

        
        # 4. Set initial conditions

        # NOTE (free drainage): the bottom boundary used to be a fixed moisture of 0.9 * theta_S at 1 m depth, i.e. a shallow water table that
        # does not exist at this site, and that kept the whole profile close to saturation through capillary rise. It is replaced by
        # free drainage (see the end of this function). With these hydraulic parameters, conductivity is already ~5 mm/day at theta = 0.31,
        # so a uniform 0.31 start would drain strongly during the first weeks. The initial profile is therefore taken from a one-column
        # water balance spin-up over the months preceding the simulation, which gives the moisture state of the season.
        if not self.no_flux_bottom_boundary and self.soil_water_spinup:
            initial_profile = self.soil_water_spinup_profile(nz=nz, dz=dz,
                                                             initial_profile=self.voxels["soil_moisture"].mean(axis=(0, 2)))
            self.voxels["soil_moisture"][:] = initial_profile[None, :, None]
            self.voxels["water_volume"] = self.voxels["voxel_volume"] * self.voxels["soil_moisture"]

        # NOTE (leaching fix): solutes are exchanged with CMF as AMOUNTS (g per voxel), no longer as concentrations derived from soil_moisture.
        # CMF stores layer water as wetness * porosity * layer volume, i.e. without the residual water theta_R, while layer.theta (used as
        # soil_moisture) includes it. Converting mass -> concentration with soil_moisture and back with a different soil_moisture therefore
        # did not conserve solutes: drying created N and wetting (rain) destroyed it (about 0.5 % for a 0.01 change in theta), even with no
        # transport at all. Amounts are conserved by construction. The ratio below only accounts for any CMF layer / voxel volume mismatch.
        self.cmf_to_voxel_volume_ratio = (dx * dy * dz) / self.voxels["voxel_volume"]
        voxel_solute_amounts = {solute_name: self.voxels[solute_name] * self.voxels["dry_soil_mass"] * self.cmf_to_voxel_volume_ratio
                                for solute_name in self.cmf_accounted_solutes}

        # Dynamic storage of rainfall nodes for each cells
        self.rainfall_nodes = {}
        # Dynamic storage of a conservative sink for each layer, used to withdraw root water uptake from cmf's
        # own mass-balanced integration instead of overwriting theta/potential from outside every hour.
        self.uptake_nodes = {}
        # Dynamic storage of an evaporation sink on the top layer of each cell (see soil_evaporation_intensity)
        self.evaporation_nodes = {}
        for ix in range(nx):
            for iy in range(ny):
                cell = self.cmf_cells[self.cmf_id_grid[ix, iy]]

                # Null at first, just to instantiate the object
                rs = cmf.ConstantRainSource(self.cmf_project, cmf.point(0, 0, 0), 0)
                rs.set_conc(nitrate, 0)
                cell.rain_source = rs
                self.rainfall_nodes[self.cmf_id_grid[ix, iy]] = rs

                for iz, l in enumerate(cell.layers):
                    # NOTE (leaching fix): moisture is now imposed through the wetness only. Two CMF conventions made the previous
                    # "l.theta = ...; l.potential = MatricPotential(...)" initialization wrong:
                    #   1) layer.potential is the TOTAL head (matric + elevation), so assigning it a matric potential overrode theta
                    #      with a wetter state increasing with depth (0.33 at the top to 0.38 at 1 m instead of 0.31), and turned the
                    #      bottom boundary into a saturated water table. This is why the whole profile was close to saturation and why
                    #      all rain percolated.
                    #   2) the layer.theta setter ignores theta_r (it sets wetness = theta / phi) while the theta getter does not, so even
                    #      without the potential assignment, 0.31 read back as 0.329.
                    # Since solute masses are converted with soil_moisture both ways, any such mismatch also created ~6 to 14 % of
                    # mineral N at the first step. Setting the wetness gives an exact round trip and a consistent potential.
                    l.wetness = self.wetness_from_theta(self.voxels["soil_moisture"][iy, iz, ix])
                    for solute_name, solute in zip(self.cmf_accounted_solutes, self.cmf_project.solutes):
                        l[solute].state = voxel_solute_amounts[solute_name][iy, iz, ix]

                    uptake_node = cmf.NeumannBoundary.create(l)
                    uptake_node.set_flux(0.)
                    # NOTE (uptake coupling fix): by default a CMF flux connection carries solutes at the concentration of the node
                    # water leaves from, so root water uptake was also removing nitrate (and DOC, DON) by mass flow. Root solute uptake
                    # is already fully computed by the plant models (active uptake, plus the apoplastic mass flow term of Root-CyNAPS),
                    # and sent to the soil as mineralN_uptake, so the CMF sink must only extract water: solutes are left in the voxel.
                    for connection in uptake_node.connections:
                        for solute in self.cmf_project.solutes:
                            connection.set_tracer_filter(solute, 0.)
                    self.uptake_nodes[(iy, iz, ix)] = uptake_node

                # Evaporation only removes water: solutes are concentrated in the top layer instead of leaving the soil
                evaporation_node = cmf.NeumannBoundary.create(cell.layers[0])
                evaporation_node.set_flux(0.)
                for connection in evaporation_node.connections:
                    for solute in self.cmf_project.solutes:
                        connection.set_tracer_filter(solute, 0.)
                self.evaporation_nodes[self.cmf_id_grid[ix, iy]] = evaporation_node

        # 4bis. Free drainage bottom boundary: the bottom layer of each column drains at its unsaturated conductivity (unit gradient)
        # into an outlet that collects drainage water and the solutes it carries (mineral_N_leaching).
        if not self.no_flux_bottom_boundary:
            self.drainage_outlet = self.cmf_project.NewOutlet("drainage", 0., 0., -(nz * dz) - 1.)
            for cell in self.cmf_cells.values():
                cmf.FreeDrainagePercolation(cell.layers[-1], self.drainage_outlet)

        # 5. Set up integrators (water + solute)
        water_integrator = cmf.ImplicitEuler(self.cmf_project, solve_tolerance)
        solute_integrator = cmf.CVodeKrylov(self.cmf_project, solve_tolerance)
        self.cmf_solver = cmf.SoluteWaterIntegrator(self.cmf_project.solutes, solute_integrator, water_integrator, self.cmf_project)


    def soil_water_spinup_profile(self, nz, dz, initial_profile):
        """
        Moisture profile at the simulation starting date, from a daily water balance of a single bare soil column with the same retention
        curve and bottom boundary as the simulated scene. Inputs are the daily rain (RR) and potential evaporation (ETPP) of the daily meteo file.
        The scene is horizontally homogeneous at initialization, so one column is enough and is copied to every column.

        Assumptions: bare soil before sowing (no crop transpiration during the spin-up), evaporation limited by top layer wetness as in
        cmf_transport, and daily rain spread evenly over the day.
        """
        project = cmf.project("")
        column = project.NewCell(0., 0., 0., 1.)
        for iz in range(nz):
            column.add_layer((iz + 1) * dz, self.r_curve)
        column.install_connection(cmf.Richards)
        for iz, layer in enumerate(column.layers):
            layer.wetness = min(max(self.wetness_from_theta(initial_profile[iz]), 1e-3), 1.)

        rain = cmf.ConstantRainSource(project, cmf.point(0, 0, 0), 0.)
        column.rain_source = rain
        evaporation = cmf.NeumannBoundary.create(column.layers[0])
        outlet = project.NewOutlet("drainage", 0., 0., -(nz * dz) - 1.)
        cmf.FreeDrainagePercolation(column.layers[-1], outlet)
        # CVODE (adaptive) rather than ImplicitEuler: ImplicitEuler fails on daily steps with non finite Richards fluxes
        solver = cmf.CVodeDense(project, 1e-9)

        meteo = self.daily_meteo_df.set_index("DATE")
        n_days = (self.simulation_initial_date - self.spinup_starting_date).days
        print(f"[INFO] soil water spin-up over {n_days} days")
        for day in range(n_days):
            date = self.spinup_starting_date + pd.Timedelta(days=day)
            rain.intensity = float(meteo.loc[date, "RR"]) if date in meteo.index else 0. # mm.day-1
            top_wetness = min(max(column.layers[0].wetness, 0.), 1.)
            evaporation_rate = float(meteo.loc[date, "ETPP"]) if date in meteo.index else 0.
            evaporation.set_flux(-evaporation_rate / 1e3 * top_wetness * column.area) # m3.day-1
            solver(solver.t + cmf.day)

        profile = np.array([layer.theta for layer in column.layers])
        print(f"[INFO] soil water spin-up done, moisture top {profile[0]:.3f}, 30 cm {profile[min(int(0.3 / dz), nz - 1)]:.3f}, bottom {profile[-1]:.3f}")
        return profile

    def initiate_campbell(self):
        self.campbell_input_df = self.daily_meteo_df

        nb_layers = self.voxels['z2'].shape[1]
        soil_depth = self.voxels['z2'].max() * 100 # m to cm
        layer_thickness = np.unique(self.voxels['z2'] - self.voxels['z1'])[0] * 100 # m to cm
        bulk_density = self.voxels['bulk_density'][0, 0, 0] # g/cm3
        SOC = (self.voxels["MAOC"] + self.voxels["POC"] + self.voxels["DOC"] + self.voxels["microbial_C"]).mean() * 100 # gC/100g of dry soil

        # Silt Loam, Saxon et Rawls 2006 NOTE expected to be different from Hydrus Theta_S and Theta_R, not the same parameters TODO to parameters
        permanent_wilting_point = 0.11
        field_capacity=0.31
        soil_water_saturated=0.48

        # TODO soil texture will be passed from the model rather than hardcoded here

        self.temperature_model = PyCampbell()

        self.static_config_dict = self.temperature_model.static_config(nb_layers=nb_layers, 
                                                    SOIL_ID="SILO",
                                                    SOIL_NAME="SiltyLoam",
                                                    layer_thickness=layer_thickness,
                                                    bulk_density=bulk_density,
                                                    soil_depth=soil_depth,
                                                    soil_organic_C=SOC,
                                                    permanent_wilting_point=permanent_wilting_point,
                                                    soil_water_saturated=soil_water_saturated,
                                                    field_capacity=field_capacity,
                                                    clay_percentage=self.clay_percentage,
                                                    sand_percentage=self.sand_percentage,
                                                    silt_percentage=self.silt_percentage,
                                                    XLAT=48.8442,
                                                    TAV=11.3, # Ljutovac 2002 + CLimatik TODO to parameters
                                                    TAMP=8.0 # based on monthly average (max-min)/2, 12.6 if min/max daily, if hourly will obviously be different TODO sort out
                                                    )

        spinup_starting_date = self.spinup_starting_date
        initial_date = self.simulation_initial_date
        self.previous_DOY = 347
        current_inputs = self.campbell_input_df.loc[self.campbell_input_df["DATE"] == spinup_starting_date]
        sunrise = current_inputs["SUNUP"].iat[0]
        sunset = current_inputs["SUNDWN"].iat[0]
        soil_moisture = self.voxels['soil_moisture'].mean(axis=(0, 2)) # cm3/cm3

        config = self.temperature_model.daily_config(self.static_config_dict,
                                    DATE=spinup_starting_date,
                                    T2M=current_inputs["TM"].iat[0], # °C
                                    TMIN=current_inputs["TN"].iat[0], # °C
                                    TMAX=current_inputs["TX"].iat[0], # °C
                                    RAIN=current_inputs["RR"].iat[0], # mm
                                    SRAD=current_inputs["RG"].iat[0] / 100, # from J/cm2 to MJ/m2
                                    DAYLD=sunset - sunrise,
                                    SUNUP=sunrise,
                                    SUNDN=sunset,
                                    plant_available_water=(soil_moisture - permanent_wilting_point) / (field_capacity - permanent_wilting_point),
                                    LAI=0., # TODO to couple with shoot
                                    albedo=0.12,
                                    irrig=0.,
                                    aboveGroundDM=0.) # TODO to couple with shoot
        
        res, previous_outputs = self.temperature_model.run(config=config, nb_steps=1)
        
        spinup_days = (initial_date - spinup_starting_date).days
        print("[INFO] starting temperature model spinup")
        for day in range(spinup_days):
            current_date = spinup_starting_date + pd.Timedelta(days=day+1)
            current_inputs = self.campbell_input_df.loc[self.campbell_input_df["DATE"] == current_date]
            sunrise = current_inputs["SUNUP"].iat[0]
            sunset = current_inputs["SUNDWN"].iat[0]
            soil_moisture = self.voxels['soil_moisture'].mean(axis=(0, 2)) # cm3/cm3

            config = self.temperature_model.daily_config(self.static_config_dict,
                                        DATE=current_date,
                                        T2M=current_inputs["TM"].iat[0], # °C
                                        TMIN=current_inputs["TN"].iat[0], # °C
                                        TMAX=current_inputs["TX"].iat[0], # °C
                                        RAIN=current_inputs["RR"].iat[0], # mm
                                        SRAD=current_inputs["RG"].iat[0] / 100, # from J/cm2 to MJ/m2
                                        DAYLD=sunset - sunrise, # h
                                        SUNUP=sunrise, # h
                                        SUNDN=sunset, # h
                                        plant_available_water=(soil_moisture - permanent_wilting_point) / (field_capacity - permanent_wilting_point),
                                        LAI=0.,
                                        albedo=0.12,
                                        irrig=0.,
                                        aboveGroundDM=0.)

            res, previous_outputs = self.temperature_model.run(config=config, nb_steps=1, previous_outputs=previous_outputs)
        print("[INFO] Finished temperature model spinup")

        self.previous_campbell_states = previous_outputs
        self.current_date = initial_date
        temperature_array = res["TSLD"].loc[res["Layer"] >= 1].to_numpy() # We ignore layer 0 that gives surface temperature
        self.voxels['soil_temperature'][:] = temperature_array[None, :, None]
        self.update_daily_evaporation()

    def daily_potential_evaporation(self, date):
        """Potential evaporation of the given day from the daily meteo file (mm.day-1), 0 if the date is missing"""
        current_inputs = self.daily_meteo_df.loc[self.daily_meteo_df["DATE"] == date]
        return float(current_inputs["ETPP"].iat[0]) if len(current_inputs) > 0 else 0.

    def update_daily_evaporation(self):
        """
        NOTE (free drainage): soil evaporation follows the season instead of a constant value. A constant 1 mm/day is about twice to five
        times the December-February potential evapotranspiration at this site, which would have dried the topsoil too much in winter.
        """
        if self.soil_evaporation_from_meteo:
            self.voxels["soil_evaporation_intensity"].fill(self.daily_potential_evaporation(self.current_date) / 24) # mm.day-1 to mm.h-1


    def voxel_grid_to_self(self, name, init_value):
        self.voxels[name] = np.zeros((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels[name].fill(init_value)
        #setattr(self, name, self.voxels[name])
    

    def add_patch_repartition_to_soil(self, property_name: str, patch_value: float, x_loc=None, y_loc=None, z_loc=None, 
                                                                        x_width=0, y_width=0, z_width=0, 
                                                                        x_dev=1e-3, y_dev=1e-3, z_dev=1e-3,
                                                                        spherical_normal_patch = False, normal_boundaries = False):
        
        if spherical_normal_patch:
            y_dev = x_dev
            z_dev = x_dev
            x_width = 0
            y_width = 0
            z_width = 0
        
        # Start with Z
        if z_loc is not None:
            
            z_mean = (self.voxels["z1"] + self.voxels["z2"]) / 2

            test = np.logical_and(z_loc - z_width/2 < z_mean, z_mean < z_loc + z_width/2)
            self.voxels[property_name][test] = patch_value
            if normal_boundaries:
                test = z_mean > z_loc + z_width/2
                new_values = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (z_dev * np.sqrt(2 * np.pi)) * np.exp(-((z_mean - (z_loc + z_width/2)) ** 2) / (2 * z_dev ** 2))
                self.voxels[property_name][test] = new_values[test]
                test = z_mean < z_loc - z_width/2
                new_values = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (z_dev * np.sqrt(2 * np.pi)) * np.exp(-((z_mean - (z_loc - z_width/2)) ** 2) / (2 * z_dev ** 2))
                self.voxels[property_name][test] = new_values[test]

        # Then x and y
        if x_loc is not None:
            x_mean = (self.voxels["x1"] + self.voxels["x2"]) / 2

            self.voxels[property_name][x_loc - x_width/2 < x_mean < x_loc + x_width/2] = patch_value
            self.voxels[property_name][x_mean > x_loc + x_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (x_dev * np.sqrt(2 * np.pi)) * np.exp(-((x_mean - (x_loc + x_width/2)) ** 2) / (2 * x_dev ** 2))
            self.voxels[property_name][x_mean < x_loc - x_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (x_dev * np.sqrt(2 * np.pi)) * np.exp(-((x_mean - (x_loc - x_width/2)) ** 2) / (2 * x_dev ** 2))

        if y_loc is not None:
            y_mean = (self.voxels["y1"] + self.voxels["y2"]) / 2

            self.voxels[property_name][y_loc - y_width/2 < y_mean < y_loc + y_width/2] = patch_value
            self.voxels[property_name][y_mean > y_loc + y_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (y_dev * np.sqrt(2 * np.pi)) * np.exp(-((y_mean - (y_loc + y_width/2)) ** 2) / (2 * y_dev ** 2))
            self.voxels[property_name][y_mean < y_loc - y_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (y_dev * np.sqrt(2 * np.pi)) * np.exp(-((y_mean - (y_loc - y_width/2)) ** 2) / (2 * y_dev ** 2))


    def compute_mtg_voxel_neighbors(self, props):

        # necessary to get updated coordinates.
        # if "angle_down" in g.properties().keys():
        #     plot_mtg(g)

        for vid in props["vertex_index"].keys():
            if (vid not in props["voxel_neighbor"].keys()) or (props["voxel_neighbor"][vid] is None) or (props["length"][vid] > props["initial_length"][vid]):
                baricenter = (np.mean((props["x1"][vid], props["x2"][vid])) % self.scene_xrange, # min value is 0
                            np.mean((props["y1"][vid], props["y2"][vid])) % self.scene_yrange, # min value is 0
                            -np.mean((props["z1"][vid], props["z2"][vid])))
                testx1 = self.voxels["x1"] <= baricenter[0]
                testx2 = baricenter[0] <= self.voxels["x2"]
                testy1 = self.voxels["y1"] <= baricenter[1]
                testy2 = baricenter[1] <= self.voxels["y2"]
                testz1 = self.voxels["z1"] <= baricenter[2]
                testz2 = baricenter[2] <= self.voxels["z2"]
                test = testx1 * testx2 * testy1 * testy2 * testz1 * testz2
                try:
                    props["voxel_neighbor"][vid] = [int(v) for v in np.where(test)]
                except:
                    print(" WARNING, issue in computing the voxel neighbor for vid ", vid)
                    props["voxel_neighbor"][vid] = None
        
        return props
    
    def compute_mtg_voxel_neighbors_fast(self, data, hs, mask,
                                         xmin=0, ymin=0, zmin=0,
                                        periodic_xy=True, flip_z=False):
        
        # barycenters (vectorized)
        bx = 0.5 * (data[hs["x1"]] + data[hs["x2"]])
        by = 0.5 * (data[hs["y1"]] + data[hs["y2"]])
        bz = 0.5 * (data[hs["z1"]] + data[hs["z2"]])

        # The integer indices of the vertices where the boolean mask need is True
        # idx = np.nonzero(mask)[0]
        # xs, ys, zs = bx[idx], by[idx], bz[idx]
        xs, ys, zs = bx[mask], by[mask], bz[mask]

        # grid params
        Ny, Nz, Nx = self.voxel_number_y, self.voxel_number_z, self.voxel_number_x
        dx, dy, dz = self.voxel_dx, self.voxel_dy, self.voxel_dz

        if flip_z:
            zs = -zs
        
        if periodic_xy:
            Lx, Ly = Nx * dx, Ny * dy
            xs = (xs - xmin) % Lx + xmin
            ys = (ys - ymin) % Ly + ymin

        ix = np.floor((xs - xmin) / dx).astype(np.int32)
        iy = np.floor((ys - ymin) / dy).astype(np.int32)
        iz = np.floor((zs - zmin) / dz).astype(np.int32)

        # clamp to valid range (if not periodic or due to tiny FP drift)
        np.clip(ix, 0, Nx - 1, out=ix)
        np.clip(iy, 0, Ny - 1, out=iy)
        np.clip(iz, 0, Nz - 1, out=iz)

        return iy, iz, ix
    
    
    def apply_to_voxel(self, props):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """

        for name in self.inputs:
            self.voxels[name].fill(0)
        
        for vid in props["vertex_index"].keys():
            if props["length"][vid] > 0:
                if props["voxel_neighbor"][vid] is not None:
                    vy, vz, vx = props["voxel_neighbor"][vid]
                    for name in self.inputs:
                        # print(name,  props[name])
                        self.voxels[name][vy][vz][vx] += props[name][vid]
                else:
                    print(f"WARNING! segment {vid} did not send its status to the soil")


    def apply_to_voxel_fast(self, iy, iz, ix, data, hs, model_name, mask):
        for name in self.inputs:
            if name in self.pullable_inputs[model_name]:
                source_variables = self.pullable_inputs[model_name][name]
                to_apply = np.zeros(mask.sum(), dtype=np.float64)
                for variable, unit_conversion in source_variables.items():
                    to_apply += unit_conversion * data[hs[variable]][mask]
            else:
                to_apply = data[hs[name]][mask]

            np.add.at(self.voxels[name], (iy, iz, ix), to_apply) 

            # print(name, self.voxels[name].sum())


    def get_from_voxel(self, props, soil_outputs):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        for vid, (vy, vz, vx) in props["voxel_neighbor"].items():
            for name in soil_outputs:
                if name != "voxel_neighbor":
                    props[name][vid] = self.voxels[name][vy][vz][vx]
        
        return props

    def get_from_voxel_fast(self, iy, iz, ix, data, hs, soil_outputs, mask):
        # cols = np.flatnonzero(mask)
        # Nx, Nz = self.voxel_number_x, self.voxel_number_z
        # linear_index = ((iy * Nz + iz) * Nx + ix)
        # print("idx shape", ix.shape)
        # print(linear_index.shape)
        for name in soil_outputs:
            # print('vs assigned', self.voxels[name].ravel())
            # print(name, ix[cols].shape, data[hs[name], mask].shape, self.voxels[name][iy, iz, ix].shape)
            data[hs[name], mask] = self.voxels[name][iy, iz, ix]
            # print(name, data[hs[name], :])
            


    def pull_available_inputs(self, props, model_name):
        # vertices = props["vertex_index"].keys()
        vertices = [vid for vid in props["vertex_index"].keys() if props["living_struct_mass"][vid] > 0]
        
        for input, source_variables in self.pullable_inputs[model_name].items():
            if input not in props:
                props[input] = {}
            # print(input, source_variables)
            props[input].update({vid: sum([props[variable][vid]*unit_conversion 
                                           for variable, unit_conversion in source_variables.items()]) 
                                 for vid in vertices})
        return props

    
    def __call__(self, queue_plants_to_soil, queues_soil_to_plants, soil_outputs: list=[], *args):

        # We get fluxes and voxel interception from the plant mtgs (If none passed, soil model can be autonomous)
        # Waiting for all plants to put their outputs
        t1 = time.time()

        batch = []

        # A1l inputs are cumulatives of each plants and should be initialized at 0. at each step
        for name in self.inputs:
            self.voxels[name].fill(0.)

        for _ in range(len(queues_soil_to_plants)):
            plant_data = queue_plants_to_soil.get()
            batch.append(plant_data)

            # Deplaced here because if one plant hangs, their is not reason not to retreive others in the meantime
            self.get_from_plant(plant_data)
        
        t2 = time.time()
        if debug: print("soil waits plants: ", t2 - t1)

        # Run the soil model
        self.choregrapher(module_family=self.__class__.__name__, *args)

        self.simulation_time_hours += 1

        homogeneize_properties = False
        if homogeneize_properties:
            v = self.voxels
            v["dissolved_mineral_N"] = np.ones_like(v["dissolved_mineral_N"]) * (v["dissolved_mineral_N"] * (v["dry_soil_mass"])).sum() / (v["dry_soil_mass"]).sum()
            v["C_mineralN_soil"] = v["dissolved_mineral_N"] * (v["dry_soil_mass"] / (v["soil_moisture"] * v["voxel_volume"])) / 14 

        t3 = time.time()
        if debug: print("soil solve: ", t3 - t2)

        for plant_data in batch:
            plant_id = plant_data["plant_id"]

            self.send_to_plant(plant_data, soil_outputs)
            
            # Send a message to plants so that they can resume with last soil state
            queues_soil_to_plants[plant_id].put("finished")
        
        t4 = time.time()
        if debug: print("soil sends plants: ", t4 - t3)


    def get_from_plant(self, plant_data):
        """
        TODO : probably transfer to composite
        """
        # Unpacking message
        plant_id = plant_data["plant_id"]
        model_name = plant_data["model_name"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        hs = plant_data["handshake"]
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention

        iy, iz, ix = self.compute_mtg_voxel_neighbors_fast(buf, hs, mask=vertices_mask, flip_z=True)
        self.apply_to_voxel_fast(iy, iz, ix, buf, hs, model_name, vertices_mask)
        # Stored for sending to plants later
        self.voxel_neighbor[plant_id] = (iy, iz, ix)

        shm.close()

        # Legacy code commented
        # props = plant_data["data"]
        # props = self.pull_available_inputs(props, model_name)
        # props = self.compute_mtg_voxel_neighbors(props)
        # self.apply_to_voxel(props)
        # voxel_neighbors[id] = props["voxel_neighbor"]


    def send_to_plant(self, plant_data, soil_outputs):
        """
        TODO : probably transfer to composite
        """
        # Then apply the states to the plants
        plant_id = plant_data["plant_id"]
        hs = plant_data["handshake"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention
        self.get_from_voxel_fast(*self.voxel_neighbor[plant_id], buf, hs, soil_outputs, vertices_mask)

        shm.close()

        # legacy_code_commented
        # # EDIT : removed modification of the input props, there should be no variable link between inputs and outputs, excepted voxel neighbors for interception 
        # outputs = {name: {} for name in soil_outputs}
        # outputs["voxel_neighbor"] = vn
        # outputs = self.get_from_voxel(outputs, soil_outputs=soil_outputs)

    
    @actual
    @rate
    def cmf_transport(self):
        """
        Water and solute transport based on the CMF model
        """
        rain_intensity = self.voxels["rain_intensity"][0, 0, 0] * 24 # mm.day-1 are used by cmf : https://philippkraft.github.io/cmf/cmf_tut_meteostation.html?utm_source=chatgpt.com
        evaporation_intensity = self.voxels["soil_evaporation_intensity"][0, 0, 0] * 24 / 1e3 # mm.h-1 to m.day-1
        mineral_N_fertilization_amount = self.voxels["mineral_N_fertilization_rate"][0, 0, 0] * self.time_step # gN
        fertilization_volume = self.scene_xrange * self.scene_yrange * rain_intensity / 24 / 1e3 # m3 for current hour, # NOTE should be included in rain input time series
        fertigation_no3 = mineral_N_fertilization_amount / fertilization_volume if fertilization_volume > 0. else 0.

        # First retreive solute amounts from the CN-biochem model (see initiate_cmf for why amounts and not concentrations are exchanged)
        voxel_solute_amounts = {solute_name: self.voxels[solute_name] * self.voxels["dry_soil_mass"] * self.cmf_to_voxel_volume_ratio
                                for solute_name in self.cmf_accounted_solutes}

        # Clamp the requested water_uptake to what is actually available in each voxel this timestep, so that a
        # locally over-demanding root never asks the CMF solver to extract more water than the voxel holds.
        available_water = np.maximum(self.voxels["voxel_volume"] * (self.voxels["soil_moisture"] - self.theta_R), 0.)
        max_uptake = self.max_uptake_fraction_of_available_water * available_water / self.time_step
        if np.any(self.voxels["water_uptake"] > max_uptake):
            print("WARNING, roots uptake exceeds a minimal available fraction: CLAMPED")
        self.voxels["actual_water_uptake"] = np.minimum(self.voxels["water_uptake"], max_uptake)

        # Adjust inputs for this specific time step
        for ix in range(self.voxel_number_x):
            for iy in range(self.voxel_number_y):
                cell = self.cmf_cells[self.cmf_id_grid[ix][iy]]
                self.rainfall_nodes[self.cmf_id_grid[ix][iy]].intensity = rain_intensity
                self.rainfall_nodes[self.cmf_id_grid[ix][iy]].set_conc(self.cmf_project.solutes[self.cmf_accounted_solutes.index("dissolved_mineral_N")], fertigation_no3)
                # NOTE (free drainage): the former fixed-moisture water table reset of the bottom layer is gone; free drainage is a CMF
                # connection installed once in initiate_cmf. (The reset also created N on every capillary supply from the water table:
                # in a 48 h test the scene gained 10 times more mineral N with it than with a closed bottom.)

                # NOTE (leaching fix): soil evaporation from the top layer. Actual evaporation is reduced proportionally to the relative
                # wetness of that layer (simple supply limitation, avoids drying it below theta_R). Canopy shading is not accounted for.
                top_wetness = min(max(self.wetness_from_theta(self.voxels["soil_moisture"][iy, 0, ix]), 0.), 1.)
                self.evaporation_nodes[self.cmf_id_grid[ix][iy]].set_flux(-evaporation_intensity * top_wetness * cell.area) # m3.day-1

                # Other layers keep evolving under cmf's own mass-conservative state between calls: we only hand it
                # the current root water uptake as a prescribed sink flux, instead of overwriting theta/potential.
                for iz, l in enumerate(cell.layers):
                    self.uptake_nodes[(iy, iz, ix)].set_flux(-self.voxels["actual_water_uptake"][iy, iz, ix] * 24 * 3600) # m3.s-1 to m3.day-1
                    for solute_name, solute in zip(self.cmf_accounted_solutes, self.cmf_project.solutes):
                        l[solute].state = voxel_solute_amounts[solute_name][iy, iz, ix]

        # Fertigation amount entering each column with rain this step (CMF units), for the leaching balance below
        rain_volume_per_cell = rain_intensity / 1e3 * self.voxel_dx * self.voxel_dy * self.time_step / (24 * 3600) # m3
        fertigation_per_cell = fertigation_no3 * rain_volume_per_cell
        mineral_N_index = self.cmf_accounted_solutes.index("dissolved_mineral_N")
        mineral_N_solute = self.cmf_project.solutes[mineral_N_index]

        # Even for a single time_step, solver needs to iterate to actually run on its own
        [t for t in self.cmf_solver.run(self.cmf_solver.t, self.cmf_solver.t + cmf.h * self.time_step/3600, cmf.h * self.time_step/3600)]

        # 7. Extract results as NumPy arrays
        # TODO : optimization only because this already runs well, but in theory, accessing arrays through indices is highly unefficient and cmf fully with arrays would be better, but didn't find how to do it yet
        self.voxels["mineral_N_leaching"].fill(0.)
        for ix in range(self.voxel_number_x):
            for iy in range(self.voxel_number_y):
                cell = self.cmf_cells[self.cmf_id_grid[ix][iy]]
                # Root water uptake and evaporation carry no solutes, so the only mineral N leaving a column is free drainage:
                # leaching = amount before + fertigation - amount after
                if not self.no_flux_bottom_boundary:
                    amount_before = voxel_solute_amounts["dissolved_mineral_N"][iy, :, ix].sum()
                    amount_after = sum(l[mineral_N_solute].state for l in cell.layers)
                    self.voxels["mineral_N_leaching"][iy, -1, ix] = (amount_before + fertigation_per_cell - amount_after) / self.cmf_to_voxel_volume_ratio[iy, -1, ix] / self.time_step
                for iz, l in enumerate(cell.layers):
                    self.voxels["soil_moisture"][iy, iz, ix] = l.theta
                    self.voxels["water_potential_soil"][iy, iz, ix] = (l.potential - l.position[2]) * 1000 * 9.81 # Convert to Pa
                    for solute_name, solute in zip(self.cmf_accounted_solutes, self.cmf_project.solutes):
                        self.voxels[solute_name][iy, iz, ix] = l[solute].state / self.cmf_to_voxel_volume_ratio[iy, iz, ix] / self.voxels["dry_soil_mass"][iy, iz, ix]

        if self.mineral_N_diffusion_between_voxels:
            self.mineral_N_diffusion_step()

    def mineral_N_diffusion_step(self):
        """
        Explicit diffusion of dissolved mineral N between face-neighboring voxels, applied after the CMF advective transport.

        NOTE: CMF moves solutes only with water fluxes. Root water uptake is small here, so nothing resupplied a voxel depleted by roots,
        while unrooted neighbors at the same depth held 2 to 5 times more N. Estimated for the 7.5 cm layer of a previous run, diffusion
        through a single face would already bring ~2e-7 gN/h against a root uptake of ~3e-7 gN/h in the rooted voxel.

        - Gradient: concentration in soil water, C = mass / (theta * voxel_volume).
        - Effective diffusion coefficient (per unit soil area): De = theta * tau * Dw, with the Millington-Quirk tortuosity
          tau = theta^(7/3) / phi^2 (phi approximated by theta_S), and Dw corrected for temperature with Stokes-Einstein (T / viscosity).
        - Face coefficient: harmonic mean of the two voxels. Boundaries: periodic in x and y (as the lateral CMF connections of a
          symmetrical scene), no flux at the soil surface and at the bottom (vertical exchanges there are advective, handled by CMF).
        - Fluxes are antisymmetric, so N is conserved. Sub-steps keep the explicit scheme stable (one step per hour in practice).
        """
        v = self.voxels
        theta = np.maximum(v["soil_moisture"], 1e-6)
        water_volume = theta * v["voxel_volume"]
        mass = v["dissolved_mineral_N"] * v["dry_soil_mass"] # gN per voxel

        T = v["soil_temperature"] + 273.15
        water_viscosity = 2.414e-5 * 10 ** (247.8 / (T - 140.)) # Pa.s, Vogel equation
        Dw = self.nitrate_diffusion_coefficient_water * (T / 298.15) * (8.90e-4 / water_viscosity)
        tortuosity = theta ** (7. / 3.) / self.theta_S ** 2
        De = theta * tortuosity * Dw # m2.s-1

        dx, dy, dz = self.voxel_dx, self.voxel_dy, self.voxel_dz
        # (axis, distance between voxel centers, exchange area, periodic)
        directions = ((2, dx, dy * dz, True), (0, dy, dx * dz, True), (1, dz, dx * dy, False))

        # Stability of the explicit scheme on concentrations: dt * D_apparent * sum(1/d^2) <= 1/2, with D_apparent = tau * Dw
        stable_time_step = 0.5 / ((tortuosity * Dw).max() * (2 / dx**2 + 2 / dy**2 + 2 / dz**2))
        n_substeps = max(1, int(np.ceil(self.time_step / stable_time_step)))
        dt = self.time_step / n_substeps

        net_gain = np.zeros_like(mass)
        for _ in range(n_substeps):
            C = mass / water_volume # gN.m-3 of soil water
            gain = np.zeros_like(mass)
            for axis, distance, area, periodic in directions:
                De_next = np.roll(De, -1, axis=axis)
                De_face = 2 * De * De_next / np.maximum(De + De_next, 1e-300)
                # flux from each voxel to its +1 neighbor along this axis (gN.s-1)
                flux = De_face * (C - np.roll(C, -1, axis=axis)) / distance * area
                if not periodic:
                    # no exchange between the last and the first layer
                    index = [slice(None)] * flux.ndim
                    index[axis] = -1
                    flux[tuple(index)] = 0.
                gain += np.roll(flux, 1, axis=axis) - flux
            mass += gain * dt
            net_gain += gain * dt

        v["dissolved_mineral_N"] = mass / v["dry_soil_mass"]
        v["mineral_N_diffusion"] = net_gain / self.time_step

    
    @actual
    @rate # Rate because it should impact the states of mimics beforehand
    def _DOC(self, DOC, dry_soil_mass, hexose_exudation_massic, phloem_hexose_exudation_massic, mucilage_secretion_massic, cells_release_massic, hexose_uptake_from_soil_massic, phloem_hexose_uptake_from_soil_massic, amino_acids_diffusion_from_roots, amino_acids_diffusion_from_xylem, amino_acids_uptake):
        C_rhizodeposition = (hexose_exudation_massic + phloem_hexose_exudation_massic + mucilage_secretion_massic + cells_release_massic - hexose_uptake_from_soil_massic - phloem_hexose_uptake_from_soil_massic 
                      + amino_acids_diffusion_from_roots + amino_acids_diffusion_from_xylem - amino_acids_uptake)
        # print(hexose_exudation_massic.max(), phloem_hexose_exudation_massic.max(), mucilage_secretion_massic.max(), cells_release_massic.max(), hexose_uptake_from_soil_massic.max(), phloem_hexose_uptake_from_soil_massic.max(), 
        #               amino_acids_diffusion_from_roots.max(), amino_acids_diffusion_from_xylem.min(), amino_acids_uptake.max())
        # print("RD", C_rhizodeposition.min(), C_rhizodeposition.mean(), C_rhizodeposition.max())
        return np.maximum(DOC + C_rhizodeposition * (self.time_step / dry_soil_mass), 0.)
        
    @actual
    @rate # Rate because it should impact the states of mimics beforehand
    def _DON(self, DON, dry_soil_mass, amino_acids_diffusion_from_roots, amino_acids_diffusion_from_xylem, amino_acids_uptake):
        N_rhizodeposition = (amino_acids_diffusion_from_roots + amino_acids_diffusion_from_xylem - amino_acids_uptake) * 14 / 12 / self.CN_ratio_amino_acids
        return np.maximum(DON + N_rhizodeposition * (self.time_step / dry_soil_mass), 0.)
    

    def litter_inputs_profile(self):
        """
        Annual residue inputs to MIMICS per voxel (mgC.cm-3.y-1), split into metabolic and structural litter as in Daycent.
        Constant over the plough layer, then exponentially decreasing with depth.
        """
        z_mean = (self.voxels["z1"] + self.voxels["z2"]) / 2
        depth_factor = np.where(z_mean <= self.litter_input_plough_depth, 1.,
                                np.exp(-(z_mean - self.litter_input_plough_depth) / self.litter_input_decay_length))
        total_input = self.litter_input_rate * depth_factor

        N_in_mass = self.litter_C_fraction / self.litter_total_CN
        fmet = 0.85 - 0.013 * self.litter_lignin_fraction / N_in_mass
        structural_CN = (self.litter_total_CN - self.litter_metabolic_CN * fmet) / (1 - fmet)

        return dict(structural=total_input * (1 - fmet), structural_CN=np.full_like(total_input, structural_CN),
                    metabolic=total_input * fmet, metabolic_CN=np.full_like(total_input, self.litter_metabolic_CN))

    @actual
    @state
    def mimics_cn_states(self):
        if self.run_mimics:
            soil_temperature = self.voxels["soil_temperature"]
            voxel_volume = self.voxels["voxel_volume"]
            dry_soil_mass = self.voxels["dry_soil_mass"]
            # The MIMICS available pool only receives what changed in its dissolved fraction since the last write
            # (rhizodeposition, root uptake, CMF transport), instead of being overwritten by it
            to_mg = 1e3 * dry_soil_mass / voxel_volume / 1e6 # g.g-1 soil to mg/cm3
            DOC = np.maximum(self.mimics.SOC_available + (self.voxels["DOC"] - self._DOC_written) * to_mg, 0.)
            DON = np.maximum(self.mimics.SON_available + (self.voxels["DON"] - self._DON_written) * to_mg, 0.)
            dissolved_mineral_N_previous = 1e3 * self.voxels["dissolved_mineral_N"] * dry_soil_mass / voxel_volume / 1e6 # gN.g-1 soil to mgN/cm3

            net_N_uptake = 1e3 * 3600 * (
                self.voxels["mineralN_uptake"]
                - self.voxels["mineralN_diffusion_from_roots"]
                - self.voxels["mineralN_diffusion_from_xylem"]) / voxel_volume / 1e6 # gN.s-1 to mgN/cm3/h

            net_N_uptake_pos = np.maximum(net_N_uptake, 0.)
            dissolved_mineral_N = dissolved_mineral_N_previous - np.minimum(net_N_uptake, 0.) # TODO, check if a negative uptake would be valid to the model input

            # NOTE (uptake availability fix): the N debt of previous steps is requested from the voxel on top of the current root uptake.
            # MIMICS then reports which part of this request could not be met (DIN_deficit) instead of silently clamping DIN to 0.
            time_step_in_hours = self.time_step / 3600
            debt_previous = 1e3 * self.voxels["mineralN_uptake_debt"] / voxel_volume / 1e6 # gN to mgN/cm3
            requested_N_uptake = net_N_uptake_pos + debt_previous / time_step_in_hours # mgN/cm3/h

            # Same depth-dependent residue inputs as for the MIMICS steady-state initialization
            li = self._litter_inputs
            hourly_C_input = (li["structural"] + li["metabolic"]) / (365 * 24)
            hourly_N_input = (li["structural"] / li["structural_CN"] + li["metabolic"] / li["metabolic_CN"]) / (365 * 24)

            initial_C = (self.mimics.Litter_DOC + self.mimics.Litter_POC + self.mimics.MBC_r + self.mimics.MBC_K + self.mimics.SOC_physical + self.mimics.SOC_chemical + DOC + hourly_C_input) * (1e6 * voxel_volume)
            initial_SOM_N = (self.mimics.Litter_DON + self.mimics.Litter_PON + self.mimics.SON_physical + self.mimics.SON_chemical + DON + hourly_N_input) * (1e6 * voxel_volume)

            self.mimics(soil_temperature=soil_temperature,
                        labile_OC=DOC, # Expects mgC/cm3
                        labile_ON=DON,
                        labile_IN=dissolved_mineral_N,
                        net_N_uptake=requested_N_uptake,
                        litter_inputs=li["structural"],
                        litter_CN=li["structural_CN"],
                        rhizodeposits_inputs=li["metabolic"],
                        rhizodeposits_CN=li["metabolic_CN"])

            concentrations_conversion = 1e6 * voxel_volume / dry_soil_mass / 1e3
            self.voxels["dissolved_mineral_N"] = self.mimics.DIN * concentrations_conversion
            # Unmet part of the request, carried over to the next time step
            self.voxels["mineralN_uptake_debt"] = self.mimics.DIN_deficit * (1e6 * voxel_volume) / 1e3 # mgN/cm3 to gN
            # Mineral N actually withdrawn from the voxel by roots during this step (mgN/cm3)
            root_N_withdrawn = requested_N_uptake * time_step_in_hours - self.mimics.DIN_deficit
            self.voxels["DOC"] = self.DOM_dissolved_fraction * self.mimics.SOC_available * concentrations_conversion
            self.voxels["DON"] = self.DOM_dissolved_fraction * self.mimics.SON_available * concentrations_conversion
            self._DOC_written, self._DON_written = self.voxels["DOC"].copy(), self.voxels["DON"].copy()
            # TODO: log litter inputs if varying
            self.voxels["MAOC"] = self.mimics.SOC_physical * concentrations_conversion
            self.voxels["MAON"] = self.mimics.SON_physical * concentrations_conversion
            self.voxels["POC"] = self.mimics.SOC_chemical * concentrations_conversion
            self.voxels["PON"] = self.mimics.SON_chemical * concentrations_conversion
            self.voxels["microbial_C"] = (self.mimics.MBC_r + self.mimics.MBC_K) * concentrations_conversion
            self.voxels["microbial_N"] = (self.mimics.MBN_r + self.mimics.MBN_K) * concentrations_conversion

            final_C = (self.mimics.Litter_DOC + self.mimics.Litter_POC + self.mimics.MBC_r + self.mimics.MBC_K + self.mimics.SOC_physical + self.mimics.SOC_chemical + self.mimics.SOC_available) * (1e6 * voxel_volume)
            final_SOM_N = (self.mimics.Litter_DON + self.mimics.Litter_PON + self.mimics.SON_physical + self.mimics.SON_chemical + self.mimics.SON_available) * (1e6 * voxel_volume)

            self.voxels["microbial_respiration"] = (initial_C - final_C) / dry_soil_mass / 1e3 / 3600
            self.voxels["N_mineralization"] = (initial_SOM_N - final_SOM_N) / dry_soil_mass / 1e3 / 3600
            # Uses the N actually withdrawn by roots: with the former clamp, unmet root demand appeared here as a negative microbial uptake
            self.voxels["microbial_N_uptake"] = (((dissolved_mineral_N - self.mimics.DIN - root_N_withdrawn)  * (1e6 * voxel_volume)) + (initial_SOM_N - final_SOM_N)) / dry_soil_mass / 1e3 / self.time_step

        

    @actual
    @state
    def update_temperature(self):
        if (self.simulation_time_hours + 12) % 24 == 0.:
            permanent_wilting_point = 0.11
            field_capacity=0.31

            self.current_date += pd.Timedelta(days=1)
            self.update_daily_evaporation()
            current_inputs = self.campbell_input_df.loc[self.campbell_input_df["DATE"] == self.current_date]
            sunrise = current_inputs["SUNUP"].iat[0]
            sunset = current_inputs["SUNDWN"].iat[0]
            soil_moisture = self.voxels['soil_moisture'].mean(axis=(0, 2)) # cm3/cm3

            config = self.temperature_model.daily_config(self.static_config_dict,
                                        DATE=self.current_date,
                                        T2M=current_inputs["TM"].iat[0], # °C
                                        TMIN=current_inputs["TN"].iat[0], # °C
                                        TMAX=current_inputs["TX"].iat[0], # °C
                                        RAIN=current_inputs["RR"].iat[0], # mm
                                        SRAD=current_inputs["RG"].iat[0] / 100, # from J/cm2 to MJ/m2
                                        DAYLD=sunset - sunrise, # h
                                        SUNUP=sunrise, # h
                                        SUNDN=sunset, # h
                                        plant_available_water=(soil_moisture - permanent_wilting_point) / (field_capacity - permanent_wilting_point),
                                        LAI=0.,
                                        albedo=0.12,
                                        irrig=0.,
                                        aboveGroundDM=0.)

            res, previous_outputs = self.temperature_model.run(config=config, nb_steps=1, previous_outputs=self.previous_campbell_states)

            self.previous_campbell_states = previous_outputs
            temperature_array = res["TSLD"].loc[res["Layer"] >= 1].to_numpy() # We ignore layer 0 that gives surface temperature
            self.voxels['soil_temperature'][:] = temperature_array[None, :, None]
    
    # @actual
    # @rate
    # Not active while cmf handles it
    def _voxel_mineral_N_fertilization(self, mineral_N_fertilization_rate, dry_soil_mass):
        return dry_soil_mass * mineral_N_fertilization_rate.mean() / dry_soil_mass.sum()

    @potential
    @rate
    def _mineral_N_net_mineralization(self, soil_temperature, voxel_volume):
        """CN-Wheat mineralization function"""
        
        # First temperature effect on Vmax
        Tref = 20 + 273.15
        Tk = soil_temperature + 273.15
        R = 8.3144  #: Physical parameter: Gas constant (J mol-1 K-1)
        deltaHa = 55  # 89.7  #: Enthalpie of activation of parameter pname (kJ mol-1)
        deltaS = 0.48  # 0.486  #: entropy term of parameter pname (kJ mol-1 K-1)
        deltaHd = 154  # 149.3 #: Enthalpie of deactivation of parameter pname (kJ mol-1)

        f_activation = np.exp((deltaHa * (Tk - Tref)) / (R * 1E-3 * Tref * Tk))  #: Energy of activation (normalized to unity)

        f_deactivation = (1 + np.exp((Tref * deltaS - deltaHd) / (Tref * R * 1E-3))) / (1 + np.exp((Tk * deltaS - deltaHd) / (Tk * R * 1E-3)))  #: Energy of deactivation (normalized to unity)

        T_effect_Vmax = f_activation * f_deactivation

        mineralization_rate = 2.05e-6 / 1e6 # mol N nitrates m-3 s-1
        return 44.44 * mineralization_rate * voxel_volume * T_effect_Vmax * 14 # expecting gN g-1 of soil # NOTE : 44 factor specific to small soil volume to match rates of CN-Wheat!
    
    
    # RhizoDep legacy functions as we don't know how to define a realistic external availability otherwise, no constant proportion of DOC
    @rate
    def _mucilage_degradation(self, Cs_mucilage_soil, root_exchange_surface, soil_temperature):
        """
        This function computes the rate of mucilage degradation outside the root (in mol of equivalent-hexose per second)
        for a given root element. Only the external surface of the root element is taken into account here, similarly to
        what is done for mucilage secretion.
        :param Cs_mucilage_soil: mucilage concentration in soil solution (equivalent hexose, mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_mucilage_degradation_rate_max = self.mucilage_degradation_rate_max * self.temperature_modification(
                                                                    soil_temperature=soil_temperature,
                                                                    T_ref=self.mucilage_degradation_rate_max_T_ref,
                                                                    A=self.mucilage_degradation_rate_max_A,
                                                                    B=self.mucilage_degradation_rate_max_B,
                                                                    C=self.mucilage_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of mucilage
        # in the soil:
        result = corrected_mucilage_degradation_rate_max * root_exchange_surface * Cs_mucilage_soil / (
                self.Km_mucilage_degradation + Cs_mucilage_soil)
        result[result < 0.] = 0.

        return result

    @rate
    def _cells_degradation(self, Cs_cells_soil, root_exchange_surface, soil_temperature):
        """
        This function computes the rate of root cells degradation outside the root (in mol of equivalent-hexose per second)
        for a given root element. Only the external surface of the root element is taken into account as the exchange
        surface, similarly to what is done for root cells release.
        :param Cs_cells_soil: released cells concentration in soil solution (equivalent hexose, mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_cells_degradation_rate_max = self.cells_degradation_rate_max * self.temperature_modification(
                                                                        soil_temperature=soil_temperature,
                                                                        T_ref=self.cells_degradation_rate_max_T_ref,
                                                                        A=self.cells_degradation_rate_max_A,
                                                                        B=self.cells_degradation_rate_max_B,
                                                                        C=self.cells_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of root cells
        # in the soil:
        result = corrected_cells_degradation_rate_max * root_exchange_surface * Cs_cells_soil / (
                self.Km_cells_degradation + Cs_cells_soil)
        result[result < 0] = 0.
        return result

    @state
    def _dissolved_mineral_N(self, dissolved_mineral_N, dry_soil_mass, mineral_N_net_mineralization, mineralN_diffusion_from_roots, mineralN_diffusion_from_xylem, mineralN_uptake):
        if self.run_mimics:
            return dissolved_mineral_N
        else:
            balance = dissolved_mineral_N + (self.time_step / dry_soil_mass) * (
                mineral_N_net_mineralization
                + mineralN_diffusion_from_roots
                + mineralN_diffusion_from_xylem
                - mineralN_uptake
                # + voxel_mineral_N_fertilization
                ) - self.voxels["mineralN_uptake_debt"] / dry_soil_mass
            
            # NOTE (uptake availability fix): same debt logic as in mimics_cn_states, instead of clamping the pool to 0 and creating N
            self.voxels["mineralN_uptake_debt"] = np.maximum(-balance, 0.) * dry_soil_mass
            balance[balance < 0] = 0

            return balance
    

    # @note Post state coupling variables

    @segmentation
    @state
    def _C_mineralN_soil(self, dissolved_mineral_N, dry_soil_mass, soil_moisture, voxel_volume):
        return dissolved_mineral_N * (dry_soil_mass / (soil_moisture * voxel_volume)) / 14

    @segmentation
    @state
    def _C_amino_acids_soil(self, DON, dry_soil_mass, soil_moisture, voxel_volume):
        f_DON_in_available = 0.1
        f_AA_in_DON = 0.2
        return f_DON_in_available * f_AA_in_DON * DON * (dry_soil_mass / (soil_moisture * voxel_volume)) / 14 / 1.4
    
    @segmentation
    @state
    def _C_hexose_soil(self, DOC, dry_soil_mass, soil_moisture, voxel_volume):
        f_DOC_in_available = 0.1
        f_hex_in_available = 0.5
        Conc = f_DOC_in_available * f_hex_in_available * DOC * (dry_soil_mass / (soil_moisture * voxel_volume)) / 12 / 6
        # print(Conc.min(), Conc.mean(), Conc.max())
        return f_DOC_in_available * f_hex_in_available * DOC * (dry_soil_mass / (soil_moisture * voxel_volume)) / 12 / 6
    
    @segmentation
    @state
    def _Cs_mucilage_soil(self, Cs_mucilage_soil, soil_moisture, voxel_volume, mucilage_secretion_massic, mucilage_degradation):
        balance = Cs_mucilage_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
            mucilage_secretion_massic
            - mucilage_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    @segmentation
    @state
    def _Cs_cells_soil(self, Cs_cells_soil, soil_moisture, voxel_volume, cells_release_massic, cells_degradation):
        balance = Cs_cells_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
                cells_release_massic
                - cells_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    @postsegmentation
    @state
    def _Cv_solutes_soil(self, C_hexose_soil, Cs_mucilage_soil, Cs_cells_soil, C_mineralN_soil, C_amino_acids_soil):
        return C_hexose_soil + Cs_mucilage_soil + Cs_cells_soil + C_mineralN_soil + C_amino_acids_soil + self.C_solutes_background # Commented until we are sure of proper initialization and balance of these different concentrations
        # return C_mineralN_soil 
    
    @state
    def _water_volume(self, soil_moisture, voxel_volume):
        return soil_moisture * voxel_volume


    def temperature_modification(self, soil_temperature=15, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
        """
        This function calculates how the value of a process should be modified according to soil temperature (in degrees Celsius).
        Parameters correspond to the value of the process at reference temperature T_ref (process_at_T_ref),
        to two empirical coefficients A and B, and to a coefficient C used to switch between different formalisms.
        If C=0 and B=1, then the relationship corresponds to a classical linear increase with temperature (thermal time).
        If C=1, A=0 and B>1, then the relationship corresponds to a classical exponential increase with temperature (Q10).
        If C=1, A<0 and B>0, then the relationship corresponds to bell-shaped curve, close to the one from Parent et al. (2010).
        :param T_ref: the reference temperature
        :param A: parameter A (may be equivalent to the coefficient of linear increase)
        :param B: parameter B (may be equivalent to the Q10 value)
        :param C: parameter C (either 0 or 1)
        :return: the new value of the process
        """
        # We compute a temperature-modified process, correspond to a Q10-modified relationship,
        # based on the work of Tjoelker et al. (2001):
        if C != 0 and C != 1:
            print("The modification of the process at T =", soil_temperature,
                  "only works for C=0 or C=1!")
            print("The modified process has been set to 0.")
            return np.zeros_like(soil_temperature)

        modified_process = process_at_T_ref * (A * (soil_temperature - T_ref) + B) ** (1 - C) \
                           * (A * (soil_temperature - T_ref) + B) ** (
                                   C * (soil_temperature - T_ref) / 10.)
        
        if C == 1:
            modified_process[(A * (soil_temperature - T_ref) + B) < 0.] = 0.

        modified_process[modified_process < 0.] = 0.

        return modified_process