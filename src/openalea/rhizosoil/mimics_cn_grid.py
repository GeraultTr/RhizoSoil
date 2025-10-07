"""
Radiocarbon implementation of MIMICS-CN v1.0 (Kyker-Snowman et al., 2020).

Functions `initialize_pools`, `get_parameters`, `FXEQ` in this file are
modified from R-code in "MIMICS-CN_forRelease.Rmd" (Kyker-Snowman, 2019).

* Source code of MIMICS-CN v1.0:
    Kyker-Snowman, E. (2019). "EmilyKykerSnowman/MIMICS-CN-for-publication
    v1.0" (v1.0). Zenodo. https://doi.org/10.5281/zenodo.3534562

* Associated manuscript:
    Kyker-Snowman, E., et al. (2020). "Stoichiometrically coupled carbon and
    nitrogen cycling in the MIcrobial-MIneral Carbon Stabilization model
    version 1.0 (MIMICS-CN v1.0)". Geoscientific Model Development, 13(9),
    4413–4434. https://doi.org/10.5194/gmd-13-4413-2020


Original work Copyright (C) 2020  Emily Kyker-Snowman  (no license specified)

Modified work Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.

This file has been further edited by Tristan Gérault 2025/09/19
"""

import numpy as np
import pandas as pd
from numba import njit
import scipy.optimize


def initialize_pools_grid(shape, dtype=np.float64):
    """
    Copied and adapted from `initializePools` function
    in "MIMICS-CN_forRelease.Rmd" (Kyker-Snowman, 2019).
    """

    state = np.empty((15, *shape), dtype=dtype)

    # LIT_1 = 1
    state[0, ...] = 1.
    # LIT_2 = 1
    state[1, ...] = 1.
    # MIC_1 = 1
    state[2, ...] = 1.
    # MIC_2 = 1
    state[3, ...] = 1.
    # SOM_1 = 1
    state[4, ...] = 1.
    # SOM_2 = 1
    state[5, ...] = 1.
    # SOM_3 = 1
    state[6, ...] = 1.

    # LIT_1_N = .1
    state[7, ...] = .1
    # LIT_2_N = .1
    state[8, ...] = .1
    # MIC_1_N = .1
    state[9, ...] = .1
    # MIC_2_N = .1
    state[10, ...] = .1
    # SOM_1_N = .1
    state[11, ...] = .1
    # SOM_2_N = .1
    state[12, ...] = .1
    # SOM_3_N = .1
    state[13, ...] = .1
    # DIN     = .1
    state[14, ...] = .1
    
    return state


# @njit
def get_parameters_rhizodeposits_grid(annual_NPP_C, annual_RD_C, clay_pct, Tsoil, input_RD_CN, input_LIT_CN):
    """
    All parameter values and comments are from `siteSpecificParameters`
    in "MIMICS-CN_forRelease.Rmd" (Kyker-Snowman, 2019).
    Comments with double-hash ## are lines of code removed by asb219.

    Comments with triple dash from Tristan Gérault
    NPP stands for net primary production, for litter inputs
    RD stands for rhizodeposition
    inputs in mgC/cm3/h
    """

    ## ANPP_C  = LTERdata$ANPP / 2       # convert to gC/m2/y from g/m2/y
    ### ANPP_C  = annual_NPP_C ### NOTE Tristan comment
    ANPP_C = annual_NPP_C + annual_RD_C
    ## strSite = as.character(LTERdata$Site)  #convert site names to string
    ## nsites  = length(strSite)
    ## npts   = 6*10*14 #6 litter * 10 years * 14 sites
    clay  = clay_pct/100 ##LTERdata$CLAY2/100 
    tsoi  = Tsoil ##LTERdata$MAT
    ## nsites = length(LTERdata$Site)
    # lig   = input_lignin_pct/100 #LTERdata$LIG/100
    # Nnew  = 1/input_CN/2.5 #1/LTERdata$CN/2.5             #N in litter additions
    # fMET1 = 0.85 - 0.013 * lig / Nnew    #as partitioned in Daycent
    fMET1 = annual_RD_C / ANPP_C

    #Parameters related to inputs
    # EST_LIT_in  = ANPP_C / (365*24)   #gC/m2/h (from g/m2/y, Knapp et al. Science 2001)
    ## BAG_LIT_in  = 100      #gC/m2/h
    # soilDepth       = 30  ## leave this as such!!
    ## h2y         = 24*365
    ## MICROtoECO  = soilDepth * 1e4 * 1e6 / 1e6   #mgC/cm3 to kgC/km2
    # EST_LIT     = EST_LIT_in  * 1e3 / 1e4    #mgC/cm2/h 
    ## BAG_LIT     = BAG_LIT_in  * 1e3 / 1e4    #mgC/cm2/h
    fMET        = fMET1
    #Litter inputs to MET/STR
    Inputs = np.stack((
        annual_RD_C / (365*24),      #partitioned to layers
        annual_NPP_C / (365*24)  ## units: mgC/cm3/h
    ), axis=0)
    FI       = np.array((0.05,0.3))#c(0.05, 0.05)#

    ## BAG      = array(NA, dim=c(6,2))              #litter BAG inputs to MET/STR
    ## for (i in 1:6) {
    ## BAG[i,1]   = (BAG_LIT / soilDepth) * bagMET[i]      #partitioned to layers
    ## BAG[i,2]   = (BAG_LIT / soilDepth) * (1-bagMET[i])
    ## }

    #Parameters related to stabilization mechanisms
    fCLAY       = clay
    fPHYS    = 0.1 * np.stack((
        .15 * np.exp(1.3*fCLAY), 
        0.1 * np.exp(0.8*fCLAY)
        ), axis=0) #Sulman et al. 2018
    fCHEM    = 3*np.stack((
        0.1 * np.exp(-3*fMET) * 1, 
        0.3 * np.exp(-3*fMET) * 1
        ), axis=0) #Sulman et al. 2018 #fraction to SOMc
    fAVAI    = 1-(fPHYS + fCHEM)
    desorb   = 2e-5      * np.exp(-4.5*(fCLAY)) #Sulman et al. 2018
    desorb   = 0.05*desorb
    Nleak   = 0.2#.1   #This is the proportion N lost from DIN pool at each hourly time step.

    #Parameters related to microbial physiology and pool stoichiometry
    CUE        = np.array((0.55, 0.25, 0.75, 0.35))  #for LITm and LITs entering MICr and MICK, respectively
    NUE        = .85         #Nitrogen stoichiometry of fixed pools
    # CN_m        = 15
    # CN_s        = (input_CN-CN_m*fMET)/(1-fMET)
    CN_m        = input_RD_CN
    CN_s        = input_LIT_CN
    ## CN_s_BAG    =  (bagCN-CN_m*bagMET)/(1-bagMET)
    CN_r        =6#5
    CN_K        =10#8

    turnover      = np.stack((
        5.2e-4*np.exp(0.3*(fMET)), 
        2.4e-4*np.exp(0.1*(fMET))
        ), axis=0) #WORKS BETTER FOR N_RESPONSE RATIO
    ## turnover_MOD1 = np.sqrt(annual_NPP/100)  #basicaily standardize against NWT
    ## turnover_MOD1[turnover_MOD1 < 0.6] = 0.6 # correction not used in LIDET resutls 
    ## turnover_MOD1[turnover_MOD1 > 1.3] = 1.3      #Sulman et al. 2018
    turnover_MOD1 = np.minimum(1.3, np.maximum(0.6, np.sqrt(ANPP_C/100)))
    turnover      = turnover * turnover_MOD1
    turnover = turnover/2.2
    turnover = turnover**2*0.55/(.45*Inputs)
    densDep = 2#1 #This exponent controls the density dependence of microbial turnover. Currently anything other than 1 breaks things.

    fracNImportr  =  0 #No N import for r strategists
    fracNImportK  =  0.2 #Only K strategists can import N

    #Parameters related to temperature-sensitive enzyme kinetics
    TSOI        = tsoi
    #Calculate Vmax & (using parameters from German 2012)
    #from "gamma" simulations "best", uses max Vslope, min Kslope
    Vslope = np.array([
        0.043, #META LIT to MIC_1
        0.043, #STRU LIT to MIC_1 
        0.063, #AVAI SOM to MIC_1 
        0.043, #META LIT to MIC_2 
        0.063, #STRU LIT to MIC_2 
        0.063  #AVAI SOM to MIC_2 
        ])[:, None, None, None]
    Vint     = 5.47
    aV       = 8e-6
    aV       = aV*.06 #Forward
    Vmax     = np.exp(TSOI * Vslope + Vint) * aV

    Kslope = np.array([
        0.017, #META LIT to MIC_1
        0.027, #STRU LIT to MIC_1 
        0.017, #AVAI SOM to MIC_1 
        0.017, #META LIT to MIC_2
        0.027, #STRU LIT to MIC_2
        0.017 #AVAI SOM to MIC_2
        ])[:, None, None, None]
    Kint     = 3.19
    aK       = 10
    aK       = aK/20 #Forward
    Km       = np.exp(Kslope * TSOI + Kint) * aK

    #Enzyme kinetic modifiers:
    k        = 2.0    #2.0            #REDUCED FROM 3 TO 1, REDUCES TEXTURE EFFECTS ON SOMa decay
    a        = 2.0    #2.2            #increased from 4.0 to 4.5
    cMAX     = 1.4                    #ORIG 1.4 Maximum CHEM SOM scalar w/   0% Clay 
    cMIN     = 1.2                    #ORIG 1.4 Minimum CHEM SOM scalar w/ 100% Clay 
    cSLOPE   = cMIN - cMAX            #Slope of linear function of cSCALAR for CHEM SOM  
    pSCALAR  = a * np.exp(-k*(np.sqrt(fCLAY)))  #Scalar for texture effects on SOMp

    #------------!!MODIFIERS AS IN MIMICS2_b!!---------------
    MOD1     = np.array((10, 2*.75, 10, 3, 3*.75, 2))[:, None, None, None]

    # Explicit stack needed since not only scalars
    ones = np.ones_like(pSCALAR)
    MOD2 = np.stack((
        8.0 * ones,              # idx 0
        2.0 * ones,              # idx 1
        4.0 * pSCALAR,           # idx 2  (texture dependent)
        2.0 * ones,              # idx 3
        4.0 * ones,              # idx 4
        6.0 * pSCALAR            # idx 5  (texture dependent)
    ), axis=0)                   # (6,nx,ny,nz)

    VMAX     = Vmax * MOD1
    KM       = Km / MOD2
    KO       = np.array((6,6))     #Values from Sulman et al. 2018

    return (
        Inputs, VMAX, KM, CUE,
        fPHYS, fCHEM, fAVAI, FI,
        turnover, ## LITmin, SOMmin, MICtrn,
        desorb, ## DEsorb, OXIDAT,
        ## LITminN, SOMminN, MICtrnN,
        ## DEsorbN, OXIDATN,
        KO,
        ## CNup, DINup, Nspill, Overflow,
        ## upMIC_1, upMIC_1_N,
        ## upMIC_2, upMIC_2_N,
        NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep
    )


# @njit
def FXEQ_grid(state,
        Inputs, VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
        KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep, RootUptakeIN=0., time_step_in_hours=1, leaching=True):
    """
    Most of the code is copied and adapted from the `FXEQ` function
    in "MIMICS-CN_forRelease.Rmd" (Kyker-Snowman, 2019).
    Addition by asb219: 14C implementation (last 4 blocks of code).
    
    All comments starting with a single # are from the original code.
    Comments starting with ## are by asb219.
    """
    # Unpack pools into arrays
    LIT_1    = state[0]
    LIT_2    = state[1]
    MIC_1    = state[2]
    MIC_2    = state[3]
    SOM_1    = state[4]
    SOM_2    = state[5]
    SOM_3    = state[6]
    LIT_1_N  = state[7]
    LIT_2_N  = state[8]
    MIC_1_N  = state[9]
    MIC_2_N  = state[10]
    SOM_1_N  = state[11]
    SOM_2_N  = state[12]
    SOM_3_N  = state[13]
    DIN      = state[14]

    Inputs_1, Inputs_2 = Inputs
    VMAX_1, VMAX_2, VMAX_3, VMAX_4, VMAX_5, VMAX_6 = VMAX
    KM_1, KM_2, KM_3, KM_4, KM_5, KM_6 = KM
    CUE_1, CUE_2, CUE_3, CUE_4 = CUE
    fPHYS_1, fPHYS_2 = fPHYS
    fCHEM_1, fCHEM_2 = fCHEM
    fAVAI_1, fAVAI_2 = fAVAI
    FI_1, FI_2 = FI
    turnover_1, turnover_2 = turnover
    KO_1, KO_2 = KO


    #Carbon fluxes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Flows to and from MIC_1
    LITmin_1 = MIC_1 * VMAX_1 * LIT_1 / (KM_1 + LIT_1)#LIT_1)   #MIC_1 decomp of MET lit
    LITmin_2 = MIC_1 * VMAX_2 * LIT_2 / (KM_2 + LIT_2)#LIT_2)   #MIC_1 decomp of STRUC lit
    SOMmin_1 = MIC_1 * VMAX_3 * SOM_3 / (KM_3 + SOM_3)#SOM_3)   #decomp of SOMa by MIC_1
    MICtrn_1 = MIC_1**densDep * turnover_1  * fPHYS_1 #MIC_1 turnover to PHYSICAL SOM
    MICtrn_2 = MIC_1**densDep * turnover_1  * fCHEM_1 #MIC_1 turnover to CHEMICAL SOM
    MICtrn_3 = MIC_1**densDep * turnover_1  * fAVAI_1 #MIC_1 turnover to AVAILABLE SOM

    #Flows to and from MIC_2
    LITmin_3 = MIC_2 * VMAX_4 * LIT_1 / (KM_4 + LIT_1)#LIT_1)   #decomp of MET litter
    LITmin_4 = MIC_2 * VMAX_5 * LIT_2 / (KM_5 + LIT_2)#LIT_2)   #decomp of SRUCTURAL litter
    SOMmin_2 = MIC_2 * VMAX_6 * SOM_3 / (KM_6 + SOM_3)#SOM_3)   #decomp of SOMa by MIC_2
    MICtrn_4 = MIC_2**densDep * turnover_2  * fPHYS_2                  #MIC_2 turnover to PHYSICAL  SOM
    MICtrn_5 = MIC_2**densDep * turnover_2  * fCHEM_2                  #MIC_2 turnover to CHEMICAL  SOM
    MICtrn_6 = MIC_2**densDep * turnover_2  * fAVAI_2                  #MIC_2 turnover to AVAILABLE SOM
    
    DEsorb    = SOM_1 * desorb  #* (MIC_1 + MIC_2)      #desorbtion of PHYS to AVAIL (function of fCLAY)
    OXIDAT    = ((MIC_2 * VMAX_5 * SOM_2 / (KO_2*KM_5 + SOM_2)) +#SOM_2)) +
                 (MIC_1 * VMAX_2 * SOM_2 / (KO_1*KM_2 + SOM_2)))#SOM_2)))  #oxidation of C to A

    #Nitrogen fluxes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Flows to and from MIC_1
    LITminN_1 =  LITmin_1*LIT_1_N/(LIT_1+1e-100)#LITmin_1/CN_m
    LITminN_2 =  LITmin_2*LIT_2_N/(LIT_2++1e-100)#LITmin_2/CN_s
    SOMminN_1 =  SOMmin_1*(SOM_3_N/(SOM_3++1e-100))#SOMmin_1*(SOM_3_N/SOM_3)#*relRateN
    MICtrnN_1 =  MICtrn_1*MIC_1_N/(MIC_1+1e-100)#MICtrn_1/CN_r
    MICtrnN_2 =  MICtrn_2*MIC_1_N/(MIC_1+1e-100)#MICtrn_2/CN_r
    MICtrnN_3 =  MICtrn_3*MIC_1_N/(MIC_1+1e-100)#MICtrn_3/CN_r

    #Flows to and from MIC_2
    LITminN_3 =  LITmin_3*LIT_1_N/(LIT_1+1e-100)#LITmin_3/CN_m
    LITminN_4 =  LITmin_4*LIT_2_N/(LIT_2+1e-100)#LITmin_4/CN_s
    SOMminN_2 =  SOMmin_2*(SOM_3_N/(SOM_3+1e-100))#SOMmin_2*(SOM_3_N/SOM_3)#*relRateN
    MICtrnN_4 =  MICtrn_4*MIC_2_N/(MIC_2+1e-100)#MICtrn_4/CN_K
    MICtrnN_5 =  MICtrn_5*MIC_2_N/(MIC_2+1e-100)#MICtrn_5/CN_K
    MICtrnN_6 =  MICtrn_6*MIC_2_N/(MIC_2+1e-100)#MICtrn_6/CN_K
    
    DEsorbN    =  DEsorb*(SOM_1_N/(SOM_1+1e-100))#*relRateN
    OXIDATN    =  OXIDAT*(SOM_2_N/(SOM_2+1e-100))#*relRateN
    DINup_1   = (1-Nleak)*DIN*MIC_1/(MIC_1+MIC_2+1e-100) #Partitions DIN between microbial pools based on relative biomass
    DINup_2   = (1-Nleak)*DIN*MIC_2/(MIC_1+MIC_2+1e-100)

    #####
    upMIC_1    = CUE_1*(LITmin_1 + SOMmin_1) + CUE_2*(LITmin_2)
    upMIC_1_N  = NUE* (LITminN_1 + SOMminN_1) + NUE*(LITminN_2) + DINup_1
    CNup_1    = (upMIC_1)/(upMIC_1_N+1e-100) #Trying to optimize run speed here by avoiding /0
    Overflow_1 = (upMIC_1) - (upMIC_1_N)*np.minimum(CN_r, CNup_1)
    Nspill_1   = (upMIC_1_N) - (upMIC_1)/np.maximum(CN_r, CNup_1)
  
    upMIC_2    = CUE_3*(LITmin_3 + SOMmin_2) + CUE_4*(LITmin_4)
    upMIC_2_N  = NUE*(LITminN_3 + SOMminN_2) + NUE*(LITminN_4) + DINup_2
    CNup_2    = (upMIC_2)/(upMIC_2_N+1e-100)
    Overflow_2 = (upMIC_2) - (upMIC_2_N)*np.minimum(CN_K, CNup_2)
    Nspill_2   = (upMIC_2_N) - (upMIC_2)/np.maximum(CN_K, CNup_2)
    ######

    dLIT_1 = Inputs_1*(1-FI_1) - LITmin_1 - LITmin_3
    dLIT_2 = Inputs_2*(1-FI_2) - LITmin_2 - LITmin_4
    dMIC_1 = CUE_1*(LITmin_1 + SOMmin_1) + CUE_2*(LITmin_2) - (MICtrn_1 + MICtrn_2 + MICtrn_3) - Overflow_1
    dMIC_2 = CUE_3*(LITmin_3 + SOMmin_2) + CUE_4*(LITmin_4) - (MICtrn_4 + MICtrn_5 + MICtrn_6) - Overflow_2 
    dSOM_1 = Inputs_1*FI_1 + MICtrn_1 + MICtrn_4 - DEsorb 
    dSOM_2 = Inputs_2*FI_2 + MICtrn_2 + MICtrn_5 - OXIDAT
    dSOM_3 = MICtrn_3 + MICtrn_6 + DEsorb + OXIDAT - SOMmin_1 - SOMmin_2

    dLIT_1_N = Inputs_1*(1-FI_1)/CN_m - LITminN_1 - LITminN_3
    dLIT_2_N = Inputs_2*(1-FI_2)/CN_s - LITminN_2 - LITminN_4
    dMIC_1_N = NUE*(LITminN_1 + SOMminN_1) + NUE*(LITminN_2) - (MICtrnN_1 + MICtrnN_2 + MICtrnN_3) + DINup_1 - Nspill_1
    dMIC_2_N = NUE*(LITminN_3 + SOMminN_2) + NUE*(LITminN_4) - (MICtrnN_4 + MICtrnN_5 + MICtrnN_6) + DINup_2 - Nspill_2
    dSOM_1_N = Inputs_1*FI_1/CN_m + MICtrnN_1 + MICtrnN_4 - DEsorbN
    dSOM_2_N = Inputs_2*FI_2/CN_s + MICtrnN_2 + MICtrnN_5 - OXIDATN
    dSOM_3_N = MICtrnN_3 + MICtrnN_6 + DEsorbN + OXIDATN - SOMminN_1 - SOMminN_2

    dDIN = (
        (1-NUE)*(LITminN_1 + LITminN_2 + SOMminN_1) +  #Inputs from r decomp
        (1-NUE)*(LITminN_3 + LITminN_4 + SOMminN_2) +  #Inputs from K decomp
        Nspill_1 + Nspill_2 - DINup_1 - DINup_2    #Uptake to microbial pools and spillage
        - RootUptakeIN # Roots uptake mgN/cm3/h
    )

    if leaching:
        LeachingLoss = Nleak*DIN ### NOTE added condition to be able to handle transport from another module
        dDIN = dDIN-LeachingLoss #N leaching losses
    
    out = np.empty((15,) + state.shape[1:], dtype=state.dtype)
    out[0]  = dLIT_1
    out[1]  = dLIT_2
    out[2]  = dMIC_1
    out[3]  = dMIC_2
    out[4]  = dSOM_1
    out[5]  = dSOM_2
    out[6]  = dSOM_3
    out[7]  = dLIT_1_N
    out[8]  = dLIT_2_N
    out[9]  = dMIC_1_N
    out[10] = dMIC_2_N
    out[11] = dSOM_1_N
    out[12] = dSOM_2_N
    out[13] = dSOM_3_N
    out[14] = dDIN
    return out * time_step_in_hours

def _solve_one_steady_state(params, bounds=(None, None), dt=1.0):
    # Solve a single-voxel 15-var system with fsolve→lsq fallback
    x0 = initialize_pools_grid((1,1,1)).ravel()  # (15,)
    def fun(x):
        s = x.reshape(15,1,1,1)
        r = FXEQ_grid(s, *params, RootUptakeIN=0.0, time_step_in_hours=dt, leaching=True)
        return r.reshape(15)
    try:
        x = scipy.optimize.fsolve(fun, x0, xtol=1e-10, maxfev=2000)
        lo, hi = bounds
        success = (lo is None or np.all(x > lo)) and (hi is None or np.all(x < hi))
        if not success:
            raise ValueError
        return x.reshape(15)
    except Exception:
        sol = scipy.optimize.least_squares(fun, x0, bounds=bounds)
        return (sol.x if sol.success else x0).reshape(15)


class MIMICS_CN:

    def __init__(self, time_step_in_hours=1, clay_percentage=np.array(())):

        self.time_step_in_hours = time_step_in_hours
        self.clay_percentage = clay_percentage
        soil_shape = clay_percentage.shape
        pools_shape = (15, *soil_shape)


        # Prepare output
        state = np.empty(pools_shape, dtype=np.float64)

        # If clay are floats, use unique levels robustly
        # (If they are exact bins like 10, 20, 30, you can drop 'round')
        clay_levels = np.unique(clay_percentage)

        for c in clay_levels:
            print(f"initializing clay level {c}")
            mask = clay_percentage == c

            # Build 1×1×1 parameter fields for this clay level
            scalar_shape = (1,1,1)
            prop = 0.5
            params = get_parameters_rhizodeposits_grid(
                annual_NPP_C = np.full(scalar_shape, 0.6), # Bolinder 2007
                annual_RD_C  = np.full(scalar_shape, 0.1), # Pausch et Kuzyakov
                clay_pct     = np.full(scalar_shape, c),
                Tsoil        = np.full(scalar_shape, 10),
                input_RD_CN  = np.full(scalar_shape, 15),
                input_LIT_CN = np.full(scalar_shape, 15),
            )

            # Solve once for this class (voxel-sized residual)
            sol_15 = _solve_one_steady_state(params, bounds=(0., 1e7), dt=time_step_in_hours)  # (15,)

            # Paste to all voxels of that class
            # arr[:, mask] view has shape (15, M) → broadcast sol[:, None]
            state[:, mask] = sol_15[:, None]

        self.state = state



    def __init__old(self, time_step_in_hours=1, clay_percentage=np.array(())):

        self.time_step_in_hours = time_step_in_hours
        self.clay_percentage = clay_percentage
        soil_shape = clay_percentage.shape
        pools_shape = (15, *soil_shape)
        prop = 0.5

        parameters = get_parameters_rhizodeposits_grid(
            annual_NPP_C = np.full(soil_shape, (1-prop)*100.), # gC/m2/y
            annual_RD_C = np.full(soil_shape, prop*100.), # gC/m2/y
            clay_pct = clay_percentage, # %
            Tsoil = np.full(soil_shape, 10.), # Celsius
            input_RD_CN = np.full(soil_shape, 15.), # adim
            input_LIT_CN = np.full(soil_shape, 15.)
        )

        # First guess of steady state
        steady_state_C_N_guess = initialize_pools_grid(soil_shape).ravel()

        # Find steady-state C and N stocks
        def func(state_C_N):
            state4d = state_C_N.reshape(pools_shape)
            deriv4d = FXEQ_grid(state4d, *parameters, RootUptakeIN=np.zeros(soil_shape), time_step_in_hours=time_step_in_hours)
            return deriv4d.ravel()
        print("entering")
        steady_state_C_N, success_C_N = self._find_steady_state(
            func, steady_state_C_N_guess, bounds=(0, 1e7), name='C,N'
        )

        self.state = (steady_state_C_N if success_C_N else steady_state_C_N_guess).reshape(pools_shape)


    def __call__(self, soil_temperature=10, labile_OC=5e-1, labile_ON=1e-2, labile_IN=1e-6, net_N_uptake=0., litter_inputs=50., litter_CN=15, rhizodeposits_inputs=50., rhizodeposits_CN=15.):
        
        state = self.state
        # Apply the input concentrations for labile pools, result of the transport processes
        state[6] = labile_OC
        state[13] = labile_ON
        state[14] = labile_IN
        
        # TODO cache parameters that do no vary with these inputs
        parameters = get_parameters_rhizodeposits_grid(
            annual_NPP_C = litter_inputs, # mgC/cm3/y
            annual_RD_C = rhizodeposits_inputs, # mgC/cm3/y
            clay_pct = self.clay_percentage, # %
            Tsoil = soil_temperature, # Celsius
            input_RD_CN = rhizodeposits_CN, # adim
            input_LIT_CN = litter_CN
        )

        deriv = FXEQ_grid(state, *parameters, RootUptakeIN=net_N_uptake, time_step_in_hours=self.time_step_in_hours, leaching=False)
        state += deriv
        np.maximum(state, 0.0, out=state)

    
    @property
    def Litter_DOC(self):
        return self.state[0]
    
    @property
    def Litter_POC(self):
        return self.state[1]
    
    @property
    def MBC_r(self):
        return self.state[2]
    
    @property
    def MBC_K(self):
        return self.state[3]
    
    @property
    def SOC_physical(self):
        return self.state[4]
    
    @property
    def SOC_chemical(self):
        return self.state[5]
    
    @property
    def SOC_available(self):
        return self.state[6]
    
    @property
    def Litter_DON(self):
        return self.state[7]
    
    @property
    def Litter_PON(self):
        return self.state[8]
    
    @property
    def MBN_r(self):
        return self.state[9]
    
    @property
    def MBN_K(self):
        return self.state[10]
    
    @property
    def SON_physical(self):
        return self.state[11]
    
    @property
    def SON_chemical(self):
        return self.state[12]
    
    @property
    def SON_available(self):
        return self.state[13]
    
    @property
    def DIN(self):
        return self.state[14]

    
    def _find_steady_state(self, func, x0, bounds=(None,None), name='C'):
        """
        Find a steady state solution `x` such that `func(x) = 0` with
        first guess `x0`. First try with `scipy.optimize.fsolve(func, x0)`,
        but if the result is out of bounds, then try again with
        `scipy.optimize.least_squares(func, x0, bounds=bounds)`.
        
        Parameters
        ----------
        func
            function taking a vector as its sole argument and returning
            a vector of same length
        x0 : np.ndarray
            first guess for the steady state solution
        bounds : (lo, hi), default (None, None)
            lower and upper bounds the solution vector; `lo` and `hi` can
            be `None`, scalar, or np.ndarray vector with length of `x0`
        name : str, default 'C'
            name of the quantity for which to find a steady state
        
        Returns
        -------
        x : np.ndarray or None
            steady state solution vector
        success : bool
            whether `x` is a good steady state solution
        """

        x = scipy.optimize.fsolve(func, x0)
        lo, hi = bounds
        success = (lo is None or all(x > lo)) and (hi is None or all(x < hi))

        if not success:
            print(
                f'scipy.optimize.fsolve failed to find steady state'
                f' of {name} for {self}.'
                ' Falling back to scipy.optimize.least_squares.'
            )

            sol = scipy.optimize.least_squares(func, x0, bounds=bounds)
            x = sol.x
            success = sol.success

            if not success:
                print(
                    f'scipy.optimize.least_squares failed to find {name}'
                    f' steady state for {self}.'
                    f' STATUS: {sol.status}. MESSAGE: {sol.message}'
                )

        return x, success

