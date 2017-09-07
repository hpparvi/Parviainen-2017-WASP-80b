# WASP-80b ground-based transmission spectroscopy
**Hannu Parviainen, E. Palle, G. Chen, F. Murgas, L. Nortmann, G. Nowak, A. Booth, M. Abazorius, S. Aigrain**
**Instituto de Astrofísica de Canarias & University of Oxford, 2017**

These repository contains the code, data, and notebooks that cover the broadband and transmission spectroscopy analysis of WASP-80b using three sets of previously published light curves (27 broadband LCs) and two nights of GTC-observed transmission spectroscopy. We start with a broadband analysis of the previously observed datasets to constrain the orbital parameters, and then continue with broad- and narrow band transmission spectroscopy.

**NOTE: We are currently reorganising and cleaning up the code. This will hopefully not take too long, but a lot of things may be broken at the moment.**

**Cleanup status 7.9.2017** 

 - **Broadband analysis has been cleaned and revised to work with george 0.3.** 
 - **The GTC transmission spectroscopy notebooks are still a mess, and the code uses george 0.2.**

## Introduction

The `notebooks` directory contains the IPython notebooks going through the analyses step by step, the `src` directory contains the code for the analysis, and the `bin` directory contains executable scripts.

## Requirements

    Python, IPython, Jupyter, pandas, astropy, SciPy, NumPy, matplotlib, emcee, george, mpi4py, tqdm
    PyTransit, PyExoTK, LDTk, PyDE

## 1 External broadband data
We combine the GTC-observed data with three external multicolour datasets by Triaud et al. (2013), Mancini et al. (2014), and Fukui et al. (2014).

The results from external data modelling are stored in results/external.h5 with results structured as

    run_name/de
    run_name/mc
    run_name/fc

where de stores the final differential evolution, mc the population of the last emcee step, and fc the flattened emcee chain.

### 1.1 White noise runs
We carry out four runs using only Triaud et al. and Mancini et al. data, with run names

    ckwn, ckwn_ldtk
    vkwn, vkwn_ldtk
    
where `wn` stands for white noise, `ck` for constant radius ratio, `vk` for wavelength-dependent radius ratio, and `ldtk` has been added to the name if the run uses LDTk to constrain the limb darkening.

### 1.2 Gaussian process hyperparameter optimisation
The white noise runs are used to optimise the GP hyperparameters for all the external datasets.

### 1.3 Red noise runs
We carry out further four runs assuming red noise modelled using Gaussian processes (GPs), with run names

    ckrn, ckrn_ldtk
    vkrn, vkrn_ldtk

where `rn` stands for red noise.

## 2 GTC data
We have two nights (July 16 and August 25) of observations. The target star (WASP-80) is placed on CCD 2 and the comparison star on CCD 1. The transmission spectroscopy analysis is carried out for two versions of the light curves, one produced by a pipeline by H. Parviainen, and another by a pipeline by G. Chen (HP and GC datasets, respectively).

The light curves reduced by the HP pipeline are stored in

    results/gtc_light_curves_hp.h5
    
and the light curves reduced by the GC pipeline are stored in

    results/gtc_light_curves_gc.h5
    
The h5 files contain two groups, night1 and night2, storing the photometry and the auxiliary data taken from the fits headers as Pandas dataframes.

---

&copy; 2017 Hannu Parviainen
