# parametricSIDM
Parametric Model Analysis Tool for Self-Interacting Dark Matter Halos

### Relevant publications 

- Dark matter-only version: arXiv:2305.16176, published in [JCAP 02, 032 (2024)](http://dx.doi.org/10.1088/1475-7516/2024/02/032)
- Dark matter plus baryon version: arXiv:24XX.XXXX

### Example applications

- <img src="https://github.com/DanengYang/parametricSIDM/blob/main/figs/demo.png" alt="Illustrate the parametric SIDM density profile" width="300"/>


- The Mathematica notebook (basicHalos.nb) is tailored for the basic approach, allowing users to obtain density and velocity dispersion profiles in SIDM for individual halos.

- Applying the integral approach requires information about the evolution histories. We provide an example application for the SIDM-796 halo. 
Run 
$ python getVRmaxplotISO.py
will generate the following figure

<img src="https://github.com/DanengYang/parametricSIDM/blob/main/figs/fig_tL_vmax_case_cdm_799_796_C4_1000bins.png" alt="The Vmax evolution of a deeply core collapsing SIDM subhalo from the parametric model with the integral approach (solid-green), basic approach (dotted-blue), and the SIDM simulation (solid-magenta), as well as the CDM counterpart (dashed-black).
" width="300"/>


Our codes are free to copy and modify. If you find them useful, please quote our papers. 

