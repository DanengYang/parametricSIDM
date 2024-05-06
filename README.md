# Parametric Model Analysis Tools for Self-Interacting Dark Matter Halos

### Relevant publications 

- Dark matter-only version: D. Yang, E. O. Nadler, H.-B. Yu, and Y.-M. Zhong, [arXiv:2305.16176](https://arxiv.org/abs/2305.16176), published in [JCAP 02, 032 (2024)](http://dx.doi.org/10.1088/1475-7516/2024/02/032)
- Dark matter plus baryon version: arXiv:24XX.XXXX
- Our method has been implemented in the [SASHIMI program for SIDM subhalos](https://github.com/shinichiroando/sashimi-si): S. Ando, S. Horigome, E. O. Nadler, D. Yang, and H.-B. Yu, [arXiv:2403.16633](https://arxiv.org/abs/2403.16633)

### Example applications

- Plot the parametric SIDM density profile 
  ```
  $ python demo.py
  ```

<img src="https://github.com/DanengYang/parametricSIDM/blob/main/figs/demo.png" alt="Illustrate the parametric SIDM density profile" width="300"/>


- The Mathematica notebook (basicHalos.nb) is tailored for the basic approach, allowing users to obtain density and velocity dispersion profiles in SIDM for individual halos.

- Applying the integral approach requires information about the evolution histories. We provide an example application for the SIDM-796 halo. 
Run (`$ python getVRmaxplotISO.py`) will generate the following figure

<img src="https://github.com/DanengYang/parametricSIDM/blob/main/figs/fig_tL_vmax_case_cdm_799_796_C4_1000bins.png" alt="The Vmax evolution of a deeply core collapsing SIDM subhalo from the parametric model with the integral approach (solid-green), basic approach (dotted-blue), and the SIDM simulation (solid-magenta), as well as the CDM counterpart (dashed-black).
" width="300"/>

- Generate a sample of isolated halos (samples/generateSamples_flatHMF.nb)

- Getting SIDM predictions for the generated samples: Hybrid approach 
   ```bash
   $ python printHybridVmaxRmaxSIDMwithbaryon_sigma0_100_w_100_DMO.py
   $ python plotHybridVmaxRmaxSIDMwithbaryon_sigma0_100_w_100_DMO.py
   $ python plotHybridVmaxRmaxSIDMwithbaryon_SIDM_sigma0_100_w_100_total.py
   ```
<img src="https://github.com/DanengYang/parametricSIDM/blob/main/figs/fig_vmax_rmax_SIDM_baryon_flat_HMF_sigma0_100_w_100_DMO.png" alt="The Vmax-Rmax distribution of the dark matter component for the velocity-dependent SIDM model" width="300"/><img src="https://github.com/DanengYang/parametricSIDM/blob/main/figs/fig_vmax_rmax_SIDM_baryon_flat_HMF_sigma0_100_w_100.png" alt="The Vmax-Rmax distribution of dark matter plus baryons for the velocity-dependent SIDM model" width="300"/>

- Getting SIDM predictions for the generated samples: Integral approach (exemplified through 14 points, adjust for processing more)
   ```bash
   $ printIntegralVmaxRmaxSIDMwithbaryon_sigma0_100_w_100_DMO.py
   $ plotIntegralVmaxRmaxSIDMwithbaryon_SIDM_sigma0_100_w_100_DMO.py
   $ plotIntegralVmaxRmaxSIDMwithbaryon_SIDM_sigma0_100_w_100_total.py
   ```

Our codes are free to copy and modify. If you find them useful, please quote our paper(s). 
If you have any questions or comments, please contact [danengy_at_ucr.edu].

