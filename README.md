# Parametric Model Analysis Tools for Self-Interacting Dark Matter Halos

The parametric model for self-interacting dark matter (SIDM) halos allows for obtaining halo density profiles based on a few calibrated equations under an assumed SIDM model, using parameters from their CDM counterparts, such as NFW scale parameters or halo mass and concentration parameters. The effect of a baryon potential can be incorporated, assuming the density profile conforms to the Hernquist model.

### Relevant works 

- [1] **Dark matter-only version**: D. Yang, E. O. Nadler, H.-B. Yu, and Y.-M. Zhong, [arXiv:2305.16176](https://arxiv.org/abs/2305.16176), published in [JCAP 02, 032 (2024)](http://dx.doi.org/10.1088/1475-7516/2024/02/032)
- [2] **Dark matter plus baryon version**: D. Yang, [arXiv:2405.03787](https://arxiv.org/abs/2405.03787), published in [Phys. Rev. D](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.103044)
- [3] **Testing** the parametric model using matched halos in **cosmological simulations**: D. Yang, E. O. Nadler, and H.-B. Yu, [arXiv:2406.10753](http://arxiv.org/abs/2406.10753), published in [Phys. Dark Universe](https://www.sciencedirect.com/science/article/pii/S2212686425000020)
- [4] For a **lensing specific** parametric model for SIDM halos, see S. Hou, D. Yang, N. Li, and G. Li, [arXiv:2502.14964](https://arxiv.org/abs/2502.14964), with [public codes on GitHub.](https://github.com/HouSiyuan2001/SIDM_Lensing_Model). 
- [5] Our method has been implemented in the [**SASHIMI program** for SIDM subhalos](https://github.com/shinichiroando/sashimi-si): S. Ando, S. Horigome, E. O. Nadler, D. Yang, and H.-B. Yu, [arXiv:2403.16633](https://arxiv.org/abs/2403.16633)


Our codes are free to copy and modify. 
Please quote the relevant paper(s), if you find them useful. 
For any questions or comments, please contact [yangdn_at_pmo.ac.cn].

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
- [Tutorial slides](https://github.com/DanengYang/parametricSIDM/blob/main/tutorial/A%20Quick%20Start%20for%20Working%20with%20SIDM%20Halos.pdf). Example scripts available under the tutorial folder. 

### Notes 

- The ratio $t/t_c$ can exceed 1 if the evolution time $t$ is greater than the core collapse time $t_c$. However, the gravothermal phase $\tau = t/t_c$ should be truncated at 1.1, as the halo has collapsed, and we assume that the outer halo region in the long mean-free-path regime maintains approximately the same configuration.
- The maximum value of $\tau$ being 1.1 instead of 1 in the parametric model reflects the uncertainty in $t_c$. Our parametrized evolution histories for $\rho_s$, $r_s$, and $r_c$ can more robustly model the shape, while the normalization of tau can be rescaled to account for uncertainties in $t_c$.
- The effective constant cross section $\sigma_{\text{eff}}$ relies on an approximate effective velocity dispersion, which does not necessarily align with the default choice. It may require adjustment based on the velocity dispersion profile of the specific SIDM model you are using.



