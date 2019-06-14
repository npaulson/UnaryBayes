# UnaryBayes
This is a collection of codes developed for Paulson et al. (2019).

## General instructions
Tools for performing the Bayesian analysis are contained within `core_compute.py` and plotting tools are in `core_plot.py`. Most of the remaining codes are scripts (e.g. `Bayesian_Inference.py` or `liquid_lin.py`) which perform a variety of Bayesian inference computations based on a common template. `example_outlier.ipynb` is a Jupyter notebook that demonstrates the use of the `core_compute.py` and `core_plot.py` for Bayesian inference.

For a quick primer to Bayesian statistics, see the [Bayesian fundamentals - model calibration and selection](https://github.com/npaulson/Bayesian-statistics-notebooks/blob/master/Bayesian%20fundamentals%20-%20model%20calibration%20and%20selection.ipynb) notebook.

## Instruction to reproduce paper results for toy problems
1. Run `Bayesian_inference.py`, changing the D['order'] parameter on line 87 to explore polynomials of varying order. This will produce Figures 9a, 10, and 11.
2. Run `outliers_normal.py` and `outliers_students-t.py` to produce Figures 12a and 12b, respectively.
3. Run `errorbars_standard.py` and `errorbars_yma.py` to produce Figures 13 and 14.
4. Run `thermo_consistency.py` and `thermo_consistency_separate.py` to produce Figure 1.

## Instructions to reproduce paper results for Hf case study (Section 3):
1. Run `data_process/data_process_4.py`
2. For each of `alpha_quart_debye.py`, `beta_quad.py`, and `liquid_lin.py`:
    * run a first time to get initial posterior
	* run a second time to use narrowed prior distributions and to evaluate the final marginal likelihoods
	* this will result in plots of the data/model-predictions with UQ, the univariate parameter distributions, a corner plot, a table with posterior statistics (used to produce Table 2), and a text output with the sampling time and marginal likelihood (used to produce Table 1)
3. Run `plot_all.py` to plot the data, model-predictions with UQ for each phase (alpha, beta, liquid)and property (Cp, H, S, G). This produces Figures 4 - 6.
4. Run `plot_model_differences.py` to plot the percent differences between the model prediction and previous Hf models (Figure 7).

## Required packages
* python/3.6.8
* emcee/2.2.1
* kombine/0.8.3
* matplotlib/3.0.2
* numpy/1.15.4
* pandas/0.23.4
* pymultinest/2.6
* scipy/1.2.1
* seaborn/0.9.0

## Paper reference
Paulson, N.H., Jennings, E., Stan, M. “Bayesian strategies for uncertainty quantification of the thermodynamic properties of materials,” International Journal of Engineering Science. 142 (2019) 74-93 https://doi.org/10.1016/j.ijengsci.2019.05.011
