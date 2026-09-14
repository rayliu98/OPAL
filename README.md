# OPAL: An Optimal Distributed Learning Framework via Mutual Validation Between Sites

================================================

# OPAL-TIT-2026-Accepted-supp.pdf

This supplementary material contains additional theoretical and numerical results, the hyperparameter settings used in the simulation and real data analysis, and alternative empirical versions of the limiting distribution.

# code documentation

There are three folders in the `code` directory: `real_data`, `simulation`, and `source`.  
The `simulation` folder contains the code for the experiments reported in Section IV and Online Appendices C-G, C-H, C-I, and C-J, while the `real_data` folder contains the code for the experiments reported in Section V.  
The `source` folder contains the necessary modules to implement OPAL.

## simulation

### sim1.R

The file `sim1.R` is the R script used to produce the experimental results in Section IV-A, which focuses on prediction accuracy under a model misspecification setting.  The results are related to Table I in the main text and Tables SIII and SIV in Online Appendix C-G.

### sim2_*.R

The files `sim2_*.R` are the R scripts used to produce the experimental results in Section IV-B, which focuses on the influence of six irregular designs denoted by $^*$. The results are related to Figures 2 and 3 in the main text and Figures S2-S6 in Online Appendix C-G.  

The specific meanings of the $^*$ symbols are provided below.

**=correlation*: correlation among covariates

**=rare&rare_extreme*: unbalanced rate of labels

**=dft*: kurtosis in covariate distribution

**=lambdae*: skewness in covariate distribution

**=sigmis*: extent of model misspecification

**=nby*: difference in sample sizes across sites

### OAPR_random_cluster.R

The file `OAPR_random_cluster.R` is the R script used to produce the experimental results in Experiment C-J in Online Appendix C-J. It conducts alternative random selection methods for clustering in OAPR, including random selection and CV-error-based clustering, to compare with the recommended Shapley value-based clustering.

### OAPR_sensitivity_G_batch_ver.R

The file `OAPR_sensitivity_G_batch_ver.R` is the R script used to produce the experimental results in Online Appendix C-I, corresponding to Experiment C-I.1. It conducts sensitivity analyses on the number of clusters $G$ to confirm the robustness of the OAPR method.

### OAPR_sensitivity_Shapley_batch_ver.R

The file `OAPR_sensitivity_Shapley_batch_ver.R` is the R script used to produce the experimental results in Online Appendix C-I, corresponding to Experiment C-I.2. It conducts sensitivity analyses on the deviation of the clustering metrics (parameter estimator $\hat{\bm\beta}^{(m)}$ or Shapley value $\widetilde{\textit{shap}}^{(m)}$) to confirm the robustness of the OAPR method.

### sim1_deep_model_batch_ver.R

The file `sim1_deep_model_batch_ver.R` is the R script used to produce the experimental results in Experiments C-H.1 and C-H.2 in Online Appendix C-H. It examines the predictive performance and computational overhead of our method when dealing with a larger number of sites and when some sites employ deep models.

### sim1_deep_model_Kfold2_batch_ver.R

The file `sim1_deep_model_Kfold2_batch_ver.R` is the companion R script of the file `sim1_deep_model_batch_ver.R`, which sets $J=2$ instead of $J=5$.

## real_data

The Fashion-MNIST dataset can be downloaded from https://github.com/zalandoresearch/fashion-mnist or accessed in R via the `dataset_fashion_mnist()` function from the *keras* package. See the R script `Fashion_MNIST_Image_data_cleaning.R` for details.

### Fashion_MNIST_Image_data_cleaning.R

The file `Fashion_MNIST_Image_data_cleaning.R` is the R script used to preprocess the Fashion-MNIST image data in the *keras* package for subsequent modeling and analysis.

### Fashion_MNIST_Image_OPAL_M_5_10_25.R

The file `Fashion_MNIST_Image_OPAL_M_5_10_25.R` is the R script used to produce the experimental results for $M=5,10,25$ presented in Table II in Section V and Figure S7 in Online Appendix C-G.

### Fashion_MNIST_Image_OPAL_M_50.R

The file `Fashion_MNIST_Image_OPAL_M_50.R` is the R script used to produce the experimental results for $M=50$ presented in Table II in Section V.

### distillation.py

The file `distillation.py` is an intermediate Python script used by `Fashion_MNIST_Image_OPAL_M_5_10_25.R` and `Fashion_MNIST_Image_OPAL_M_50.R` to implement CNN1 and CNN2 at local sites.

### distillation_Full.py

The file `distillation_Full.py` is an intermediate Python script used by `Fashion_MNIST_Image_OPAL_M_5_10_25.R` and `Fashion_MNIST_Image_OPAL_M_50.R` to implement CNN1 and CNN2 using the Full method (pooled data).

## source

### ci_OARFISH.R

The file `ci_OARFISH.R` is used to construct confidence intervals for GLM models under OPAL.

### coef_ncvreg.R

The file `coef_ncvreg.R` supplements the built-in function `coef_ncvreg` in the *ncvreg* package.

### lam_names.R

The file `lam_names.R` defines a representative method for selecting $\lambda$ in SCAD.

### logistic_solvers_forR.R

The file `logistic_solvers_forR.R` contains code implementing the methods CSL, CEASE, AGD, ADMM, and GIANT.

### meta.R

The file `meta.R` contains code for implementing the Meta method.

### OARFISH_solver.R

The file `OARFISH_solver.R` contains code for implementing OPAL, including weight solvers for OAP and OAPR, a clustering algorithm (K-means) for OAPR, and a cluster-number ($G$) selector using the "elbow" and "distortion" methods.

### predict_ncvreg.R

The file `predict_ncvreg.R` supplements the built-in function `predict_ncvreg` in the *ncvreg* package.

