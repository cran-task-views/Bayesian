---
name: Bayesian
topic: Bayesian Inference
maintainer: Jong Hee Park, Michela Cameletti, Xun Pang, Kevin M. Quinn
email: jongheepark@snu.ac.kr
version: 2022-04-06
source: https://github.com/cran-task-views/Bayesian/
output: pdf_document
---

## CRAN Task View: Bayesian Inference

Applied researchers interested in Bayesian statistics are increasingly
attracted to R because of the ease of which one can code algorithms to
sample from posterior distributions as well as the significant number of
packages contributed to the Comprehensive R Archive Network (CRAN) that
provide tools for Bayesian inference. This task view catalogs these
tools. In this task view, we divide those packages into four groups
based on the scope and focus of the packages. We first review R packages
that provide Bayesian estimation tools for a wide range of models. We
then discuss packages that address specific Bayesian models or
specialized methods in Bayesian statistics. This is followed by a
description of packages used for post-estimation analysis. Finally, we
review packages that link R to other Bayesian sampling engines such as
[JAGS](http://mcmc-jags.sourceforge.net/),
[OpenBUGS](http://www.openbugs.net/),
[WinBUGS](http://www.mrc-bsu.cam.ac.uk/software/bugs/),
[Stan](http://mc-stan.org/), and
[TensorFlow](https://www.tensorflow.org).

### General Purpose Model-Fitting Packages

-   The `r pkg("arm", priority = "core")` package contains R
    functions for Bayesian inference using lm, glm, mer and polr
    objects.
-   `r pkg("BACCO", priority = "core")` is an R bundle for
    Bayesian analysis of random functions. `r pkg("BACCO")`
    contains three sub-packages: emulator, calibrator, and approximator,
    that perform Bayesian emulation and calibration of computer
    programs.
-   `r pkg("bayesforecast", priority = "core")` provides various functions for Bayesian time series analysis using 'Stan' for full Bayesian inference. A wide range of distributions and models are supported, allowing users to fit Seasonal ARIMA, ARIMAX, Dynamic Harmonic Regression, GARCH, t-student innovation GARCH models, asymmetric GARCH, Random Walks, stochastic volatility models for univariate time series.    
-   `r pkg("bayesm", priority = "core")` provides R functions
    for Bayesian inference for various models widely used in marketing
    and micro-econometrics. The models include linear regression models,
    multinomial logit, multinomial probit, multivariate probit,
    multivariate mixture of normals (including clustering), density
    estimation using finite mixtures of normals as well as Dirichlet
    Process priors, hierarchical linear models, hierarchical multinomial
    logit, hierarchical negative binomial regression models, and linear
    instrumental variable models.

-   `r pkg("BayesianTools")` is an R package for
    general-purpose MCMC and SMC samplers, as well as plot and
    diagnostic functions for Bayesian statistics, with a particular
    focus on calibrating complex system models. Implemented samplers
    include various Metropolis MCMC variants (including adaptive and/or
    delayed rejection MH), the T-walk, two differential evolution MCMCs,
    two DREAM MCMCs, and a sequential Monte Carlo (SMC) particle filter.
-   `r pkg("LaplacesDemon")` seeks to provide a complete
    Bayesian environment, including numerous MCMC algorithms, Laplace
    Approximation with multiple optimization algorithms, scores of
    examples, dozens of additional probability distributions, numerous
    MCMC diagnostics, Bayes factors, posterior predictive checks, a
    variety of plots, elicitation, parameter and variable importance,
    and numerous additional utility functions.
-   `r pkg("loo")` provides functions for efficient
    approximate leave-one-out cross-validation (LOO) for Bayesian models
    using Markov chain Monte Carlo. The approximation uses Pareto
    smoothed importance sampling (PSIS), a new procedure for
    regularizing importance weights. As a byproduct of the calculations,
    `r pkg("loo")` also provides standard errors for
    estimated predictive errors and for the comparison of predictive
    errors between models. The package also provides methods for using
    stacking and other model weighting techniques to average Bayesian
    predictive distributions.
-   `r pkg("MCMCpack", priority = "core")` provides
    model-specific Markov chain Monte Carlo (MCMC) algorithms for wide
    range of models commonly used in the social and behavioral sciences.
    It contains R functions to fit a number of regression models (linear
    regression, logit, ordinal probit, probit, Poisson regression,
    etc.), measurement models (item response theory and factor models),
    changepoint models (linear regression, binary probit, ordinal
    probit, Poisson, panel), and models for ecological inference. It
    also contains a generic Metropolis sampler that can be used to fit
    arbitrary models.
-   The `r pkg("mcmc", priority = "core")` package consists
    of an R function for a random-walk Metropolis algorithm for a
    continuous random vector.
-   The `r pkg("nimble", priority = "core")` package provides
    a general MCMC system that allows customizable MCMC for models
    written in the BUGS/JAGS model language. Users can choose samplers
    and write new samplers. Models and samplers are automatically
    compiled via generated C++. The package also supports other methods
    such as particle filtering or whatever users write in its algorithm
    language.

### Application-Specific Packages

#### ANOVA
-   `r pkg("bayesanova")` provides a Bayesian version of the analysis of variance based on a three-component Gaussian mixture for which a Gibbs sampler produces posterior draws. 
-   `r pkg("AovBay")` provides the classical analysis of variance, the nonparametric equivalent of Kruskal Wallis, and the Bayesian approach. 


#### Bayes factor/model comparison/Bayesian model averaging
-   `r pkg("bain")` computes approximated adjusted fractional Bayes factors for equality, inequality, and about equality constrained hypotheses. 
-   `r pkg("BayesFactor")` provides a suite of functions for
    computing various Bayes factors for simple designs, including
    contingency tables, one- and two-sample designs, one-way designs,
    general ANOVA designs, and linear regression.
-   `r pkg("BayesVarSel")` calculate Bayes factors in linear
    models and then to provide a formal Bayesian answer to testing and
    variable selection problems.
-   The `r pkg("BMA")` package has functions for Bayesian
    model averaging for linear models, generalized linear models, and
    survival models. The complementary package
    `r pkg("ensembleBMA")` uses the
    `r pkg("BMA")` package to create probabilistic forecasts
    of ensembles using a mixture of normal distributions.
-   `r pkg("BMS")` is Bayesian Model Averaging library for
    linear models with a wide choice of (customizable) priors. Built-in
    priors include coefficient priors (fixed, flexible and hyper-g
    priors), and 5 kinds of model priors.
-   `r pkg("bridgesampling")` provides R functions for
    estimating marginal likelihoods, Bayes factors, posterior model
    probabilities, and normalizing constants in general, via different
    versions of bridge sampling (Meng and Wong, 1996).
-   `r pkg("RoBMA")` implements Bayesian model-averaging for
    meta-analytic models, including models correcting for publication
    bias.


#### Bayesian tree models
-   `r pkg("dbarts")` fits Bayesian additive regression trees (Chipman, George, and McCulloch 2010).
-   The `r pkg("bartBMA")` offers functions for Bayesian additive regression trees using Bayesian model averaging.     
-   `r pkg("bartCause")` contains a variety of methods to generate typical causal inference estimates using Bayesian Additive Regression Trees (BART) as the underlying regression model (Hill 2012). 

#### Causal inference
-   `r pkg("bama")` performs mediation analysis in the presence of high-dimensional mediators based on the potential outcome framework. Bayesian Mediation Analysis (BAMA), developed by Song et al (2019).
-   `r pkg("bartCause")` contains a variety of methods to generate typical causal inference estimates using Bayesian Additive Regression Trees (BART) as the underlying regression model (Hill 2012). 
-   `r pkg("BayesCACE")` performs CACE (Complier Average Causal Effect analysis) on either a single study or meta-analysis of datasets with binary outcomes, using either complete or incomplete noncompliance information. 
-   `r pkg("baycn")` is a package for a Bayesian hybrid approach for inferring Directed Acyclic Graphs (DAGs) for continuous, discrete, and mixed data.
-   `r pkg("BayesTree")` implements BART (Bayesian Additive
    Regression Trees) by Chipman, George, and McCulloch (2006).
-   `r pkg("BDgraph")` provides statistical tools for
    Bayesian structure learning in undirected graphical models for
    multivariate continuous, discrete, and mixed data.
-   `r pkg("blavaan")` fits a variety of Bayesian latent
    variable models, including confirmatory factor analysis, structural
    equation models, and latent growth curve models.
-   `r pkg("causact")` provides R functions for visualizing
    and running inference on generative directed acyclic graphs (DAGs).
    Once a generative DAG is created, the package automates Bayesian
    inference via the `r pkg("greta")` package and
    **TensorFlow** .
-   `r pkg("CausalImpact")` implements a Bayesian approach to causal impact estimation in time series, as described in Brodersen et al. (2015). 

#### Computational methods 
-   `r pkg("abc")` package implements several ABC algorithms
    for performing parameter estimation and model selection.
    Cross-validation tools are also available for measuring the accuracy
    of ABC estimates, and to calculate the misclassification
    probabilities of different models.
-   `r pkg("abcrf")` performs Approximate Bayesian Computation (ABC) model choice and parameter inference via random forests.
-   `r pkg("bamlss")` provides an infrastructure for
    estimating probabilistic distributional regression models in a
    Bayesian framework. The distribution parameters may capture
    location, scale, shape, etc. and every parameter may depend on
    complex additive terms similar to a generalized additive model.
-   `r pkg("bang")` provides functions for the Bayesian analysis of some simple commonly-used models, without using Markov Chain Monte Carlo (MCMC) methods such as Gibbs sampling. 
-   `r pkg("bayesboot")` provides functions for performing the Bayesian bootstrap as introduced by Rubin (1981).
- `r pkg("bayesian")` fits Bayesian models using 'brms'/'Stan' with 'parsnip'/'tidymodels.'
-   `r pkg("BayesLN")` allows to easily carry out a proper Bayesian inferential procedure by fixing a suitable distribution (the generalized inverse Gaussian) as prior for the variance. 
-   `r pkg("dclone")` provides low level functions for
    implementing maximum likelihood estimating procedures for complex
    models using data cloning and MCMC methods.
-   `r pkg("EntropyMCMC")` is an R package for MCMC
    simulation and convergence evaluation using entropy and
    Kullback-Leibler divergence estimation.
-   The `r pkg("HI")` package has functions to implement a
    geometric approach to transdimensional MCMC methods and random
    direction multivariate Adaptive Rejection Metropolis Sampling.
-   `r pkg("iterLap")` performs an iterative Laplace
    approximation to build a global approximation of the posterior
    (using mixture distributions) and then uses importance sampling for
    simulation based inference.
-   The `r pkg("mcmcensemble")` package provides ensemble
    samplers for affine-invariant Monte Carlo Markov Chain, which allow
    a faster convergence for badly scaled estimation problems. Two
    samplers are proposed: the 'differential.evolution' sampler and
    the 'stretch' sampler.
-   `r pkg("mcmcse")` allows estimation of multivariate
    effective sample size and calculation of Monte Carlo standard
    errors.
-   The `hitro.new()` function in `r pkg("Runuran")`
    provides an MCMC sampler based on the Hit-and-Run algorithm in
    combination with the Ratio-of-Uniforms method.


#### Discrete data
-   `r pkg("ammiBayes")` offers flexible multi-environment trials analysis via MCMC method for Additive Main Effects and Multiplicative Model (AMMI) for ordinal data.     
-   `r pkg("BANOVA")` includes functions for Hierarchical Bayes ANOVA models with normal response, t response, Binomial (Bernoulli) response, Poisson response, ordered multinomial response and multinomial response variables.
-   The `r pkg("BART")` package provide flexible
    nonparametric modeling of covariates for continuous, binary,
    categorical and time-to-event outcomes.
-   `r pkg("bayesbr")` fits the beta regression model using Bayesian inference. 
-   `r pkg("BayesComm")` performs Bayesian multivariate binary (probit) regression models for analysis of ecological communities.
-   `r pkg("bayescopulareg")` provides tools for Bayesian copula generalized linear models (GLMs).
-   `r pkg("bayescount")` provides a set of functions to allow analysis of count data (such as faecal egg count data) using Bayesian MCMC methods. 
-   `r pkg("BayesGWQS")` fits Bayesian grouped weighted quantile sum (BGWQS) regressions for one or more chemical groups with binary outcomes. 
-   `r pkg("BayesLogit")` provides tools for sampling from the PolyaGamma distribution based on Polson, Scott, and Windle (2013).
-   The `r pkg("mlogitBMA")` Provides a modified function
    `bic.glm()` of the `r pkg("BMA")` package that can be
    applied to multinomial logit (MNL) data.
-   The `r pkg("MNP")` package fits multinomial probit
    models using MCMC methods.
-   `r bioc("vbmp")` is a package for variational Bayesian
    multinomial probit regression with Gaussian process priors. It
    estimates class membership posterior probability employing
    variational and sparse approximation to the full posterior. This
    software also incorporates feature weighting by means of Automatic
    Relevance Determination.
-   `r pkg("zic")` provides functions for an MCMC analysis
    of zero-inflated count models including stochastic search variable
    selection.

#### Experiment/Contingency table/meta analysis/AB testing methods
-   `r pkg("abtest")` provides functions for Bayesian A/B testing including prior elicitation options based on Kass and Vaidyanathan (1992). 
-   `r pkg("acebayes")` finds optimal Bayesian experimental
    design using the approximate coordinate exchange (ACE) algorithm.
-   `r pkg("APFr")` implements a multiple testing approach to the choice of a threshold gamma on the p-values using the Average Power Function (APF) and Bayes False Discovery Rate (FDR) robust estimation.
-   `r pkg("ashr")` implements an Empirical Bayes approach for large-scale hypothesis testing and false discovery rate (FDR) estimation based on the methods proposed in Stephens (2016).
-   `r pkg("bamdit")` provides functions for Bayesian meta-analysis of diagnostic test data which are based on a scale mixtures bivariate random-effects model.
-   `r pkg("BASS")` is a package for Bayesian fitting and sensitivity analysis methods for adaptive spline surfaces. 
-   The `r pkg("bayefdr")` implements the Bayesian FDR control described by Newton et al. (2004).
-   The `r pkg("bayesAB")` provides a suite of functions that allow the user to analyze A/B test data in a Bayesian framework. 
-   `r pkg("BayesCombo")` combines diverse evidence across multiple studies to test a high level scientific theory. The methods can also be used as an alternative to a standard meta-analysis.
- The `r pkg("bayesloglin")` package is for Bayesian analysis of contingency table data.
-   `r pkg("bayesmeta")` is an R package to perform
    meta-analyses within the common random-effects model framework.
-   `r pkg("BEST")` provides an alternative to t-tests,
    producing posterior estimates for group means and standard
    deviations and their differences and effect sizes.
-   `r pkg("bspmma")` is a package for Bayesian
    semiparametric models for meta-analysis.
-   `r pkg("CPBayes")` performs a Bayesian meta-analysis method for studying cross-phenotype genetic associations. 
-   `r pkg("openEBGM")` calculates Empirical Bayes Geometric
    Mean (EBGM) and quantile scores from the posterior distribution
    using the Gamma-Poisson Shrinker (GPS) model to find unusually large
    cell counts in large, sparse contingency tables.
-   `r pkg("RxCEcolInf")` fits the R x C inference model
    described in Greiner and Quinn (2009).

#### Graphics
-   `r pkg("basicMCMCplots")` provides methods for examining posterior MCMC samples from a single chain using trace plots and density plots, and from multiple chains by comparing posterior medians and credible intervals from each chain. 
-   `r pkg("bayeslincom")` computes point estimates, standard deviations, and credible intervals for linear combinations of posterior samples.
-   `r pkg("ggmcmc")` is a tool for assessing and diagnosing
    convergence of Markov Chain Monte Carlo simulations, as well as for
    graphically display results from full MCMC analysis.
-   `r pkg("SamplerCompare")` provides a framework for
    running sets of MCMC samplers on sets of distributions with a
    variety of tuning parameters, along with plotting functions to
    visualize the results of those simulations.


#### Hierarchical models
-   `r pkg("baggr")` compares meta-analyses of data with hierarchical Bayesian models in Stan, including convenience functions for formatting data, plotting and pooling measures specific to meta-analysis. 
-   `r pkg("dirichletprocess")` performs nonparametric Bayesian analysis using Dirichlet processes without the need to program the inference algorithms. 
-   The `r pkg("lmm")` package contains R functions to fit
    linear mixed models using MCMC methods.
-   `r pkg("MCMCglmm")` is package for fitting Generalised
    Linear Mixed Models using MCMC methods.
-   `r pkg("RSGHB")` can be used to estimate models using a
    hierarchical Bayesian framework and provides flexibility in allowing
    the user to specify the likelihood function directly instead of
    assuming predetermined model structures.

#### High dimensional methods/machine learning methods 
-   `r pkg("abglasso")` implements a Bayesian adaptive graphical lasso data-augmented block Gibbs sampler. 
-   `r pkg("autohd")` performs mediation analysis for time to event high-dimensional data. Mediation Analysis proposed by Miocevic et al.(2017) as a statistical tool in the Bayesian framework. 
-   `r pkg("bartMachine")` allows an advanced implementation of Bayesian Additive Regression Trees with expanded features for data analysis and visualization.
-   The `r pkg("bayesGAM")`package is designed to provide a user friendly option to fit univariate and multivariate response Generalized Additive Models (GAM) using Hamiltonian Monte Carlo (HMC) with few technical burdens. 
-   `r pkg("BCBCSF")` provides functions to predict the
    discrete response based on selected high dimensional features, such
    as gene expression data.


#### Factor analysis/item response theory models
-   `r pkg("LAWBL")` is an R package latent (variable)
    analysis with with different Bayesian learning methods, including
    the partially confirmatory factor analysis, its generalized version,
    and the partially confirmatory item response model.
-   The `r pkg("pscl")` package provides R functions to fit
    item-response theory models using MCMC methods and to compute
    highest density regions for the Beta distribution and the inverse
    gamma distribution.


#### Missing data
-   `r pkg("sbgcop")` estimates parameters of a Gaussian
    copula, treating the univariate marginal distributions as nuisance
    parameters as described in Hoff(2007). It also provides a
    semiparametric imputation procedure for missing multivariate data.

#### Mixture models
-   `r pkg("AdMit")` provides functions to perform the
    fitting of an adapative mixture of Student-t distributions to a
    target density through its kernel function. The mixture
    approximation can be used as the importance density in importance
    sampling or as the candidate density in the Metropolis-Hastings
    algorithm.
-   `r pkg("BayesBinMix")` provides a fully Bayesian inference for estimating the number of clusters and related parameters to heterogeneous binary data.
-   `r pkg("BayesBinMix")` performs fully Bayesian inference for estimating the number of clusters and related parameters to heterogeneous binary data.
-   `r pkg("bmixture")` provides statistical tools for
    Bayesian estimation for the finite mixture of distributions, mainly
    mixture of Gamma, Normal and t-distributions.
-   `r pkg("Bmix")` is a bare-bones implementation of
    sampling algorithms for a variety of Bayesian stick-breaking
    (marginally DP) mixture models, including particle learning and
    Gibbs sampling for static DP mixtures, particle learning for dynamic
    BAR stick-breaking, and DP mixture regression.
-   `r pkg("REBayes")` is a package for empirical Bayes estimation using Kiefer-Wolfowitz maximum likelihood estimation. 


#### Network models/Matrix-variate distribution
- `r pkg("BayesianNetwork")` provides a 'Shiny' web application for creating interactive Bayesian Network models, learning the structure and parameters of Bayesian networks, and utilities for classic network analysis.
-   `r pkg("Bergm")` performs Bayesian analysis for
    exponential random graph models using advanced computational
    algorithms.
-   `r pkg("bnlearn")` is a package for Bayesian network
    structure learning (via constraint-based, score-based and hybrid
    algorithms), parameter learning (via ML and Bayesian estimators) and
    inference.
-   `r pkg("ebdbNet")` can be used to infer the adjacency
    matrix of a network from time course data using an empirical Bayes
    estimation procedure based on Dynamic Bayesian Networks.
-   `r pkg("eigenmodel")` estimates the parameters of a
    model for symmetric relational data (e.g., the above-diagonal part
    of a square matrix), using a model-based eigenvalue decomposition
    and regression using MCMC methods.
-   `r pkg("gRain")` is a package for probability
    propagation in graphical independence networks, also known as
    Bayesian networks or probabilistic expert systems.
-   `r pkg("NetworkChange")` is an R package for change
    point analysis in longitudinal network data. It implements a hidden
    Markovmultilinear tensor regression model. Model diagnostic tools
    using marginal likelihoods and WAIC are provided.
-   `r pkg("rstiefel")` simulates random orthonormal
    matrices from linear and quadratic exponential family distributions
    on the Stiefel manifold using the Gibbs sampling method. The most
    general type of distribution covered is the matrix-variate
    Bingham-von Mises-Fisher distribution.
-   `r pkg("sna")`, an R package for social network
    analysis, contains functions to generate posterior samples from
    Butt's Bayesian network accuracy model using Gibbs sampling.
-   `r pkg("ssgraph")` is for Bayesian inference in
    undirected graphical models using spike-and-slab priors for
    multivariate continuous, discrete, and mixed data.

#### Quantile regression
-   `r pkg("bayesQR")` supports Bayesian quantile regression
    using the asymmetric Laplace distribution, both continuous as well
    as binary dependent variables.


#### Shrinkage/Variable selection/Gaussian process
-   `r pkg("BAS")` is a package for Bayesian Variable
    Selection and Model Averaging in linear models and generalized
    linear models using stochastic or deterministic sampling without
    replacement from posterior distributions. Prior distributions on
    coefficients are from Zellner's g-prior or mixtures of g-priors
    corresponding to the Zellner-Siow Cauchy Priors or the mixture of
    g-priors for linear models or mixtures of g-priors in generalized
    linear models.
-   `r pkg("basad")` provides a Bayesian variable selection approach using continuous spike and slab prior distributions. 
-   `r pkg("BayesGPfit")` performs Bayesian inferences on nonparametric regression via Gaussian Processes with a modified exponential square kernel using a basis expansion approach.
-   `r pkg("BayesianGLasso")` implements a data-augmented block Gibbs sampler for simulating the posterior distribution of concentration matrices for specifying the topology and parameterization of a Gaussian Graphical Model (GGM). 
-   `r pkg("BLR")` provides R functions to fit parametric
    regression models using different types of shrinkage methods.
-   `r pkg("BNSP")` is a package for Bayeisan non- and
    semi-parametric model fitting. It handles Dirichlet process mixtures
    and spike-slab for multivariate (and univariate) response analysis,
    with nonparametric models for the means, the variances and the
    correlation matrix.
-   `r pkg("BoomSpikeSlab")` provides functions to do spike
    and slab regression via the stochastic search variable selection
    algorithm. It handles probit, logit, poisson, and student T data.
-   `r pkg("bsamGP")` provides functions to perform Bayesian
    inference using a spectral analysis of Gaussian process priors.
    Gaussian processes are represented with a Fourier series based on
    cosine basis functions. Currently the package includes parametric
    linear models, partial linear additive models with/without shape
    restrictions, generalized linear additive models with/without shape
    restrictions, and density estimation model.
-   `r pkg("spikeslab")` provides functions for prediction
    and variable selection using spike and slab regression.
-   `r pkg("spikeSlabGAM")` implements Bayesian variable
    selection, model choice, and regularized estimation in
    (geo-)additive mixed models for Gaussian, binomial, and Poisson
    responses.

#### Spatial models
-   `r pkg("CARBayes")` implements a class of univariate and multivariate spatial generalised linear mixed models for areal unit data, with inference in a Bayesian setting using Markov chain Monte Carlo (MCMC) simulation. Also, see `r pkg("CARBayesdata")`. 
-   `r pkg("CARBayesST")`, which implements a class of univariate and multivariate spatio-temporal generalised linear mixed models for areal unit data, with inference in a Bayesian setting using Markov chain Monte Carlo (MCMC) simulation. 
-   `r pkg("CircSpaceTime")` implementation of Bayesian models for spatial and spatio-temporal interpolation of circular data using Gaussian Wrapped and Gaussian Projected distributions.
-   The function `krige.bayes()` in the `r pkg("geoR")`
    package performs Bayesian analysis of geostatistical data allowing
    specification of different levels of uncertainty in the model
    parameters. See the `r view("Spatial")` view for more
    information.
-   `r pkg("spBayes")` provides R functions that fit
    Gaussian spatial process models for univariate as well as
    multivariate point-referenced data using MCMC methods.
-   `r pkg("spTimer")` fits, spatially predict and
    temporally forecast large amounts of space-time data using Bayesian
    Gaussian Process Models, Bayesian Auto-Regressive (AR) Models, and
    Bayesian Gaussian Predictive Processes based AR Models.
-   The `r pkg("tgp")` package implements Bayesian treed
    Gaussian process models: a spatial modeling and regression package
    providing fully Bayesian MCMC posterior inference for models ranging
    from the simple linear model, to nonstationary treed Gaussian
    process, and others in between.


#### Survival models
-   The `r pkg("BMA")` package has functions for Bayesian
    model averaging for linear models, generalized linear models, and
    survival models. The complementary package
    `r pkg("ensembleBMA")` uses the
    `r pkg("BMA")` package to create probabilistic forecasts
    of ensembles using a mixture of normal distributions.

#### Time series models
-   `r pkg("BaPreStoPro")` is a R package for Bayesian estimation and prediction for stochastic processes based on the Euler approximation.
-   `r pkg("BayesARIMAX")` is a package for Bayesian estimation of ARIMAX model. Autoregressive Integrated Moving Average (ARIMA) model is very popular univariate time series model. Its application has been widened by the incorporation of exogenous variable(s) (X) in the model and modified as ARIMAX by Bierens (1987). 
-   `r pkg("bayesDccGarch")` performs Bayesian estimation of dynamic conditional correlation GARCH model for multivariate time series volatility (Fioruci et al. 2014). 
-   `r pkg("bayesdfa")` implements Bayesian dynamic factor analysis with 'Stan'. 
-   The `r pkg("bayesGARCH")` package provides a function
    which perform the Bayesian estimation of the GARCH(1,1) model with
    Student's t innovations.
-   `r pkg("bayeslongitudinal")` adjusts longitudinal regression models using Bayesian methodology for covariance structures of composite symmetry (SC), autoregressive ones of order 1 AR (1) and autoregressive moving average of order (1,1) ARMA (1,1).
-   `r pkg("BAYSTAR")` provides functions for Bayesian
    estimation of threshold autoregressive models.
-   `r pkg("bcp")` implements a Bayesian analysis of
    changepoint problem using Barry and Hartigan product partition
    model.
-   `r pkg("bspec")` performs Bayesian inference on the
    (discrete) power spectrum of time series.
-   `r pkg("bsts")` is a package for time series regression
    using dynamic linear models using MCMC.
-   `r pkg("BVAR")` is a package for estimating hierarchical
    Bayesian vector autoregressive models.
-   `r pkg("DIRECT")` provides a Bayesian clustering method for replicated time series or replicated measurements from multiple experimental conditions. 
-   `r pkg("dlm")` is a package for Bayesian (and likelihood) analysis of dynamic linear models. It includes the calculations of the Kalman filter and smoother, and the forward
    filtering backward sampling algorithm.
-   `r pkg("EbayesThresh")` implements Bayesian estimation
    for thresholding methods. Although the original model is developed
    in the context of wavelets, this package is useful when researchers
    need to take advantage of possible sparsity in a parameter set.
-   `r pkg("NetworkChange")` is an R package for change
    point analysis in longitudinal network data. It implements a hidden
    Markovmultilinear tensor regression model. Model diagnostic tools
    using marginal likelihoods and WAIC are provided.
-   `r pkg("NGSSEML")` gives codes for formulating and
    specifying the non-Gaussian state-space models in the R language.
    Inferences for the parameters of the model can be made under the
    classical and Bayesian.
-   `r pkg("Rbeast")` implements  a Bayesian model averaging method via RJMCMC to decompose time series into abrupt changes, trend, and seasonality, useful for changepoint detection, time series decomposition,  nonlinear trend analysis, and time series segmentation.
-   `r pkg("spTimer")` fits, spatially predict and
    temporally forecast large amounts of space-time data using Bayesian
    Gaussian Process Models, Bayesian Auto-Regressive (AR) Models, and
    Bayesian Gaussian Predictive Processes based AR Models.
-   `r pkg("ssMousetrack")` estimates previously compiled
    state-space modeling for mouse-tracking experiment using the
    `r pkg("rstan")` package, which provides the R interface
    to the Stan C++ library for Bayesian estimation.
-   `r pkg("stochvol")` provides efficient algorithms for
    fully Bayesian estimation of stochastic volatility (SV) models.


#### Other models
-   `r pkg("bayesammi")` performs Bayesian estimation of the additive main effects and multiplicative interaction (AMMI) model. 
-   `r pkg("BayesBP")` is a package for Bayesian estimation using Bernstein polynomial fits rate matrix.
-   `r pkg("BayesCR")` proposes a parametric fit for censored linear regression models based on SMSN distributions, from a Bayesian perspective. 
-   `r pkg("bayesdistreg")` implements Bayesian Distribution Regression methods. This package contains functions for three estimators (non-asymptotic, semi-asymptotic and asymptotic) and related routines for Bayesian Distribution Regression in Huang and Tsyawo (2018).
-   `r pkg("bayesDP")` provides functions for data augmentation using the Bayesian discount prior method for single arm and two-arm clinical trials in Haddad et al. (2017).
-   `r pkg("BayesFM")` provides a collection of procedures to perform Bayesian analysis on a variety of factor models. 
-   `r pkg("BayesGOF")` performs four interconnected tasks: (i) characterizes the uncertainty of the elicited parametric prior; (ii) provides exploratory diagnostic for checking prior-data conflict; (iii) computes the final statistical prior density estimate; and (iv) executes macro- and micro-inference.
-   `r pkg("Bayesiangammareg")` adjusts the Gamma regression models from a Bayesian perspective described by Cepeda and Urdinola (2012).
-   `r pkg("BayesLCA")` performs Bayesian Latent Class Analysis using several different methods. 
-   `r pkg("BayesMallows")` performs Bayesian preference learning with the Mallows rank model. 
- `r pkg("BayesMassBal")` is a package for Bayesian data reconciliation of separation processes. 
-   `r pkg("bayestestR")` provides utilities to describe
    posterior distributions and Bayesian models. It includes
    point-estimates such as Maximum A Posteriori (MAP), measures of
    dispersion (Highest Density Interval) and indices used for
    null-hypothesis testing (such as ROPE percentage, pd and Bayes
    factors).
-   `r pkg("bbricks")` provides a class of frequently used
    Bayesian parametric and nonparametric model structures,as well as a
    set of tools for common analytical tasks.
-   `r pkg("coalescentMCMC")` provides a flexible framework
    for coalescent analyses in R.
-   `r pkg("deBInfer")` provides R functions for Bayesian
    parameter inference in differential equations using MCMC methods.
-   `r pkg("densEstBayes")` provides Bayesian density estimates for univariate continuous random samples are provided using the Bayesian inference engine paradigm. 
-   `r pkg("errum")` performs a Bayesian estimation of the
    exploratory reduced reparameterized unified model (ErRUM).
    `r pkg("rrum")` implements Gibbs sampling algorithm for
    Bayesian estimation of the Reduced Reparameterized Unified Model
    (rrum).
-   `r pkg("FME")` provides functions to help in fitting
    models to data, to perform Monte Carlo, sensitivity and
    identifiability analysis. It is intended to work with models be
    written as a set of differential equations that are solved either by
    an integration routine from deSolve, or a steady-state solver from
    rootSolve.
-   The `gbayes()` function in `r pkg("Hmisc")` derives the
    posterior (and optionally) the predictive distribution when both the
    prior and the likelihood are Gaussian, and when the statistic of
    interest comes from a two-sample problem.
-   The `r pkg("hbsae")` package provides functions to
    compute small area estimates based on a basic area or unit-level
    model. The model is fit using restricted maximum likelihood, or in a
    hierarchical Bayesian way.
-   `r pkg("matchingMarkets")` implements a structural model
    based on a Gibbs sampler to correct for the bias from endogenous
    matching (e.g. group formation or two-sided matching).
-   `r pkg("mombf")` performs model selection based on
    non-local priors, including MOM, eMOM and iMOM priors..
-   `r pkg("prevalence")` provides functions for the
    estimation of true prevalence from apparent prevalence in a Bayesian
    framework. MCMC sampling is performed via JAGS/rjags.
-   `r pkg("PReMiuM")` is a package for profile regression,
    which is a Dirichlet process Bayesian clustering where the response
    is linked non-parametrically to the covariate profile.
-   `r pkg("revdbayes")` provides functions for the Bayesian
    analysis of extreme value models using direct random sampling from
    extreme value posterior distributions.
-   The `vcov.gam()` function the `r pkg("mgcv")` package
    can extract a Bayesian posterior covariance matrix of the parameters
    from a fitted `gam` object.


### Bayesian models for specific disciplines
-   `r pkg("AnaCoDa")` is a collection of models to analyze genome scale codon data using a Bayesian framework.
-   `r pkg("ArchaeoChron")` provides a list of functions for the Bayesian modeling of archaeological chronologies.
-   The `r pkg("BACCT")` implements the Bayesian Augmented Control (BAC, a.k.a. Bayesian historical data borrowing) method under clinical trial setting by calling 'Just Another Gibbs Sampler' ('JAGS') software.  
-   `r pkg("BaSkePro")` provides tools to perform Bayesian inference of carcass processing/transport strategy and bone attrition from archaeofaunal skeletal profiles characterized by percentages of MAU (Minimum Anatomical Units). 
-   `r pkg("bayesbio")` provides miscellaneous functions for bioinformatics and Bayesian statistics.
-   `r pkg("bayesCT")` performs simulation and analysis of Bayesian adaptive clinical trials for binomial, Gaussian, and time-to-event data types, incorporates historical data and allows early stopping for futility or early success. 
-   `r pkg("BayesCTDesign")` provides a set of functions to help clinical trial researchers calculate power and sample size for two-arm Bayesian randomized clinical trials that do or do not incorporate historical control data.
-   `r pkg("bayes4psy")` contains several Bayesian models for data analysis of psychological tests.
- `r pkg("bayesianETAS")` is a package for Bayesian estimation of the Epidemic Type Aftershock Sequence (ETAS) model for earthquake occurrences. 
-   `r pkg("BayesianFROC")` provides new methods for the so-called Free-response Receiver Operating Characteristic (FROC) analysis. 
- `r pkg("BayesianLaterality")` provides functions to implement a Bayesian model for predicting hemispheric dominance from observed laterality indices (Sorensen and Westerhausen 2020). 
-   `r pkg("bayesImageS")` is an R package for Bayesian image analysis using the hidden Potts model.
-   `r pkg("bayesLife")` makes probabilistic projections of life expectancy for all countries of the world, using a Bayesian hierarchical model.
-   `r pkg("BCE")` contains function to estimates taxonomic
    compositions from biomarker data using a Bayesian approach.
-   `r pkg("bqtl")` can be used to fit quantitative trait
    loci (QTL) models. This package allows Bayesian estimation of
    multi-gene models via Laplace approximations and provides tools for
    interval mapping of genetic loci. The package also contains
    graphical tools for QTL analysis.
-    `r pkg("coalitions")` implements a Bayesian framework for the opinion poll based estimation of event probabilities in multi-party electoral systems (Bender and Bauer 2018). 
-   `r pkg("dfpk")` provides statistical methods involving PK measures are provided, in the dose allocation process during a Phase I clinical trials. 
-   `r pkg("dina")` estimates the Deterministic Input, Noisy
    "And" Gate (DINA) cognitive diagnostic model parameters using the
    Gibbs sampler. `r pkg("edina")` performs a Bayesian
    estimation of the exploratory deterministic input, noisy and gate
    (EDINA) cognitive diagnostic model.




### Post-estimation tools

-   `r pkg("MCMCvis")` performs key functions (visualizes,
    manipulates, and summarizes) for MCMC analysis. Functions support
    simple and straightforward subsetting of model parameters within the
    calls, and produce presentable and 'publication-ready' output.
    MCMC output may be derived from Bayesian model output fit with JAGS,
    Stan, or other MCMC samplers.
-   The `r pkg("boa", priority = "core")` package provides
    functions for diagnostics, summarization, and visualization of MCMC
    sequences. It imports draws from BUGS format, or from plain
    matrices. `r pkg("boa")` provides the Gelman and Rubin,
    Geweke, Heidelberger and Welch, and Raftery and Lewis diagnostics,
    the Brooks and Gelman multivariate shrink factors.
-   The `r pkg("coda", priority = "core")` (Convergence
    Diagnosis and Output Analysis) package is a suite of functions that
    can be used to summarize, plot, and and diagnose convergence from
    MCMC samples. `r pkg("coda")` also defines an `mcmc`
    object and related methods which are used by other packages. It can
    easily import MCMC output from WinBUGS, OpenBUGS, and JAGS, or from
    plain matrices. `r pkg("coda")` contains the Gelman and
    Rubin, Geweke, Heidelberger and Welch, and Raftery and Lewis
    diagnostics.
-   `r pkg("plotMCMC")` extends `r pkg("coda")`
    by adding convenience functions to make it easier to create
    multipanel plots. The graphical parameters have sensible defaults
    and are easy to modify via top-level arguments.
-   `r pkg("ramps")` implements Bayesian geostatistical
    analysis of Gaussian processes using a reparameterized and
    marginalized posterior sampling algorithm.

### Packages for learning Bayesian statistics
-   `r pkg("BaM")` provide functions and datasets for "Bayesian Methods: A Social and Behavioral Sciences Approach" by Jeff Gill (Chapman and Hall/CRC, 2002/2007/2014).
-   `r pkg("BayesDA")` provides R functions and datasets for
    "Bayesian Data Analysis, Second Edition" (CRC Press, 2003) by
    Andrew Gelman, John B. Carlin, Hal S. Stern, and Donald B. Rubin.
-   The `r pkg("Bolstad")` package contains a set of R
    functions and data sets for the book Introduction to Bayesian
    Statistics, by Bolstad, W.M. (2007).
-   The `r pkg("LearnBayes")` package contains a collection
    of functions helpful in learning the basic tenets of Bayesian
    statistical inference. It contains functions for summarizing basic
    one and two parameter posterior distributions and predictive
    distributions and MCMC algorithms for summarizing posterior
    distributions defined by the user. It also contains functions for
    regression models, hierarchical models, Bayesian tests, and
    illustrations of Gibbs sampling.

### Packages that link R to other sampling engines

-   `r pkg("bayesmix")` is an R package to fit Bayesian
    mixture models using [JAGS](http://mcmc-jags.sourceforge.net/).
-   `r pkg("BayesX")` provides functionality for exploring
    and visualizing estimation results obtained with the software
    package [BayesX](http://www.BayesX.org/).
-   `r pkg("Boom")` provides a C++ library for Bayesian
    modeling, with an emphasis on Markov chain Monte Carlo.
-   **BRugs** provides an R interface to
    [OpenBUGS](http://www.openbugs.net/). It works under Windows and
    Linux. **BRugs** used to be available from CRAN, now it is located
    at the [CRANextras](http://www.stats.ox.ac.uk/pub/RWin/) repository.
-   `r pkg("brms")` implements Bayesian multilevel models in
    R using [Stan](http://mc-stan.org/). A wide range of distributions
    and link functions are supported, allowing users to fit linear,
    robust linear, binomial, Poisson, survival, response times,
    ordinal, quantile, zero-inflated, hurdle, and even non-linear models
    all in a multilevel context. `r pkg("shinybrms")` is a
    graphical user interface (GUI) for fitting Bayesian regression
    models using the package `r pkg("brms")`.
-   `r pkg("greta")` allows users to write statistical
    models in R and fit them by MCMC and optimisation on CPUs and GPUs,
    using Google **'TensorFlow'** . `r pkg("greta")` lets
    you write your own model like in BUGS, JAGS and Stan, except that
    you write models right in R, it scales well to massive datasets, and
    it is easy to extend and build on.
-   There are two packages that can be used to interface R with
    [WinBUGS](http://www.mrc-bsu.cam.ac.uk/software/bugs/).
    `r pkg("R2WinBUGS")` provides a set of functions to call
    WinBUGS on a Windows system and a Linux system.
-   There are three packages that provide R interface with [Just Another
    Gibbs Sampler (JAGS)](http://mcmc-jags.sourceforge.net/) :
    `r pkg("rjags")`, `r pkg("R2jags")`, and
    `r pkg("runjags")`.
-   All of these BUGS engines use graphical models for model
    specification. As such, the `r view("GraphicalModels")` task view may
    be of interest.
-   `r pkg("rstan")` provides R functions to parse, compile,
    test, estimate, and analyze Stan models by accessing the header-only
    Stan library provided by the `StanHeaders' package. The
    [Stan](http://mc-stan.org/) project develops a probabilistic
    programming language that implements full Bayesian statistical
    inference via MCMC and (optionally penalized) maximum likelihood
    estimation via optimization. 
-   `r pkg("rstanarm")` estimates previously compiled regression models 
    using the `r pkg("rstan")` package,  which provides the R interface 
    to the Stan C++ library for Bayesian estimation.
-   `r pkg("pcFactorStan")` provides convenience functions
    and pre-programmed Stan models related to the paired comparison
    factor model. Its purpose is to make fitting paired comparison data
    using Stan easy.

The Bayesian Inference Task View is written by Jong Hee Park (Seoul
National University, South Korea), Andrew D. Martin (Washington
University in St. Louis, MO, USA), and Kevin M. Quinn (UC Berkeley,
Berkeley, CA, USA). Please e-mail the maintainer with suggestion
or by submitting an issue or pull request in the GitHub repository linked above.


### Links
-   [Bayesian Statistics and Marketing (bayesm)](http://www.perossi.org/home/bsm-1)
-   [BayesX](http://www.BayesX.org/)
-   [BOA](http://www.public-health.uiowa.edu/boa/)
-   [BRugs in CRANextras](http://www.stats.ox.ac.uk/pub/RWin/src/contrib/)
-   [Just Another Gibbs Sampler (JAGS)](http://mcmc-jags.sourceforge.net/)
-   [MCMCpack](http://mcmcpack.berkeley.edu/)
-   [NIMBLE](http://r-nimble.org/)
-   [OpenBUGS](http://www.openbugs.net/)
-   [Stan](http://mc-stan.org/)
-   [TensorFlow](https://www.tensorflow.org)
-   [The BUGS Project (WinBUGS)](http://www.mrc-bsu.cam.ac.uk/software/bugs/)
