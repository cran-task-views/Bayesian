---
name: Bayesian
topic: Bayesian Inference
maintainer: Jong Hee Park
email: jongheepark@snu.ac.kr
version: 2022-02-22
source: https://github.com/cran-task-views/Bayesian/
---

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

### Bayesian packages for general model fitting

-   The `r pkg("arm", priority = "core")` package contains R
    functions for Bayesian inference using lm, glm, mer and polr
    objects.
-   `r pkg("BACCO", priority = "core")` is an R bundle for
    Bayesian analysis of random functions. `r pkg("BACCO")`
    contains three sub-packages: emulator, calibrator, and approximator,
    that perform Bayesian emulation and calibration of computer
    programs.
-   `r pkg("bayesm", priority = "core")` provides R functions
    for Bayesian inference for various models widely used in marketing
    and micro-econometrics. The models include linear regression models,
    multinomial logit, multinomial probit, multivariate probit,
    multivariate mixture of normals (including clustering), density
    estimation using finite mixtures of normals as well as Dirichlet
    Process priors, hierarchical linear models, hierarchical multinomial
    logit, hierarchical negative binomial regression models, and linear
    instrumental variable models.
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

### Bayesian packages for specific models or methods

-   `r pkg("abc")` package implements several ABC algorithms
    for performing parameter estimation and model selection.
    Cross-validation tools are also available for measuring the accuracy
    of ABC estimates, and to calculate the misclassification
    probabilities of different models.
-   `r pkg("acebayes")` finds optimal Bayesian experimental
    design using the approximate coordinate exchange (ACE) algorithm.
-   `r pkg("AdMit")` provides functions to perform the
    fitting of an adapative mixture of Student-t distributions to a
    target density through its kernel function. The mixture
    approximation can be used as the importance density in importance
    sampling or as the candidate density in the Metropolis-Hastings
    algorithm.
-   The `r pkg("BaBooN")` package contains two variants of
    Bayesian Bootstrap Predictive Mean Matching to multiply impute
    missing data.
-   `r pkg("bamlss")` provides an infrastructure for
    estimating probabilistic distributional regression models in a
    Bayesian framework. The distribution parameters may capture
    location, scale, shape, etc. and every parameter may depend on
    complex additive terms similar to a generalized additive model.
-   The `r pkg("BART")` package provide flexible
    nonparametric modeling of covariates for continuous, binary,
    categorical and time-to-event outcomes.
-   `r pkg("BAS")` is a package for Bayesian Variable
    Selection and Model Averaging in linear models and generalized
    linear models using stochastic or deterministic sampling without
    replacement from posterior distributions. Prior distributions on
    coefficients are from Zellner's g-prior or mixtures of g-priors
    corresponding to the Zellner-Siow Cauchy Priors or the mixture of
    g-priors for linear models or mixtures of g-priors in generalized
    linear models.
-   The `r pkg("bayesGARCH")` package provides a function
    which perform the Bayesian estimation of the GARCH(1,1) model with
    Student's t innovations.
-   `r pkg("BayesianTools")` is an R package for
    general-purpose MCMC and SMC samplers, as well as plot and
    diagnostic functions for Bayesian statistics, with a particular
    focus on calibrating complex system models. Implemented samplers
    include various Metropolis MCMC variants (including adaptive and/or
    delayed rejection MH), the T-walk, two differential evolution MCMCs,
    two DREAM MCMCs, and a sequential Monte Carlo (SMC) particle filter.
-   `r pkg("bayesImageS")` is an R package for Bayesian
    image analysis using the hidden Potts model.
-   `r pkg("bayesmeta")` is an R package to perform
    meta-analyses within the common random-effects model framework.
-   `r pkg("BayesTree")` implements BART (Bayesian Additive
    Regression Trees) by Chipman, George, and McCulloch (2006).
-   `r pkg("bayesQR")` supports Bayesian quantile regression
    using the asymmetric Laplace distribution, both continuous as well
    as binary dependent variables.
-   `r pkg("BayesFactor")` provides a suite of functions for
    computing various Bayes factors for simple designs, including
    contingency tables, one- and two-sample designs, one-way designs,
    general ANOVA designs, and linear regression.
-   `r pkg("bayestestR")` provides utilities to describe
    posterior distributions and Bayesian models. It includes
    point-estimates such as Maximum A Posteriori (MAP), measures of
    dispersion (Highest Density Interval) and indices used for
    null-hypothesis testing (such as ROPE percentage, pd and Bayes
    factors).
-   `r pkg("BayesVarSel")` calculate Bayes factors in linear
    models and then to provide a formal Bayesian answer to testing and
    variable selection problems.
-   `r pkg("BayHaz")` contains a suite of R functions for
    Bayesian estimation of smooth hazard rates via Compound Poisson
    Process (CPP) priors.
-   `r pkg("BAYSTAR")` provides functions for Bayesian
    estimation of threshold autoregressive models.
-   `r pkg("bbemkr")` implements Bayesian bandwidth
    estimation for Nadaraya-Watson type multivariate kernel regression
    with Gaussian error.
-   `r pkg("bbricks")` provides a class of frequently used
    Bayesian parametric and nonparametric model structures,as well as a
    set of tools for common analytical tasks.
-   `r pkg("BCE")` contains function to estimates taxonomic
    compositions from biomarker data using a Bayesian approach.
-   `r pkg("BCBCSF")` provides functions to predict the
    discrete response based on selected high dimensional features, such
    as gene expression data.
-   `r pkg("bcp")` implements a Bayesian analysis of
    changepoint problem using Barry and Hartigan product partition
    model.
-   `r pkg("BDgraph")` provides statistical tools for
    Bayesian structure learning in undirected graphical models for
    multivariate continuous, discrete, and mixed data.
-   `r pkg("Bergm")` performs Bayesian analysis for
    exponential random graph models using advanced computational
    algorithms.
-   `r pkg("BEST")` provides an alternative to t-tests,
    producing posterior estimates for group means and standard
    deviations and their differences and effect sizes.
-   `r pkg("blavaan")` fits a variety of Bayesian latent
    variable models, including confirmatory factor analysis, structural
    equation models, and latent growth curve models.
-   `r pkg("BLR")` provides R functions to fit parametric
    regression models using different types of shrinkage methods.
-   The `r pkg("BMA")` package has functions for Bayesian
    model averaging for linear models, generalized linear models, and
    survival models. The complementary package
    `r pkg("ensembleBMA")` uses the
    `r pkg("BMA")` package to create probabilistic forecasts
    of ensembles using a mixture of normal distributions.
-   `r pkg("bmixture")` provides statistical tools for
    Bayesian estimation for the finite mixture of distributions, mainly
    mixture of Gamma, Normal and t-distributions.
-   `r pkg("BMS")` is Bayesian Model Averaging library for
    linear models with a wide choice of (customizable) priors. Built-in
    priors include coefficient priors (fixed, flexible and hyper-g
    priors), and 5 kinds of model priors.
-   `r pkg("Bmix")` is a bare-bones implementation of
    sampling algorithms for a variety of Bayesian stick-breaking
    (marginally DP) mixture models, including particle learning and
    Gibbs sampling for static DP mixtures, particle learning for dynamic
    BAR stick-breaking, and DP mixture regression.
-   `r pkg("bnlearn")` is a package for Bayesian network
    structure learning (via constraint-based, score-based and hybrid
    algorithms), parameter learning (via ML and Bayesian estimators) and
    inference.
-   `r pkg("BNSP")` is a package for Bayeisan non- and
    semi-parametric model fitting. It handles Dirichlet process mixtures
    and spike-slab for multivariate (and univariate) response analysis,
    with nonparametric models for the means, the variances and the
    correlation matrix.
-   `r pkg("BoomSpikeSlab")` provides functions to do spike
    and slab regression via the stochastic search variable selection
    algorithm. It handles probit, logit, poisson, and student T data.
-   `r pkg("bqtl")` can be used to fit quantitative trait
    loci (QTL) models. This package allows Bayesian estimation of
    multi-gene models via Laplace approximations and provides tools for
    interval mapping of genetic loci. The package also contains
    graphical tools for QTL analysis.
-   `r pkg("bridgesampling")` provides R functions for
    estimating marginal likelihoods, Bayes factors, posterior model
    probabilities, and normalizing constants in general, via different
    versions of bridge sampling (Meng and Wong, 1996).
-   `r pkg("bsamGP")` provides functions to perform Bayesian
    inference using a spectral analysis of Gaussian process priors.
    Gaussian processes are represented with a Fourier series based on
    cosine basis functions. Currently the package includes parametric
    linear models, partial linear additive models with/without shape
    restrictions, generalized linear additive models with/without shape
    restrictions, and density estimation model.
-   `r pkg("bspec")` performs Bayesian inference on the
    (discrete) power spectrum of time series.
-   `r pkg("bspmma")` is a package for Bayesian
    semiparametric models for meta-analysis.
-   `r pkg("bsts")` is a package for time series regression
    using dynamic linear models using MCMC.
-   `r pkg("BVAR")` is a package for estimating hierarchical
    Bayesian vector autoregressive models.
-   `r pkg("causact")` provides R functions for visualizing
    and running inference on generative directed acyclic graphs (DAGs).
    Once a generative DAG is created, the package automates Bayesian
    inference via the `r pkg("greta")` package and
    **TensorFlow** .
-   `r pkg("coalescentMCMC")` provides a flexible framework
    for coalescent analyses in R.
-   `r pkg("conting")` performs Bayesian analysis of
    complete and incomplete contingency tables.
-   `r pkg("dclone")` provides low level functions for
    implementing maximum likelihood estimating procedures for complex
    models using data cloning and MCMC methods.
-   `r pkg("deBInfer")` provides R functions for Bayesian
    parameter inference in differential equations using MCMC methods.
-   `r pkg("dina")` estimates the Deterministic Input, Noisy
    "And" Gate (DINA) cognitive diagnostic model parameters using the
    Gibbs sampler. `r pkg("edina")` performs a Bayesian
    estimation of the exploratory deterministic input, noisy and gate
    (EDINA) cognitive diagnostic model.
-   `r pkg("dlm")` is a package for Bayesian (and
    likelihood) analysis of dynamic linear models. It includes the
    calculations of the Kalman filter and smoother, and the forward
    filtering backward sampling algorithm.
-   `r pkg("EbayesThresh")` implements Bayesian estimation
    for thresholding methods. Although the original model is developed
    in the context of wavelets, this package is useful when researchers
    need to take advantage of possible sparsity in a parameter set.
-   `r pkg("ebdbNet")` can be used to infer the adjacency
    matrix of a network from time course data using an empirical Bayes
    estimation procedure based on Dynamic Bayesian Networks.
-   `r pkg("eigenmodel")` estimates the parameters of a
    model for symmetric relational data (e.g., the above-diagonal part
    of a square matrix), using a model-based eigenvalue decomposition
    and regression using MCMC methods.
-   `r pkg("EntropyMCMC")` is an R package for MCMC
    simulation and convergence evaluation using entropy and
    Kullback-Leibler divergence estimation.
-   `r pkg("errum")` performs a Bayesian estimation of the
    exploratory reduced reparameterized unified model (ErRUM).
    `r pkg("rrum")` implements Gibbs sampling algorithm for
    Bayesian estimation of the Reduced Reparameterized Unified Model
    (rrum).
-   `r pkg("evdbayes")` provides tools for Bayesian analysis
    of extreme value models.
-   `r pkg("exactLoglinTest")` provides functions for
    log-linear models that compute Monte Carlo estimates of conditional
    P-values for goodness of fit tests.
-   `r pkg("factorQR")` is a package to fit Bayesian
    quantile regression models that assume a factor structure for at
    least part of the design matrix.
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
-   `r pkg("ggmcmc")` is a tool for assessing and diagnosing
    convergence of Markov Chain Monte Carlo simulations, as well as for
    graphically display results from full MCMC analysis.
-   `r pkg("gRain")` is a package for probability
    propagation in graphical independence networks, also known as
    Bayesian networks or probabilistic expert systems.
-   The `r pkg("HI")` package has functions to implement a
    geometric approach to transdimensional MCMC methods and random
    direction multivariate Adaptive Rejection Metropolis Sampling.
-   The `r pkg("hbsae")` package provides functions to
    compute small area estimates based on a basic area or unit-level
    model. The model is fit using restricted maximum likelihood, or in a
    hierarchical Bayesian way.
-   `r pkg("iterLap")` performs an iterative Laplace
    approximation to build a global approximation of the posterior
    (using mixture distributions) and then uses importance sampling for
    simulation based inference.
-   The function `krige.bayes()` in the `r pkg("geoR")`
    package performs Bayesian analysis of geostatistical data allowing
    specification of different levels of uncertainty in the model
    parameters. See the `r view("Spatial")` view for more
    information.
-   `r pkg("LAWBL")` is an R package latent (variable)
    analysis with with different Bayesian learning methods, including
    the partially confirmatory factor analysis, its generalized version,
    and the partially confirmatory item response model.
-   The `r pkg("lmm")` package contains R functions to fit
    linear mixed models using MCMC methods.
-   `r pkg("matchingMarkets")` implements a structural model
    based on a Gibbs sampler to correct for the bias from endogenous
    matching (e.g. group formation or two-sided matching).
-   The `r pkg("mcmcensemble")` package provides ensemble
    samplers for affine-invariant Monte Carlo Markov Chain, which allow
    a faster convergence for badly scaled estimation problems. Two
    samplers are proposed: the 'differential.evolution' sampler and
    the 'stretch' sampler.
-   `r pkg("MCMCglmm")` is package for fitting Generalised
    Linear Mixed Models using MCMC methods.
-   `r pkg("mcmcse")` allows estimation of multivariate
    effective sample size and calculation of Monte Carlo standard
    errors.
-   `r pkg("MHadaptive")` performs general
    Metropolis-Hastings Markov Chain Monte Carlo sampling of a user
    defined function which returns the un-normalized value (likelihood
    times prior) of a Bayesian model. The proposal variance-covariance
    structure is updated adaptively for efficient mixing when the
    structure of the target distribution is unknown.
-   The `r pkg("mlogitBMA")` Provides a modified function
    `bic.glm()` of the `r pkg("BMA")` package that can be
    applied to multinomial logit (MNL) data.
-   The `r pkg("MNP")` package fits multinomial probit
    models using MCMC methods.
-   `r pkg("mombf")` performs model selection based on
    non-local priors, including MOM, eMOM and iMOM priors..
-   `r pkg("NetworkChange")` is an R package for change
    point analysis in longitudinal network data. It implements a hidden
    Markovmultilinear tensor regression model. Model diagnostic tools
    using marginal likelihoods and WAIC are provided.
-   `r pkg("NGSSEML")` gives codes for formulating and
    specifying the non-Gaussian state-space models in the R language.
    Inferences for the parameters of the model can be made under the
    classical and Bayesian.
-   `r pkg("openEBGM")` calculates Empirical Bayes Geometric
    Mean (EBGM) and quantile scores from the posterior distribution
    using the Gamma-Poisson Shrinker (GPS) model to find unusually large
    cell counts in large, sparse contingency tables.
-   `r pkg("pacbpred")` perform estimation and prediction in
    high-dimensional additive models, using a sparse PAC-Bayesian point
    of view and a MCMC algorithm.
-   `r pkg("predmixcor")` provides functions to predict the
    binary response based on high dimensional binary features modeled
    with Bayesian mixture models.
-   `r pkg("prevalence")` provides functions for the
    estimation of true prevalence from apparent prevalence in a Bayesian
    framework. MCMC sampling is performed via JAGS/rjags.
-   The `r pkg("pscl")` package provides R functions to fit
    item-response theory models using MCMC methods and to compute
    highest density regions for the Beta distribution and the inverse
    gamma distribution.
-   `r pkg("PReMiuM")` is a package for profile regression,
    which is a Dirichlet process Bayesian clustering where the response
    is linked non-parametrically to the covariate profile.
-   `r pkg("revdbayes")` provides functions for the Bayesian
    analysis of extreme value models using direct random sampling from
    extreme value posterior distributions.
-   The `hitro.new()` function in `r pkg("Runuran")`
    provides an MCMC sampler based on the Hit-and-Run algorithm in
    combination with the Ratio-of-Uniforms method.
-   `r pkg("RoBMA")` implements Bayesian model-averaging for
    meta-analytic models, including models correcting for publication
    bias.
-   `r pkg("RSGHB")` can be used to estimate models using a
    hierarchical Bayesian framework and provides flexibility in allowing
    the user to specify the likelihood function directly instead of
    assuming predetermined model structures.
-   `r pkg("rstiefel")` simulates random orthonormal
    matrices from linear and quadratic exponential family distributions
    on the Stiefel manifold using the Gibbs sampling method. The most
    general type of distribution covered is the matrix-variate
    Bingham-von Mises-Fisher distribution.
-   `r pkg("RxCEcolInf")` fits the R x C inference model
    described in Greiner and Quinn (2009).
-   `r pkg("SamplerCompare")` provides a framework for
    running sets of MCMC samplers on sets of distributions with a
    variety of tuning parameters, along with plotting functions to
    visualize the results of those simulations.
-   `r pkg("SampleSizeMeans")` contains a set of R functions
    for calculating sample size requirements using three different
    Bayesian criteria in the context of designing an experiment to
    estimate a normal mean or the difference between two normal means.
-   `r pkg("SampleSizeProportions")` contains a set of R
    functions for calculating sample size requirements using three
    different Bayesian criteria in the context of designing an
    experiment to estimate the difference between two binomial
    proportions.
-   `r pkg("sbgcop")` estimates parameters of a Gaussian
    copula, treating the univariate marginal distributions as nuisance
    parameters as described in Hoff(2007). It also provides a
    semiparametric imputation procedure for missing multivariate data.
-   `r pkg("sna")`, an R package for social network
    analysis, contains functions to generate posterior samples from
    Butt's Bayesian network accuracy model using Gibbs sampling.
-   `r pkg("spBayes")` provides R functions that fit
    Gaussian spatial process models for univariate as well as
    multivariate point-referenced data using MCMC methods.
-   `r pkg("spikeslab")` provides functions for prediction
    and variable selection using spike and slab regression.
-   `r pkg("spikeSlabGAM")` implements Bayesian variable
    selection, model choice, and regularized estimation in
    (geo-)additive mixed models for Gaussian, binomial, and Poisson
    responses.
-   `r pkg("spTimer")` fits, spatially predict and
    temporally forecast large amounts of space-time data using Bayesian
    Gaussian Process Models, Bayesian Auto-Regressive (AR) Models, and
    Bayesian Gaussian Predictive Processes based AR Models.
-   `r pkg("ssgraph")` is for Bayesian inference in
    undirected graphical models using spike-and-slab priors for
    multivariate continuous, discrete, and mixed data.
-   `r pkg("ssMousetrack")` estimates previously compiled
    state-space modeling for mouse-tracking experiment using the
    `r pkg("rstan")` package, which provides the R interface
    to the Stan C++ library for Bayesian estimation.
-   `r pkg("stochvol")` provides efficient algorithms for
    fully Bayesian estimation of stochastic volatility (SV) models.
-   The `r pkg("tgp")` package implements Bayesian treed
    Gaussian process models: a spatial modeling and regression package
    providing fully Bayesian MCMC posterior inference for models ranging
    from the simple linear model, to nonstationary treed Gaussian
    process, and others in between.
-   `r bioc("vbmp")` is a package for variational Bayesian
    multinomial probit regression with Gaussian process priors. It
    estimates class membership posterior probability employing
    variational and sparse approximation to the full posterior. This
    software also incorporates feature weighting by means of Automatic
    Relevance Determination.
-   The `vcov.gam()` function the `r pkg("mgcv")` package
    can extract a Bayesian posterior covariance matrix of the parameters
    from a fitted `gam` object.
-   `r pkg("zic")` provides functions for an MCMC analysis
    of zero-inflated count models including stochastic search variable
    selection.

### Post-estimation tools

-   `r pkg("BayesValidate")` implements a software
    validation method for Bayesian softwares.
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
    robust linear, binomial, Pois- son, survival, response times,
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
    specification. As such, the `r view("gR")` task view may
    be of interest.
-   `r pkg("rstan")` provides R functions to parse, compile,
    test, estimate, and analyze Stan models by accessing the header-only
    Stan library provided by the `StanHeaders' package. The
    [Stan](http://mc-stan.org/) project develops a probabilistic
    programming language that implements full Bayesian statistical
    inference via MCMC and (optionally penalized) maximum likelihood
    estimation via optimization.
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
