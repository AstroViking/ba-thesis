library(reticulate)
conda_create('ba-thesis')
conda_install('ba-thesis', c('matplotlib', 'tensorflow', 'tensorflow-probability', 'seaborn'))
conda_install('ba-thesis', c('zfit', 'zfit_physics'), pip = TRUE)
conda_install('ba-thesis', '-e git+https://github.com/zfit/benchmarks#egg=zfit_benchmarks', pip = TRUE)