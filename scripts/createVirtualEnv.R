library(reticulate)
conda_create('ba-thesis')
conda_install('ba-thesis', '-e src/tf-kde', pip = TRUE)
