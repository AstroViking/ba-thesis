library(devtools)
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/reticulate")

# Install tensorflow with R to avoid crashes of RStudio
library(tensorflow)
install_tensorflow(method = "conda", envname = "ba-thesis")

# Create conda env ba-thesis and install tf-kde
library(reticulate)
conda_create('ba-thesis')
conda_install('ba-thesis', 'pip')
conda_install('ba-thesis', '-e src/tf-kde', pip = TRUE)
