library(devtools)
devtools::install_github("rstudio/reticulate")
devtools::install_github("rstudio/tensorflow")

library(reticulate)
library(tensorflow)


# Create conda env ba-thesis
conda_create('ba-thesis')

# Install tensorflow with R to avoid crashes of RStudio
library(tensorflow)
install_tensorflow(version= '2.2.0', method = "conda", envname = "ba-thesis")

# Install tf_kde
conda_install('ba-thesis', '-e src/tf-kde', pip = TRUE)