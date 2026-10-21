dir.create("reference/R-library", recursive = TRUE, showWarnings = FALSE)
install.packages("deconvolveR", lib = "reference/R-library",
                 repos = "https://cloud.r-project.org", type = "source")
