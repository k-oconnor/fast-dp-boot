.libPaths(c("reference/R-library", .libPaths()))
library(deconvolveR)
dir.create("tests/fixtures/deconvolver", recursive = TRUE, showWarnings = FALSE)
folder <- "tests/fixtures/deconvolver"
x <- as.matrix(read.csv("reference/benchmark_inputs/noisy_releases.csv", header = FALSE))[,1]
sigma <- scan("reference/benchmark_inputs/noise_sigma.csv", quiet = TRUE)
z <- x / sigma
q <- quantile(z, c(.25, .75))
tau <- seq(q[1] - 3*diff(q), q[2] + 3*diff(q), length.out = 101)
fit <- deconv(tau = tau, X = z, family = "Normal", pDegree = 5, c0 = .1)
write.table(x, file.path(folder, "release.csv"), row.names = FALSE, col.names = FALSE, sep = ",")
write.table(sigma, file.path(folder, "sigma.csv"), row.names = FALSE, col.names = FALSE)
write.table(fit$Q, file.path(folder, "basis.csv"), row.names = FALSE, col.names = FALSE, sep = ",")
write.table(fit$P, file.path(folder, "likelihood.csv"), row.names = FALSE, col.names = FALSE, sep = ",")
write.csv(fit$stats, file.path(folder, "stats.csv"), row.names = FALSE)
write.table(fit$mle, file.path(folder, "coefficients.csv"), row.names = FALSE, col.names = FALSE)
write.table(fit$cov, file.path(folder, "covariance.csv"), row.names = FALSE, col.names = FALSE, sep = ",")
write.table(fit$cov.g, file.path(folder, "mass_covariance.csv"), row.names = FALSE, col.names = FALSE, sep = ",")
write.table(as.numeric(fit$loglik(fit$mle)), file.path(folder, "objective.csv"), row.names = FALSE, col.names = FALSE)
writeLines(c(paste("R:", R.version.string), paste("deconvolveR:", packageVersion("deconvolveR")),
             "Synthetic release; grid_size=101, pDegree=5, c0=.1, default observation edges=40."),
           file.path(folder, "PROVENANCE.txt"))
