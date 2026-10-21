.libPaths(c("reference/R-library", .libPaths()))
library(deconvolveR)
x <- as.matrix(read.csv("reference/benchmark_inputs/noisy_releases.csv", header = FALSE))
sigma <- scan("reference/benchmark_inputs/noise_sigma.csv", quiet = TRUE)
rows <- list()
for (i in seq_len(ncol(x))) {
  # Exact settings and preprocessing in the authors' func_deconvolveR_CI.
  scaled <- x[,i] / sigma
  q <- quantile(scaled, c(.25, .75))
  tau <- seq(q[1] - 3*diff(q), q[2] + 3*diff(q), length.out = 1000)
  start <- proc.time()[[3]]
  fit <- deconv(tau = tau, X = scaled, family = "Normal", pDegree = 5, c0 = .1)
  elapsed <- proc.time()[[3]] - start
  cdf <- fit$stats[, "G"]
  theta <- fit$stats[, "theta"] * sigma
  low <- max(1, which(cdf > .05)[1] - 1)
  high <- which(cdf > .95)[1]
  rows[[i]] <- data.frame(trial = i, seconds = elapsed, lower = theta[low], upper = theta[high],
                          observed_bins = nrow(fit$P), latent_points = ncol(fit$P))
}
write.csv(do.call(rbind, rows), "reference/benchmark_inputs/r_deconvolution.csv", row.names = FALSE)
cat("deconvolveR", as.character(packageVersion("deconvolveR")), "\n")
