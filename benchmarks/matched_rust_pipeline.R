# Persistent single-worker reference, driven one trial at a time by Python.
# CSV I/O and the stdin/stdout coordination are outside the measured pipeline.
.libPaths(c("reference/R-library", .libPaths()))
suppressPackageStartupMessages(library(deconvolveR))
source("benchmarks/load_upstream_boot.R")
args <- commandArgs(trailingOnly = TRUE)
folder <- args[[1]]
data <- as.matrix(read.csv(file.path(folder, "inputs.csv"), header = FALSE))
B <- 2000
sigma <- 1.125 * sqrt(B) / nrow(data)

pipeline <- function(i, seed) {
  set.seed(seed)
  start <- proc.time()[[3]]
  clean <- as.vector(func_boot(data[, i]))
  noisy <- clean + rnorm(B, 0, sigma)
  release_elapsed <- proc.time()[[3]] - start
  scaled <- noisy / sigma
  q <- quantile(scaled, c(.25, .75))
  tau <- seq(q[1] - 3 * diff(q), q[2] + 3 * diff(q), length.out = 1000)
  fit <- deconv(tau = tau, X = scaled, family = "Normal", pDegree = 5, c0 = .1)
  total_elapsed <- proc.time()[[3]] - start
  # Interval extraction and validation are outside both implementations' timers.
  cdf <- fit$stats[, "G"]
  theta <- fit$stats[, "theta"] * sigma
  low <- max(1, which(cdf > .05)[1] - 1)
  high <- which(cdf > .95)[1]
  stopifnot(nrow(fit$stats) == 1000,
            all(is.finite(fit$stats[, c("theta", "g", "G", "SE.g", "Bias.g")])),
            identical(dim(fit$cov.g), c(1000L, 1000L)),
            all(is.finite(fit$cov.g)))
  list(release_seconds = release_elapsed, total_seconds = total_elapsed,
       lower = theta[low], upper = theta[high], noisy = noisy,
       nonfinite_cdf_se = sum(!is.finite(fit$stats[, "SE.G"])))
}

# Load/compile/warm allocations before the trial protocol starts.
invisible(pipeline(1, 0))
writeLines(c(R.version.string,
             paste("deconvolveR", packageVersion("deconvolveR")),
             paste("boot", packageVersion("boot")),
             paste("BLAS", extSoftVersion()[["BLAS"]])),
           file.path(folder, "r_environment.txt"))
releases <- matrix(NA_real_, nrow = B, ncol = ncol(data))
cat("READY\n")
flush(stdout())
input <- file("stdin", open = "r")
repeat {
  line <- readLines(input, n = 1, warn = FALSE)
  if (length(line) == 0 || line == "STOP") break
  i <- as.integer(line)
  stopifnot(length(i) == 1, !is.na(i), i >= 1, i <= ncol(data))
  result <- pipeline(i, i)
  releases[, i] <- result$noisy
  cat(sprintf("RESULT,%d,%.17g,%.17g,%.17g,%.17g,%d\n", i,
              result$release_seconds, result$total_seconds,
              result$lower, result$upper, result$nonfinite_cdf_se))
  flush(stdout())
}
close(input)
write.table(releases, file.path(folder, "r_releases.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
