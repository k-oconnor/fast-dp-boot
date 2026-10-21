.libPaths(c("reference/R-library", .libPaths()))
library(deconvolveR)
source("benchmarks/load_upstream_boot.R")
data <- as.matrix(read.csv("reference/benchmark_inputs/pipeline_inputs.csv", header = FALSE))
B <- 2000
sigma <- 1.125*sqrt(B)/nrow(data)
rows <- list()
for (i in seq_len(ncol(data))) {
  set.seed(i)
  start <- proc.time()[[3]]
  clean <- as.vector(func_boot(data[,i]))
  bootstrap_elapsed <- proc.time()[[3]] - start
  noisy <- clean + rnorm(B,0,sigma)
  scaled <- noisy/sigma
  q <- quantile(scaled,c(.25,.75))
  tau <- seq(q[1]-3*diff(q),q[2]+3*diff(q),length.out=1000)
  fit <- deconv(tau=tau,X=scaled,family="Normal",pDegree=5,c0=.1)
  elapsed <- proc.time()[[3]] - start
  rows[[i]] <- data.frame(trial=i,bootstrap_seconds=bootstrap_elapsed,total_seconds=elapsed)
}
write.csv(do.call(rbind,rows),"reference/benchmark_inputs/r_pipeline.csv",row.names=FALSE)
