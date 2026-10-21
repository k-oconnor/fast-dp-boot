# Uses the authors' function bodies, extracted without executing experiment loops.
library(boot)
tree <- parse("reference/upstream/clamp_normal_mean.r")
find_assignment <- function(x, name) {
  if (missing(x)) return(NULL)
  if (is.call(x) && identical(x[[1]], as.name("<-")) &&
      identical(x[[2]], as.name(name))) return(x)
  if (is.call(x) || is.expression(x) || is.pairlist(x)) {
    for (child in as.list(x)) {
      found <- find_assignment(child, name)
      if (!is.null(found)) return(found)
    }
  }
  NULL
}
eval(find_assignment(tree, "func_param"))
eval(find_assignment(tree, "func_boot"))
cases <- read.csv("reference/benchmark_inputs/cases.csv")
rows <- list()
for (i in seq_len(nrow(cases))) {
  x <- scan(cases$file[i], quiet = TRUE)
  B <- cases$replicates[i]
  times <- numeric(3)
  for (j in 1:3) {
    set.seed(42)
    start <- proc.time()[[3]]
    estimates <- func_boot(x)
    times[j] <- proc.time()[[3]] - start
  }
  rows[[i]] <- data.frame(case = cases$case[i], seconds = median(times),
                          estimate_mean = mean(estimates), estimate_variance = var(as.vector(estimates)))
}
write.csv(do.call(rbind, rows), "reference/benchmark_inputs/r_bootstrap.csv", row.names = FALSE)
