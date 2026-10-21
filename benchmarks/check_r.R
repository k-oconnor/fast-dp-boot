cat(R.version.string, "\n")
for (p in c("boot", "deconvolveR")) {
  cat(p, requireNamespace(p, quietly = TRUE), "\n")
}
