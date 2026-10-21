# Load only the two authors' function definitions; do not execute experiment loops.
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
