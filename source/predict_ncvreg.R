predict.ncvreg <- function (object, X, type = c("link", "response", "class", "coefficients", 
                              "vars", "nvars"), lambda, which = 1:length(object$lambda), intercept=object$intercept, 
          ...) 
{
  type <- match.arg(type)
  beta <- coef.ncvreg(object, lambda = lambda, which = which, 
                      drop = FALSE)
  if (type == "coefficients") 
    return(beta)
  if (!inherits(object, "ncvsurv") && intercept) {
    alpha <- beta[1, ]
    beta <- beta[-1, , drop = FALSE]
  }
  else {
    alpha <- 0
    beta <- beta
  }
  if (type == "nvars") 
    return(apply(beta != 0, 2, sum))
  if (type == "vars") 
    return(drop(apply(beta != 0, 2, FUN = which)))
  eta <- sweep(X %*% beta, 2, alpha, "+")
  if (type == "link" || object$family == "gaussian") 
    return(drop(eta))
  resp <- switch(object$family, binomial = exp(eta)/(1 + exp(eta)), 
                 poisson = exp(eta))
  if (type == "response") 
    return(drop(resp))
  if (type == "class") {
    if (object$family == "binomial") {
      return(drop(1 * (eta > 0)))
    }
    else {
      stop("type='class' can only be used with family='binomial'", 
           call. = FALSE)
    }
  }
}
