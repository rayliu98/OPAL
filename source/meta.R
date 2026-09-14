# This is the code written by Jessie originally for dCLR
# revised by Ray Liu at 2024/6/12

library(meta)
library(logistf)
library(stats)

###############################
#### Step Three: clean the real data
###############################
# K: number of hospitals
# n: a list number of patients in each hospital (deprecated)
# y: outcome
#   1. each row represents a hospital
#   2. each column represents a patient
# x_all: covariates
#   1. each row represents a hospital
#   2. each column represents a patient
#   3. all the covariate are column-combined together (one patient can take length_par columns)
# length_par: number of covariates
# Please refer the sample data i shared: outcome.csv and variables.csv for reference



meta_dCLR <- function(K, n, y, x_all, length_par){
  beta_meta_list = matrix(0,nrow = dim(y)[1],ncol = length_par+1)
  se_meta_list = matrix(0,nrow = dim(y)[1],ncol = length_par+1)
  # Jessie changed the code here: 04.07.2020
  for (i in c(1:dim(y)[1])){
    X_each = matrix(unlist(x_all[i,])[!is.na(unlist(x_all[i,]))], ncol = length_par)
    Y_each = unlist(y[i,])[!is.na(unlist(y[i,]))]
    n=dim(X_each)[1]
    p=dim(X_each)[2]
    
    fit_each= tryCatch(logistf(Y_each ~ X_each),error=function(e)print(e))
    if(!is.null(fit_each$message)){
      # When X_each is singular, if you add intercept column manually, probably
      # the gim.fit will not converge and the coefficient estimates explode.
      # So we are going to fit the model without intercept and give the intercept
      # an estimate as the average of the residuals. It may be not a perfect solution
      # but better than nothing.
      
      fit_each= try(glm.fit(X_each,Y_each,family=binomial(),intercept=F))
      # X_each=cbind(rep(1,n),X_each)
      # minimize_logistic_loss(X_each,Y_each,learning_rate = 0.1,maxiter = 10000)$coefs
      intercept=mean(residuals.glm(fit_each))
      intercept.sd=sd(residuals.glm(fit_each))
      beta_meta_list[i,] = c(intercept,fit_each$coefficients)
      se_meta_list[i,] = c(intercept.sd,sqrt(diag(summary.glm(fit_each)$cov.unscaled)))
    } else {
      beta_meta_list[i,] = fit_each$coefficients
      se_meta_list[i,] = sqrt(diag(fit_each$var))
    }
  }
  
  
  #estimate from meta-analysis
  beta_meta_fix = c()
  beta_meta_random = c()
  beta_meta_fix_lower = c()
  beta_meta_fix_upper = c()
  beta_meta_random_lower = c()
  beta_meta_random_upper = c()
  for (i in 1:dim(beta_meta_list)[2]){
    ################### fixed effect meta method ######################
    tmp = tryCatch(metagen(beta_meta_list[,i], se_meta_list[,i],
                     comb.fixed = TRUE,comb.random = TRUE,sm="OR"),error=function(e)print(e))
    if(!is.null(tmp$message)){
      tmp = tryCatch(metagen(beta_meta_list[,i], se_meta_list[,i],
                       comb.fixed = TRUE,comb.random = TRUE,sm="OR",control=list(stepadj=0.5)),
               error=function(e)print(e))
      if(!is.null(tmp$message)){
        tmp = tryCatch(metagen(beta_meta_list[,i], se_meta_list[,i],
                                     comb.fixed = TRUE,comb.random = TRUE,sm="OR",control=list(stepadj=0.01)),
                       error=function(e)print(e))
        if(!is.null(tmp$message)){
          stop("step has already adjusted to 0.01!")
        }
      }
    }
    
    beta_meta_fix[i] = tmp$TE.fixed
    beta_meta_fix_lower[i] = tmp$lower.fixed
    beta_meta_fix_upper[i] = tmp$upper.fixed
    ################### random effect meta method ######################
    beta_meta_random[i] = tmp$TE.random
    beta_meta_random_lower[i] = tmp$lower.random
    beta_meta_random_upper[i] = tmp$upper.random
  }
  return(list(beta_meta_fix = beta_meta_fix,
              beta_meta_random = beta_meta_random,
              beta_meta_fix_lower = beta_meta_fix_lower,
              beta_meta_fix_upper = beta_meta_fix_upper,
              beta_meta_random_lower = beta_meta_random_lower,
              beta_meta_random_upper = beta_meta_random_upper))
}


