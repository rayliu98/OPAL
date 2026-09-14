# This is the simulation in Section 4.2 in OPAL
# author: Ray Liu
# date: 2024/11/28

###############################
#### Notes 
###############################
#### Notes for fitting SCAD model: 
#1. "ncpen" can fit the nonconcave penalized model, which is more powerful
#in aspect of the user-defined arguments "x.standardize" and "intercept". For
#the same lambda, "ncvreg" and "ncpen" get almost the same result for scad in
#logistic regression. However, 'intercept' does affect the estimation of the 
#coefficient in contrast to the first point.
#2. "coef.ncpen" returns the coefficients matrix for all lambda values! So we
#avail "coef.ncvreg" to extract the beta estimation corresponding to the lambda 
#selected by cv that we are really interested in.
#3. "coef.cv.ncpen" gets two loss function for choosing lambda, "rmse" and "like".
#Generally, "like" impose greater penalty (larger lambda) than "rmse".
###############################
#### Compared methods (confidence interval constructed for the star-marked methods *)
###############################
#1. OAPR* (Optimal Aggregation Prediction with the weight Restriction)
#2. OAP* (Optimal Aggregation Prediction)
#3. Full* (gold standard)
#4. Naive averaging
#5. Cubic rate averaging (Shi et al., 2018)
#6. AGD (Accelerated Gradient Descent, Nesterov, 1983)
#7. Meta analysis*
#8. dCLR* (distributed Conditional Logistic Regression, Tong et al., 2022)
#9. CEASE* (Communication-Efficient Accurate Statistical Estimators, Fan et al., 2021)
#10. CEASE with averaging* (Fan et al., 2021)
#11. CSL* (Communication-efficient Surrogate Likelihood, Jordan et al., 2019)
#12. CSL with averaging* (Jordan et al., 2019 & Fan et al., 2021)
#13. ADMM (Alternating Direction Method of Multipliers, Boyd et al., 2011)
#14. GIANT (Globally Improved Approximate NewTon method, Wang et al., 2018))
#15. Weighted averaging with inverse variance weights
#16. Single site

###############################
#### install/load library
###############################
#setwd("~/DCGLM/sim_fan")
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
Sys.setenv(NUMEXPR_NUM_THREADS = "1")
library(snowfall)
library(ncvreg)
library(glmnet)
library(caret)
library(ncpen)
library(ClusterR)
library(CVXR)
library(osqp)
source("coef_ncvreg.R")
source("lam_names.R")
source("predict_ncvreg.R")
source("OARFISH_solver.R")
source("ci_OARFISH.R")
source("logistic_solvers_forR.R")
source("meta.R")

###############################
#### DGP and settings
###############################

# DGP 2.2 Label imbalance varies
# (modified from JASA paper "Communication-Efficient Accurate Statistical
# Estimation" by Jianqing Fan et al.)

# The dimensionality d, true parameter \theta^*, design matrix X, response Y
# local sample size nn and site number M.

set.seed(202402)
md_basic=3 
K_fixed=vector(length = 3) # K is (M-2), (M-2)/2, and (M-2)/4, respectively
N <- 10000 # the total sample size N is fixed
ns <- c(2000,1000)
Ms <- round(N/ns)
ds <- c(21)

by=c(0,0) #difference parameter, used for 'seq' function, seq(from,by,length.out), 
          #to set local sample size and allow it different across different site
G<-100 #the setting for replicates
corrx <- 0 #control the correlation between the covariates in design matrix
rares <- seq(0.5,0.9,0.05) #control the proportion of positive labels (to produce unbalanced data)
df.t <- 0 #control the degree of freedom of t distribution
lambda.e <- 0 #control the lambda of exponential distribution
sig.mis <- 0 #control the signal strength of the misspecified parameter
Kfold <- 10 #when predicting the probability of in-site observations, how many folds we set?
cpunum <- 50 #the setting for the amount of CPUs requested for the cluster

md_ci_name <- c(paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP"),"b1y_schurO1","type1",sep="_"),
                paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP"),"b2","type1",sep="_"),
                "Full","Meta_f","Meta_r",
                paste(c(paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
                        paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_")),"s1",sep="_"),
                paste(c(paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
                        paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_")),"s2",sep="_"))
M_name <- paste("M",Ms,sep="_")
md_basic_name <- c("scad","lasso","glm")
d_name <- paste("d",ds,sep="_")
md_ci=length(md_ci_name)

# Recording lists for the mse, running time and weights, where 'timeM' records  
# the time for specific term in 'timename' and timecostM records the total running
# tiem for G replications.
# weights are computed using osqp solver, which produces more dense solution.
# Each element of the 'wM' array store a list of the weights distributed to M sites
# (G=100 replicates) with local sample size n_m and parameter dimension d.

alphaM <-array(dim=c(21,length(Ms),md_ci,md_basic,length(ds),length(rares))) #value of coverage rate for true parameters (focus on the first 21 coefs) 
lengthciM <-array(dim=c(21,length(Ms),md_ci,md_basic,length(ds),length(rares))) #length of ci (focus on the first 21 coefs)
timecostM <-array(0,c(length(Ms),3,length(ds),length(rares)))#'3' means the output of 'proc.time'

dimnames(alphaM) <- dimnames(lengthciM) <- list(NULL,M_name,md_ci_name,md_basic_name,d_name,as.character(rares))

simfunc <- function(g){
  sfCat(paste0("Replicate ",g,": starts"), sep="\n")
  set.seed(2024+g)
  #generate the coefficients (Partial strong signals, partial signals located on 
  #                           a uniform sphere, and partial zero signals.)
  d_sphere_zero <- round((d-5)/2)
  theta_uniform <- rnorm(d_sphere_zero)
  theta_uniform <- theta_uniform/norm(as.matrix(theta_uniform),type = "2")
  theta_star <- c(1,seq(2,1.25,-0.25),theta_uniform,rep(0,d_sphere_zero))
  
  XX<-list() #save the design matrix at each site
  YY<-list()
  MU<-list()
  Obs_p <- list()
  Lambda_series.scad <- vector(length = M)
  M.contaminate <- round(M/2)
  M.overlap <- round(M/4)
  
  #DGP for M sites
  for (j in 1:M){
    nm<-nn[j]#the sample size of jth site
    nm.positive <- round(nm*(1-rare))
    nm.negative <- nm-nm.positive
    SigmaX <- diag(sqrt(c(10,5,2,rep(1,d-4))),d-1)
    CorrX <- corrx*tcrossprod(rep(1,d-1))+(1-corrx)*diag(d-1)
    SigmaX <- SigmaX%*%CorrX%*%SigmaX
    pos=neg=0
    Xm=Ym=NULL
    while(pos<nm.positive|neg<nm.negative){
      Covariates.normal <- mvrnorm(1,rep(0,d-1),SigmaX)
      Covariates.exp <- Covariates.t <- 0
      if(j>=1&j<=M.contaminate){
        if(lambda.e==0){Covariates.exp=0} else {Covariates.exp <- rexp(1*(d-1),lambda.e)-1/lambda.e}
      }
      if(j>=(1+M.overlap)&j<=(M.contaminate+M.overlap)){
        if(df.t<3){Covariates.t=0} else {Covariates.t <- rt(1*(d-1),df.t)}
      }
      Covariates <- Covariates.normal+Covariates.exp+Covariates.t
      xm<-c(1,Covariates) #generate the covariates at the jth site
      x.mis <- rnorm(1)
      etam <- xm %*% theta_star + sig.mis*x.mis #the true linear explanatory part at the jth site
      probm <- 1/(1 + exp(-etam)) #the true probability of success at the jth site
      ym <- rbinom(1,1,probm) #the response Y_m
      if(ym==1&pos<nm.positive){pos=pos+1;Xm=rbind(Xm,xm);Ym=c(Ym,ym)}
      if(ym==0&neg<nm.negative){neg=neg+1;Xm=rbind(Xm,xm);Ym=c(Ym,ym)}
    }
    obs_partition <- createFolds(Ym,k=Kfold)
    

    ###############################
    #### SCAD
    scad_mod <- ncpen(y.vec=Ym, x.mat=Xm[,-1], family="binomial", penalty="scad",
                      x.standardize=T, intercept=T)
    lambda_scad.fit <- cv.ncpen(y.vec=Ym, x.mat=Xm[,-1], family="binomial", penalty="scad",
                                x.standardize=T, intercept=T)
    if(d>50){
      lambda_scad <- coef.cv.ncpen(lambda_scad.fit,type="like")$lambda
    } else {
      lambda_scad <- coef.cv.ncpen(lambda_scad.fit,type="rmse")$lambda
    }
    
    if(lambda_scad>scad_mod$lambda[1]){
      betahatm_scad <- coef.ncvreg(scad_mod, lambda=scad_mod$lambda[1])
      warning("lambda_scad>scad_mod$lambda[1]")
    } else if(lambda_scad<scad_mod$lambda[length(scad_mod$lambda)]) {
      betahatm_scad <- coef.ncvreg(scad_mod, lambda=scad_mod$lambda[length(scad_mod$lambda)])
      warning("lambda_scad<scad_mod$lambda[length(scad_mod$lambda)]")
    } else {
      betahatm_scad <- coef.ncvreg(scad_mod, lambda=lambda_scad)
    }
    
    Lambda_series.scad[j] <- lambda_scad
    betahat[,1,j] <- as.matrix(betahatm_scad)
    
    lambda_series.scad <- scad_mod$lambda
    for (k in 1:Kfold) {
      scad_mod_cv <- ncpen(x.mat=Xm[-obs_partition[[k]],-1],y.vec=Ym[-obs_partition[[k]]],
                           family="binomial", penalty="scad",x.standardize=T, intercept=T,
                           lambda = lambda_series.scad)
      #remarks: 
      #1. Here we fit 'scad_mod_cv' with the same lambda series produced in 'scad_mod',
      #and you may get warnings of "In control.ncpen(y.vec, x.mat, family, penalty,
      #x.standardize,  ... :lambda is extended up to ...". Never mind, this means 
      #ncpen add some 'lambda' that is greater than the maximum of 'lambda_series'
      #for implementing the gradient descent method.
      #2. When predicting for 'MUtildecv', we use 'lambda_scad' as the input of lambda
      #and do not tune it anymore. NOTICE you would better to assign the lambda_series,
      #otherwise 'ncpen' would assign it according to the samples and 'lambda_scad' with
      #probability will not be contained in this series, and finally you get nothing.
      #Although "scad_mod_cv" takes 'lambda_scad' as the input of lambda, it would probably
      #not compute the beta corresponding to each lambda instead of a piece of interval
      #of the lambda series, which satisfies that the max lambda makes all coefficients 
      #estimations equal to 0, and the min lambda makes all coefficients estimation
      #either not equal to 0 or too irrational LARGE(remember it is a non-convex 
      #optimization problem!), in the sense it will stop at a suitable small lambda 
      #you provide rather the minimum of the lambda series. So we solve it by trimming
      #(directly use the maximum/minimum of this lambda path "scad_mod_cv" finally get),
      #although it is not the best because "lambda_scad" do changes for this fold.
      #3. Another choice is refiting the 'cv.ncpen' to select the tuning for each fold,
      #which follows the same fitting procedure that applies to the full model, if
      #you do not mind time cost, here's the codes:
      #for (k in 1:Kfold) {
      #scad_mod_cv <- ncpen(x.mat=Xm[-obs_partition[[k]],],y.vec=Ym[-obs_partition[[k]]],
      #family="binomial", penalty="scad",x.standardize=T, intercept=F)
      #lambda_scad.fit_cv <- cv.ncpen(y.vec=Ym[-obs_partition[[k]]], x.mat=Xm[-obs_partition[[k]],],
      #                               family="binomial", penalty="scad", x.standardize=T, intercept=F)
      #lambda_scad_cv <- coef.cv.ncpen(lambda_scad.fit_cv,type="like")$lambda
      #BetaHatcv[,k,j]<-coef.ncvreg(scad_mod_cv, lambda=lambda_scad_cv)
      #MUtildecv[obs_partition[[k]],j] <- predict.ncvreg(scad_mod_cv,Xm[obs_partition[[k]],],
      #                                                  type="response",lambda=lambda_scad_cv)
      if(lambda_scad>scad_mod_cv$lambda[1]){
        betahatm_scad_cv <- coef.ncvreg(scad_mod_cv, lambda=scad_mod_cv$lambda[1])
        BetaHatcv[,1,k,j]<-betahatm_scad_cv
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatm_scad_cv)
        warning("lambda_scad>scad_mod_cv$lambda[1]")
      } else if(lambda_scad<scad_mod_cv$lambda[length(scad_mod_cv$lambda)]) {
        betahatm_scad_cv <- coef.ncvreg(scad_mod_cv, lambda=scad_mod_cv$lambda[length(scad_mod_cv$lambda)])
        BetaHatcv[,1,k,j]<-betahatm_scad_cv
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatm_scad_cv)
        warning("lambda_scad<scad_mod_cv$lambda[length(scad_mod_cv$lambda)]")
      } else {
        betahatm_scad_cv <- coef.ncvreg(scad_mod_cv, lambda=lambda_scad)
        BetaHatcv[,1,k,j]<-betahatm_scad_cv
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatm_scad_cv)
      }
      if(sum(is.nan(MUtildecv[[1,j]][obs_partition[[k]]]))>0){
        # Modify mu when the situation that exp(testeta)/(1+exp(testeta)) produces 'nan' happens
        MUtildecv[[1,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[1,j]][obs_partition[[k]]])==1)] <- 1
      }
    }
    
    ###############################
    #### lasso
    lasso_mod = glmnet(Xm[,-1], Ym, family = "binomial", intercept=T, standardize = T) 
    #here compute the lasso estimates(which is also served as the initial value of CSL)
    lambda_lasso.fit = cv.glmnet(Xm[,-1], Ym, family = "binomial", intercept=T, standardize = T)
    lambda_lasso = lambda_lasso.fit$lambda.min
    betahat[,2,j]<-as.matrix(coef(lasso_mod, s=lambda_lasso))
    #store the lasso estimates of beta at jth site
    #Notes:
    #1. We apply 'standardize=TRUE' here for two functions above.
    #2. The number of folds is 10 by default for 'cv,glmnet', and this function
    #partition the samples randomly rather than divide from the start with equal length.
    
    lambda_series.lasso <- lasso_mod$lambda
    for (k in 1:Kfold) {
      lasso_mod_cv <- glmnet(Xm[-obs_partition[[k]],-1],Ym[-obs_partition[[k]]],
                             family="binomial", intercept=T, lambda = lambda_series.lasso)
      BetaHatcv[,2,k,j]<-as.matrix(coef(lasso_mod_cv, s=lambda_lasso))
      MUtildecv[[2,j]][obs_partition[[k]]] <- predict(lasso_mod_cv,Xm[obs_partition[[k]],-1],
                                                      type="response",s=lambda_lasso)
      if(sum(is.nan(MUtildecv[[2,j]][obs_partition[[k]]]))>0){
        MUtildecv[[2,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[2,j]][obs_partition[[k]]])==1)] <- 1
      }
    }
    
    ###############################
    #### glm
    betahat[,3,j]<-minimize_logistic_loss(Xm,Ym,learning_rate = 0.1,maxiter = 10000)$coefs
    
    for (k in 1:Kfold) {
      BetaHatcv[,3,k,j]<-minimize_logistic_loss(Xm[-obs_partition[[k]],],Ym[-obs_partition[[k]]],learning_rate = 0.1,maxiter = 10000)$coefs
      MUtildecv[[3,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%BetaHatcv[,3,k,j])
      if(sum(is.nan(MUtildecv[[3,j]][obs_partition[[k]]]))>0){
        MUtildecv[[3,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[3,j]][obs_partition[[k]]])==1)] <- 1
      }
    }
    
    ###############################
    #### saving
    XX[[j]]<-   Xm#store the design matrix on hand at jth site into XX list
    YY[[j]]<-   Ym#store the response vector at jth site into YY list
    MU[[j]]<-   probm#store the mean vector at jth site into MU list
    Obs_p[[j]]<- obs_partition
  }
  sfCat(paste0("Replicate ",g,": model fitting at local sites completes"), sep="\n")
  
  ###############################
  #### Clustering
  ###############################
  #### K-selection
  # Temporarily we consider three schemes of selecting K: (1) K=log(M); 
  # (2) elbow method; (3) distortion method.
  # Before applying K-means, betahat is standardized.
  K_logM <- round(log(M))
  M_for_Kmeans <- min(M-2,100)
  K_fixed <- c(round(M_for_Kmeans/4),round(M_for_Kmeans/2),M_for_Kmeans)
  
  betahat_s <- array(0,c(M,d,3))#save the scaled betahat: scad, lasso, glm
  
  K.table <- matrix(0,3+length(K_fixed),3)
  for (md in 1:3) {
    betahat_s_tmp <- scale(t(drop0(betahat[,md,])))
    betahat_s[,,md] <- betahat_s_tmp
    K.table[2,] <- K_selection(betahat_s[,,md],"elbow",M-2)
    K.table[3,] <- K_selection(betahat_s[,,md],"distortion",M-2)
  }
  K.table[1,] <- K_logM
  for (kseries in 1:length(K_fixed)) {
    K.table[kseries+3,] <- K_fixed[kseries]
  }
  dimnames(K.table) <- list(
    c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
    c("scad","lasso","glm"))
  
  ###############################
  #### Implement K-means
  kmeans_clusters <- array(0,c(M,3+length(K_fixed),3),
                           dimnames = list(NULL, c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                           c("scad","lasso","glm")))
  for (kseries in 1:(3+length(K_fixed))) {
    for (md in 1:3) {
      if(K.table[kseries,md]==M){
        kmeans_clusters[,kseries,md] <- 1:M
      } else {
        non_zero_cols <- apply(betahat_s[,,md], 2, function(col) any(col != 0) & any(!is.na(col)))
        betahatMat_dropcols <- betahat_s[,,md][, non_zero_cols, drop = FALSE] # 'drop' coerce it to be a matrix
        
        kmeans_tmp <- KMeans_rcpp(betahatMat_dropcols, K.table[kseries,md])
        kmeans_clusters[,kseries,md] <- kmeans_tmp$clusters
      }
    }
  }
  sfCat(paste0("Replicate ",g,": Kmeans completes"), sep="\n")
  
  ###############################
  #### Method OAPR 
  ###############################
  betahatOAPR <- array(0,c(d,3+length(K_fixed),3),dimnames = list(NULL,
                                                                  c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                                                  c("scad","lasso","glm")))
  iter.center_opt <- array(0,c(3+length(K_fixed),3),dimnames = 
                             list(c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                  c("scad","lasso","glm")))
  # Record the estimated weights
  w.table <- array(0,c(M,3+length(K_fixed),3),dimnames = 
                     list(NULL,c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                          c("scad","lasso","glm")))
  ci_betahatOAPR_glm <- array(0,c(d,3+length(K_fixed),2,2)) # only for scad and glm
  ci_betahatOAPR_scad.r <- array(0,c(d,3+length(K_fixed),2,2))
  # dimension, kmeans, lower/upper bound, b1y_schurO1_type1/b2
  
  for (md in 1:3) {
    for (kseries in 1:(3+length(K_fixed))) {
      res <- betahat_OARFISH(XX,YY,betahat[,md,],MUtildecv[md,],kmeans_clusters[,kseries,md])
      coefs.OAPR <- res$coefs
      #coefs.OAPR[abs(coefs.OAPR)<1e-8] <- 0
      betahatOAPR[,kseries,md]=coefs.OAPR # OAPR coefficient estimator with osqp solver
      iter.center_opt[kseries,md] <- res$iter
      w.table[,kseries,md] <- res$w
      
      if(md==1){
        # Omega1_est="b1y_schurO1", method_type="type1"
        OAPR.supp <- betahatOAPR[,kseries,md]!=0
        OAPR.r <- t(diag(OAPR.supp)[OAPR.supp,])
        res.ci <- ci.OARFISH(XX,YY,betahatOAPR[,kseries,md],w.table[,kseries,md],
                             betahat[,md,],kmeans_clusters[,kseries,md],"b1y_schurO1","type1",FALSE,
                             0.05,500,Lambda_series.scad,Kfold)
        
        ci_betahatOAPR_scad.r[,kseries,1,1] <- OAPR.r%*%res.ci$ci_l
        ci_betahatOAPR_scad.r[,kseries,2,1] <- OAPR.r%*%res.ci$ci_u
        # Omega1_est="b2", method_type="type1"
        res.ci <- ci.OARFISH(XX,YY,betahatOAPR[,kseries,md],w.table[,kseries,md],
                             betahat[,md,],kmeans_clusters[,kseries,md],"b2","type1",FALSE,
                             0.05,500,Lambda_series.scad,Kfold)
        
        ci_betahatOAPR_scad.r[,kseries,1,2] <- OAPR.r%*%res.ci$ci_l
        ci_betahatOAPR_scad.r[,kseries,2,2] <- OAPR.r%*%res.ci$ci_u
      }
      if(md==3){
        # Omega1_est="b1y_schurO1", method_type="type1"
        res.ci <- ci.OARFISH(XX,YY,betahatOAPR[,kseries,md],w.table[,kseries,md],
                             betahat[,md,],kmeans_clusters[,kseries,md],"b1y_schurO1","type1",FALSE,
                             0.05,500,rep(0,M),Kfold)
        ci_betahatOAPR_glm[,kseries,1,1] <- res.ci$ci_l
        ci_betahatOAPR_glm[,kseries,2,1] <- res.ci$ci_u
        # Omega1_est="b2", method_type="type1"
        res.ci <- ci.OARFISH(XX,YY,betahatOAPR[,kseries,md],w.table[,kseries,md],
                             betahat[,md,],kmeans_clusters[,kseries,md],"b2","type1",FALSE,
                             0.05,500,rep(0,M),Kfold)
        ci_betahatOAPR_glm[,kseries,1,2] <- res.ci$ci_l
        ci_betahatOAPR_glm[,kseries,2,2] <- res.ci$ci_u
      }
    }
  }
  sfCat(paste0("Replicate ",g,": OARP completes"), sep="\n")
  
  
  ###############################
  #### Method OAP
  ###############################
  betahatOAP <- matrix(0,d,3,dimnames = list(NULL,c("scad","lasso","glm")))
  iter.OAP <- matrix(0,3,1,dimnames = list(c("scad","lasso","glm"),NULL))
  # Record the estimated weights
  w.OAP <- matrix(0,M,3,dimnames = list(NULL,c("scad","lasso","glm")))
  ci_betahatOAP_glm <- array(0,c(d,2,2)) # only for scad and glm
  ci_betahatOAP_scad.r <- array(0,c(d,2,2))
  # dimension, lower/upper bound, b1y_schurO1_type1/b2
  
  for (md in 1:3) {
    res <- betahat_OARFISH(XX,YY,betahat[,md,],MUtildecv[md,],1:M)
    coefs.OAP <- res$coefs
    #coefs.OAP[abs(coefs.OAP)<1e-8] <- 0
    betahatOAP[,md]=coefs.OAP # OAPR coefficient estimator with osqp solver
    iter.OAP[md,1] <- res$iter
    w.OAP[,md] <- res$w
    
    if(md==1){
      OAP.supp <- betahatOAP[,md]!=0
      OAP.r <- t(diag(OAP.supp)[OAP.supp,])
      # Omega1_est="b1y_schurO1", method_type="type1"
      res.ci <- ci.OARFISH(XX,YY,betahatOAP[,md],w.OAP[,md],
                           betahat[,md,],1:M,"b1y_schurO1","type1",FALSE,
                           0.05,500,Lambda_series.scad,Kfold)
      
      ci_betahatOAP_scad.r[,1,1] <- OAP.r%*%res.ci$ci_l
      ci_betahatOAP_scad.r[,2,1] <- OAP.r%*%res.ci$ci_u
      # Omega1_est="b2", method_type="type1"
      res.ci <- ci.OARFISH(XX,YY,betahatOAP[,md],w.OAP[,md],
                           betahat[,md,],1:M,"b2","type1",FALSE,
                           0.05,500,Lambda_series.scad,Kfold)
      
      ci_betahatOAP_scad.r[,1,2] <- OAP.r%*%res.ci$ci_l
      ci_betahatOAP_scad.r[,2,2] <- OAP.r%*%res.ci$ci_u
    }
    if(md==3){
      # Omega1_est="b1y_schurO1", method_type="type1"
      res.ci <- ci.OARFISH(XX,YY,betahatOAP[,md],w.OAP[,md],
                           betahat[,md,],1:M,"b1y_schurO1","type1",FALSE,
                           0.05,500,rep(0,M),Kfold)
      ci_betahatOAP_glm[,1,1] <- res.ci$ci_l
      ci_betahatOAP_glm[,2,1] <- res.ci$ci_u
      # Omega1_est="b2", method_type="type1"
      res.ci <- ci.OARFISH(XX,YY,betahatOAP[,md],w.OAP[,md],
                           betahat[,md,],1:M,"b2","type1",FALSE,
                           0.05,500,rep(0,M),Kfold)
      ci_betahatOAP_glm[,1,2] <- res.ci$ci_l
      ci_betahatOAP_glm[,2,2] <- res.ci$ci_u
    }
  }
  sfCat(paste0("Replicate ",g,": OAP completes"), sep="\n")
  
  ###############################
  #### Method Equal Weight (also used as the warm start)
  ###############################
  betahatEW <- array(0,c(d,3),dimnames = list(NULL,c("scad","lasso","glm")))
  for (md in 1:3) {
    w0 <- matrix(1,nrow=M,ncol=1)/M #the initial value of weights
    betahatEW[,md]=betahat[,md,]%*%w0; #EMA coefficient estimator
  }
  
  
  ###############################
  #### Method Full
  ###############################
  betahatFull <- array(0,c(d,3),dimnames = list(NULL,c("scad","lasso","glm")))
  
  X<-c() 
  Y<-c() 
  for (j in 1:M){
    X<-rbind(X,XX[[j]])
    Y<-c(Y,YY[[j]])
  } 
  for (md in 1:3) {
    if(md==1){
      scad_mod <- ncpen(Y, X[,-1], family="binomial", penalty="scad",
                        x.standardize=T, intercept=T)
      lambda_scad.fit <- cv.ncpen(Y, X[,-1], family="binomial", penalty="scad",
                                  x.standardize=T, intercept=T)
      if(d>50){
        lambda_scad <- coef.cv.ncpen(lambda_scad.fit,type="like")$lambda
      } else {
        lambda_scad <- coef.cv.ncpen(lambda_scad.fit,type="rmse")$lambda
      }
      betahatFull[,md] <- coef.ncvreg(scad_mod, lambda=lambda_scad)
      
      full_supp <- betahatFull[,md]!=0
      full.r <- t(diag(full_supp)[full_supp,])
      qfull_supp <- sum(full_supp)
      ci_betahatFull_scad <- array(0,c(qfull_supp,2)) 
      ci_betahatFull_scad.r <- array(0,c(d,2))
      Sigma_full <- diag(full_supp)[full_supp,]%*%Sigmahat(betahatFull[,md],lambda = lambda_scad)%*%t(diag(full_supp)[full_supp,])
      Omega1_full <- diag(full_supp)[full_supp,]%*%Omega1_b2(betahatFull[,md],X)%*%t(diag(full_supp)[full_supp,])
      I1inv_full <- solve(Omega1_full + Sigma_full)
      filling_full <- diag(full_supp)[full_supp,]%*%Omega1_b1y(betahatFull[,md],X,Y)%*%t(diag(full_supp)[full_supp,])
      sandwitch_full <- I1inv_full%*%filling_full%*%I1inv_full
      
      b_full <- diag(full_supp)[full_supp,]%*%bhat(betahatFull[,md],lambda = lambda_scad)
      Fhat_full <- I1inv_full%*%b_full
      
      ci_betahatFull_scad[,1] <- betahatFull[,md][full_supp] + Fhat_full - qnorm(0.05/2,lower.tail = FALSE)*sqrt(diag(sandwitch_full))/sqrt(N)
      ci_betahatFull_scad[,2] <- betahatFull[,md][full_supp] + Fhat_full + qnorm(0.05/2,lower.tail = FALSE)*sqrt(diag(sandwitch_full))/sqrt(N)
      
      ci_betahatFull_scad.r[,1] <- full.r%*%ci_betahatFull_scad[,1]
      ci_betahatFull_scad.r[,2] <- full.r%*%ci_betahatFull_scad[,2]
    }
    if(md==2){
      lasso_mod = glmnet(X[,-1], Y, family = "binomial", intercept=T, standardize = T) 
      lambda_lasso.fit = cv.glmnet(X[,-1], Y, family = "binomial" ,intercept=T, standardize=T)
      betahatFull[,md]<-as.matrix(coef(lasso_mod, s=lambda_lasso.fit$lambda.min))
    }
    if(md==3){
      glm_mod = glm.fit(X, Y, family = binomial(), intercept=F) 
      betahatFull[,md]<-coef(glm_mod)
      
      ci_betahatFull_glm <- array(0,c(d,2)) 
      Omega1_full <- Omega1_b2(betahatFull[,md],X)
      I1inv_full <- solve(Omega1_full)
      filling_full <- Omega1_b1y(betahatFull[,md],X,Y)
      sandwitch_full <- I1inv_full%*%filling_full%*%I1inv_full
      
      ci_betahatFull_glm[,1] <- betahatFull[,md] - qnorm(0.05/2,lower.tail = FALSE)*sqrt(diag(sandwitch_full))/sqrt(N)
      ci_betahatFull_glm[,2] <- betahatFull[,md] + qnorm(0.05/2,lower.tail = FALSE)*sqrt(diag(sandwitch_full))/sqrt(N)
    }
  }
  sfCat(paste0("Replicate ",g,": Full completes"), sep="\n")
  
  ###############################
  #### Method Meta
  ###############################
  betahatMeta <- array(0,c(d,2),dimnames = list(NULL,c("meta_fixed","meta_random")))
  ci_betahatMeta <- array(0,c(d,2,2)) # dim, ci, fixed/random
  y.dCLR <- matrix(nrow = M,ncol = max(nn))
  x_all.dCLR <- matrix(nrow = M,ncol = max(nn)*(d-1))
  for (i in 1:M) {
    y.dCLR[i,1:nn[i]] <- YY[[i]]
    x_all.dCLR[i,1:(nn[i]*(d-1))] <- unlist(c(XX[[i]][,-1]))
  }
  res.meta=meta_dCLR(M,nn,y.dCLR,x_all.dCLR,d-1)
  betahatMeta[,1] <- res.meta$beta_meta_fix
  betahatMeta[,2] <- res.meta$beta_meta_random
  ci_betahatMeta[,1,1] <- res.meta$beta_meta_fix_lower
  ci_betahatMeta[,2,1] <- res.meta$beta_meta_fix_upper
  ci_betahatMeta[,1,2] <- res.meta$beta_meta_random_lower
  ci_betahatMeta[,2,2] <- res.meta$beta_meta_random_upper
  sfCat(paste0("Replicate ",g,": Meta completes"), sep="\n")
  
  ###############################
  #### Method CEASE(a) (a is alpha)
  ###############################
  betahatCEASEa <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  ci1_betahatCEASEa <- array(0,c(d,2,3))
  ci2_betahatCEASEa <- array(0,c(d,2,3))
  for (md in 1:3) {
    if(md==3){
      res_CEASEa <- logistic_DANE(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                  theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                                  learning_rate=0.01, maxiter=500, tol=1e-3)
      betahatCEASEa[,md,]=res_CEASEa$coefs
      ci1_betahatCEASEa[,1,] <- res_CEASEa$ci1_l
      ci1_betahatCEASEa[,2,] <- res_CEASEa$ci1_u
      ci2_betahatCEASEa[,1,] <- res_CEASEa$ci2_l
      ci2_betahatCEASEa[,2,] <- res_CEASEa$ci2_u
    }
    if(md==2){
      betahatCEASEa[,md,]=logistic_DANE(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                        theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "lasso",
                                        trunconst = 1e-6,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
    if(md==1){
      betahatCEASEa[,md,]=logistic_DANE(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                        theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "scad",
                                        trunconst = 0.01,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
  }
  sfCat(paste0("Replicate ",g,": CEASEa completes"), sep="\n")
  ###############################
  #### Method CEASE(0)
  ###############################
  betahatCEASE0 <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  ci1_betahatCEASE0 <- array(0,c(d,2,3))
  ci2_betahatCEASE0 <- array(0,c(d,2,3))
  for (md in 1:3) {
    if(md==3){
      res_CEASE0 <- logistic_DANE(XX,YY,t=3,alpha = 0,
                                  theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                                  learning_rate=0.01, maxiter=500, tol=1e-3)
      betahatCEASE0[,md,]=res_CEASE0$coefs
      ci1_betahatCEASE0[,1,] <- res_CEASE0$ci1_l
      ci1_betahatCEASE0[,2,] <- res_CEASE0$ci1_u
      ci2_betahatCEASE0[,1,] <- res_CEASE0$ci2_l
      ci2_betahatCEASE0[,2,] <- res_CEASE0$ci2_u
    }
    if(md==2){
      betahatCEASE0[,md,]=logistic_DANE(XX,YY,t=3,alpha = 0,
                                        theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "lasso",
                                        trunconst = 1e-6,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
    if(md==1){
      betahatCEASE0[,md,]=logistic_DANE(XX,YY,t=3,alpha = 0,
                                        theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "scad",
                                        trunconst = 0.01,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
  }
  sfCat(paste0("Replicate ",g,": CEASE0 completes"), sep="\n")
  ###############################
  #### Method CEASE-single(a)
  ###############################
  betahatCEASEsa <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  ci1_betahatCEASEsa <- array(0,c(d,2,3))
  ci2_betahatCEASEsa <- array(0,c(d,2,3))
  for (md in 1:3) {
    if(md==3){
      res_CEASEsa <- logistic_CEASE1(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                     theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                                     trunconst = 0.01,learning_rate=0.01, maxiter=500, tol=1e-3)
      betahatCEASEsa[,md,]=res_CEASEsa$coefs
      ci1_betahatCEASEsa[,1,] <- res_CEASEsa$ci1_l
      ci1_betahatCEASEsa[,2,] <- res_CEASEsa$ci1_u
      ci2_betahatCEASEsa[,1,] <- res_CEASEsa$ci2_l
      ci2_betahatCEASEsa[,2,] <- res_CEASEsa$ci2_u
    }
    if(md==2){
      betahatCEASEsa[,md,]=logistic_CEASE1(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                           theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "lasso",
                                           trunconst = 1e-6,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
    if(md==1){
      betahatCEASEsa[,md,]=logistic_CEASE1(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                           theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "scad",
                                           trunconst = 0.01,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
  }
  sfCat(paste0("Replicate ",g,": CEASEsa completes"), sep="\n")
  ###############################
  #### Method CEASE-single(0) (CSL)
  ###############################
  betahatCSL <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  ci1_betahatCSL <- array(0,c(d,2,3))
  ci2_betahatCSL <- array(0,c(d,2,3))
  for (md in 1:3) {
    if(md==3){
      res_CSL <- logistic_CSL1(XX,YY,t=3, theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                               trunconst = 0.01,learning_rate=0.01, maxiter=500, tol=1e-3)
      betahatCSL[,md,]=res_CSL$coefs
      ci1_betahatCSL[,1,] <- res_CSL$ci1_l
      ci1_betahatCSL[,2,] <- res_CSL$ci1_u
      ci2_betahatCSL[,1,] <- res_CSL$ci2_l
      ci2_betahatCSL[,2,] <- res_CSL$ci2_u
    }
    if(md==2){
      betahatCSL[,md,]=logistic_CSL1(XX,YY,t=3, theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "lasso",
                                     trunconst = 1e-6,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
    if(md==1){
      betahatCSL[,md,]=logistic_CSL1(XX,YY,t=3, theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "scad",
                                     trunconst = 0.01,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
  }
  sfCat(paste0("Replicate ",g,": CSL completes"), sep="\n")
  
  #### integrate all betahat into a large array
  md_ci_name <- c(paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP"),"b1y_schurO1","type1",sep="_"),
                  paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP"),"b2","type1",sep="_"),
                  "Full","Meta_f","Meta_r",
                  paste(c(paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
                          paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_")),"s1",sep="_"),
                  paste(c(paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
                          paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_")),"s2",sep="_"))
  
  betahatci_all <- array(dim = c(d,md_ci,md_basic,2),dimnames = list(NULL,md_ci_name,c("scad","lasso","glm"),c("l","u")))
  
  for (md in 1:3) {
    # fill betahatcil_all/betahatciu_all
    for (lu in 1:2) {
      if(md==1){
        betahatci_all[,1:(3+length(K_fixed)),md,lu] <- ci_betahatOAPR_scad.r[,,lu,1] # OAPR_b1y_schurO1_type1
        betahatci_all[,(3+length(K_fixed))+1,md,lu] <- ci_betahatOAP_scad.r[,lu,1] # OAP_b1y_schurO1_type1
        betahatci_all[,(3+length(K_fixed))+1+1:(3+length(K_fixed)),md,lu] <- ci_betahatOAPR_scad.r[,,lu,2] # OAPR_b2_type1
        betahatci_all[,((3+length(K_fixed))+1)*2,md,lu] <- ci_betahatOAP_scad.r[,lu,2] # OAP_b2_type1
        betahatci_all[,((3+length(K_fixed))+1)*2+1,md,lu] <- ci_betahatFull_scad.r[,lu] # Full
      }
      if(md==3){
        betahatci_all[,1:(3+length(K_fixed)),md,lu] <- ci_betahatOAPR_glm[,,lu,1] # OAPR_b1y_schurO1_type1
        betahatci_all[,(3+length(K_fixed))+1,md,lu] <- ci_betahatOAP_glm[,lu,1] # OAP_b1y_schurO1_type1
        betahatci_all[,(3+length(K_fixed))+1+1:(3+length(K_fixed)),md,lu] <- ci_betahatOAPR_glm[,,lu,2] # OAPR_b2_type1
        betahatci_all[,((3+length(K_fixed))+1)*2,md,lu] <- ci_betahatOAP_glm[,lu,2] # OAP_b2_type1
        betahatci_all[,((3+length(K_fixed))+1)*2+1,md,lu] <- ci_betahatFull_glm[,lu] # Full
        betahatci_all[,((3+length(K_fixed))+1)*2+2,md,lu] <- ci_betahatMeta[,lu,1] # Meta_f
        betahatci_all[,((3+length(K_fixed))+1)*2+3,md,lu] <- ci_betahatMeta[,lu,2] # Meta_r
        betahatci_all[,((3+length(K_fixed))+1)*2+4:6,md,lu] <- ci1_betahatCEASEa[,lu,] # CEASEa_s1
        betahatci_all[,((3+length(K_fixed))+1)*2+7:9,md,lu] <- ci1_betahatCEASE0[,lu,] # CEASE0_s1
        betahatci_all[,((3+length(K_fixed))+1)*2+10:12,md,lu] <- ci1_betahatCEASEsa[,lu,] # CEASEsa_s1
        betahatci_all[,((3+length(K_fixed))+1)*2+13:15,md,lu] <- ci1_betahatCSL[,lu,] # CSL_s1
        betahatci_all[,((3+length(K_fixed))+1)*2+16:18,md,lu] <- ci2_betahatCEASEa[,lu,] # CEASEa_s2
        betahatci_all[,((3+length(K_fixed))+1)*2+19:21,md,lu] <- ci2_betahatCEASE0[,lu,] # CEASE0_s2
        betahatci_all[,((3+length(K_fixed))+1)*2+22:24,md,lu] <- ci2_betahatCEASEsa[,lu,] # CEASEsa_s2
        betahatci_all[,((3+length(K_fixed))+1)*2+25:27,md,lu] <- ci2_betahatCSL[,lu,] # CSL_s2
      }
    }
  }
  
  ###############################
  #### Coverage rate of confidence interval &
  #### Length of confidence interval
  ###############################
  cierror <- apply(betahatci_all,c(2,3),function(x){x[,1]<=theta_star&x[,2]>=theta_star})
  cilen <- apply(betahatci_all,c(2,3),function(x){x[,2]-x[,1]})
  
  return(list(cierror=cierror,cilen=cilen))
}

###############################
#### Load library in parallel way (using snowfall)
###############################
sfInit(parallel = T,cpus = cpunum,slaveOutfile = "sim_fan_3_rare.txt")
sfLibrary(snowfall)
sfLibrary(glmnet)
sfLibrary(ncvreg)
sfLibrary(ncpen)
sfLibrary(caret)
sfLibrary(ClusterR)
sfLibrary(CVXR)
sfLibrary(osqp)
sfSource("coef_ncvreg.R")
sfSource("lam_names.R")
sfSource("predict_ncvreg.R")
sfSource("OARFISH_solver.R")
sfSource("ci_OARFISH.R")
sfSource("logistic_solvers_forR.R")
sfSource("meta.R")
sfExport('md_basic')
sfExport('md_ci')
sfExport('N')
sfExport('ns')
sfExport('by')
sfExport('G')
sfExport('Kfold')
sfExport('corrx')
#sfExport('rare')
sfExport('df.t')
sfExport('lambda.e')
sfExport('sig.mis')

###############################
#### Compute the statistics in situations 
#### with different d and n1
###############################
#g=dind=1;nsi=2;ra=8
for (dind in 1:length(ds)) {
  d <- ds[dind]
  #generate the coefficients (sparse)
  #theta_star <- c(rep(1,10),rep(0,d-10))
  
  sfExport('d')
  #sfExport('theta_star')
  
  for (nsi in 1:length(ns)) {
    n1 <- ns[nsi]
    M <- round(N/n1)
    nn<-seq(n1, by=by[nsi],length.out=M) - by[nsi]*(M-1)/2
    #distribute sample size for every site
    
    #betahat: the coefficients length of betahat for two kinds of estimations - 
    #SCAD and lasso and total M sites.
    betahat<-array(0,c(d,3,M))
    BetaHatcv<-array(0,c(d,3,Kfold,M))
    #dimension; methods:scad, lasso and glm; cv folds; sites
    MUtildecv <- list()#record the coefficient estimation
    length(MUtildecv) <- 3*M
    dim(MUtildecv) <- c(3,M) 
    for (md in 1:3) {
      for (j in 1:M) {
        MUtildecv[[md,j]] <- vector(length = nn[j])
      }
    }
    
    #prediction loss for listed methods
    cierror <- array(dim=c(d,md_ci,md_basic),dimnames=list(NULL,md_ci_name,c("scad","lasso","glm")))
    cilen <- array(dim=c(d,md_ci,md_basic),dimnames=list(NULL,md_ci_name,c("scad","lasso","glm")))
    
    sfExport('M')
    sfExport('nn')
    sfExport('n1')
    sfExport('betahat')
    sfExport('BetaHatcv')
    sfExport('MUtildecv')
    sfExport('cierror')
    sfExport('cilen')

    for (ra in 1:length(rares)) {
      rare <- rares[ra]
      sfExport('rare')
      #Store1 <- sfSapply(1:cpunum,simfunc)#warm start for 'proc.time'
      ptm <- proc.time()
      Store <- sfSapply(1:G,simfunc)
      timecost.whole <- proc.time()-ptm
      timecost.whole <- round(timecost.whole,3)
      
      alphaM[,nsi,,,dind,ra]<-(Reduce("+",Store[1,])/length(Store[1,]))[1:21,,]
      lengthciM[,nsi,,,dind,ra]<-(Reduce("+",Store[2,])/length(Store[2,]))[1:21,,]
      
      timecostM[nsi,,dind,ra]<-timecost.whole[1:3]
      
      print(paste0("sitesamplesize=",n1," sitenumbers=",M," d=",d))
      print(paste0(names(timecost.whole[1]),"=",timecost.whole[1],",",names(timecost.whole[2]),"=",timecost.whole[2],",",names(timecost.whole[3]),"=",timecost.whole[3]))
      #the best way to avoid crush: save your data!
      save.image(file = paste("sim_fan_3_rare","_G_",G,"_cpu_",cpunum,".Rdata",sep=""))
    }
  }
}
sfStop()

