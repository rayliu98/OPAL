# This is the sensitivity analysis for OAPR
# author: Ray Liu
# date: 2026/02/28

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
#-------------------------------------------------
#Base learner: only logistic model
#1. OAPR* (Optimal Aggregation Prediction with the weight Restriction)
#2. OAP* (Optimal Aggregation Prediction)
#3. Full* (gold standard)
#4. Naive averaging on betahat
#5. Single site
#-------------------------------------------------
#Base learners: xgboost, MLP and logistic model
#6. OAPR (Optimal Aggregation Prediction with the weight Restriction)
#7. OAP (Optimal Aggregation Prediction)
#8. Full (gold standard)
#9. Naive averaging on probability estimate

###############################
#### install/load library
###############################
# change to your working directory
#setwd("~/DCGLM/new_submission")
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
Sys.setenv(NUMEXPR_NUM_THREADS = "1")
library(parallel)
library(snowfall)
library(ncvreg)
library(glmnet)
library(caret)
library(ncpen)
#library(neuralnet)
library(xgboost)
#library(iml)
#library(shapr)
#library(fastshap)
#library(keras3)
library(keras) 
# if the error 'py_call_impl(callable, call_args$unnamed, call_args$named): 
# ValueError: Only input tensors may be passed as positional arguments.' happens,
# try 'keras3' instead of 'keras'.
library(tensorflow)
library(reticulate)
#library(NeuralNetTools)
#library(e1071)
#library(randomForest)
library(shapviz)
library(ClusterR)
library(cluster)
library(CVXR)
library(osqp)
library(pROC)
source("coef_ncvreg.R")
source("lam_names.R")
source("predict_ncvreg.R")
source("OARFISH_solver.R")
source("logistic_solvers_forR.R")
#source("meta.R")

###############################
#### self-defined functions
###############################
predictfun.glm <- function(beta1, newdata) {
  link(as.numeric(beta1 %*% t(as.matrix(newdata))))  
}

###############################
#### DGP and settings
###############################

# DGP 1 (modified according to JASA paper "Communication-Efficient Accurate Statistical
# Estimation" by Jianqing Fan et al., and "Least Squares Model Averaging" by Bruce Hansen)

# Notations:
# The dimensionality d, true parameter \theta^*, design matrix X, response Y
# local sample size n and site number M.

set.seed(202402)
md_basic=3 
K_fixed_num=3
d0s <- c(51)
#ns <- c(100,1000)
ns <- c(100,200,400,1000)
Ms <- c(20,100)
N_test=2000
ds <- c(0.2,0.5,0.8)
decays<-c(1) #the decay rate of coefficient, where -.5 means uniform setting in the JASA paper
sigs <- c(3)
md_ml=5 #when base models at each site are different, how many base machine learning methods do we use
by=0 #difference parameter, used for 'seq' function, seq(from,by,length.out), 
     #to set local sample size and allow it different across different site
delta=c(0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28) #perturbation on betahat/shapley when clustering
G<-100 #the setting for replicates
#G<-5 #the setting for replicates
Kfold <- 5 #when predicting the probability of in-site observations, how many folds we set?
#cpunum <- c(20,10,8,5) #the setting for the amount of CPUs requested for the cluster
#cpunum <- c(34,8)
cpunum=matrix(c(25,20,20,34,10,8,10,15),4,2)
#cpunum <- c(5,5)

# specific model parameter
MLP_hidden1=5
MLP_hidden2=c(3,2)
xgb_nrounds1=2
xgb_maxdepth1=6
xgb_nrounds2=8
xgb_maxdepth2=3

# Recording lists for the mse, running time and weights, where 'timeM' records  
# the time for specific term in 'timename' and timecostM records the total running
# time for G replications.
# weights are computed using osqp solver, which produces more dense solution.
# Each element of the 'wM' array store a list of the weights distributed to M sites
# (G=100 replicates) with local sample size n_m and parameter dimension d.
md_b_name <- c("elbow","distortion",paste("fixed",1:K_fixed_num,sep="_"),"OAP",
               "EW","Partial","Full")
md_p_name <- c(md_b_name,"MLPFull","MLP2Full","xgbFull","xgb2Full",
               "MLPp","MLP2p","xgbp","xgb2p",
               paste0(c("mix"),c("elbow","distortion",paste("fixed",1:K_fixed_num,sep="_"))),
               paste0(c("mix"),c("OAP","EW")),
               paste(c("elbow","distortion",paste("fixed",1:K_fixed_num,sep="_"),"OAP",'EW'),'p',sep="_"))
md_z_name <- md_b_name
M_name <- paste("M",Ms,sep="_")
n_name <- paste("n",ns,sep="_")
md_basic_name <- c("scad","lasso","glm")
d0_name <- paste("d0",d0s,sep="_")
d_name <- paste("d",ds,sep="_")
decay_name <- c(paste("decay",decays[1],sep="_"))
sig_name <- paste("sig",sigs,sep="_")
delta_name=paste("delta",delta,sep="_")


md_b=length(md_b_name)
md_p=length(md_p_name)
md_a=length(md_p_name)
md_z=length(md_z_name)

# Ms: site number; md_b/md_p/md_a/md_z: comparison method; md_basic: base learner;
# d0s: the dimensionality of true parameter; ds: the number of variables used; alphaM: coefficient pattern
msebM <-array(dim=c(length(Ms),length(ns),md_b,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #MSE for coefficient estimation
msepM <-array(dim=c(length(Ms),length(ns),md_p,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #MSE for probability estimation
accM <-array(dim=c(length(Ms),length(ns),md_a,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #accuracy index
aucM <-array(dim=c(length(Ms),length(ns),md_a,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #auc index
msebM.sd <-array(dim=c(length(Ms),length(ns),md_b,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #MSE for coefficient estimation
msepM.sd <-array(dim=c(length(Ms),length(ns),md_p,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #MSE for probability estimation
accM.sd <-array(dim=c(length(Ms),length(ns),md_a,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #accuracy index
aucM.sd <-array(dim=c(length(Ms),length(ns),md_a,md_basic,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #auc index
zerorightM <-array(dim=c(length(Ms),length(ns),md_z,md_basic-1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #correct zero estimated by penalized methods
zerowrongM <-array(dim=c(length(Ms),length(ns),md_z,md_basic-1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) #incorrect zero estimated by penalized methods
timecostM <-array(0,c(length(Ms),length(ns),3,length(delta),length(d0s),length(ds),length(decays),length(sigs)))#'3' means the output of 'proc.time'
timecostMd <-array(0,c(length(Ms),length(ns),md_p,md_basic+1,length(d0s),length(ds),length(decays),length(sigs)))
K.tableM <-array(dim=c(length(Ms),length(ns),2,length(delta),md_basic+1,length(d0s),length(ds),length(decays),length(sigs)))
K.tableM.sd <-array(dim=c(length(Ms),length(ns),2,length(delta),md_basic+1,length(d0s),length(ds),length(decays),length(sigs))) 

wl1M <-array(dim=c(length(Ms),length(ns),2+K_fixed_num,md_basic+1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) 
wl2M <-array(dim=c(length(Ms),length(ns),2+K_fixed_num,md_basic+1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) 
wlmaxM <-array(dim=c(length(Ms),length(ns),2+K_fixed_num,md_basic+1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) 
wl1M.sd <-array(dim=c(length(Ms),length(ns),2+K_fixed_num,md_basic+1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) 
wl2M.sd <-array(dim=c(length(Ms),length(ns),2+K_fixed_num,md_basic+1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) 
wlmaxM.sd <-array(dim=c(length(Ms),length(ns),2+K_fixed_num,md_basic+1,length(delta),length(d0s),length(ds),length(decays),length(sigs))) 

dimnames(msebM) <- dimnames(msebM.sd) <- list(M_name,n_name,md_b_name,md_basic_name,delta_name,d0_name,d_name,decay_name,sig_name)
dimnames(msepM) <- dimnames(accM) <- dimnames(aucM) <- dimnames(msepM.sd) <- dimnames(accM.sd) <- dimnames(aucM.sd) <- 
  list(M_name,n_name,md_p_name,md_basic_name,delta_name,d0_name,d_name,decay_name,sig_name)
dimnames(zerorightM) <- dimnames(zerowrongM) <- list(M_name,n_name,md_z_name,md_basic_name[-3],delta_name,d0_name,d_name,decay_name,sig_name)
dimnames(timecostMd) <- list(M_name,n_name,md_p_name,c(md_basic_name,'mix'),d0_name,d_name,decay_name,sig_name)
dimnames(K.tableM)=dimnames(K.tableM.sd)=list(M_name,n_name,c('elbow','distortion'),delta_name,c('scad','lasso','glm','mix'),d0_name,d_name,decay_name,sig_name)
dimnames(wl1M)=dimnames(wl2M)=dimnames(wlmaxM)=dimnames(wl1M.sd)=dimnames(wl2M.sd)=dimnames(wlmaxM.sd)=
  list(M_name,n_name,md_b_name[1:(2+K_fixed_num)],c('scad','lasso','glm','mix'),delta_name,d0_name,d_name,decay_name,sig_name)

wM.OAPR <- list()
length(wM.OAPR) <- length(Ms)*length(ns)*length(d0s)*length(ds)*length(decays)*length(sigs)
dim(wM.OAPR) <- c(length(Ms),length(ns),length(d0s),length(ds),length(decays),length(sigs)) 
woM.OAPR <- list()
length(woM.OAPR) <- length(Ms)*length(ns)*length(d0s)*length(ds)*length(decays)*length(sigs)
dim(woM.OAPR) <- c(length(Ms),length(ns),length(d0s),length(ds),length(decays),length(sigs)) 

simfunc <- function(g){
  shap <- reticulate::import("shap")
  #ptm <- proc.time()
  sfCat(paste0("Replicate ",g,": starts"), sep="\n")
  set.seed(2024+g)
  if(decay==-0.5){
    #generate the coefficients (uniform on sphere)
    theta_star <- rnorm(d0)
  } else {
    #generate the coefficients (decay series)
    theta_star <- sqrt(2*decay)*c((1:d0)^(-decay-0.5))
  }
  theta_star <- sig*theta_star/norm(as.matrix(theta_star),type = "2")
  
  XX<-list() #save the design matrix at each site
  YY<-list()
  MU<-list()
  Obs_p <- list()
  Lambda_series.scad <- vector(length = M)
  modelsMLP <- list()
  modelsxg2 <- list()
  modelsxg <- list()
  modelsMLP2 <- list()
  MUtildecvMLP <- list()
  MUtildecvMLP2 <- list()
  MUtildecvxg <- list()
  MUtildecvxg2 <- list()
  shapleyrecord_mix <- array(0,c(M,d-1))
  length(MUtildecvMLP)=length(MUtildecvMLP2)=length(MUtildecvxg)=length(MUtildecvxg2)=M
  for (j in 1:M) {
    MUtildecvMLP[[j]] <- vector(length = nn[j])
    MUtildecvxg2[[j]] <- vector(length = nn[j])
    MUtildecvxg[[j]] <- vector(length = nn[j])
    MUtildecvMLP2[[j]] <- vector(length = nn[j])
  }
  # randomly dsitribute methods to local site
  shuffled_ml <- 1:M
  ml_partition <- split(shuffled_ml,cut(1:M, md_ml, labels = FALSE))
  # epoch used in the training process of MLP at the mth site
  epochm <- rep(50,M)
  # Test data
  Yst<-c();Xst<-c();probst<-c()
  for (j in 1:M) {
    nm<-N_test/M#the sample size of jth site
    SigmaX <- diag(sqrt(c(10,5,2,rep(1,d0-4))),d0-1)
    X0m<-cbind(rep(1,nm),matrix(rnorm(nm*(d0-1),0,1),nm,d0-1)%*%SigmaX) #generate the covariates at the jth site
    Xm<-X0m[,1:d] #take the first d covariates as the ones on hand
    etam <- X0m %*% theta_star+tan(X0m[,2])+X0m[,3]*X0m[,4]*X0m[,5]+X0m[,6]*X0m[,7]*X0m[,8]*X0m[,9] #the true linear explanatory part at the jth site
    probm <- 1-exp(-etam*(etam>0))
    Ym <- rbinom(nm,1,probm) #the response Y_m
    Xst<-rbind(Xst,Xm)#pool the design matrices from all sites altogether
    probst<-rbind(probst,probm)
    Yst<-c(Yst,Ym)
  }
  dimnames(Xst) <- list(NULL,c("intercept",1:(ncol(Xst)-1)))
  
  #DGP for M sites
  time.fitting=matrix(NA,nrow=M,ncol=4)#scad,lasso,glm,mix
  time.fitting.cv=matrix(NA,nrow=M,ncol=4)#scad,lasso,glm,mix
  for (j in 1:M){
    nm<-nn[j]#the sample size of jth site
    SigmaX <- diag(sqrt(c(10,5,2,rep(1,d0-4))),d0-1)
    X0m<-cbind(rep(1,nm),matrix(rnorm(nm*(d0-1),0,1),nm,d0-1)%*%SigmaX) #generate the covariates at the jth site
    Xm<-X0m[,1:d] #take the first d covariates as the ones on hand
    etam <- X0m %*% theta_star+tan(X0m[,2])+X0m[,3]*X0m[,4]*X0m[,5]+X0m[,6]*X0m[,7]*X0m[,8]*X0m[,9] #the true linear explanatory part at the jth site
    probm <- 1-exp(-etam*(etam>0))
    Ym <- rbinom(nm,1,probm) #the response Y_m
    obs_partition <- createFolds(Ym,k=Kfold)
    ###############################
    #### saving
    XX[[j]]<-   Xm#store the design matrix on hand at jth site into XX list
    YY[[j]]<-   Ym#store the response vector at jth site into YY list
    MU[[j]]<-   probm#store the mean vector at jth site into MU list
    Obs_p[[j]]<- obs_partition
    
    ###############################
    #### GLM
    ###############################
    ###############################
    #### SCAD
    ptm1=proc.time()
    scad_mod <- ncpen(y.vec=Ym, x.mat=Xm[,-1], family="binomial", penalty="scad",
                      x.standardize=T, intercept=T)
    lambda_scad.fit <- cv.ncpen(y.vec=Ym, x.mat=Xm[,-1], family="binomial", penalty="scad",
                                x.standardize=T, intercept=T)
    lambda_scad <- coef.cv.ncpen(lambda_scad.fit,type="rmse")$lambda
    
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
    time.fitting[j,1]=(proc.time()-ptm1)[3]
    
    ptm2=proc.time()
    lambda_series.scad <- scad_mod$lambda
    for (k in 1:Kfold) {
      scad_mod_cv <- ncpen(x.mat=Xm[-obs_partition[[k]],-1],y.vec=Ym[-obs_partition[[k]]],
                           family="binomial", penalty="scad",x.standardize=T, intercept=T,
                           lambda = lambda_series.scad)
      if(lambda_scad>scad_mod_cv$lambda[1]){
        betahatm_scad_cv <- coef.ncvreg(scad_mod_cv, lambda=scad_mod_cv$lambda[1])
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatm_scad_cv)
        warning("lambda_scad>scad_mod_cv$lambda[1]")
      } else if(lambda_scad<scad_mod_cv$lambda[length(scad_mod_cv$lambda)]) {
        betahatm_scad_cv <- coef.ncvreg(scad_mod_cv, lambda=scad_mod_cv$lambda[length(scad_mod_cv$lambda)])
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatm_scad_cv)
        warning("lambda_scad<scad_mod_cv$lambda[length(scad_mod_cv$lambda)]")
      } else {
        betahatm_scad_cv <- coef.ncvreg(scad_mod_cv, lambda=lambda_scad)
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatm_scad_cv)
      }
      if(sum(is.nan(MUtildecv[[1,j]][obs_partition[[k]]]))>0){
        # Modify mu when the situation that exp(testeta)/(1+exp(testeta)) produces 'nan' happens
        MUtildecv[[1,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[1,j]][obs_partition[[k]]])==1)] <- 1
      }
    }
    time.fitting.cv[j,1]=(proc.time()-ptm2)[3]
    
    ###############################
    #### lasso
    ptm1=proc.time()
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
    time.fitting[j,2]=(proc.time()-ptm1)[3]
    
    ptm2=proc.time()
    lambda_series.lasso <- lasso_mod$lambda
    for (k in 1:Kfold) {
      lasso_mod_cv <- glmnet(Xm[-obs_partition[[k]],-1],Ym[-obs_partition[[k]]],
                             family="binomial", intercept=T, lambda = lambda_series.lasso)
      MUtildecv[[2,j]][obs_partition[[k]]] <- predict(lasso_mod_cv,Xm[obs_partition[[k]],-1],
                                                      type="response",s=lambda_lasso)
      if(sum(is.nan(MUtildecv[[2,j]][obs_partition[[k]]]))>0){
        MUtildecv[[2,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[2,j]][obs_partition[[k]]])==1)] <- 1
      }
    }
    time.fitting.cv[j,2]=(proc.time()-ptm2)[3]
    
    ###############################
    #### glm
    ptm1=proc.time()
    betahat[,3,j]<-minimize_logistic_loss(Xm,Ym,learning_rate = 0.1,maxiter = 10000)$coefs
    time.fitting[j,3]=(proc.time()-ptm1)[3]
    
    ptm2=proc.time()
    for (k in 1:Kfold) {
      betahatglmcv<-minimize_logistic_loss(Xm[-obs_partition[[k]],],Ym[-obs_partition[[k]]],learning_rate = 0.1,maxiter = 10000)$coefs
      MUtildecv[[3,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatglmcv)
      if(sum(is.nan(MUtildecv[[3,j]][obs_partition[[k]]]))>0){
        MUtildecv[[3,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[3,j]][obs_partition[[k]]])==1)] <- 1
      }
    }
    time.fitting.cv[j,3]=(proc.time()-ptm2)[3]
    
    ###############################
    #### lasso for mix
    ###############################
    if(j %in% ml_partition[[1]]){
      time.fitting[j,4]=time.fitting[j,2]
      time.fitting.cv[j,4]=time.fitting.cv[j,2]
    }
    ###############################
    #### MLP (para=d*MLP_hidden+(MLP_hidden+1)*2=26*5+(5+1)*2=142)
    ###############################
    if(j %in% ml_partition[[2]]){
      Neuralm1 <- keras_model_sequential() %>%
        layer_dense(units = MLP_hidden1, activation = 'relu', 
                    input_shape = c(ncol(Xm) - 1)) %>%
        layer_dense(units = 1, activation = 'sigmoid')
      Neuralm1 %>% compile(loss = 'binary_crossentropy', 
                           optimizer = 'adam', 
                           metrics = c('accuracy'))
      ptm1=proc.time() 
      # 'ptm1' should been placed before 'Neuralm1', but
      # starting Keras takes time. Not affected after the first startup.
      Neuralm1 %>% fit(Xm[,-1], Ym, epochs = epochm[j], batch_size = 10,verbose=0)
      modelsMLP[[j]] <- Neuralm1
      #Neuralm1$count_params()
      time.fitting[j,4]=(proc.time()-ptm1)[3]
      
      ptm2=proc.time()
      for (k in 1:Kfold) {
        Neuralmcv <- keras_model_sequential() %>%
          layer_dense(units = MLP_hidden1, activation = 'relu', 
                      input_shape = c(ncol(Xm) - 1)) %>%
          layer_dense(units = 1, activation = 'sigmoid')
        Neuralmcv %>% compile(loss = 'binary_crossentropy', 
                              optimizer = 'adam', 
                              metrics = c('accuracy'))
        Neuralmcv %>% fit(Xm[-obs_partition[[k]],-1], Ym[-obs_partition[[k]]], epochs = epochm[j], batch_size = 10,verbose=0)
        MUtildecvMLP[[j]][obs_partition[[k]]] <- Neuralmcv %>% predict(Xm[obs_partition[[k]],-1])
      }  
      time.mlpcv <- (proc.time()-ptm2)[3]
      time.fitting.cv[j,4]=time.mlpcv
    }
    ###############################
    #### MLP2 (para=d*MLP_hidden1+(MLP_hidden1+1)*MLP_hidden2+
    #### (MLP_hidden2+1)*2=26*3+(3+1)*3+(3+1)*2=98)
    ###############################
    if(j %in% ml_partition[[3]]){
      Neuralm2 <- keras_model_sequential() %>%
        layer_dense(units = MLP_hidden2[1], activation = 'relu', 
                    input_shape = c(ncol(Xm) - 1)) %>%
        layer_dense(units = MLP_hidden2[2], activation = 'relu') %>%
        layer_dense(units = 1, activation = 'sigmoid')
      Neuralm2 %>% compile(loss = 'binary_crossentropy', 
                           optimizer = 'adam', 
                           metrics = c('accuracy'))
      ptm1=proc.time()
      Neuralm2 %>% fit(Xm[,-1], Ym, epochs = epochm[j], batch_size = 10,verbose=0)
      modelsMLP2[[j]] <- Neuralm2
      #Neuralm2$count_params()
      time.fitting[j,4]=(proc.time()-ptm1)[3]
      
      ptm2=proc.time()
      for (k in 1:Kfold) {
        Neuralmcv <- keras_model_sequential() %>%
          layer_dense(units = MLP_hidden2[1], activation = 'relu', 
                      input_shape = c(ncol(Xm) - 1)) %>%
          layer_dense(units = MLP_hidden2[2], activation = 'relu') %>%
          layer_dense(units = 1, activation = 'sigmoid')
        Neuralmcv %>% compile(loss = 'binary_crossentropy', 
                              optimizer = 'adam', 
                              metrics = c('accuracy'))
        Neuralmcv %>% fit(Xm[-obs_partition[[k]],-1], Ym[-obs_partition[[k]]], epochs = epochm[j], batch_size = 10,verbose=0)
        MUtildecvMLP2[[j]][obs_partition[[k]]] <- Neuralmcv %>% predict(Xm[obs_partition[[k]],-1])
      }  
      time.mlp2cv <- (proc.time()-ptm2)[3]
      time.fitting.cv[j,4]=time.mlp2cv
    }
    ###############################
    #### xgboost (para<=((64-1)*4+64)*2=632)
    ###############################
    if(j %in% ml_partition[[4]]){
      ptm1=proc.time()
      param <- list(max_depth = xgb_maxdepth1, eta = 1, nthread = 2, 
                    objective = "binary:logistic", eval_metric = "error")
      xgbm <- xgboost(data=as.matrix(Xm[,-1]),label = Ym,params = param,nrounds = xgb_nrounds1,verbose = 0)
      #xgb.dump(xgbm)
      modelsxg[[j]] <- xgbm
      time.fitting[j,4]=(proc.time()-ptm1)[3]
      
      ptm2=proc.time()
      for (k in 1:Kfold) {
        xgbmcv <- xgboost(data=as.matrix(Xm[-obs_partition[[k]],-1]),label = Ym[-obs_partition[[k]]],params = param,nrounds = xgb_nrounds1,verbose = 0)
        MUtildecvxg[[j]][obs_partition[[k]]] <- predict(xgbmcv,newdata=Xm[obs_partition[[k]],-1])
      }
      time.fitting.cv[j,4]=(proc.time()-ptm2)[3]
    }
    ###############################
    #### xgboost2 (para<=((8-1)*4+8)*8=288)
    ###############################
    if(j %in% ml_partition[[5]]){
      ptm1=proc.time()
      param2 <- list(max_depth =xgb_maxdepth2, eta = 1, nthread = 2, 
                     objective = "binary:logistic", eval_metric = "error")
      xgbm2 <- xgboost(data=as.matrix(Xm[,-1]),label = Ym,params = param2,nrounds = xgb_nrounds2,verbose = 0)
      #xgb.dump(xgbm2)
      modelsxg2[[j]] <- xgbm2
      time.fitting[j,4]=(proc.time()-ptm1)[3]
      
      ptm2=proc.time()
      for (k in 1:Kfold) {
        xgbmcv <- xgboost(data=as.matrix(Xm[-obs_partition[[k]],-1]),label = Ym[-obs_partition[[k]]],params = param2,nrounds = xgb_nrounds2,verbose = 0)
        MUtildecvxg2[[j]][obs_partition[[k]]] <- predict(xgbmcv,newdata=Xm[obs_partition[[k]],-1])
      }
      time.fitting.cv[j,4]=(proc.time()-ptm2)[3]
    }
  }
  sfCat(paste0("Replicate ",g,": model fitting at local sites completes"), sep="\n")
  
  ###############################
  #### Compute Shapley
  ###############################
  #### five methods are randomly and equally distributed to M sites
  modelsmix=MUtildecvmix=list()
  length(modelsmix)=length(MUtildecvmix)=M
  time.shapley=vector(length = M)
  for (ml in 1:md_ml) {
    for (ml2 in ml_partition[[ml]]) {
      ptm=proc.time()
      dimnames(XX[[ml2]]) <- list(NULL,c("intercept",1:(ncol(XX[[ml2]])-1)))
      if(ml==1){
        custom_predict <- function(newdata) {
          predictfun.glm(beta1 = betahat[,2,ml2], newdata = newdata)
        }
        explainer_glm <- shap$Explainer(custom_predict,XX[[ml2]])
        shap_values_glm=explainer_glm(XX[[ml2]][1:100,])
        shapleyrecord_mix[ml2,]=apply(shap_values_glm$values[,-1],2,mean)
        modelsmix[[ml2]]=betahat[,2,ml2];MUtildecvmix[[ml2]]=MUtildecv[[2,ml2]]
        }
      if(ml==2){
        explainer_keras <- shap$Explainer(modelsMLP[[ml2]],XX[[ml2]][,-1])
        #explainer_keras <- shap$KernelExplainer(modelsMLP[[ml2]],XX[[ml2]][,-1])
        shap_values_keras <- explainer_keras(XX[[ml2]][1:100,-1])
        shapleyrecord_mix[ml2,]=apply(shap_values_keras$values,2,mean)
        modelsmix[[ml2]]=modelsMLP[[ml2]];MUtildecvmix[[ml2]]=MUtildecvMLP[[ml2]]
        }
      if(ml==3){
        explainer_keras <- shap$Explainer(modelsMLP2[[ml2]],XX[[ml2]][,-1])
        #explainer_keras <- shap$KernelExplainer(modelsMLP[[ml2]],XX[[ml2]][,-1])
        shap_values_keras <- explainer_keras(XX[[ml2]][1:100,-1])
        shapleyrecord_mix[ml2,]=apply(shap_values_keras$values,2,mean)
        modelsmix[[ml2]]=modelsMLP2[[ml2]];MUtildecvmix[[ml2]]=MUtildecvMLP2[[ml2]]
        }
      if(ml==4){
        shapleyrecord_mix[ml2,]=apply(as.matrix(shapviz(modelsxg[[ml2]],XX[[ml2]][1:100,-1])$S),2,mean)
        modelsmix[[ml2]]=modelsxg[[ml2]];MUtildecvmix[[ml2]]=MUtildecvxg[[ml2]]
        }
      if(ml==5){
        shapleyrecord_mix[ml2,]=apply(as.matrix(shapviz(modelsxg2[[ml2]],XX[[ml2]][1:100,-1])$S),2,mean)
        modelsmix[[ml2]]=modelsxg2[[ml2]];MUtildecvmix[[ml2]]=MUtildecvxg2[[ml2]]
      }
      time.shapley[ml2]=(proc.time()-ptm)[3]
    }
  }
  rm(shap)
  sfCat(paste0("Replicate ",g,": shapley computation completes"), sep="\n")
  
  ###############################
  #### Clustering
  ###############################
  #### K-selection
  # Temporarily we consider three schemes of selecting K: (1) K=1:M (fixed values); 
  # (2) elbow method; (3) distortion method.
  # Before applying K-means, betahat is standardized.
  M_for_Kmeans <- 1:K_fixed_num
  K_fixed_sep=floor((M-2)/K_fixed_num)
  K_fixed = M_for_Kmeans*K_fixed_sep
  
  ###############################
  #### glm (cluster index: coefficient)
  ###############################
  time.k_selection=matrix(NA,2,4)#G_selector: elbow, distortion; predictors: scad, lasso, glm, mix
  time.scale=vector(length = 4)
  betahat_s <- array(0,c(M,d,3,length(delta)))#save the scaled betahat: scad, lasso, glm
  
  K.table <- array(0,c(2+length(K_fixed),3,length(delta)))
  for (md in 1:3) {
    for (del in 1:length(delta)) {
      ptm=proc.time()
      betahat_s_tmp <- scale(t(drop0(betahat[,md,]+
                                      t(t(matrix(rnorm(d*M,0,delta[del]),d,M))*
                                          apply(betahat[,md,], 2, norm,type='2'))/d)))
      betahat_s[,,md,del] <- betahat_s_tmp
      time.scale[md]=(proc.time()-ptm)[3]
      
      ptm1=proc.time()
      K.table[1,md,del] <- K_selection_new(betahat_s[,,md,del],"elbow",M-2,frac=0.5)
      time.k_selection[1,md]=(proc.time()-ptm1)[3]
      
      ptm2=proc.time()
      K.table[2,md,del] <- K_selection_new(betahat_s[,,md,del],"distortion",M-2,distortion_ratio=1)
      time.k_selection[2,md]=(proc.time()-ptm2)[3]
    }
  }
  for (kseries in 1:length(K_fixed)) {
    for (del in 1:length(delta)) {
      K.table[kseries+2,,del] <- K_fixed[kseries]
    }
  }
  dimnames(K.table) <- list(
    c("elbow","distortion",paste("fixed",K_fixed)),
    c("scad","lasso","glm"),
    paste0('delta_',delta))
  
  ###############################
  #### mix (cluster index: shapley)
  ###############################
  #add_noise_simple <- function(w, sd, normalizing=FALSE) {
  #  w2 <- w + rnorm(length(w), 0, sd)
  #  if(normalizing==FALSE){
  #    return(w2)
  #  } else {
  #    w2[w2 < 0] <- 0   
  #    return(t(apply(w2, 1, function(x){x/sum(x)})))
  #  }
  #}
  
  #shapleyrecord_mix1 <- t(apply(shapleyrecord_mix,1,function(x){abs(x)/sum(abs(x))}))
  #shapleymix_s <- array(0,c(M,d-1,length(delta)))
  #for (del in 1:length(delta)) {
  #  ptm=proc.time()
  #  shapleymix_s[,,del]=add_noise_simple(shapleyrecord_mix1,delta[del])
  #  time.scale[4]=(proc.time()-ptm)[3]
  #}
  
  shapleymix_s <- array(0,c(M,d-1,length(delta)))
  
  for (del in 1:length(delta)) {
    ptm=proc.time()
    shapleymix_s[,,del]=scale(drop0(shapleyrecord_mix+
                                      matrix(rnorm((d-1)*M,0,delta[del]),M,d-1)*
                                      apply(shapleyrecord_mix, 1, norm,type='2')/(d-1)))
    time.scale[4]=(proc.time()-ptm)[3]
  }
  
  K.table.ml <- array(0,c(2+length(K_fixed),1,length(delta)),list(
    c("elbow","distortion",paste("fixed",K_fixed)),
    c("mix"),paste0('delta_',delta)))
  
  for (md in 1:1) {
    for (del in 1:length(delta)) {
      ptm1=proc.time()
      K.table.ml[1,md,del] <- K_selection_new(shapleymix_s[,,del],"elbow",M-2,frac=0.5)
      time.k_selection[1,4]=(proc.time()-ptm1)[3]
      ptm2=proc.time()
      K.table.ml[2,md,del] <- K_selection_new(shapleymix_s[,,del],"distortion",M-2,distortion_ratio=1)
      time.k_selection[2,4]=(proc.time()-ptm2)[3]
    }
  }
  
  #sum(is.na(K.table.ml))
  for (kseries in 1:length(K_fixed)) {
    for (del in 1:length(delta)) {
      K.table.ml[kseries+2,,del] <- K_fixed[kseries]
    }
  }
  
  ###############################
  #### Implement K-means
  ###############################
  #### glm
  ###############################
  time.cluster=matrix(NA,2+length(K_fixed),4)# predictors: scad, lasso, glm, mix
  kmeans_clusters <- array(0,c(M,2+length(K_fixed),3,length(delta)),
                           dimnames = list(NULL, c("elbow","distortion",paste("fixed",K_fixed)),
                                           c("scad","lasso","glm"),paste0('delta_',delta)))
  for (kseries in 1:(2+length(K_fixed))) {
    for (md in 1:3) {
      for (del in 1:length(delta)) {
        ptm=proc.time()
        if(K.table[kseries,md,del]==M){
          kmeans_clusters[,kseries,md,del] <- 1:M
        } else {
          non_zero_cols <- apply(betahat_s[,,md,del], 2, function(col) any(col != 0) & any(!is.na(col)))
          betahatMat_dropcols <- betahat_s[,,md,del][, non_zero_cols, drop = FALSE] # 'drop' coerce it to be a matrix
          
          kmeans_tmp <- KMeans_rcpp(betahatMat_dropcols, K.table[kseries,md,del])
          kmeans_clusters[,kseries,md,del] <- kmeans_tmp$clusters
        }
        time.cluster[kseries,md]=(proc.time()-ptm)[3]
      }
    }
  }
  ###############################
  #### mix
  ###############################
  kmeans_other_clusters <- array(0,c(M,2+length(K_fixed),1,length(delta)),
                                 dimnames = list(NULL,c("elbow","distortion",paste("fixed",K_fixed)),
                                                 c("mix"),paste0('delta_',delta)))
  for (kseries in 1:(2+length(K_fixed))) {
    for (md in 1:1) {
      for (del in 1:length(delta)) {
        ptm=proc.time()
        KK=K.table.ml[kseries,md,del]
        shaps=shapleymix_s[,,del]
        if(KK==M){
          kmeans_other_clusters[,kseries,md,del] <- 1:M
        } else {
          non_zero_cols <- apply(shaps, 2, function(col) any(col != 0) & any(!is.na(col)))
          shap_dropcols <- shaps[, non_zero_cols, drop = FALSE] # 'drop' coerce it to be a matrix
          
          kmeans_tmp <- KMeans_rcpp(shap_dropcols, KK)
          kmeans_other_clusters[,kseries,md,del] <- kmeans_tmp$clusters
        }
        time.cluster[kseries,4]=(proc.time()-ptm)[3]
      }
    }
  }
  sfCat(paste0("Replicate ",g,": Kmeans completes"), sep="\n")
  
  ###############################
  #### Method OAPR 
  ###############################
  ###############################
  #### glm
  ###############################
  betahatOAPR <- array(0,c(d,2+length(K_fixed),3,length(delta)),dimnames = list(NULL,
                                                                  c("elbow","distortion",paste("fixed",K_fixed)),
                                                                  c("scad","lasso","glm"),
                                                                  paste0('delta_',delta)))
  iter.center_opt <- array(0,c(2+length(K_fixed),3),dimnames = 
                             list(c("elbow","distortion",paste("fixed",K_fixed)),
                                  c("scad","lasso","glm")))
  # Record the estimated weights
  w.table <- array(0,c(M,2+length(K_fixed),3,length(delta)),dimnames = 
                     list(NULL,c("elbow","distortion",paste("fixed",K_fixed)),
                          c("scad","lasso","glm"),paste0('delta_',delta)))
  time.weight.OAPR=array(0,c(2+length(K_fixed),4),dimnames = 
                           list(c("elbow","distortion",paste("fixed",K_fixed)),
                                c("scad","lasso","glm","mix")))
  for (md in 1:3) {
    for (kseries in 1:(2+length(K_fixed))) {
      for (del in 1:length(delta)) {
        res <- betahat_OARFISH(XX,YY,betahat[,md,],MUtildecv[md,],kmeans_clusters[,kseries,md,del])
        coefs.OAPR <- res$coefs
        #coefs.OAPR[abs(coefs.OAPR)<1e-8] <- 0
        betahatOAPR[,kseries,md,del]=coefs.OAPR # OAPR coefficient estimator with osqp solver
        iter.center_opt[kseries,md] <- res$iter
        w.table[,kseries,md,del] <- res$w
        time.weight.OAPR[kseries,md]=res$time
      }
    }
  }

  ###############################
  #### mix
  ###############################
  # Record the estimated weights
  wo.OAPR <- array(0,c(M,2+length(K_fixed),1,length(delta)),
                   list(NULL,
                        c("elbow","distortion",paste("fixed",K_fixed)),
                        c("mix"),paste0('delta_',delta)))
  
  for (md in 1:1) {
    if(md==1){model_list=modelsmix;MUtildecvo=MUtildecvmix}
    for (kseries in 1:(2+length(K_fixed))) {
      for (del in 1:length(delta)) {
        res <- betahat_OARFISH(XX,YY,MUtilde_list=MUtildecvo,cluster_vec=kmeans_other_clusters[,kseries,md,del],model_list=model_list)
        wo.OAPR[,kseries,md,del] <- res$w
        time.weight.OAPR[kseries,4]=res$time
      }
    }
  }
  sfCat(paste0("Replicate ",g,": OARP completes"), sep="\n")
  
  ###############################
  #### Computing the change of weight
  ###############################
  dimnames(wl1loss)=dimnames(wl2loss)=dimnames(wlmaxloss)=
    list(c("elbow","distortion",paste("fixed",K_fixed)),
         c("scad","lasso","glm",'mix'),paste0('delta_',delta))
  for (md in 1:4) {
    for (kseries in 1:(2+length(K_fixed))) {
      for (del in 1:length(delta)) {
        if(md<4){
          w.delta0=w.table[,kseries,md,1]
          wl1loss[kseries,md,del]<-sum(abs(w.table[,kseries,md,del]-w.delta0))
          wl2loss[kseries,md,del]<-norm(w.table[,kseries,md,del]-w.delta0,type = "2")
          wlmaxloss[kseries,md,del]<-max(abs(w.table[,kseries,md,del]-w.delta0))
        } else {
          w.delta0=wo.OAPR[,kseries,md-3,1]
          wl1loss[kseries,md,del]<-sum(abs(wo.OAPR[,kseries,md-3,del]-w.delta0))
          wl2loss[kseries,md,del]<-norm(wo.OAPR[,kseries,md-3,del]-w.delta0,type = "2")
          wlmaxloss[kseries,md,del]<-max(abs(wo.OAPR[,kseries,md-3,del]-w.delta0))
        }
      }
    }
  }
  
  ###############################
  #### Method OAP
  ###############################
  ###############################
  #### glm
  ###############################
  time.weight.OAP=vector(length = 4)
  betahatOAP <- matrix(0,d,3,dimnames = list(NULL,c("scad","lasso","glm")))
  iter.OAP <- matrix(0,3,1,dimnames = list(c("scad","lasso","glm"),NULL))
  # Record the estimated weights
  w.OAP <- matrix(0,M,3,dimnames = list(NULL,c("scad","lasso","glm")))
  
  for (md in 1:3) {
    res <- betahat_OARFISH(XX,YY,betahat[,md,],MUtildecv[md,],1:M)
    coefs.OAP <- res$coefs
    #coefs.OAP[abs(coefs.OAP)<1e-8] <- 0
    betahatOAP[,md]=coefs.OAP # OAPR coefficient estimator with osqp solver
    iter.OAP[md,1] <- res$iter
    w.OAP[,md] <- res$w
    time.weight.OAP[md]=res$time
  }
  
  ###############################
  #### mix
  ###############################
  # Record the estimated weights
  wo.OAP <- matrix(0,M,1,dimnames = 
                     list(NULL,c("mix")))
  for (md in 1:1) {
    if(md==1){model_list=modelsmix;MUtildecvo=MUtildecvmix}
    res <- betahat_OARFISH(XX,YY,MUtilde_list=MUtildecvo,cluster_vec=1:M,model_list=model_list)
    wo.OAP[,md] <- res$w
    time.weight.OAP[4]=res$time
  }
  sfCat(paste0("Replicate ",g,": OAP completes"), sep="\n")
  
  ###############################
  #### Method Equal Weight (also used as the warm start)
  ###############################
  time.ew=vector(length = 3)
  betahatEW <- array(0,c(d,3),dimnames = list(NULL,c("scad","lasso","glm")))
  for (md in 1:3) {
    ptm=proc.time()
    w0 <- matrix(1,nrow=M,ncol=1)/M #the initial value of weights
    betahatEW[,md]=betahat[,md,]%*%w0; #EMA coefficient estimator
    time.ew[md]=(proc.time()-ptm)[3]
  }
  
  
  ###############################
  #### Method Full
  ###############################
  betahatFull <- array(0,c(d,3),dimnames = list(NULL,c("scad","lasso","glm")))
  time.full_glm=vector(length = 3)
  X<-c() 
  Y<-c() 
  for (j in 1:M){
    X<-rbind(X,XX[[j]])
    Y<-c(Y,YY[[j]])
  } 
  #rm(XX,YY)
  for (md in 1:3) {
    ptm=proc.time()
    if(md==1){
      scad_mod <- ncpen(Y, X[,-1], family="binomial", penalty="scad",
                        x.standardize=T, intercept=T)
      lambda_scad.fit <- cv.ncpen(Y, X[,-1], family="binomial", penalty="scad",
                                  x.standardize=T, intercept=T)
      lambda_scad <- coef.cv.ncpen(lambda_scad.fit,type="rmse")$lambda
      
      if(lambda_scad>scad_mod$lambda[1]){
        betahatFull[,md] <- coef.ncvreg(scad_mod, lambda=scad_mod$lambda[1])
        warning("lambda_scad>scad_mod$lambda[1]")
      } else if(lambda_scad<scad_mod$lambda[length(scad_mod$lambda)]) {
        betahatFull[,md] <- coef.ncvreg(scad_mod, lambda=scad_mod$lambda[length(scad_mod$lambda)])
        warning("lambda_scad<scad_mod$lambda[length(scad_mod$lambda)]")
      } else {
        betahatFull[,md] <- coef.ncvreg(scad_mod, lambda=lambda_scad)
      }
    }
    if(md==2){
      lasso_mod = glmnet(X[,-1], Y, family = "binomial", intercept=T, standardize = T) 
      lambda_lasso.fit = cv.glmnet(X[,-1], Y, family = "binomial" ,intercept=T, standardize=T)
      betahatFull[,md]<-as.matrix(coef(lasso_mod, s=lambda_lasso.fit$lambda.min))
    }
    if(md==3){
      glm_mod = glm.fit(X, Y, family = binomial(), intercept=F) 
      betahatFull[,md]<-coef(glm_mod)
    }
    time.full_glm[md]=(proc.time()-ptm)[3]
  }
  sfCat(paste0("Replicate ",g,": Full completes"), sep="\n")
  
  ###############################
  #### Method Full_MLP
  ###############################
  ptm=proc.time()
  NeuralFull <- keras_model_sequential() %>%
    layer_dense(units = MLP_hidden1, activation = 'relu', 
                input_shape = c(ncol(X) - 1)) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  NeuralFull %>% compile(loss = 'binary_crossentropy', 
                         optimizer = 'adam', 
                         metrics = c('accuracy'))
  NeuralFull %>% fit(X[,-1], Y, epochs = 50, batch_size = 10,verbose=0)
  time.full_mlp=(proc.time()-ptm)[3]
  ###############################
  #### Method Full_MLP2
  ###############################
  ptm=proc.time()
  NeuralFull2 <- keras_model_sequential() %>%
    layer_dense(units = MLP_hidden2[1], activation = 'relu', 
                input_shape = c(ncol(X) - 1)) %>%
    layer_dense(units = MLP_hidden2[2], activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
  NeuralFull2 %>% compile(loss = 'binary_crossentropy', 
                          optimizer = 'adam', 
                          metrics = c('accuracy'))
  NeuralFull2 %>% fit(X[,-1], Y, epochs = 50, batch_size = 10,verbose=0)
  time.full_mlp2=(proc.time()-ptm)[3]
  ###############################
  #### Method Full_xgboost 
  ###############################
  ptm=proc.time()
  param <- list(max_depth = xgb_maxdepth1, eta = 1, nthread = 2, 
                objective = "binary:logistic", eval_metric = "error")
  xgbFull <- xgboost(data=as.matrix(X[,-1]),label = Y,params = param,nrounds = xgb_nrounds1,verbose = 0)
  time.full_xgb=(proc.time()-ptm)[3]
  ###############################
  #### Method Full_xgboost2 
  ###############################
  ptm=proc.time()
  param2 <- list(max_depth = xgb_maxdepth2, eta = 1, nthread = 2, 
                 objective = "binary:logistic", eval_metric = "error")
  xgbFull2 <- xgboost(data=as.matrix(X[,-1]),label = Y,params = param2,nrounds = xgb_nrounds2,verbose = 0)
  time.full_xgb2=(proc.time()-ptm)[3]
  
  #rm(X,Y)
  
  
  #### integrate all betahat into a large array
  md_b_name <- c("elbow","distortion",paste("fixed",K_fixed,sep="_"),"OAP",
                 "EW","Partial","Full")
  betahat_all <- array(dim = c(d,md_b,md_basic,length(delta)),
                       dimnames = list(NULL,md_b_name,c("scad","lasso","glm"),paste0('delta_',delta)))
  
  for (md in 1:3) {
    for (del in 1:length(delta)) {
      # fill betahat_all
      betahat_all[,1:(2+length(K_fixed)),md,del] <- betahatOAPR[,,md,del] # OAPR
      betahat_all[,(2+length(K_fixed))+1,md,del] <- betahatOAP[,md] # OAP
      betahat_all[,(2+length(K_fixed))+2,md,del] <- betahatEW[,md] # EW
      betahat_all[,(2+length(K_fixed))+3,md,del] <- betahat[,md,1] # Partial
      betahat_all[,(2+length(K_fixed))+4,md,del] <- betahatFull[,md] # Full
    }
  }
  
  betahat_allz <- betahat_all[,,-3,]
  
  ###############################
  #### Squared estimation error of theta &
  #### Coverage rate of confidence interval &
  #### Length of confidence interval
  ###############################
  storelossb <- apply(betahat_all,c(2,3,4),function(x){sum((x - theta_star[1:d])^2)})
  
  ###############################
  #### Correct/incorrect zero estimation rate of theta
  ###############################
  zeror <- apply(betahat_allz,c(2,3,4),function(x){sum((x==theta_star[1:d])[theta_star[1:d]==0])})
  zerow <- apply(betahat_allz,c(2,3,4),function(x){sum((x==0)[theta_star[1:d]!=0])})
  
  ###############################
  #### Predictive risk &
  #### Predictive accuracy (Bayes criterion: if p^hat_i>0.5, y^hat_i=1) &
  #### Predictive auc
  ###############################
  #### first predict, then weight predictions
  time.pred_M=matrix(NA,M,3)
  time.pred_OPAL=matrix(NA,2+length(K_fixed)+2,3)
  pred_prob1 <- array(0,c(nrow(Xst),2+length(K_fixed)+2,3,length(delta)))
  pred_probs1 <- array(0,c(nrow(Xst),M,3))
  for (md in 1:3) {
    for (j in 1:M) {
      ptm=proc.time()
      pred_probs1[,j,md] <- link(Xst%*%betahat[,md,j])
      time.pred_M[j,md]=(proc.time()-ptm)[3]
    }
  }
  for (mdp in 1:3) {
    for (kseries in 1:(2+length(K_fixed)+2)) {
      for (del in 1:length(delta)) {
        ptm=proc.time()
        if(kseries<(2+length(K_fixed)+1)){
          pred_prob1[,kseries,mdp,del] <- pred_probs1[,,mdp]%*%w.table[,kseries,mdp,del]
        }
        if(kseries==(2+length(K_fixed)+1)){
          pred_prob1[,kseries,mdp,del] <- pred_probs1[,,mdp]%*%w.OAP[,mdp]
        }
        if(kseries==(2+length(K_fixed)+2)){
          pred_prob1[,kseries,mdp,del] <- pred_probs1[,,mdp]%*%w0
        }
        time.pred_OPAL[kseries,mdp]=(proc.time()-ptm)[3]
      }
    }
  }
  
  for (del in 1:length(delta)) {
    for (mdp in 1:3) {
      storelossp[(md_p-(2+length(K_fixed)+2)+1):md_p,mdp,del] <- colMeans((sweep(-pred_prob1[,,mdp,del],1,probst,"+"))^2)
      pred_y <- pred_prob1[,,mdp,del]>=0.5
      if(any(pred_prob1[,,mdp,del]==0.5,na.rm=T)){
        pred_y[pred_prob1[,,mdp,del]==0.5&!is.na(pred_prob1[,,mdp,del])] <- sample(c(FALSE,TRUE),sum(pred_prob1[,,mdp,del]==0.5),replace=T)
      }
      storelossaac[(md_p-(2+length(K_fixed)+2)+1):md_p,mdp,del] <- colMeans(abs(sweep(-pred_y,1,Yst,"+")),na.rm=T)
      for (j in 1:dim(pred_prob1)[2]) {
        pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_prob1[,j,mdp,del]),quiet=T)
        pred_auc <- auc(pred_roc)
        storelossauc[(md_p-(2+length(K_fixed)+2))+j,mdp,del] <- pred_auc
      }
    }
  }
  
  #### first weight parameter estimators, then predict
  time.pred_OPAL_coef=matrix(NA,md_b,3)
  for (del in 1:length(delta)) {
    for (mdb in 1:md_b) {
      for (mdp in 1:3) {
        ptm=proc.time()
        pred_eta <- Xst%*%betahat_all[,mdb,mdp,del]
        pred_prob <- 1/(1+exp(-pred_eta))
        storelossp[mdb,mdp,del] <- mean((probst- pred_prob)^2)
        
        if(is.na(storelossp[mdb,mdp,del])){
          storelossaac[mdb,mdp,del] <- NA
          storelossauc[mdb,mdp,del] <- NA
        } else {
          pred_y <- pred_prob>=0.5
          pred_y[pred_prob==0.5] <- sample(c(FALSE,TRUE),sum(pred_prob==0.5),replace=T)
          storelossaac[mdb,mdp,del] <- mean(abs(pred_y - Yst))
          
          pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_prob),quiet=T)
          pred_auc <- auc(pred_roc)
          storelossauc[mdb,mdp,del] <- pred_auc
        }
        time.pred_OPAL_coef[mdb,mdp]=(proc.time()-ptm)[3]
      }
    }
  }
  
  ###############################
  #### Method others
  ###############################
  #### single site
  time.pred_mix=matrix(NA,(2+length(K_fixed)+2),1)
  time.pred_ml=matrix(NA,md_p-(2+length(K_fixed)+2)*2-md_b,1)
  time.pred_M_ml=matrix(NA,M,1)
  pred_prob <- array(0,c(nrow(Xst),md_p-(2+length(K_fixed)+2)-md_b,length(delta)))
  pred_probs <- array(0,c(nrow(Xst),M))
  for (ml in 1:md_ml) {
    for (ml2 in ml_partition[[ml]]) {
      ptm=proc.time()
      if(ml==1){pred_probs[,ml2] <- pred_probs1[,ml2,2]}
      if(ml==2){pred_probs[,ml2] <- modelsMLP[[ml2]] %>% predict(Xst[,-1])}
      if(ml==3){pred_probs[,ml2] <- modelsMLP2[[ml2]] %>% predict(Xst[,-1])}
      if(ml==4){pred_probs[,ml2] <- predict(modelsxg[[ml2]],newdata=Xst[,-1])}
      if(ml==5){pred_probs[,ml2] <- predict(modelsxg2[[ml2]],newdata=Xst[,-1])}
      if(ml!=1){
        time.pred_M_ml[ml2]=(proc.time()-ptm)[3]
      } else {
        time.pred_M_ml[ml2]=time.pred_M[ml2,2]
      }
    }
  }

  for (del in 1:length(delta)) {
    #### Full MLP/xgboost
    ptm=proc.time()
    pred_prob[,1,del] <- NeuralFull %>% predict(Xst[,-1])
    time.pred_ml[1]=(proc.time()-ptm)[3]
    ptm=proc.time()
    pred_prob[,2,del] <- NeuralFull2 %>% predict(Xst[,-1])
    time.pred_ml[2]=(proc.time()-ptm)[3]
    ptm=proc.time()
    pred_prob[,3,del] <- predict(xgbFull,newdata=Xst[,-1])
    time.pred_ml[3]=(proc.time()-ptm)[3]
    ptm=proc.time()
    pred_prob[,4,del] <- predict(xgbFull2,newdata=Xst[,-1])
    time.pred_ml[4]=(proc.time()-ptm)[3]
    #### Partial (take the est at the first site)
    pred_prob[,5:8,del] <- pred_probs[,unname(sapply(ml_partition, function(x) x[1]))[-1]]
    time.pred_ml[5:8]=time.pred_M_ml[unname(sapply(ml_partition, function(x) x[1]))[-1]]
  }
  
  #### OAPR/OAP for mixed learners
  for (del in 1:length(delta)) {
    for (kseries in 1:(2+length(K_fixed)+2)) {
      ptm=proc.time()
      if(kseries<(2+length(K_fixed)+1)){
        pred_prob[,kseries+8,del] <- pred_probs[,]%*%wo.OAPR[,kseries,1,del]
      }
      if(kseries==(2+length(K_fixed)+1)){
        pred_prob[,kseries+8,del] <- pred_probs[,]%*%wo.OAP[,1]
      }
      if(kseries==(2+length(K_fixed)+2)){
        pred_prob[,kseries+8,del] <- rowMeans(pred_probs[,])
      }
      time.pred_mix[kseries]=(proc.time()-ptm)[3]
    }
  }
  
  #Two evaluation criteria: probst for Risk, and Yst for accuracy
  for (del in 1:length(delta)) {
    storelossp[(md_b+1):(md_p-(2+length(K_fixed)+2)),3,del] <- colMeans((sweep(-pred_prob[,,del],1,probst,"+"))^2)
    pred_y <- pred_prob[,,del]>=0.5
    if(any(pred_prob[,,del]==0.5,na.rm=T)){
      pred_y[pred_prob[,,del]==0.5&!is.na(pred_prob[,,del])] <- sample(c(FALSE,TRUE),sum(pred_prob[,,del]==0.5),replace=T)
    }
    storelossaac[(md_b+1):(md_p-(2+length(K_fixed)+2)),3,del] <- colMeans(abs(sweep(-pred_y,1,Yst,"+")),na.rm=T)
    for (j in 1:(md_p-(2+length(K_fixed)+2)-md_b)) {
      pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_prob[,,del][,j]),quiet=T)
      pred_auc <- auc(pred_roc)
      storelossauc[md_b+j,3,del] <- pred_auc
    }
  }
  
  
  dimnames(storelossp) <- dimnames(storelossaac) <- dimnames(storelossauc) <- list(md_p_name,c("scad","lasso","glm"),paste0("delta_",delta))
  
  #### time table
  timeTable=matrix(NA,md_p,4)
  dimnames(timeTable)=list(md_p_name,c("scad","lasso","glm","mix"))

  time.fitting1=time.fitting
  time.fitting1[,4]=time.fitting1[,4]+time.shapley
  
  time.before_weight=sweep(time.cluster,2,time.scale,'+')
  time.before_weight[1:2,]=time.before_weight[1:2,]+time.k_selection
  time.before_weight1=pmax(time.before_weight,matrix(apply(time.fitting.cv,2,max),dim(time.before_weight)[1],dim(time.before_weight)[2],byrow = T))
  time.before_weight1=sweep(time.before_weight1,2,apply(time.fitting1,2,max),'+')
  time.before_weight1=rbind(time.before_weight1,sweep(rbind(apply(time.fitting.cv,2,max),rep(0,4)),2,apply(time.fitting,2,max),'+'))
  
  time.after_weight=time.before_weight1
  time.after_weight[1:(2+length(K_fixed)),]=time.after_weight[1:(2+length(K_fixed)),]+time.weight.OAPR
  time.after_weight[(2+length(K_fixed))+1,]=time.after_weight[(2+length(K_fixed))+1,]+time.weight.OAP
  time.after_weight[(2+length(K_fixed))+2,1:3]=time.after_weight[(2+length(K_fixed))+2,1:3]+time.ew
  
  timeTable[1:(2+length(K_fixed)+2),1:3]=time.after_weight[1:(2+length(K_fixed)+2),1:3]+time.pred_OPAL_coef[1:(2+length(K_fixed)+2),1:3]
  timeTable[(2+length(K_fixed)+2+1),1:3]=apply(time.fitting,2,max)[1:3]+time.pred_M[1,]
  timeTable[(2+length(K_fixed)+2+2),1:3]=time.full_glm+time.pred_OPAL_coef[(2+length(K_fixed)+2+2),1:3]
  timeTable[md_b+1,4]=time.full_mlp
  timeTable[md_b+2,4]=time.full_mlp2
  timeTable[md_b+3,4]=time.full_xgb
  timeTable[md_b+4,4]=time.full_xgb2
  timeTable[(md_b+5):(md_b+8),4]=time.fitting[,4][unname(sapply(ml_partition, function(x) x[1]))[-1]]
  timeTable[(md_b+1):(md_b+8),4]=timeTable[(md_b+1):(md_b+8),4]+time.pred_ml
  timeTable[(md_b+9):(md_b+8+2+length(K_fixed)+2),4]=time.after_weight[1:(2+length(K_fixed)+2),4]+time.pred_mix+max(time.pred_M_ml)
  timeTable[(md_p-(2+length(K_fixed)+2)+1):md_p,1:3]=time.after_weight[1:(2+length(K_fixed)+2),1:3]+time.pred_OPAL+matrix(apply(time.pred_M,2,max),2+K_fixed_num+2,3,byrow=T)
  
  # save the data-driven K by methods 'elbow' and 'distortion'
  K.table_summary=array(
    c(K.table[1:2,1,],K.table[1:2,2,],K.table[1:2,3,],K.table.ml[1:2,,]),
    dim = c(nrow(K.table[1:2,1,]), ncol(K.table[1:2,1,]),4),
    dimnames = list(c("elbow","distortion"),paste0('delta_',delta),
                    c("scad","lasso","glm",'mix'))
  )
  # clear memory
  rm(modelsMLP, modelsMLP2, modelsxg, modelsxg2,
     NeuralFull, NeuralFull2, xgbFull, xgbFull2,
     shapleyrecord_mix, modelsmix, MUtildecvMLP, MUtildecvMLP2,
     MUtildecvxg, MUtildecvxg2)
  rm(Xst, Yst, probst, MU, betahat, MUtildecv)
  gc()

  return(list(storelossb=storelossb,storelossp=storelossp,storelossaac=storelossaac,
              storelossauc=storelossauc,zeror=zeror,zerow=zerow,timeTable=timeTable,
              K.table=K.table_summary,wl1loss=wl1loss,wl2loss=wl2loss,wlmaxloss=wlmaxloss,
              w.table=w.table,wo.OAPR=wo.OAPR))
}

###############################
#### Load library in parallel way (using snowfall)
###############################
sfSetMaxCPUs(cpunum)

###############################
#### Compute the statistics in situations 
#### with different d and n1
###############################
# d0ind=dind=decayi=nsi=sigi=g=1;mi=2
for (d0ind in 1:length(d0s)) {
  for (dind in 1:length(ds)) {
    for (decayi in 1:length(decays)) {
      for (sigi in 1:length(sigs)){
        for (nsi in 1:length(ns)) {
          for (mi in 1:length(Ms)) {

            ptm <- proc.time()
            num_batches <- ceiling(G / cpunum[nsi,mi])  
            Store=c()
            for (b in 1:num_batches) {
              start_idx <- (b - 1) * cpunum[nsi,mi] + 1
              end_idx <- min(b * cpunum[nsi,mi], G)
              
              sfInit(parallel = T,cpus = cpunum[nsi,mi],slaveOutfile = "OAPR_sensitivity_Shapley.txt")
              sfLibrary(snowfall)
              sfLibrary(glmnet)
              sfLibrary(ncvreg)
              sfLibrary(ncpen)
              sfLibrary(MASS)
              sfLibrary(keras)
              sfLibrary(tensorflow)
              sfLibrary(reticulate)
              sfLibrary(xgboost)
              sfLibrary(caret)
              sfLibrary(ClusterR)
              sfLibrary(KneeArrower)
              sfLibrary(pROC)
              sfLibrary(CVXR)
              sfLibrary(osqp)
              sfLibrary(shapviz)
              sfSource("coef_ncvreg.R")
              sfSource("lam_names.R")
              sfSource("predict_ncvreg.R")
              sfSource("OARFISH_solver.R")
              #sfSource("logistic_solvers_forR.R")
              #sfSource("meta.R")
              sfExport('predictfun.glm')
              sfExport('md_basic')
              sfExport('md_b')
              sfExport('md_p')
              sfExport('md_a')
              sfExport('md_z')
              sfExport('delta')
              sfExport('ns')
              sfExport('by')
              sfExport('G')
              sfExport('Kfold')
              sfExport('N_test')
              sfExport('K_fixed_num')
              
              d0 <- d0s[d0ind]
              d <- (d0-1)*(ds[dind])+1
              decay <- decays[decayi]
              sig <- sigs[sigi]
              n1 <- ns[nsi]
              M <- Ms[mi]
              nn<-seq(n1, by=by,length.out=M)
              #distribute sample size for every site
              
              betahat<-array(0,c(d,3,M)) #dimension; methods:scad, lasso and glm; the number of sites
              MUtildecv <- list()#record the coefficient estimation
              length(MUtildecv) <- 3*M
              dim(MUtildecv) <- c(3,M) 
              for (md in 1:3) {
                for (j in 1:M) {
                  MUtildecv[[md,j]] <- vector(length = nn[j])
                }
              }
              
              #prediction loss for listed methods
              storelossp <-array(0,c(md_p,md_basic,length(delta)),dimnames = list(NULL,c("scad","lasso","glm"),paste0('delta_',delta)))
              storelossaac <-array(0,c(md_a,md_basic,length(delta)),dimnames=list(NULL,c("scad","lasso","glm"),paste0('delta_',delta)))
              storelossb <-array(0,c(md_b,md_basic,length(delta)),dimnames = list(NULL,c("scad","lasso","glm"),paste0('delta_',delta)))
              storelossauc <-array(0,c(md_a,md_basic,length(delta)),dimnames=list(NULL,c("scad","lasso","glm"),paste0('delta_',delta)))
              zeror <- array(0,c(md_z,md_basic-1,length(delta)),dimnames = list(NULL,c("scad","lasso"),paste0('delta_',delta)))
              zerow <- array(0,c(md_z,md_basic-1,length(delta)),dimnames = list(NULL,c("scad","lasso"),paste0('delta_',delta)))
              wl1loss <-array(0,c(2+K_fixed_num,md_basic+1,length(delta)))
              wl2loss <-array(0,c(2+K_fixed_num,md_basic+1,length(delta)))
              wlmaxloss<-array(0,c(2+K_fixed_num,md_basic+1,length(delta)))
              
              sfExport('d0')
              sfExport('d')
              sfExport('decay')
              sfExport('sig')
              sfExport('M')
              sfExport('nn')
              sfExport('n1')
              sfExport('betahat')
              sfExport('MUtildecv')
              sfExport('storelossp')
              sfExport('storelossb')
              sfExport('storelossaac')
              sfExport('storelossauc')
              sfExport('zeror')
              sfExport('zerow')
              sfExport('wl1loss')
              sfExport('wl2loss')
              sfExport('wlmaxloss')
              sfExport('md_p_name')
              sfExport('MLP_hidden1','MLP_hidden2')
              sfExport('md_ml')
              sfExport('xgb_nrounds1','xgb_nrounds2')
              sfExport('xgb_maxdepth1','xgb_maxdepth2')
              
              #Store1 <- sfSapply(1:cpunum,simfunc)#warm start for 'proc.time'
              Store1 <- sfSapply(start_idx:end_idx,simfunc)
              sfStop()
              
              Store=cbind(Store,Store1)
            }
            timecost.whole <- proc.time()-ptm
            timecost.whole <- round(timecost.whole,3)
            
            msebM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[1,])/G
            msepM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[2,])/G
            accM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[3,])/G
            aucM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[4,])/G
            msebM.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[1,],function(X){(X-msebM[mi,nsi,,,,d0ind,dind,decayi,sigi])^2}))/(G-1)/G)
            msepM.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[2,],function(X){(X-msepM[mi,nsi,,,,d0ind,dind,decayi,sigi])^2}))/(G-1)/G)
            accM.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[3,],function(X){(X-accM[mi,nsi,,,,d0ind,dind,decayi,sigi])^2}))/(G-1)/G)
            aucM.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[4,],function(X){(X-aucM[mi,nsi,,,,d0ind,dind,decayi,sigi])^2}))/(G-1)/G)
            zerorightM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[5,])/length(Store[5,])
            zerowrongM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[6,])/length(Store[6,])
            timecostMd[mi,nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[7,])/length(Store[7,])
            timecostM[mi,nsi,,,d0ind,dind,decayi,sigi]<-timecost.whole[1:3]
            K.tableM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[8,])/G
            K.tableM.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[8,],function(X){(X-K.tableM[mi,nsi,,,,d0ind,dind,decayi,sigi])^2}))/(G-1)/G)
            
            wl1M[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[9,])/G
            wl2M[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[10,])/G
            wlmaxM[mi,nsi,,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[11,])/G
            wl1M.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+", lapply(Store[9,], function(M) (M-wl1M[mi,nsi,,,,d0ind,dind,decayi,sigi])^2))/G/(G-1))
            wl2M.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+", lapply(Store[10,], function(M) (M-wl2M[mi,nsi,,,,d0ind,dind,decayi,sigi])^2))/G/(G-1))
            wlmaxM.sd[mi,nsi,,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+", lapply(Store[11,], function(M) (M-wlmaxM[mi,nsi,,,,d0ind,dind,decayi,sigi])^2))/G/(G-1))
            
            wM.OAPR[mi,nsi,d0ind,dind,decayi,sigi]<-list(simplify2array(Store[12,]))
            woM.OAPR[mi,nsi,d0ind,dind,decayi,sigi]<-list(simplify2array(Store[13,]))
            
            print(paste0("sitesamplesize=",n1," sitenumbers=",M," d=",d))
            print(paste0(names(timecost.whole[1]),"=",timecost.whole[1],",",names(timecost.whole[2]),"=",timecost.whole[2],",",names(timecost.whole[3]),"=",timecost.whole[3]))
            #the best way to avoid crush: save your data!
            save.image(file = paste("OAPR_sensitivity_Shapley","_cpu_",cpunum[nsi,mi],"_d0_",d0,"_d_",d,"_M_",M,"_n_",n1,".Rdata",sep=""))
            sfStop()
          }
        }
      }
    }
  }
}
sfStop()



