# This is the simulation in Section 4.1 in OPAL
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
#-------------------------------------------------
#Base learner: only logitic model
#1. OAPR* (Optimal Aggregation Prediction with the weight Restriction)
#2. OAP* (Optimal Aggregation Prediction)
#3. Full* (gold standard)
#4. Naive averaging on betahat
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
#-------------------------------------------------
#Base learners: xgboost, MLP and logistic model
#17. OAPR (Optimal Aggregation Prediction with the weight Restriction)
#18. OAP (Optimal Aggregation Prediction)
#19. Full (gold standard)
#20. Naive averaging on probability estimate

###############################
#### install/load library
###############################
#setwd("~/DCGLM/sim_fan")
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
library(tensorflow)
library(reticulate)
#library(NeuralNetTools)
#library(e1071)
#library(randomForest)
library(shapviz)
library(ClusterR)
library(CVXR)
library(osqp)
library(pROC)
source("coef_ncvreg.R")
source("lam_names.R")
source("predict_ncvreg.R")
source("OARFISH_solver.R")
source("logistic_solvers_forR.R")
source("meta.R")

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
K_fixed=vector(length = 3) # K is (M-2), (M-2)/2, and (M-2)/4, respectively

d0s <- c(51)
N <- 12000 # the total sample size N is fixed
N_test=2000
Ms <- c(5,10,20,40)
ns <- round(N/Ms)
ds <- c(0.2,0.5,0.8)
decays<-c(1) #the decay rate of coefficient, where -.5 means uniform setting in the JASA paper
sigs <- c(3)
md_ml=5 #when base models at each site are different, how many base machine learning methods do we use
by=0 #difference parameter, used for 'seq' function, seq(from,by,length.out), 
     #to set local sample size and allow it different across different site
G<-100 #the setting for replicates
Kfold <- 5 #when predicting the probability of in-site observations, how many folds we set?
cpunum <- c(20,10,8,5) #the setting for the amount of CPUs requested for the cluster

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
md_b_name <- c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP",
               "Partial","EW","CR","Full","Meta_f","Meta_r",paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
               paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_"),paste("ADMM",1:3,sep="_"),
               paste("GIANT",1:3,sep="_"),paste("AGD",1:3,sep="_"))
md_p_name <- c(md_b_name,"Inv_var","MLPFull","MLP2Full","xgbFull","xgb2Full",
               "MLPp","MLP2p","xgbp","xgb2p",
               paste0(c("mix"),c("logM","elbow","distortion",paste("fixed",1:3,sep="_"))),
               paste0(c("mix"),c("OAP","EW")),
               paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP",'EW'),'p',sep="_"))
col_to_remove <- match(c("Meta_f","Meta_r",paste("ADMM",1:3,sep="_")
                         ,paste("GIANT",1:3,sep="_"),paste("AGD",1:3,sep="_")), md_b_name)
md_z_name <- md_b_name[-col_to_remove]
M_name <- paste("M",Ms,sep="_")
md_basic_name <- c("scad","lasso","glm")
d0_name <- paste("d0",d0s,sep="_")
d_name <- paste("d",ds,sep="_")
decay_name <- c(paste("decay",decays[1],sep="_"))
sig_name <- paste("sig",sigs,sep="_")
timed_name=c('fitting','shapley','cluster','weight.OAPR','weight.OAPR.ml',
             'weight.OAP','weight.OAP.ml','Full','meta','CEASEa',
             'CEASE0','CEASEsa','CSL','mlpcv','mlp2cv')

md_b=length(md_b_name)
md_p=length(md_p_name)
md_a=length(md_p_name)
md_z=length(md_z_name)

# Ms: site number; md_b/md_p/md_a/md_z: comparison method; md_basic: base learner;
# d0s: the dimensionality of true parameter; ds: the number of variables used; alphaM: coefficient pattern
msebM <-array(dim=c(length(Ms),md_b,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #MSE for coefficient estimation
msepM <-array(dim=c(length(Ms),md_p,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #MSE for probability estimation
accM <-array(dim=c(length(Ms),md_a,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #accuracy index
aucM <-array(dim=c(length(Ms),md_a,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #auc index
msebM.sd <-array(dim=c(length(Ms),md_b,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #MSE for coefficient estimation
msepM.sd <-array(dim=c(length(Ms),md_p,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #MSE for probability estimation
accM.sd <-array(dim=c(length(Ms),md_a,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #accuracy index
aucM.sd <-array(dim=c(length(Ms),md_a,md_basic,length(d0s),length(ds),length(decays),length(sigs))) #auc index
zerorightM <-array(dim=c(length(Ms),md_z,md_basic-1,length(d0s),length(ds),length(decays),length(sigs))) #correct zero estimated by penalized methods
zerowrongM <-array(dim=c(length(Ms),md_z,md_basic-1,length(d0s),length(ds),length(decays),length(sigs))) #incorrect zero estimated by penalized methods
timecostM <-array(0,c(length(Ms),3,length(d0s),length(ds),length(decays),length(sigs)))#'3' means the output of 'proc.time'
timecostMd <-array(0,c(length(Ms),15,length(d0s),length(ds),length(decays),length(sigs)))#'3' means the output of 'proc.time'

dimnames(msebM) <- dimnames(msebM.sd) <- list(M_name,md_b_name,md_basic_name,d0_name,d_name,decay_name,sig_name)
dimnames(msepM) <- dimnames(accM) <- dimnames(aucM) <- dimnames(msepM.sd) <- dimnames(accM.sd) <- dimnames(aucM.sd) <- 
  list(M_name,md_p_name,md_basic_name,d0_name,d_name,decay_name,sig_name)
dimnames(zerorightM) <- dimnames(zerowrongM) <- list(M_name,md_z_name,md_basic_name[-3],d0_name,d_name,decay_name,sig_name)
dimnames(timecostMd) <- list(M_name,timed_name,d0_name,d_name,decay_name,sig_name)

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
  ptm=proc.time()
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
      betahatglmcv<-minimize_logistic_loss(Xm[-obs_partition[[k]],],Ym[-obs_partition[[k]]],learning_rate = 0.1,maxiter = 10000)$coefs
      MUtildecv[[3,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahatglmcv)
      if(sum(is.nan(MUtildecv[[3,j]][obs_partition[[k]]]))>0){
        MUtildecv[[3,j]][obs_partition[[k]]][which(is.nan(MUtildecv[[3,j]][obs_partition[[k]]])==1)] <- 1
      }
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
      Neuralm1 %>% fit(Xm[,-1], Ym, epochs = epochm[j], batch_size = 10,verbose=0)
      modelsMLP[[j]] <- Neuralm1
      #Neuralm1$count_params()
      ptm=proc.time()
      for (k in 1:Kfold) {
        ptm <- proc.time()
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
      time.mlpcv <- proc.time()-ptm
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
      Neuralm2 %>% fit(Xm[,-1], Ym, epochs = epochm[j], batch_size = 10,verbose=0)
      modelsMLP2[[j]] <- Neuralm2
      #Neuralm2$count_params()
      ptm=proc.time()
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
      time.mlp2cv <- proc.time()-ptm
    }
    ###############################
    #### xgboost (para<=((64-1)*4+64)*2=632)
    ###############################
    if(j %in% ml_partition[[4]]){
      param <- list(max_depth = xgb_maxdepth1, eta = 1, nthread = 2, 
                    objective = "binary:logistic", eval_metric = "error")
      xgbm <- xgboost(data=as.matrix(Xm[,-1]),label = Ym,params = param,nrounds = xgb_nrounds1,verbose = 0)
      #xgb.dump(xgbm)
      modelsxg[[j]] <- xgbm
      for (k in 1:Kfold) {
        xgbmcv <- xgboost(data=as.matrix(Xm[-obs_partition[[k]],-1]),label = Ym[-obs_partition[[k]]],params = param,nrounds = xgb_nrounds1,verbose = 0)
        MUtildecvxg[[j]][obs_partition[[k]]] <- predict(xgbmcv,newdata=Xm[obs_partition[[k]],-1])
      }
    }
    ###############################
    #### xgboost2 (para<=((8-1)*4+8)*8=288)
    ###############################
    if(j %in% ml_partition[[5]]){
      param2 <- list(max_depth =xgb_maxdepth2, eta = 1, nthread = 2, 
                     objective = "binary:logistic", eval_metric = "error")
      xgbm2 <- xgboost(data=as.matrix(Xm[,-1]),label = Ym,params = param2,nrounds = xgb_nrounds2,verbose = 0)
      #xgb.dump(xgbm2)
      modelsxg2[[j]] <- xgbm2
      for (k in 1:Kfold) {
        xgbmcv <- xgboost(data=as.matrix(Xm[-obs_partition[[k]],-1]),label = Ym[-obs_partition[[k]]],params = param2,nrounds = xgb_nrounds2,verbose = 0)
        MUtildecvxg2[[j]][obs_partition[[k]]] <- predict(xgbmcv,newdata=Xm[obs_partition[[k]],-1])
      }
    }
  }
  
  time.fitting=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": model fitting at local sites completes"), sep="\n")
  
  ### five methods are randomly and equally distributed to M sites
  ptm=proc.time()
  modelsmix=MUtildecvmix=list()
  length(modelsmix)=length(MUtildecvmix)=M
  for (ml in 1:md_ml) {
    for (ml2 in ml_partition[[ml]]) {
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
    }
  }
  time.shapley=proc.time()-ptm
  rm(shap)
  sfCat(paste0("Replicate ",g,": shapley computation completes"), sep="\n")
  
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
  
  ###############################
  #### glm (cluster index: coefficient)
  ###############################
  ptm <- proc.time()
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
  #### mix (cluster index: shapley)
  ###############################
  shapleymix_s=scale(drop0(shapleyrecord_mix))
  shapley_s=list(shapleymix_s)
  
  K.table.ml <- array(0,c(3+length(K_fixed),1),list(
    c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
    c("mix")))
  
  for (md in 1:1) {
    K.table.ml[2,md] <- K_selection(shapley_s[[md]],"elbow",M-2)
    K.table.ml[3,md] <- K_selection(shapley_s[[md]],"distortion",M-2)
  }
  #sum(is.na(K.table.ml))
  K.table.ml[1,] <- K_logM
  for (kseries in 1:length(K_fixed)) {
    K.table.ml[kseries+3,] <- K_fixed[kseries]
  }
  
  ###############################
  #### Implement K-means
  ###############################
  #### glm
  ###############################
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
  ###############################
  #### mix
  ###############################
  kmeans_other_clusters <- array(0,c(M,3+length(K_fixed),1),
                                 dimnames = list(NULL,c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                                 c("mix")))
  for (kseries in 1:(3+length(K_fixed))) {
    for (md in 1:1) {
      KK=K.table.ml[kseries,md]
      shaps=shapley_s[[md]]
      if(KK==M){
        kmeans_other_clusters[,kseries,md] <- 1:M
      } else {
        non_zero_cols <- apply(shaps, 2, function(col) any(col != 0) & any(!is.na(col)))
        shap_dropcols <- shaps[, non_zero_cols, drop = FALSE] # 'drop' coerce it to be a matrix
        
        kmeans_tmp <- KMeans_rcpp(shap_dropcols, KK)
        kmeans_other_clusters[,kseries,md] <- kmeans_tmp$clusters
      }
    }
  }
  time.cluster=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": Kmeans completes"), sep="\n")
  
  ###############################
  #### Method OAPR 
  ###############################
  ###############################
  #### glm
  ###############################
  ptm=proc.time()
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
  
  for (md in 1:3) {
    for (kseries in 1:(3+length(K_fixed))) {
      res <- betahat_OARFISH(XX,YY,betahat[,md,],MUtildecv[md,],kmeans_clusters[,kseries,md])
      coefs.OAPR <- res$coefs
      #coefs.OAPR[abs(coefs.OAPR)<1e-8] <- 0
      betahatOAPR[,kseries,md]=coefs.OAPR # OAPR coefficient estimator with osqp solver
      iter.center_opt[kseries,md] <- res$iter
      w.table[,kseries,md] <- res$w
    }
  }
  time.weight.OAPR=proc.time()-ptm

  ###############################
  #### mix
  ###############################
  # Record the estimated weights
  ptm=proc.time()
  wo.OAPR <- array(0,c(M,3+length(K_fixed),1),
                   list(NULL,
                        c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                        c("mix")))
  
  for (md in 1:1) {
    if(md==1){model_list=modelsmix;MUtildecvo=MUtildecvmix}
    for (kseries in 1:(3+length(K_fixed))) {
      res <- betahat_OARFISH(XX,YY,MUtilde_list=MUtildecvo,cluster_vec=kmeans_other_clusters[,kseries,md],model_list=model_list)
      wo.OAPR[,kseries,md] <- res$w
    }
  }
  time.weight.OAPR.ml=proc.time()-ptm
  
  sfCat(paste0("Replicate ",g,": OARP completes"), sep="\n")
  
  ###############################
  #### Method OAP
  ###############################
  ###############################
  #### glm
  ###############################
  ptm=proc.time()
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
  }
  time.weight.OAP=proc.time()-ptm
  ###############################
  #### mix
  ###############################
  # Record the estimated weights
  ptm=proc.time()
  wo.OAP <- matrix(0,M,1,dimnames = 
                     list(NULL,c("mix")))
  for (md in 1:1) {
    if(md==1){model_list=modelsmix;MUtildecvo=MUtildecvmix}
    res <- betahat_OARFISH(XX,YY,MUtilde_list=MUtildecvo,cluster_vec=1:M,model_list=model_list)
    wo.OAP[,md] <- res$w
  }
  time.weight.OAP.ml=proc.time()-ptm
  
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
  #### Method Cubic Rate (Shi et al.)
  ###############################
  betahatCR <- array(0,c(d,3),dimnames = list(NULL,c("scad","lasso","glm")))
  for (md in 1:3) {
    wcr <- matrix(nn^(2/3)/sum(nn^(2/3)),nrow=M,ncol=1) 
    betahatCR[,md]=betahat[,md,]%*%wcr; #CR coefficient estimator
  }
  
  ###############################
  #### Method Full
  ###############################
  ptm=proc.time()
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
  }
  
  sfCat(paste0("Replicate ",g,": Full completes"), sep="\n")
  
  ###############################
  #### Method Full_MLP
  ###############################
  NeuralFull <- keras_model_sequential() %>%
    layer_dense(units = MLP_hidden1, activation = 'relu', 
                input_shape = c(ncol(X) - 1)) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  NeuralFull %>% compile(loss = 'binary_crossentropy', 
                         optimizer = 'adam', 
                         metrics = c('accuracy'))
  NeuralFull %>% fit(X[,-1], Y, epochs = 50, batch_size = 10,verbose=0)
  
  ###############################
  #### Method Full_MLP2
  ###############################
  NeuralFull2 <- keras_model_sequential() %>%
    layer_dense(units = MLP_hidden2[1], activation = 'relu', 
                input_shape = c(ncol(X) - 1)) %>%
    layer_dense(units = MLP_hidden2[2], activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
  NeuralFull2 %>% compile(loss = 'binary_crossentropy', 
                          optimizer = 'adam', 
                          metrics = c('accuracy'))
  NeuralFull2 %>% fit(X[,-1], Y, epochs = 50, batch_size = 10,verbose=0)
  
  ###############################
  #### Method Full_xgboost 
  ###############################
  param <- list(max_depth = xgb_maxdepth1, eta = 1, nthread = 2, 
                objective = "binary:logistic", eval_metric = "error")
  xgbFull <- xgboost(data=as.matrix(X[,-1]),label = Y,params = param,nrounds = xgb_nrounds1,verbose = 0)
  
  ###############################
  #### Method Full_xgboost2 
  ###############################
  param2 <- list(max_depth = xgb_maxdepth2, eta = 1, nthread = 2, 
                 objective = "binary:logistic", eval_metric = "error")
  xgbFull2 <- xgboost(data=as.matrix(X[,-1]),label = Y,params = param2,nrounds = xgb_nrounds2,verbose = 0)
  time.Full=proc.time()-ptm
  ###############################
  #### Method AGD
  ###############################
  betahatAGD <- array(0,c(d,1,3),dimnames = list(NULL,c("glm"),paste0("t",1:3)))
  betahatAGD[,1,]=logistic_AGD(X,Y,t=3, theta_initial = betahatEW[,3],learning_rate = 0.01)
  
  sfCat(paste0("Replicate ",g,": AGD completes"), sep="\n")
  
  ###############################
  #### Method Meta
  ###############################
  ptm=proc.time()
  betahatMeta <- array(0,c(d,2),dimnames = list(NULL,c("meta_fixed","meta_random")))
  y.dCLR <- matrix(nrow = M,ncol = n1)
  x_all.dCLR <- matrix(nrow = M,ncol = n1*(d-1))
  for (i in 1:M) {
    y.dCLR[i,1:n1] <- YY[[i]]
    x_all.dCLR[i,1:(n1*(d-1))] <- unlist(c(XX[[i]][,-1]))
  }
  res.meta=meta_dCLR(M,nn,y.dCLR,x_all.dCLR,d-1)
  betahatMeta[,1] <- res.meta$beta_meta_fix
  betahatMeta[,2] <- res.meta$beta_meta_random
  
  time.meta=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": Meta completes"), sep="\n")
  
  ###############################
  #### Method CEASE(a) (a is alpha)
  ###############################
  ptm=proc.time()
  betahatCEASEa <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  for (md in 1:3) {
    if(md==3){
      res_CEASEa <- logistic_DANE(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                  theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                                  learning_rate=0.01, maxiter=500, tol=1e-3, ci=F)
      betahatCEASEa[,md,]=res_CEASEa$coefs
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
  time.CEASEa=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": CEASEa completes"), sep="\n")
  
  ###############################
  #### Method CEASE(0)
  ###############################
  ptm=proc.time()
  betahatCEASE0 <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  for (md in 1:3) {
    if(md==3){
      res_CEASE0 <- logistic_DANE(XX,YY,t=3,alpha = 0,
                                  theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                                  learning_rate=0.01, maxiter=500, tol=1e-3, ci=F)
      betahatCEASE0[,md,]=res_CEASE0$coefs
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
  time.CEASE0=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": CEASE0 completes"), sep="\n")
  
  ###############################
  #### Method CEASE-single(a)
  ###############################
  ptm=proc.time()
  betahatCEASEsa <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  for (md in 1:3) {
    if(md==3){
      res_CEASEsa <- logistic_CEASE1(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                     theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                                     trunconst = 0.01,learning_rate=0.01, maxiter=500, tol=1e-3, ci=F)
      betahatCEASEsa[,md,]=res_CEASEsa$coefs
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
  time.CEASEsa=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": CEASEsa completes"), sep="\n")
  
  ###############################
  #### Method CEASE-single(0) (CSL)
  ###############################
  ptm=proc.time()
  betahatCSL <- array(0,c(d,3,3),dimnames = list(NULL,c("scad","lasso","glm"),paste0("t",1:3)))
  for (md in 1:3) {
    if(md==3){
      res_CSL <- logistic_CSL1(XX,YY,t=3, theta_initial = betahatEW[,md], lambda=0, penalty = "null",
                               trunconst = 0.01,learning_rate=0.01, maxiter=500, tol=1e-3, ci=F)
      betahatCSL[,md,]=res_CSL$coefs
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
  time.CSL=proc.time()-ptm
  sfCat(paste0("Replicate ",g,": CSL completes"), sep="\n")
  
  
  ###############################
  #### Method ADMM
  ###############################
  betahatADMM <- array(0,c(d,1,3),dimnames = list(NULL,c("glm"),paste0("t",1:3)))
  betahatADMM[,1,]=logistic_ADMM(XX,YY,t=3, theta_initial = betahatEW[,3], rho=0.01, 
                                 learning_rate=0.01, maxiter=500, tol=1e-3)
  
  sfCat(paste0("Replicate ",g,": ADMM completes"), sep="\n")
  
  ###############################
  #### Method GIANT
  ###############################
  betahatGIANT <- array(0,c(d,1,3),dimnames = list(NULL,c("glm"),paste0("t",1:3)))
  betahatGIANT[,1,]=logistic_GIANT(XX,YY,t=3, theta_initial = betahatEW[,3])
  
  sfCat(paste0("Replicate ",g,": GIANT completes"), sep="\n")
  
  #### integrate all betahat into a large array
  md_b_name <- c(paste("logM",K_logM,sep="_"),"elbow","distortion",paste("fixed",K_fixed,sep="_"),"OAP",
                 "Partial","EW","CR","Full","Meta_f","Meta_r",paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
                 paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_"),paste("ADMM",1:3,sep="_"),
                 paste("GIANT",1:3,sep="_"),paste("AGD",1:3,sep="_"))
  betahat_all <- array(dim = c(d,md_b,md_basic),dimnames = list(NULL,md_b_name,c("scad","lasso","glm")))
  
  for (md in 1:3) {
    # fill betahat_all
    betahat_all[,1:(3+length(K_fixed)),md] <- betahatOAPR[,,md] # OAPR
    betahat_all[,(3+length(K_fixed))+1,md] <- betahatOAP[,md] # OAP
    betahat_all[,(3+length(K_fixed))+2,md] <- betahat[,md,1] # Partial
    betahat_all[,(3+length(K_fixed))+3,md] <- betahatEW[,md] # EW
    betahat_all[,(3+length(K_fixed))+4,md] <- betahatCR[,md] # CR
    betahat_all[,(3+length(K_fixed))+5,md] <- betahatFull[,md] # Full
    if(md==3){
      betahat_all[,(3+length(K_fixed))+6,md] <- betahatMeta[,1] # Meta_f
      betahat_all[,(3+length(K_fixed))+7,md] <- betahatMeta[,2] # Meta_r
    }
    betahat_all[,(3+length(K_fixed))+8:10,md] <- betahatCEASEa[,md,] # CEASEa
    betahat_all[,(3+length(K_fixed))+11:13,md] <- betahatCEASE0[,md,] # CEASE0
    betahat_all[,(3+length(K_fixed))+14:16,md] <- betahatCEASEsa[,md,] # CEASEsa
    betahat_all[,(3+length(K_fixed))+17:19,md] <- betahatCSL[,md,] # CSL
    if(md==3){
      betahat_all[,(3+length(K_fixed))+20:22,md] <- betahatADMM[,,] # ADMM
      betahat_all[,(3+length(K_fixed))+23:25,md] <- betahatGIANT[,,] # GIANT
      betahat_all[,(3+length(K_fixed))+26:28,md] <- betahatAGD[,,] # AGD
    }
  }
  
  
  col_to_remove <- match(c("Meta_f","Meta_r",paste("ADMM",1:3,sep="_")
                           ,paste("GIANT",1:3,sep="_"),paste("AGD",1:3,sep="_")), md_b_name)
  betahat_allz <- betahat_all[,-col_to_remove,-3]
  
  
  ###############################
  #### Squared estimation error of theta &
  #### Coverage rate of confidence interval &
  #### Length of confidence interval
  ###############################
  storelossb <- apply(betahat_all,c(2,3),function(x){sum((x - theta_star[1:d])^2)})
  
  ###############################
  #### Correct/incorrect zero estimation rate of theta
  ###############################
  zeror <- apply(betahat_allz,c(2,3),function(x){sum((x==theta_star[1:d])[theta_star[1:d]==0])})
  zerow <- apply(betahat_allz,c(2,3),function(x){sum((x==0)[theta_star[1:d]!=0])})
  
  ###############################
  #### Predictive risk &
  #### Predictive accuracy (Bayes criterion: if p^hat_i>0.5, y^hat_i=1) &
  #### Predictive auc
  ###############################
  pred_prob1 <- array(0,c(nrow(Xst),8,3))
  pred_probs1 <- array(0,c(nrow(Xst),M,3))
  pred_probs1[,,1] <- apply(Xst%*%betahat[,1,],2,link)
  pred_probs1[,,2] <- apply(Xst%*%betahat[,2,],2,link)
  pred_probs1[,,3] <- apply(Xst%*%betahat[,3,],2,link)
  for (mdp in 1:3) {
    pred_prob1[,1:6,mdp] <- pred_probs1[,,mdp]%*%w.table[,,mdp]
    pred_prob1[,7,mdp] <- pred_probs1[,,mdp]%*%w.OAP[,mdp]
    pred_prob1[,8,mdp] <- pred_probs1[,,mdp]%*%w0
  }
  
  for (mdp in 1:3) {
    storelossp[(md_p-7):md_p,mdp] <- colMeans((sweep(-pred_prob1[,,mdp],1,probst,"+"))^2)
    pred_y <- pred_prob1[,,mdp]>=0.5
    if(any(pred_prob1[,,mdp]==0.5,na.rm=T)){
      pred_y[pred_prob1[,,mdp]==0.5&!is.na(pred_prob1[,,mdp])] <- sample(c(FALSE,TRUE),sum(pred_prob1[,,mdp]==0.5),replace=T)
    }
    storelossaac[(md_p-7):md_p,mdp] <- colMeans(abs(sweep(-pred_y,1,Yst,"+")),na.rm=T)
    for (j in 1:dim(pred_prob1)[2]) {
      pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_prob1[,j,mdp]),quiet=T)
      pred_auc <- auc(pred_roc)
      storelossauc[(md_p-8)+j,mdp] <- pred_auc
    }
  }
  
  for (mdb in 1:md_b) {
    for (mdp in 1:3) {
      pred_eta <- Xst%*%betahat_all[,mdb,mdp]
      pred_prob <- 1/(1+exp(-pred_eta))
      storelossp[mdb,mdp] <- mean((probst- pred_prob)^2)
      
      if(is.na(storelossp[mdb,mdp])){
        storelossaac[mdb,mdp] <- NA
        storelossauc[mdb,mdp] <- NA
      } else {
        pred_y <- pred_prob>=0.5
        pred_y[pred_prob==0.5] <- sample(c(FALSE,TRUE),sum(pred_prob==0.5),replace=T)
        storelossaac[mdb,mdp] <- mean(abs(pred_y - Yst))
        
        pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_prob),quiet=T)
        pred_auc <- auc(pred_roc)
        storelossauc[mdb,mdp] <- pred_auc
      }
    }
  }
  ###############################
  #### Method Inverse Variance 
  ###############################
  for (mdp in 1:3) {
    pred_p_final <- vector(length = N_test)
    for (i in 1:N_test) {
      pred_eta <- Xst[i,]%*%betahat[,mdp,]
      pred_p <- 1/(1+exp(-pred_eta))
      singular_point0 <- which(pred_p==0)
      singular_point1 <- which(pred_p==1)
      pred_var <- pred_p*(1-pred_p)
      pred_var_w <- pred_var/sum(pred_var)
      if(length(singular_point1)!=0){
        pred_var_w[singular_point1] <- 1/length(singular_point1)
        pred_var_w[-singular_point1] <- 0
      } else {
        if(length(singular_point0)!=0){
          pred_var_w[singular_point0] <- 0
          if(length(singular_point0)!=length(pred_var_w)){
            pred_var_w[-singular_point0] <- pred_var_w[-singular_point0]/sum(pred_var_w[-singular_point0])
          } else {
            singular_point0_best <- which.max(pred_eta)
            pred_var_w[singular_point0_best] <- 1
            pred_var_w[-singular_point0_best] <- 0
          }
        }
      }
      pred_p_final[i] <- crossprod(as.vector(pred_p),as.vector(pred_var_w))
    }
    storelossp[md_b+1,mdp] <- mean((probst- pred_p_final)^2)
    
    pred_y <- pred_p_final>=0.5
    #if(any(pred_p_final==0.5,na.rm=T)){
    #  pred_y[pred_p_final==0.5&!is.na(pred_p_final)] <- sample(c(FALSE,TRUE),length(pred_p_final==0.5),replace=T)
    #}
    storelossaac[md_b+1,mdp] <- mean(abs(pred_y - Yst),na.rm=T)
    
    pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_p_final),quiet=T)
    pred_auc <- auc(pred_roc)
    storelossauc[md_b+1,mdp] <- pred_auc
  }
  
  sfCat(paste0("Replicate ",g,": Inverse method completes"), sep="\n")
  
  ###############################
  #### Method others
  ###############################
  #### single site
  pred_prob <- matrix(0,nrow(Xst),md_p-8-md_b-1)
  pred_probs <- array(0,c(nrow(Xst),M))
  for (ml in 1:md_ml) {
    for (ml2 in ml_partition[[ml]]) {
      if(ml==1){pred_probs[,ml2] <- pred_probs1[,ml2,2]}
      if(ml==2){pred_probs[,ml2] <- modelsMLP[[ml2]] %>% predict(Xst[,-1])}
      if(ml==3){pred_probs[,ml2] <- modelsMLP2[[ml2]] %>% predict(Xst[,-1])}
      if(ml==4){pred_probs[,ml2] <- predict(modelsxg[[ml2]],newdata=Xst[,-1])}
      if(ml==5){pred_probs[,ml2] <- predict(modelsxg2[[ml2]],newdata=Xst[,-1])}
    }
  }

  #### Full MLP/xgboost
  pred_prob[,1] <- NeuralFull %>% predict(Xst[,-1])
  pred_prob[,2] <- NeuralFull2 %>% predict(Xst[,-1])
  pred_prob[,3] <- predict(xgbFull,newdata=Xst[,-1])
  pred_prob[,4] <- predict(xgbFull2,newdata=Xst[,-1])
  #### Partial (take the est at the first site)
  pred_prob[,5:8] <- pred_probs[,unname(sapply(ml_partition, function(x) x[1]))[-1]]
  #### OAPR/OAP for mixed learners
  pred_prob[,9:14] <- pred_probs[,]%*%wo.OAPR[,,1]
  pred_prob[,15] <- pred_probs[,]%*%wo.OAP[,1]
  pred_prob[,16] <- rowMeans(pred_probs[,])
  
  #Two evaluation criteria: probst for Risk, and Yst for accuracy
  storelossp[(md_b+2):(md_p-8),3] <- colMeans((sweep(-pred_prob,1,probst,"+"))^2)
  pred_y <- pred_prob>=0.5
  if(any(pred_prob==0.5,na.rm=T)){
    pred_y[pred_prob==0.5&!is.na(pred_prob)] <- sample(c(FALSE,TRUE),sum(pred_prob==0.5),replace=T)
  }
  storelossaac[(md_b+2):(md_p-8),3] <- colMeans(abs(sweep(-pred_y,1,Yst,"+")),na.rm=T)
  for (j in 1:(md_p-8-md_b-1)) {
    pred_roc <- roc(factor(Yst,levels = c(0,1)),as.numeric(pred_prob[,j]),quiet=T)
    pred_auc <- auc(pred_roc)
    storelossauc[(md_b+1)+j,3] <- pred_auc
  }
  
  dimnames(storelossp) <- dimnames(storelossaac) <- dimnames(storelossauc) <- list(md_p_name,c("scad","lasso","glm"))
  
  timetable <- c(time.fitting[3],time.shapley[3],time.cluster[3],time.weight.OAPR[3],
                 time.weight.OAPR.ml[3],time.weight.OAP[3],time.weight.OAP.ml[3],time.Full[3],
                 time.meta[3],time.CEASEa[3],time.CEASE0[3],time.CEASEsa[3],time.CSL[3],time.mlpcv[3],time.mlp2cv[3])
  names(timetable) <- c('fitting','shapley','cluster','weight.OAPR','weight.OAPR.ml',
                        'weight.OAP','weight.OAP.ml','Full','meta','CEASEa',
                        'CEASE0','CEASEsa','CSL','mlpcv','mlp2cv')
  
  return(list(storelossb=storelossb,storelossp=storelossp,storelossaac=storelossaac,
              storelossauc=storelossauc,zeror=zeror,zerow=zerow,timetable=timetable))
}

###############################
#### Load library in parallel way (using snowfall)
###############################
sfSetMaxCPUs(cpunum)


###############################
#### Compute the statistics in situations 
#### with different d and n1
###############################

for (d0ind in 1:length(d0s)) {
  for (dind in 1:length(ds)) {
    for (decayi in 1:length(decays)) {
      for (sigi in 1:length(sigs)){
        for (nsi in 1:length(ns)) {
          sfInit(parallel = T,cpus = cpunum[nsi],slaveOutfile = "sim_fan1.txt")
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
          sfLibrary(pROC)
          sfLibrary(CVXR)
          sfLibrary(osqp)
          sfLibrary(shapviz)
          sfSource("coef_ncvreg.R")
          sfSource("lam_names.R")
          sfSource("predict_ncvreg.R")
          sfSource("OARFISH_solver.R")
          sfSource("logistic_solvers_forR.R")
          sfSource("meta.R")
          sfExport('predictfun.glm')
          sfExport('md_basic')
          sfExport('md_b')
          sfExport('md_p')
          sfExport('md_a')
          sfExport('md_z')
          sfExport('N')
          sfExport('ns')
          sfExport('by')
          sfExport('G')
          sfExport('Kfold')
          sfExport('N_test')
          
          d0 <- d0s[d0ind]
          d <- (d0-1)*(ds[dind])+1
          decay <- decays[decayi]
          sig <- sigs[sigi]
          n1 <- ns[nsi]
          M <- round(N/n1)
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
          storelossp <-matrix(nrow=md_p,ncol=md_basic,dimnames = list(NULL,c("scad","lasso","glm")))
          storelossaac <-matrix(nrow=md_a,ncol=md_basic,dimnames=list(NULL,c("scad","lasso","glm")))
          storelossb <-matrix(nrow=md_b,ncol=md_basic,dimnames = list(NULL,c("scad","lasso","glm")))
          storelossauc <-matrix(nrow=md_a,ncol=md_basic,dimnames=list(NULL,c("scad","lasso","glm")))
          zeror <- matrix(nrow=md_z,ncol=md_basic-1,dimnames = list(NULL,c("scad","lasso")))
          zerow <- matrix(nrow=md_z,ncol=md_basic-1,dimnames = list(NULL,c("scad","lasso")))
          
          
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
          sfExport('md_p_name')
          sfExport('MLP_hidden1','MLP_hidden2')
          sfExport('md_ml')
          sfExport('xgb_nrounds1','xgb_nrounds2')
          sfExport('xgb_maxdepth1','xgb_maxdepth2')
          
          #Store1 <- sfSapply(1:cpunum,simfunc)#warm start for 'proc.time'
          ptm <- proc.time()
          Store <- sfSapply(1:G,simfunc)
          timecost.whole <- proc.time()-ptm
          timecost.whole <- round(timecost.whole,3)
          
          msebM[nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[1,])/length(Store[1,])
          msepM[nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[2,])/length(Store[2,])
          accM[nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[3,])/length(Store[3,])
          aucM[nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[4,])/length(Store[4,])
          msebM.sd[nsi,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[1,],function(X){(X-msebM[nsi,,,d0ind,dind,decayi,sigi])^2}))/(length(Store[1,])-1))
          msepM.sd[nsi,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[2,],function(X){(X-msepM[nsi,,,d0ind,dind,decayi,sigi])^2}))/(length(Store[2,])-1))
          accM.sd[nsi,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[3,],function(X){(X-accM[nsi,,,d0ind,dind,decayi,sigi])^2}))/(length(Store[3,])-1))
          aucM.sd[nsi,,,d0ind,dind,decayi,sigi]<-sqrt(Reduce("+",lapply(Store[4,],function(X){(X-aucM[nsi,,,d0ind,dind,decayi,sigi])^2}))/(length(Store[4,])-1))
          zerorightM[nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[5,])/length(Store[5,])
          zerowrongM[nsi,,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[6,])/length(Store[6,])
          timecostMd[nsi,,d0ind,dind,decayi,sigi]<-Reduce("+",Store[7,])/length(Store[7,])
          timecostM[nsi,,d0ind,dind,decayi,sigi]<-timecost.whole[1:3]
          
          
          print(paste0("sitesamplesize=",n1," sitenumbers=",M," d=",d))
          print(paste0(names(timecost.whole[1]),"=",timecost.whole[1],",",names(timecost.whole[2]),"=",timecost.whole[2],",",names(timecost.whole[3]),"=",timecost.whole[3]))
          #the best way to avoid crush: save your data!
          save.image(file = paste("sim1_global","_G_",G,"_cpu_",cpunum[nsi],"_d0_",d0,"_d_",d,"_M_",M,"_decay_",decayi,".Rdata",sep=""))
          sfStop()
        }
      }
    }
  }
}
sfStop()



