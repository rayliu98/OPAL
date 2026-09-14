# Author: Ray Liu
###############################
#### install/load library
###############################
#setwd("~/DCGLM/sim_fan/Fashion_MNIST")
load("~/Fashion_MNIST_Image.Rdata")
#library(snowfall)
library(ncvreg)
library(glmnet)
library(ncpen)
library(xgboost)
library(reticulate)
library(keras3)
library(tensorflow)
library(shapviz)
library(ClusterR)
library(caret)
#library(Rsolnp)
library(CVXR)
library(osqp)
library(pROC)
#library(glm2)
source("coef_ncvreg.R")
source("lam_names.R")
source("predict_ncvreg.R")
source("OARFISH_solver.R")
source("logistic_solvers_forR.R")
source("ci_OARFISH.R")
shap <- reticulate::import("shap")

###############################
#### self-defined functions
###############################
predictfun.glm <- function(beta1, newdata) {
  link(as.numeric(beta1 %*% t(as.matrix(newdata))))  
}

###############################
#### settings
###############################
#following (Fan et al., 2021), we set (n,m)=(1200, 10), (480, 25) and (240, 50)
N_train <- N <- 12000
N_test <- 2000
Ms <- c(50)
ns <- N_train/Ms
d <- dim(train_x)[2]
by=0
Kfold=10
md_basic=2
md_ml=5 #when base models at each site are different, how many base machine learning methods do we use?
cpunum=5

md_b_name <- c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP",
               "Partial","EW","CR","Full",paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
               paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_"))
md_ci_name <- c(paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP"),"b2",sep="_"),
                paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP"),"b1y",sep="_"),
                "Full")
md_p_name <- c(md_b_name,"Inv_var","MLPFull","MLP2Full","xgbFull","xgb2Full",
               "MLPEW","MLP2EW","xgbEW","xgb2EW","MLPp","MLP2p","xgbp","xgb2p",
               paste0(c("MLP"),c("logM","elbow","distortion",paste("fixed",1:3,sep="_"))),
               paste0(c("MLP2"),c("logM","elbow","distortion",paste("fixed",1:3,sep="_"))),
               paste0(c("xgb"),c("logM","elbow","distortion",paste("fixed",1:3,sep="_"))),
               paste0(c("xgb2"),c("logM","elbow","distortion",paste("fixed",1:3,sep="_"))),
               paste0(c("MLP","MLP2","xgb","xgb2"),"OAP"),
               paste0(c("mix"),c("logM","elbow","distortion",paste("fixed",1:3,sep="_"))),
               paste0(c("mix"),c("OAP","EW")),
               paste(c("logM","elbow","distortion",paste("fixed",1:3,sep="_"),"OAP",'EW'),'p',sep="_"))

M_name <- paste("M",Ms,sep="_")
md_basic_name <- c("scad","lasso")
d_name <- paste("d",d,sep="_")

md_b=length(md_b_name)
md_p=length(md_p_name)
md_a=length(md_p_name)
md_ci=length(md_ci_name)

#prediction loss for listed methods
storelossaac <-matrix(nrow=md_a,ncol=md_basic,dimnames=list(NULL,c("scad","lasso")))
storelossauc <-matrix(nrow=md_a,ncol=md_basic,dimnames=list(NULL,c("scad","lasso")))
cihypo <- array(dim=c(d,md_ci,md_basic),dimnames=list(NULL,NULL,c("scad","lasso")))
cilen <- array(dim=c(d,md_ci,md_basic),dimnames=list(NULL,NULL,c("scad","lasso")))
#zero <- matrix(nrow=md_z,ncol=md_basic,dimnames = list(NULL,c("scad","lasso")))

# specific model parameter
xgb_nrounds1=2
xgb_maxdepth1=6
xgb_nrounds2=8
xgb_maxdepth2=3
scad.imp=lasso.imp=50

for (nsi in 1:length(Ms)) {
  set.seed(1924)
  ptm1 <- proc.time()
  
  M=Ms[nsi]
  n1=ns[nsi]
  nn<-seq(n1, by=by,length.out=M) - by*(M-1)/2
  XX <- list()
  XX_nn <- list()
  YY <- list()
  Obs_p <- list()
  train_partition <- createFolds(train_y,k=M)
  
  print(paste0("M = ",M,": starts"), sep="\n")
  
  ptm=proc.time()
  for (j in 1:M) {
    Xm <- train_x[train_partition[[j]],]
    Xm_nn <- train_x_nn[train_partition[[j]],,]
    Ym <- train_y[train_partition[[j]]]
    obs_partition <- createFolds(Ym,k=Kfold)
    
    Obs_p[[j]]<- obs_partition
    XX[[j]] <- Xm
    XX_nn[[j]] <- Xm_nn
    YY[[j]] <- Ym
  }
  
  ###############################
  #### CNN/CNN2 
  ###############################
  py_save_object(r_to_py(XX_nn),"CNNX.pkl")
  py_save_object(r_to_py(YY),"CNNY.pkl")
  py_save_object(r_to_py(Obs_p),"CNNObsp.pkl")
  py_save_object(r_to_py(test_x_nn),"CNNXtest.pkl")
  py_save_object(r_to_py(test_y),"CNNYtest.pkl")
  
  ptm=proc.time()
  py_run_file("distillation.py")
  cnn_arrays=py_load_object("cnn_arrays.pkl")
  time.cnn=proc.time()-ptm
  
  betahat<-array(0,c(d,md_basic,M))
  MUtildecv <- list()
  length(MUtildecv) <- md_basic*M
  dim(MUtildecv) <- c(md_basic,M) 
  for (md in 1:md_basic) {
    for (j in 1:M) {
      MUtildecv[[md,j]] <- vector(length = nn[j])
    }
  }
  
  mu_mat_lasso=array(dim = c(M,n1,M))
  mu_test_lasso=array(dim = c(M,length(test_y)))
  mu_mat_all=array(dim = c(5,M,n1,M))
  mu_test_all=array(dim = c(5,M,length(test_y)))
  mu_mat_mix=array(dim = c(M,n1,M))
  mu_test_mix=array(dim = c(M,length(test_y)))
  
  Error <- list()
  Lambda_series.scad <- vector(length=M)
  modelsxg2 <- list()
  modelsxg <- list()
  MUtildecvxg <- list()
  MUtildecvxg2 <- list()
  length(MUtildecvxg)=length(MUtildecvxg2)=M
  for (j in 1:M) {
    MUtildecvxg2[[j]] <- vector(length = nn[j])
    MUtildecvxg[[j]] <- vector(length = nn[j])
  }
  # randomly dsitribute methods to local site
  shuffled_ml <- 1:M
  ml_partition <- split(shuffled_ml,cut(1:M, md_ml, labels = FALSE))
  
  for (j in 1:M) {
    Xm <- XX[[j]]
    Ym <- YY[[j]]
    obs_partition <- Obs_p[[j]]
    
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
    for (l in 1:M) {
      if(l!=j){
        mu_mat_all[1,j,,l]=link(XX[[l]]%*% betahat[,1,j])
      } 
    }
    mu_test_all[1,j,]=link(test_x%*%betahat[,1,j])
    
    lambda_series.scad <- scad_mod$lambda
    for (k in 1:Kfold) {
      scad_mod_cv <- tryCatch(ncpen(x.mat=Xm[-obs_partition[[k]],-1],y.vec=Ym[-obs_partition[[k]]],
                                    family="binomial", penalty="scad",x.standardize=T, intercept=T,
                                    lambda = lambda_series.scad),error=function(e)print(e))
      if(!is.null(scad_mod_cv$message)){
        MUtildecv[[1,j]][obs_partition[[k]]] <- link(Xm[obs_partition[[k]],]%*%betahat[,1,j])
        Error <- c(Error,scad_mod_cv$message)
      } else {
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
    }
    mu_mat_all[1,j,,j]=MUtildecv[[1,j]]
    
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
    for (l in 1:M) {
      if(l!=j){
        mu_mat_lasso[j,,l]=link(XX[[l]]%*% betahat[,2,j])
      } 
    }
    
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
    mu_mat_lasso[j,,j]=MUtildecv[[2,j]]
    
    ###############################
    #### xgboost 
    ###############################
    param <- list(max_depth = xgb_maxdepth1, eta = 1, nthread = 2, 
                  objective = "binary:logistic", eval_metric = "error")
    xgbm <- xgboost(data=as.matrix(Xm[,-1]),label = Ym,params = param,nrounds = xgb_nrounds1,verbose = 0)
    #xgb.dump(xgbm)
    modelsxg[[j]] <- xgbm
    
    for (l in 1:M) {
      if(l!=j){
        mu_mat_all[4,j,,l]=predict(modelsxg[[j]],newdata=XX[[l]][,-1])
      } 
    }
    mu_test_all[4,j,]=predict(modelsxg[[j]],newdata=test_x[,-1])
    for (k in 1:Kfold) {
      xgbmcv <- xgboost(data=as.matrix(Xm[-obs_partition[[k]],-1]),label = Ym[-obs_partition[[k]]],params = param,nrounds = xgb_nrounds1,verbose = 0)
      MUtildecvxg[[j]][obs_partition[[k]]] <- predict(xgbmcv,newdata=Xm[obs_partition[[k]],-1])
    }
    mu_mat_all[4,j,,j]=MUtildecvxg[[j]]
    
    ###############################
    #### xgboost2 
    ###############################
    param2 <- list(max_depth =xgb_maxdepth2, eta = 1, nthread = 2, 
                   objective = "binary:logistic", eval_metric = "error")
    xgbm2 <- xgboost(data=as.matrix(Xm[,-1]),label = Ym,params = param2,nrounds = xgb_nrounds2,verbose = 0)
    #xgb.dump(xgbm2)
    modelsxg2[[j]] <- xgbm2
    for (l in 1:M) {
      if(l!=j){
        mu_mat_all[5,j,,l]=predict(modelsxg2[[j]],newdata=XX[[l]][,-1])
      } 
    }
    mu_test_all[5,j,]=predict(modelsxg2[[j]],newdata=test_x[,-1])
    for (k in 1:Kfold) {
      xgbmcv <- xgboost(data=as.matrix(Xm[-obs_partition[[k]],-1]),label = Ym[-obs_partition[[k]]],params = param2,nrounds = xgb_nrounds2,verbose = 0)
      MUtildecvxg2[[j]][obs_partition[[k]]] <- predict(xgbmcv,newdata=Xm[obs_partition[[k]],-1])
    }
    mu_mat_all[5,j,,j]=MUtildecvxg2[[j]]
  }
  
  ### fill mu_cnn into mu_mat
  mu_mat_all[2:3,,,]=cnn_arrays$mu_mat
  mu_test_all[2:3,,]=cnn_arrays$mu_test
  time.fitting=proc.time()-ptm
  print(paste0("M = ",M,": model fitting at local sites completes"), sep="\n")
  
  ### Compute the shapley value
  ### five methods are randomly and equally distributed to M sites
  ptm=proc.time()
  shapleyrecord <- array(0,c(M,d-1,md_ml))
  shapleyrecord_mix <- array(0,c(M,d-1))
  
  for (ml in 1:md_ml) {
    for (j in 1:M) {
      if(ml==1){
        # we choose lasso because:
        # super sparse for scad, lasso can extract more information on support
        lasso.nonzero=betahat[,2,j]!=0
        custom_predict <- function(newdata) {
          predictfun.glm(beta1 = betahat[,2,j][lasso.nonzero], newdata = newdata)
        }
        explainer_glm <- shap$Explainer(custom_predict,XX[[j]][,lasso.nonzero])
        shap_values_glm=tryCatch(explainer_glm(XX[[j]][1:100,lasso.nonzero]),
                                 warning = function(w)print(w),error=function(e)print(e))
        #if(grepl("ValueError",as.character(shap_values_glm$add_note))[1]){
        #  shap_values_glm=tryCatch(explainer_glm(XX[[j]][1:100,lasso.nonzero],max_evals = as.integer(sum(lasso.nonzero)*2+1)),
        #                           warning = function(w)print(w),error=function(e)print(e))
        #}
        shapleyrecord[j,lasso.nonzero[-1],ml]=apply(shap_values_glm$values[,-1],2,mean)
      }
      if(ml==4){
        shapleyrecord[j,,ml]=apply(as.matrix(shapviz(modelsxg[[j]],XX[[j]][1:100,-1])$S),2,mean)
      }
      if(ml==5){
        shapleyrecord[j,,ml]=apply(as.matrix(shapviz(modelsxg2[[j]],XX[[j]][1:100,-1])$S),2,mean)
      }
    }
  }
  
  keep_pixel=dimnames(train_x)[[2]][-1]
  shap_cnn_n=cnn_arrays$shap_cnn %>% array_reshape(c(2,M,28*28))
  for (md in 2:3) {
    shap_cnn1=shap_cnn_n[md-1,,]
    dimnames(shap_cnn1)=list(NULL,colname_pixel)
    shap_cnn1=shap_cnn1[,keep_pixel]
    shapleyrecord[,,md]=shap_cnn1
  }
  
  for (ml in 1:md_ml) {
    for (ml2 in ml_partition[[ml]]) {
      shapleyrecord_mix[ml2,]=shapleyrecord[ml2,,ml]
      mu_mat_mix[ml2,,]=mu_mat_all[ml,ml2,,]
      mu_test_mix[ml2,]=mu_test_all[ml,ml2,]
    }
  }
  
  ### if CNN/xgb is constructed at some sites, we use the variables corresponding 
  ### to top largest Shapley value for the dimension reduction of 
  ### Shapley value and subsequent clustering analysis.
  
  cnn1.shap.ind=order(abs(apply(matrix(shapleyrecord[,,2],M),2,mean)),decreasing=T)[1:lasso.imp]
  cnn2.shap.ind=order(abs(apply(matrix(shapleyrecord[,,3],M),2,mean)),decreasing=T)[1:lasso.imp]
  xgb1.shap.ind=order(abs(apply(matrix(shapleyrecord[,,4],M),2,mean)),decreasing=T)[1:lasso.imp]
  xgb2.shap.ind=order(abs(apply(matrix(shapleyrecord[,,5],M),2,mean)),decreasing=T)[1:lasso.imp]
  
  ### if mixed is used, we only use those sites with lasso to filter the 
  ### important features.
  
  lasso.shap.ind=order(abs(apply(matrix(shapleyrecord[1:(M/md_ml),,1],M/md_ml),2,mean)),decreasing=T)[1:lasso.imp]
  time.shapley=proc.time()-ptm
  print(paste0("M = ",M,": Shapley computation completes"), sep="\n")
  ###############################
  #### Clustering
  ###############################
  #### K-selection
  # Temporarily we consider three schemes of selecting K: (1) K=log(M); 
  # (2) elbow method; (3) distortion method.
  # Before applying K-means, betahat is standardized.
  ptm=proc.time()
  K_logM <- round(log(M))
  M_for_Kmeans <- min(M-2,100)
  K_fixed <- c(round(M_for_Kmeans/4),round(M_for_Kmeans/2),M_for_Kmeans)
  
  ###############################
  #### glm (cluster index: coefficient)
  ###############################
  betahat_s <- array(0,c(M,d,md_basic))#save the scaled betahat: scad, lasso, glm
  
  K.table <- matrix(0,3+length(K_fixed),md_basic)
  for (md in 1:md_basic) {
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
    c("scad","lasso"))
  
  ###############################
  #### CNN//xgboost/mix (cluster index: shapley)
  ###############################
  shapleyCNN_s=scale(drop0(shapleyrecord[,cnn1.shap.ind,2]))
  shapleyCNN2_s=scale(drop0(shapleyrecord[,cnn2.shap.ind,3]))
  shapleyxgb_s=scale(drop0(shapleyrecord[,xgb1.shap.ind,4]))
  shapleyxgb2_s=scale(drop0(shapleyrecord[,xgb2.shap.ind,5]))
  shapleymix_s=scale(drop0(shapleyrecord_mix[,lasso.shap.ind]))
  shapley_s=list(shapleyCNN_s,shapleyCNN2_s,shapleyxgb_s,
                 shapleyxgb2_s,shapleymix_s)
  
  K.table.ml <- array(0,c(3+length(K_fixed),5),list(
    c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
    c("MLP","MLP2","xgb","xgb2","mix")))
  
  for (md in 1:5) {
    K.table.ml[2,md] <- K_selection(shapley_s[[md]],"elbow",M-2)
    K.table.ml[3,md] <- K_selection(shapley_s[[md]],"distortion",M-2)
    #ClusterR::Optimal_Clusters_KMeans(shapley_s[[md]],3)
    #ClusterR::Optimal_Clusters_KMeans(shapleyCNN_s,3)
    #ClusterR::Optimal_Clusters_KMeans(shapleyCNN2_s,18)
    #ClusterR::Optimal_Clusters_KMeans(shapleyxgb_s,18)
    #ClusterR::Optimal_Clusters_KMeans(shapleyxgb2_s,18)
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
  kmeans_clusters <- array(0,c(M,3+length(K_fixed),md_basic),
                           dimnames = list(NULL, c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                           c("scad","lasso")))
  for (kseries in 1:(3+length(K_fixed))) {
    for (md in 1:md_basic) {
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
  #### CNN/xgboost/mix
  ###############################
  kmeans_other_clusters <- array(0,c(M,3+length(K_fixed),5),
                                 dimnames = list(NULL,c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                                 c("MLP","MLP2","xgb","xgb2","mix")))
  for (kseries in 1:(3+length(K_fixed))) {
    for (md in 1:5) {
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
  print(paste0("M = ",M,": Kmeans completes"), sep="\n")
  
  ###############################
  #### Method OAPR 
  ###############################
  ###############################
  #### glm
  ###############################
  ptm=proc.time()
  betahatOAPR <- array(0,c(d,3+length(K_fixed),md_basic),dimnames = list(NULL,
                                                                         c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                                                         c("scad","lasso")))
  iter.center_opt <- array(0,c(3+length(K_fixed),md_basic),dimnames = 
                             list(c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                                  c("scad","lasso")))
  # Record the estimated weights
  w.table <- array(0,c(M,3+length(K_fixed),md_basic),dimnames = 
                     list(NULL,c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                          c("scad","lasso")))
  # only construct ci for scad
  ci_betahatOAPR_scad.r <- array(0,c(d,3+length(K_fixed),2,2))
  # dimension, kmeans, lower/upper bound, Omega1_est
  
  for (md in 1:md_basic) {
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
  #### CNN/xgboost/mix
  ###############################
  # Record the estimated weights
  ptm=proc.time()
  wo.OAPR <- array(0,c(M,3+length(K_fixed),5),
                   list(NULL,
                        c(paste("logM",K_logM),"elbow","distortion",paste("fixed",K_fixed)),
                        c("MLP","MLP2","xgb","xgb2","mix")))
  
  for (md in 1:5) {
    for (kseries in 1:(3+length(K_fixed))){
      if(md<5){
        res <- betahat_OARFISH_mumat(YY,mu_mat_all[md+1,,,],kmeans_other_clusters[,kseries,md])
      } else {
        res <- betahat_OARFISH_mumat(YY,mu_mat_mix,kmeans_other_clusters[,kseries,md])
      }
      wo.OAPR[,kseries,md] <- res$w
    }
  }
  time.weight.OAPR.ml=proc.time()-ptm
  
  print(paste0("M = ",M,": OAPR completes"), sep="\n")
  
  ###############################
  #### Method OAP
  ###############################
  ###############################
  #### glm
  ###############################
  ptm=proc.time()
  betahatOAP <- matrix(0,d,md_basic,dimnames = list(NULL,c("scad","lasso")))
  iter.OAP <- matrix(0,md_basic,1,dimnames = list(c("scad","lasso"),NULL))
  # Record the estimated weights
  w.OAP <- matrix(0,M,md_basic,dimnames = list(NULL,c("scad","lasso")))
  # only construct ci for scad
  ci_betahatOAP_scad.r <- array(0,c(d,2,2))
  
  for (md in 1:md_basic) {
    res <- betahat_OARFISH(XX,YY,betahat[,md,],MUtildecv[md,],1:M)
    coefs.OAP <- res$coefs
    #coefs.OAP[abs(coefs.OAP)<1e-8] <- 0
    betahatOAP[,md]=coefs.OAP # OAPR coefficient estimator with osqp solver
    iter.OAP[md,1] <- res$iter
    w.OAP[,md] <- res$w
  }
  time.weight.OAP=proc.time()-ptm
  
  ###############################
  #### CNN/xgboost/mix
  ###############################
  # Record the estimated weights
  ptm=proc.time()
  wo.OAP <- matrix(0,M,5,dimnames = 
                     list(NULL,c("MLP","MLP2","xgb","xgb2","mix")))
  for (md in 1:5) {
    if(md<5){
      res <- betahat_OARFISH_mumat(YY,mu_mat_all[md+1,,,],1:M)
    } else {
      res <- betahat_OARFISH_mumat(YY,mu_mat_mix,1:M)
    }
    wo.OAP[,md] <- res$w
  }
  time.weight.OAP.ml=proc.time()-ptm
  print(paste0("M = ",M,": OAP completes"), sep="\n")
  
  ###############################
  #### Method Equal Weight (also used as the warm start)
  ###############################
  betahatEW <- array(0,c(d,md_basic),dimnames = list(NULL,c("scad","lasso")))
  for (md in 1:md_basic) {
    w0 <- matrix(1,nrow=M,ncol=1)/M #the initial value of weights
    betahatEW[,md]=betahat[,md,]%*%w0; #EMA coefficient estimator
  }
  
  ###############################
  #### Method Cubic Rate (Shi et al.)
  ###############################
  betahatCR <- array(0,c(d,md_basic),dimnames = list(NULL,c("scad","lasso")))
  for (md in 1:md_basic) {
    wcr <- matrix(nn^(2/3)/sum(nn^(2/3)),nrow=M,ncol=1) 
    betahatCR[,md]=betahat[,md,]%*%wcr; #CR coefficient estimator
  }
  
  ###############################
  #### Method Full
  ###############################
  ptm=proc.time()
  betahatFull <- array(0,c(d,md_basic),dimnames = list(NULL,c("scad","lasso")))
  
  for (md in 1:md_basic) {
    if(md==1){
      scad_mod <- ncpen(train_y, train_x[,-1], family="binomial", penalty="scad",
                        x.standardize=T, intercept=T)
      lambda_scad.fit <- cv.ncpen(train_y, train_x[,-1], family="binomial", penalty="scad",
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
      Omega1_full <- diag(full_supp)[full_supp,]%*%Omega1_b2(betahatFull[,md],train_x)%*%t(diag(full_supp)[full_supp,])
      I1inv_full <- solve(Omega1_full + Sigma_full)
      filling_full <- diag(full_supp)[full_supp,]%*%Omega1_b1y(betahatFull[,md],train_x,train_y)%*%t(diag(full_supp)[full_supp,])
      sandwitch_full <- I1inv_full%*%filling_full%*%I1inv_full
      
      b_full <- diag(full_supp)[full_supp,]%*%bhat(betahatFull[,md],lambda = lambda_scad)
      Fhat_full <- I1inv_full%*%b_full
      
      ci_betahatFull_scad[,1] <- betahatFull[,md][full_supp] + Fhat_full - qnorm(0.05/2,lower.tail = FALSE)*diag(sandwitch_full)/sqrt(N)
      ci_betahatFull_scad[,2] <- betahatFull[,md][full_supp] + Fhat_full + qnorm(0.05/2,lower.tail = FALSE)*diag(sandwitch_full)/sqrt(N)
      
      ci_betahatFull_scad.r[,1] <- full.r%*%ci_betahatFull_scad[,1]
      ci_betahatFull_scad.r[,2] <- full.r%*%ci_betahatFull_scad[,2]
    }
    if(md==2){
      lasso_mod = glmnet(train_x[,-1], train_y, family = "binomial", intercept=T, standardize = T) 
      lambda_lasso.fit = cv.glmnet(train_x[,-1], train_y, family = "binomial" ,intercept=T, standardize=T)
      betahatFull[,md]<-as.matrix(coef(lasso_mod, s=lambda_lasso.fit$lambda.min))
    }
  }
  
  ###############################
  #### Method Full_CNN/Full_CNN2
  ###############################
  ptm2=proc.time()
  py_run_file("distillation_Full.py")
  cnn_arrays_Full=py_load_object("cnn_arrays_Full.pkl")
  time.cnn_Full=proc.time()-ptm2
  
  ###############################
  #### Method Full_xgboost 
  ###############################
  param <- list(max_depth = xgb_maxdepth1, eta = 1, nthread = 2, 
                objective = "binary:logistic", eval_metric = "error")
  xgbFull <- xgboost(data=as.matrix(train_x[,-1]),label = train_y,params = param,nrounds = xgb_nrounds1,verbose = 0)
  
  ###############################
  #### Method Full_xgboost2 
  ###############################
  param2 <- list(max_depth = xgb_maxdepth2, eta = 1, nthread = 2, 
                 objective = "binary:logistic", eval_metric = "error")
  xgbFull2 <- xgboost(data=as.matrix(train_x[,-1]),label = train_y,params = param2,nrounds = xgb_nrounds2,verbose = 0)
  
  time.Full=proc.time()-ptm
  print(paste0("M = ",M,": Full_other completes"), sep="\n")
  
  ###############################
  #### Method CEASE(a) (a is alpha)
  ###############################
  ptm=proc.time()
  betahatCEASEa <- array(0,c(d,md_basic,3),dimnames = list(NULL,c("scad","lasso"),paste0("t",1:3)))
  for (md in 1:md_basic) {
    if(md==2){
      betahatCEASEa[,md,]=logistic_DANE(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                        theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "lasso",
                                        trunconst = 1e-6,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs#maxiter=3
    }
    if(md==1){
      betahatCEASEa[,md,]=logistic_DANE(XX,YY,t=3,alpha = 0.15*d/mean(nn),
                                        theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "scad",
                                        trunconst = 0.01,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
  }
  
  ###############################
  #### Method CEASE(0)
  ###############################
  betahatCEASE0 <- array(0,c(d,md_basic,3),dimnames = list(NULL,c("scad","lasso"),paste0("t",1:3)))
  for (md in 1:md_basic) {
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
  
  ###############################
  #### Method CEASE-single(a)
  ###############################
  betahatCEASEsa <- array(0,c(d,md_basic,3),dimnames = list(NULL,c("scad","lasso"),paste0("t",1:3)))
  for (md in 1:md_basic) {
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
  
  ###############################
  #### Method CEASE-single(0) (CSL)
  ###############################
  betahatCSL <- array(0,c(d,md_basic,3),dimnames = list(NULL,c("scad","lasso"),paste0("t",1:3)))
  for (md in 1:md_basic) {
    if(md==2){
      betahatCSL[,md,]=logistic_CSL1(XX,YY,t=3, theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "lasso",
                                     trunconst = 1e-6,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
    if(md==1){
      betahatCSL[,md,]=logistic_CSL1(XX,YY,t=3, theta_initial = betahatEW[,md], lambda=.5*sqrt(log(d)/sum(nn)), penalty = "scad",
                                     trunconst = 0.01,learning_rate=0.01, maxiter=3, tol=1e-3)$coefs
    }
  }
  time.CSLfamily=proc.time()-ptm
  print(paste0("M = ",M,": CSLfamily completes"), sep="\n")
  
  #### integrate all betahat into a large array
  md_b_name <- c(paste("logM",K_logM,sep="_"),"elbow","distortion",paste("fixed",K_fixed,sep="_"),"OAP",
                 "Partial","EW","CR","Full",paste("CEASEa",1:3,sep="_"),paste("CEASE0",1:3,sep="_"),
                 paste("CEASEsa",1:3,sep="_"),paste("CSL",1:3,sep="_"))
  md_ci_name <- c(paste(c(paste("logM",K_logM,sep="_"),"elbow","distortion",paste("fixed",K_fixed,sep="_"),"OAP"),"b2",sep="_"),
                  paste(c(paste("logM",K_logM,sep="_"),"elbow","distortion",paste("fixed",K_fixed,sep="_"),"OAP"),"b1y",sep="_"),
                  "Full")
  betahat_all <- array(dim = c(d,md_b,md_basic),dimnames = list(NULL,md_b_name,c("scad","lasso")))
  betahatci_all <- array(dim = c(d,md_ci,md_basic,2),dimnames = list(NULL,md_ci_name,c("scad","lasso"),c("l","u")))
  betahathypo_all <- array(dim = c(d,md_ci,md_basic),dimnames = list(NULL,md_ci_name,c("scad","lasso")))
  
  for (md in 1:md_basic) {
    # fill betahat_all
    betahat_all[,1:(3+length(K_fixed)),md] <- betahatOAPR[,,md] # OAPR
    betahat_all[,(3+length(K_fixed))+1,md] <- betahatOAP[,md] # OAP
    betahat_all[,(3+length(K_fixed))+2,md] <- betahat[,md,1] # Partial
    betahat_all[,(3+length(K_fixed))+3,md] <- betahatEW[,md] # EW
    betahat_all[,(3+length(K_fixed))+4,md] <- betahatCR[,md] # CR
    betahat_all[,(3+length(K_fixed))+5,md] <- betahatFull[,md] # Full
    betahat_all[,(3+length(K_fixed))+6:8,md] <- betahatCEASEa[,md,] # CEASEa
    betahat_all[,(3+length(K_fixed))+9:11,md] <- betahatCEASE0[,md,] # CEASE0
    betahat_all[,(3+length(K_fixed))+12:14,md] <- betahatCEASEsa[,md,] # CEASEsa
    betahat_all[,(3+length(K_fixed))+15:17,md] <- betahatCSL[,md,] # CSL
    
    # fill betahathypo_all
    if(md==1){
      betahathypo_all[,1:(3+length(K_fixed)),md] <- betahatOAPR[,,md] # OAPR_b2
      betahathypo_all[,(3+length(K_fixed))+1,md] <- betahatOAP[,md] # OAP_b2
      betahathypo_all[,(3+length(K_fixed))+1+1:(3+length(K_fixed)),md] <- betahatOAPR[,,md] # OAPR_b1y
      betahathypo_all[,((3+length(K_fixed))+1)*2,md] <- betahatOAP[,md] # OAP_b1y
      betahathypo_all[,((3+length(K_fixed))+1)*2+1,md] <- betahatFull[,md] # Full
    }
    
    # fill betahatcil_all/betahatciu_all
    for (lu in 1:2) {
      if(md==1){
        betahatci_all[,1:(3+length(K_fixed)),md,lu] <- ci_betahatOAPR_scad.r[,,lu,1] # OAPR_b2
        betahatci_all[,(3+length(K_fixed))+1,md,lu] <- ci_betahatOAP_scad.r[,lu,1] # OAP_b2
        betahatci_all[,(3+length(K_fixed))+1+1:(3+length(K_fixed)),md,lu] <- ci_betahatOAPR_scad.r[,,lu,2] # OAPR_b1y
        betahatci_all[,((3+length(K_fixed))+1)*2,md,lu] <- ci_betahatOAP_scad.r[,lu,2] # OAP_b1y
        betahatci_all[,((3+length(K_fixed))+1)*2+1,md,lu] <- ci_betahatFull_scad.r[,lu] # Full
      }
    }
  }
  
  ###############################
  #### Length of confidence interval &
  #### Whether CI covers the estimator
  ###############################
  cilen <- apply(betahatci_all,c(2,3),function(x){x[,2]-x[,1]})
  cihypo <- (betahatci_all[,,,1]<=betahathypo_all)&(betahatci_all[,,,2]>=betahathypo_all)
  
  ###############################
  #### Squared prediction error of test data
  ###############################
  ###############################
  #### Predictive accuracy (Bayes criterion: if p^hat_i>0.5, y^hat_i=1) &
  #### Predictive auc
  ###############################
  pred_prob1 <- array(0,c(nrow(test_x),8,3))
  pred_probs1 <- array(0,c(nrow(test_x),M,3))
  pred_probs1[,,1] <- apply(test_x%*%betahat[,1,],2,link)
  pred_probs1[,,2] <- apply(test_x%*%betahat[,2,],2,link)
  for (mdp in 1:md_basic) {
    pred_prob1[,1:6,mdp] <- pred_probs1[,,mdp]%*%w.table[,,mdp]
    pred_prob1[,7,mdp] <- pred_probs1[,,mdp]%*%w.OAP[,mdp]
    pred_prob1[,8,mdp] <- pred_probs1[,,mdp]%*%w0
  }
  
  for (mdp in 1:md_basic) {
    pred_y <- pred_prob1[,,mdp]>=0.5
    if(any(pred_prob1[,,mdp]==0.5,na.rm=T)){
      pred_y[pred_prob1[,,mdp]==0.5&!is.na(pred_prob1[,,mdp])] <- sample(c(FALSE,TRUE),sum(pred_prob1[,,mdp]==0.5),replace=T)
    }
    storelossaac[(md_p-7):md_p,mdp] <- colMeans(abs(sweep(-pred_y,1,test_y,"+")),na.rm=T)
    for (j in 1:dim(pred_prob1)[2]) {
      pred_roc <- roc(factor(test_y,levels = c(0,1)),as.numeric(pred_prob1[,j,mdp]),quiet=T)
      pred_auc <- auc(pred_roc)
      storelossauc[(md_p-8)+j,mdp] <- pred_auc
    }
  }
  
  for (mdb in 1:md_b) {
    for (mdp in 1:md_basic) {
      pred_eta <- test_x%*%betahat_all[,mdb,mdp]
      pred_prob <- 1/(1+exp(-pred_eta))
      
      pred_y <- pred_prob>=0.5
      pred_y[pred_prob==0.5] <- sample(c(FALSE,TRUE),sum(pred_prob==0.5),replace=T)
      storelossaac[mdb,mdp] <- mean(abs(pred_y - test_y))
      
      pred_roc <- roc(factor(test_y,levels = c(0,1)),as.numeric(pred_prob),quiet=T)
      pred_auc <- auc(pred_roc)
      storelossauc[mdb,mdp] <- pred_auc
    }
  }
  ###############################
  #### Method Inverse Variance 
  ###############################
  for (mdp in 1:md_basic) {
    pred_p_final <- vector(length = N_test)
    for (i in 1:N_test) {
      pred_eta <- test_x[i,]%*%betahat[,mdp,]
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
    
    pred_y <- pred_p_final>=0.5
    if(any(pred_p_final==0.5,na.rm=T)){
      pred_y[pred_p_final==0.5&!is.na(pred_p_final)] <- sample(c(FALSE,TRUE),sum(pred_p_final==0.5),replace=T)
    }
    storelossaac[length(md_b_name)+1,mdp] <- mean(abs(pred_y - test_y),na.rm=T)
    
    pred_roc <- roc(factor(test_y,levels = c(0,1)),as.numeric(pred_p_final),quiet=T)
    pred_auc <- auc(pred_roc)
    storelossauc[length(md_b_name)+1,mdp] <- pred_auc
  }
  print(paste0("M = ",M,": Inverse method completes"), sep="\n")
  
  ###############################
  #### Method others
  ###############################
  #### single site
  pred_prob <- matrix(0,nrow(test_x),md_p-8-md_b-1)#remove prob_ave, beta_ave and Inv methods
  pred_probs <- array(0,c(nrow(test_x),M,5))
  for (j in 1:M) {
    for (md in 2:md_ml) {
      pred_probs[,j,md-1]=mu_test_all[md,j,]
    }
    pred_probs[,j,5]=mu_test_mix[j,]
  }
  
  #### Full CNN/xgboost
  pred_prob[,1] <- cnn_arrays_Full$mu_test[1,]
  pred_prob[,2] <- cnn_arrays_Full$mu_test[2,]
  pred_prob[,3] <- predict(xgbFull,newdata=test_x[,-1])
  pred_prob[,4] <- predict(xgbFull2,newdata=test_x[,-1])
  #### Equal weight on prob est at single site
  pred_prob[,5:8] <- apply(pred_probs[,,2:5],3,function(X){rowMeans(X)})
  #### Partial (take the est at the first site)
  pred_prob[,9:12] <- pred_probs[,1,2:5]
  #### OAPR
  for (j in 1:4) {
    pred_prob[,13:18+6*(j-1)] <- pred_probs[,,j]%*%wo.OAPR[,,j]
  }
  #### OAP
  for (j in 1:4) {
    pred_prob[,36+j] <- pred_probs[,,j]%*%wo.OAP[,j]
  }
  #### OAPR/OAP for mixed learners
  pred_prob[,41:46] <- pred_probs[,,5]%*%wo.OAPR[,,5]
  pred_prob[,47] <- pred_probs[,,5]%*%wo.OAP[,5]
  pred_prob[,48] <- rowMeans(pred_probs[,,5])
  
  pred_y <- pred_prob>=0.5
  if(any(pred_prob==0.5,na.rm=T)){
    pred_y[pred_prob==0.5&!is.na(pred_prob)] <- sample(c(FALSE,TRUE),sum(pred_prob==0.5),replace=T)
  }
  storelossaac[(md_b+2):(md_p-8),2] <- colMeans(abs(sweep(-pred_y,1,test_y,"+")),na.rm=T)
  for (j in 1:(md_p-8-md_b-1)) {
    pred_roc <- roc(factor(test_y,levels = c(0,1)),as.numeric(pred_prob[,j]),quiet=T)
    pred_auc <- auc(pred_roc)
    storelossauc[(md_b+1)+j,2] <- pred_auc
  }
  
  dimnames(storelossaac) <- dimnames(storelossauc) <- list(md_p_name,c("scad","lasso"))
  
  timecost.whole <- proc.time()-ptm1
  timecost.whole <- round(timecost.whole,3)
  
  
  timetable <- c(time.fitting[3],time.shapley[3],time.cluster[3],time.weight.OAPR[3],
                 time.weight.OAPR.ml[3],time.weight.OAP[3],time.weight.OAP.ml[3],time.Full[3],
                 time.CSLfamily[3],time.cnn[3],time.cnn_Full[3])
  names(timetable) <- c('fitting','shapley','cluster','weight.OAPR','weight.OAPR.ml',
                        'weight.OAP','weight.OAP.ml','Full','CSLfamily',
                        'cnn','cnn_full')
  
  save.image(file = paste("Fashion_MNIST","_M_",M,".Rdata",sep="")) 
  print(paste0("M=",M,",n=",n1))
  print(paste0(names(timecost.whole[1]),"=",timecost.whole[1],",",names(timecost.whole[2]),"=",timecost.whole[2],",",names(timecost.whole[3]),"=",timecost.whole[3]))
}



