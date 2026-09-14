###############################
## Arguments:
## ***(for functions 'Omega1_b2', 'Omega1_b1y', 'Omega2', 'Omega3', 'Sigmahat' and 'bhat')
## beta: coefficients estimate, p x 1 vector
## X: design matrix, n x d matrix
## Y: response vector, n x 1 vector
## lambda: penalty in scad, scalar
## a: hyperparameter in scad, scalar
## ***(for functions 'betahat_OARFISH', 'betahat_OARFISH_mumat')
## betahatMat: coefficients estimates matrix, with the m-th row betahatm, M x p matrix
## Y_list: response list in which n x 1 vector is stored, M x 1 list
## X_list: design list in which n x d matrix is stored, list with the same dimension as Y_list
## MUtilde_arr: prediction matrix, M x n x M matrix, with the first dimension
##              the model at the mth site, the second dimension the local sample 
##              size at the mth site, and the third dimension the data at the mth
##              site to be evaluate 
## MUtilde_list: prediction on self site, M x 1 list with each one a n x 1 vector
## model_list: model at each site, M x 1 list
## cluster_vec: a vector denotes the clustering results of M sites, M x 1 vector
## ***(for function 'ci.OARFISH1')
## betahatOARFISH: coefficient estimate by 'betahat_OARFISH' function, d x 1 vector
## wOARFISH: optimal weight estimate by 'betahat_OARFISH' function, M x 1 vector
## Omega1_est: the estimating method for Omega1, character.
##             'b2': n_m^{-1}\bm{X}_{m}^\top \mathbf{D}_m^1(\hat{\bm\beta}(\tilde{\bm w})) \bm{X}_{m}^{},
##                   which is the direct plug-in estimator;
##             'b1y': \bm X_m^\top[\operatorname{vec}\{b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w})),\ldots,b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w}))\}-\bm Y_m]
##                    \cdot[\operatorname{vec}\{b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w})),\ldots,b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w}))\}-\bm Y_m]^\top\bm X_m
##                   which is the empirical sample estimator, which is used as the filling in 'sandwitch' estimator.
## method: three ways of concatenating the non-zero coefficient estimator for M sites
##         in betahatMat, character.
## first_err: first-type error to be controlled that determine the coverage of confidence interval, scalar.
## B: sampling times for constructing the empirical limiting distribution, scalar.
## lambda_series: lambda used in each site for scad estimator, M x 1 vector.
## Kfold: the folds used for cross validation within each site, scalar.
## ***(for function 'K_selection')
## criterion: method for selecting 'K' in K-means algorithm, character.
## max_K: the maximum 'K' designated for K-means algorithm, scalar.
## elbow_diff1: first order difference threshold for 'elbow' criterion in K-means algorithm, scalar.
## elbow_diff2: second order difference threshold for 'elbow' criterion in K-means algorithm, scalar.
## distortion_threshold: the threshold under which the corresponding 'K' will be 
##                       considered for 'distortion' criterion in K-means algorithm, scalar.
## distortion_ratio: a tuning for whether choosing the 'K' with minimum distortion or
##                   the largset 'K' satiefied, for 'distortion' criterion, scalar.
###############################
## Notes:
## 1. Temporarily function 'betahat_OARFISH_mumat' only support equal sample size
##    at each site.

library(MASS)
library(Matrix)
library(ClusterR)
library(osqp)
source("logistic_solvers_forR.R")

Omega1_b2 <- function(beta,X){
  t(X)%*%diag(c(exp(-X%*%beta)/(1+exp(-X%*%beta))^2))%*%X/dim(X)[1]
}

Omega1_b1y <- function(beta,X,Y){
  t(X)%*%diag(c((link(X%*%beta)-Y)^2))%*%X/dim(X)[1]-gradient(beta,Y,X)%*%t(gradient(beta,Y,X))
}

Omega2 <- function(beta,X){
  t(X)%*%diag(c((exp(-X%*%beta)/(1+exp(-X%*%beta))^2)^2))%*%X/dim(X)[1]
}

Omega3 <- function(beta,X){
  t(X)%*%diag(c((exp(-X%*%beta)/(1+exp(-X%*%beta))^2)^3))%*%X/dim(X)[1]
}

Sigmahat <- function(theta,lambda,a=3.7){
  diag(c(p2_lambda_scad(theta,lambda,a)))
}

bhat <- function(theta,lambda,a=3.7){
  as.matrix(p1_lambda_scad(theta,lambda,a)*sign(theta))
}



betahat_OARFISH <- function(X_list, Y_list, betahatMat, MUtilde_list, cluster_vec, model_list=NULL){
  M <- length(X_list)
  dim_n_p <- sapply(X_list,dim)
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])
  model=vector(length=M)
  time.em=vector(length=M)
  time.ee=vector(length=M)
  time.w=c()
  if(!is.null(model_list)){
    for (j in 1:M) {
      if(is.vector(model_list[[j]])){model[j]="vector"} else
      {model[j]=attributes(model_list[[j]])$class[1]}
    }
  }
  
  K.ind <- sort(unique(cluster_vec))
  K <- length(K.ind)
  cluster_vec_new=factor(cluster_vec,labels = 1:K)
  ee <- matrix(0,K,K)
  
  for (j in 1:M) {
    ptm=proc.time()
    Xm <- X_list[[j]]
    Ym <- Y_list[[j]]
    
    em <- matrix(0,dim_n_p[1,j],M)
    for (i in 1:M) {
      if (i!=j) {
        if(is.null(model_list)){
          testeta <- Xm%*%betahatMat[,i]
          testp <- link(testeta)
          if(sum(is.nan(testp))>0){testp[which(is.nan(testp)==1)]<-1}
        } else {
          if(model[j]=="vector"){
            testeta <- Xm%*%model_list[[j]]
            testp <- link(testeta)
            if(sum(is.nan(testp))>0){testp[which(is.nan(testp)==1)]<-1}
          }
          if(model[j]=="nn"){
            testp1 <- predict(model_list[[j]],newdata=Xm[,-1])
            num2prob <- testp1/rowSums(testp1)
            testp <- num2prob[,2]
          }
          if(model[j]=="keras.src.models.sequential.Sequential"){
            testp <- model_list[[j]] %>% predict(Xm[,-1])
          }
          if(model[j]=="randomForest"){testp <- predict(model_list[[j]],newdata=Xm[,-1],type="prob")[,2]}
          if(model[j]=="xgb.Booster"){testp <- predict(model_list[[j]],newdata=Xm[,-1])}
          if(model[j]=="lda"){testp <- predict(model_list[[j]],newdata=Xm[,-1])$posterior[,'1']}
          if(sum(is.nan(testp))>0){testp[which(is.nan(testp)==1)]<-1}
        }
      }
      if (i==j) {
        testp <- MUtilde_list[[i]]
      }
      em[,i]<-Ym-testp
    }
    time.em[j]=(proc.time()-ptm)[3]
    ptm1=proc.time()
    if(K==M){
      Bl <- em
    } else {
      Bl <- matrix(0,dim_n_p[1,j],K)
      for (kno in 1:K) {
        Bl[,kno] <- rowSums(as.matrix(em[,which(cluster_vec_new==kno)]))/M
      }
    }
    ee <- ee + t(Bl)%*%Bl
    time.ee[j]=(proc.time()-ptm1)[3]
  }
  time.ee_sum=sum(time.ee)
  time.em_max=max(time.em)
  
  ptm2=proc.time()
  kmeans_count <- table(cluster_vec_new)
  w.dim <- K
  a1 <- ee
  A <- rbind(kmeans_count,diag(1,w.dim))
  l <- c(1-0.0001,rep(0,w.dim))
  u <- c(1+0.0001,rep(1,w.dim))
  res <- osqp::solve_osqp(P=a1*2,A=A,l=l,u=u,
                    pars = osqpSettings(eps_abs=1e-8,eps_rel=1e-8,
                                        polish = TRUE,verbose = FALSE))
  wg <- res$x
  w1 <- wg[cluster_vec_new]
  w1[w1<0] <- 0
  w1 <- w1/sum(w1)
  time.w=(proc.time()-ptm2)[3]
  
  if(is.null(model_list)){
  betahatOARFISH <- betahatMat%*%w1
  iter.center_opt <- res$info$iter
  theta <- list(coefs=betahatOARFISH,iter=iter.center_opt,w=w1,
                time=time.em_max+time.ee_sum+time.w)
  } else {
  theta <- list(w=w1,time=time.em_max+time.ee_sum+time.w)
  }
  return(theta)
}

betahat_OARFISH_mumat <- function(Y_list, MUtilde_arr, cluster_vec){
  M <- dim(MUtilde_arr)[1]
  n <- dim(MUtilde_arr)[2]
  N <- M*n
  K.ind <- sort(unique(cluster_vec))
  K <- length(K.ind)
  cluster_vec_new=factor(cluster_vec,labels = 1:K)
  ee <- matrix(0,K,K)
  
  for (j in 1:M) {
    Ym <- Y_list[[j]]
    em <- matrix(Ym,n,M) - t(MUtilde_arr[,,j])
    if(K==M){
      Bl <- em
    } else {
      Bl <- matrix(0,n,K)
      for (kno in 1:K) {
        Bl[,kno] <- rowSums(as.matrix(em[,which(cluster_vec_new==kno)]))/M
      }
    }
    ee <- ee + t(Bl)%*%Bl
  }
  
  kmeans_count <- table(cluster_vec_new)
  w.dim <- K
  a1 <- ee
  A <- rbind(kmeans_count,diag(1,w.dim))
  l <- c(1-0.0001,rep(0,w.dim))
  u <- c(1+0.0001,rep(1,w.dim))
  res <- osqp::solve_osqp(P=a1*2,A=A,l=l,u=u,
                          pars = osqpSettings(eps_abs=1e-8,eps_rel=1e-8,
                                              polish = TRUE,verbose = FALSE))
  wg <- res$x
  w1 <- wg[cluster_vec_new]
  w1[w1<0] <- 0
  w1 <- w1/sum(w1)
  
  theta <- list(w=w1)
  return(theta)
}

ci.OARFISH1 <- function(X_list, Y_list, betahatOARFISH, wOARFISH, betahatMat, cluster_vec, 
                        Omega1_est=c("b2","b1y"), method=c("type1","type2","type3"),
                        first_err=0.05, B=500, lambda_series, Kfold){
  M <- length(X_list)
  dim_n_p <- sapply(X_list,dim)
  nmax <- max(dim_n_p[1,])
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])
  
  K.ind <- sort(unique(cluster_vec))
  K <- length(K.ind)
  cluster_vec_new=factor(cluster_vec,labels = 1:K)
  
  A_supp_u <- betahatOARFISH!=0
  qA_u <- sum(A_supp_u)
  if(qA_u==0)stop("all models do not have non-zero coefficient estimates")
  SMat_A_supp_u <- matrix(diag(A_supp_u)[A_supp_u,],nrow=qA_u)
  if(method!="type1"){
    A_supp <- (betahatMat!=0)&A_supp_u;qA <- apply(A_supp,2,sum)
    if(any(qA==0))stop("there exist models that coefficient estimates are all zero!")
  }
  
  Omega1_list <- I1inv_list <- I2_list <- I3_list <- Sigma_list <- b_list <- Cov <- list()
  ZL <- Zmj3 <- Zm3 <- Zm <- Zmnj <- list()
  Delta <- matrix(0,qA_u,B)
  
  for (j in 1:M) {
    I2_list[[j]] <- diag(A_supp_u)[A_supp_u,]%*%Omega2(betahatOARFISH,X_list[[j]])%*%t(diag(A_supp_u)[A_supp_u,])
    I3_list[[j]] <- diag(A_supp_u)[A_supp_u,]%*%Omega3(betahatOARFISH,X_list[[j]])%*%t(diag(A_supp_u)[A_supp_u,])
    Sigma_list[[j]] <- diag(A_supp_u)[A_supp_u,]%*%Sigmahat(betahatOARFISH,lambda = lambda_series[j])%*%t(diag(A_supp_u)[A_supp_u,])
    b_list[[j]] <- diag(A_supp_u)[A_supp_u,]%*%bhat(betahatOARFISH,lambda = lambda_series[j])
  }
  if(Omega1_est=="b2"){
    for (j in 1:M) {
      Omega1_list[[j]] <- diag(A_supp_u)[A_supp_u,]%*%Omega1_b2(betahatOARFISH,X_list[[j]])%*%t(diag(A_supp_u)[A_supp_u,])
      I1inv_list[[j]] <- solve(Omega1_list[[j]] + Sigma_list[[j]])
    }
  } else {
    for (j in 1:M) {
      Omega1_list[[j]] <- diag(A_supp_u)[A_supp_u,]%*%Omega1_b1y(betahatOARFISH,X_list[[j]],Y_list[[j]])%*%t(diag(A_supp_u)[A_supp_u,])
      I1inv_list[[j]] <- solve(Omega1_list[[j]] + Sigma_list[[j]])
    }
  }
  for (j in 1:M) {
    Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2_list[[j]]),cbind(I2_list[[j]],I3_list[[j]]))
    if(Omega1_est=="b1y"){
      Cov.eig <- eigen(Cov.tmp)
      if(sum(Cov.eig$values<=0)>0){
        Cov.tmp <- Cov.eig$vectors[,Cov.eig$values>0]%*%diag(Cov.eig$values[Cov.eig$values>0]+1e-2)%*%t(Cov.eig$vectors[,Cov.eig$values>0])
        # Cov.tmp <- Cov.tmp + diag(1e-2,2*qA_u)
      }
    }
    Cov[[j]] <- Cov.tmp
  }
  # bias matrix F
  Fhat <- mapply(function(X,y){X%*%y},I1inv_list,b_list)
  
  for (b in 1:B) {
    for (j in 1:M) {
      ZL[[j]] <- MASS::mvrnorm(Kfold,rep(0,2*qA_u),Cov[[j]])
      Zmj3[[j]] <- ZL[[j]][,(qA_u+1):(2*qA_u)]
      Zm3[[j]] <- apply(Zmj3[[j]],2,sum)/sqrt(Kfold)
      Zm[[j]] <- apply(ZL[[j]][,1:qA_u],2,sum)/sqrt(Kfold)
      zmnj_tmp <- matrix(0,Kfold,qA_u)
      for (i in 1:Kfold) {
        zmnj_tmp[i,] <- apply(matrix(ZL[[j]][-i,1:qA_u],ncol=qA_u),2,sum)/sqrt(Kfold-1)
      }
      Zmnj[[j]] <- zmnj_tmp
    }
    
    AMM <- matrix(0,M,M)
    aM <- vector(length = M)
    for (r in 1:M) {
      for (t in 1:M) {
        if(t==r){
          AMMl <- 0
          for (l in 1:M) {
            if(l!=r){AMMl <- AMMl + dim_n_p[1,l]/dim_n_p[1,r]*t(Zm[[r]])%*%I1inv_list[[r]]%*%I2_list[[l]]%*%I1inv_list[[r]]%*%Zm[[r]]}
            if(l==r){AMMl <- AMMl + sum(apply(Zmnj[[r]],1,function(x){t(x)%*%I1inv_list[[r]]%*%I2_list[[r]]%*%I1inv_list[[r]]%*%x}))/(Kfold-1)}
          }
          AMM[r,t] <- AMMl
        } else {
          AMMl <- 0
          for (l in 1:M) {
            if(l==r){
              AMMl <- AMMl + sqrt(dim_n_p[1,r]/dim_n_p[1,t])*t(apply(Zmnj[[r]],2,sum)/sqrt(Kfold*(Kfold-1)))%*%
                I1inv_list[[r]]%*%I2_list[[r]]%*%I1inv_list[[t]]%*%Zm[[t]]
            } else if(l==t){
              AMMl <- AMMl + sqrt(dim_n_p[1,t]/dim_n_p[1,r])*t(Zm[[r]])%*%
                I1inv_list[[r]]%*%I2_list[[t]]%*%I1inv_list[[t]]%*%apply(Zmnj[[t]],2,sum)/sqrt(Kfold*(Kfold-1))
            } else {
              AMMl <- AMMl + dim_n_p[1,l]/sqrt(dim_n_p[1,t]*dim_n_p[1,r])*t(Zm[[r]])%*%
                I1inv_list[[r]]%*%I2_list[[l]]%*%I1inv_list[[t]]%*%Zm[[t]]
            }
          }
          AMM[r,t] <- AMMl
        }
      }
    }
    
    for (r in 1:M) {
      aMl <- 0
      for (l in 1:M) {
        if(l==r){
          tmp <- 0
          for (j in 1:Kfold) {
            tmp <- tmp + t(Zmj3[[r]][j,])%*%I1inv_list[[r]]%*%Zmnj[[r]][j,]
          }
          aMl <- aMl + tmp/sqrt(Kfold-1)
        } else {
          aMl <- aMl + sqrt(dim_n_p[1,l]/dim_n_p[1,r])*t(Zm3[[l]])%*%I1inv_list[[r]]%*%Zm[[r]]
        }
      }
      aM[r] <- aMl
    }
    
    AGG <- matrix(0,K,K)
    aG <- vector(length = K)
    for (r in 1:K) {
      for (t in 1:K) {
        AGG[r,t] <- sum(AMM[cluster_vec_new==r,cluster_vec_new==t])
        aG[r] <- sum(aM[cluster_vec_new==r])
      }
    }
    
    kmeans_count <- table(cluster_vec_new)
    w.dim <- K
    A <- rbind(kmeans_count,diag(1,w.dim))
    l <- c(1-0.0001,rep(0,w.dim))
    u <- c(1+0.0001,rep(1,w.dim))
    res <- osqp::solve_osqp(P=AGG/M,q=-aG/M,A=A,l=l,u=u,
                            pars = osqpSettings(eps_abs=1e-8,eps_rel=1e-8,
                                                polish = TRUE,verbose = FALSE))
    wg <- res$x
    w1 <- wg[cluster_vec_new]
    w1[w1<0] <- 0
    w1 <- w1/sum(w1)
    
    delta <- 0
    for (j in 1:M) {
      delta <- delta + sqrt(nmax/dim_n_p[1,j])*w1[j]*I1inv_list[[j]]%*%Zm[[j]]
    }
    Delta[,b] <- delta
  }
  qb_l <- apply(Delta,1,quantile,prob=first_err/2)
  qb_u <- apply(Delta,1,quantile,prob=1-first_err/2)
  ci_l <- betahatOARFISH[A_supp_u] + Fhat%*%wOARFISH - qb_u/sqrt(nmax)
  ci_u <- betahatOARFISH[A_supp_u] + Fhat%*%wOARFISH - qb_l/sqrt(nmax)
  return(list(ci_l=ci_l,ci_u=ci_u))
}

K_selection <- function(betahatMat,criterion=c("elbow","distortion"),max_K=dim(betahatMat)[1]-2,
                        elbow_diff1=-0.02,elbow_diff2=0,distortion_threshold=0.85,distortion_ratio=0.9){
  p <- dim(betahatMat)[2]
  M <- dim(betahatMat)[1]
  criterion <- match.arg(criterion)
  non_zero_cols <- apply(betahatMat, 2, function(col) any(col != 0) & any(!is.na(col)))
  betahatMat_dropcols <- betahatMat[, non_zero_cols, drop = FALSE]
  if(M-2<max_K){max_K=M-2;warning('max_K is adjusted to M-2')}
  if(max_K<2){return(opt_K=NULL);warning('no need for selecting K')}
  if(criterion=="elbow"){
    elbow <- ClusterR::Optimal_Clusters_KMeans(betahatMat_dropcols,max_K,plot_clusters=FALSE)
    K <- min(which(diff(elbow, difference=2)<elbow_diff2)[1]+1,
             which(diff(elbow)>elbow_diff1)[1],na.rm = T)
    if(is.infinite(K)) {
      K=which.min(elbow)
      warning(paste0("elbow can not be detected for ","max_K=",max_K,", temporarily set opt_K=which.min(elbow). Please turn down elbow_diff1 smaller than ",elbow_diff1))
      }
  } else {
    distortion <- ClusterR::Optimal_Clusters_KMeans(betahatMat_dropcols,max_K,plot_clusters=FALSE, criterion = "distortion_fK")
    K_min <- which.min(distortion)
    K_long <- max(which(distortion<=distortion_threshold))
    if(is.infinite(K_long)) {
      K=K_min
      warning(paste0("distortion_threshold should be set above 0.85"))
    } else {
      K = ifelse(distortion[K_min]/distortion[K_long]<=distortion_ratio,K_min,K_long)
    }
  }
  return(opt_K=K)
}


K_selection_new <- function(betahatMat,criterion=c("elbow","distortion"),max_K=dim(betahatMat)[1]-2,
                        frac=0.5,distortion_threshold=0.85,distortion_ratio=0.9){
  p <- dim(betahatMat)[2]
  M <- dim(betahatMat)[1]
  criterion <- match.arg(criterion)
  non_zero_cols <- apply(betahatMat, 2, function(col) any(col != 0) & any(!is.na(col)))
  betahatMat_dropcols <- betahatMat[, non_zero_cols, drop = FALSE]
  if(M-2<max_K){max_K=M-2;warning('max_K is adjusted to M-2')}
  if(max_K<2){return(opt_K=NULL);warning('no need for selecting K')}
  if(criterion=="elbow"){
    elbow <- ClusterR::Optimal_Clusters_KMeans(betahatMat_dropcols,max_K,plot_clusters=FALSE)
    cutoff.point=KneeArrower::findCutoff(1:max_K, elbow, method="first", frac)
    K=round(cutoff.point$x)
    if(K<1) {
      K=1
      warning(paste0("elbow: ","cutoff point is out of range ",1,":",max_K,", which is set to 1."))
    }
    if(K>max_K) {
      K=max_K
      warning(paste0("elbow: ","cutoff point is out of range ",1,":",max_K,", which is set to ",max_K,"."))
    }
  } else {
    distortion <- ClusterR::Optimal_Clusters_KMeans(betahatMat_dropcols,max_K,plot_clusters=FALSE, criterion = "distortion_fK")
    K_min <- which.min(distortion)
    K_long <- max(which(distortion<=distortion_threshold))
    if(is.infinite(K_long)) {
      K=K_min
      warning(paste0("distortion_threshold should be set above 0.85"))
    } else {
      K = ifelse(distortion[K_min]/distortion[K_long]<=distortion_ratio,K_min,K_long)
    }
  }
  return(opt_K=K)
}
