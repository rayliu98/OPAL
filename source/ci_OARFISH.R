###############################
## Arguments:
## betahatMat: coefficients estimates matrix, with the m-th row betahatm, M x p matrix
## Y_list: response list in which n x 1 vector is stored, M x 1 list
## X_list: design list in which n x d matrix is stored, list with the same dimension as Y_list
## cluster_vec: a vector denotes the clustering results of M sites, M x 1 vector
## betahatOARFISH: coefficient estimate by 'betahat_OARFISH' function, d x 1 vector
## wOARFISH: optimal weight estimate by 'betahat_OARFISH' function, M x 1 vector
## Omega1_est: the estimating method for Omega1, character.
##             'b2': n_m^{-1}\bm{X}_{m}^\top \mathbf{D}_m^1(\hat{\bm\beta}(\tilde{\bm w})) \bm{X}_{m}^{},
##                   which is the direct plug-in estimator. In this case, 'Cov' is semi positive definite;
##             'b1y': \bm X_m^\top[\operatorname{vec}\{b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w})),\ldots,b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w}))\}-\bm Y_m]
##                    \cdot[\operatorname{vec}\{b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w})),\ldots,b^{(1)}(\bm x_{m,1}^\top \hat{\bm\beta}(\tilde{\bm w}))\}-\bm Y_m]^\top\bm X_m
##                   which is the empirical sample estimator, which is used as the filling in 'sandwitch' estimator.
##                   In this case, 'Cov' is likely not semi positive definite, we need to
##                   modify 'Cov' by adding the largest negative eigenvalue on the diagonal:
##                   - 'b1y_whole': decompose 'Cov' and subtract its largest negative 
##                                  eigenvalue on the diagonal of 'Cov';
##                   - 'b1y_schurI3': decompose Schur complement of 'Omega1' and subtract its largest negative 
##                                  eigenvalue on the diagonal of 'I3';
##                   - 'b1y_schurO3': decompose Schur complement of 'I3' and subtract its largest negative 
##                                  eigenvalue on the diagonal of 'Omega1';
## update: indicator of whether to use the modified 'Cov' to construct empirical limiting distribution.
## method: three ways of concatenating the non-zero coefficient estimator for M sites
##         in betahatMat, character.
## first_err: first-type error to be controlled that determine the coverage of confidence interval, scalar.
## B: sampling times for constructing the empirical limiting distribution, scalar.
## lambda_series: lambda used in each site for scad estimator, M x 1 vector.
## Kfold: the folds used for cross validation within each site, scalar.
## 
## Notes:
## 1. Correction ratio means the absolute value of the ratio of the largest negative 
## eigenvalue (which we add on the diagonal to make the covariance semi positive definite) 
## to the largest positive one of 'Cov' for the sampling purpose. 
## 2. Correction magnitude means the difference between the absolute value of largest negative 
## eigenvalue and the smallest positive one of 'Cov'.
## 3. Correction ratio/magnitude for I3/O1 is similar as described in 1 & 2 which
## focuses on effect on I3/O1 about the addition on the diagonal of 'Cov'.

library(MASS)
library(Matrix)
library(ClusterR)
library(osqp)
source("OARFISH_solver.R")
source("logistic_solvers_forR.R")


ci.OARFISH <- function(X_list, Y_list, betahatOARFISH, wOARFISH, betahatMat, cluster_vec, 
                        Omega1_est=c("b2","b1y_whole","b1y_schurI3","b1y_schurO1"), method=c("type1","type2","type3"),
                        update=TRUE, first_err=0.05, B=500, lambda_series, Kfold){
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
  
  Omega1_list <- I1inv_list <- I1_list <- I2_list <- I3_list <- list()
  Sigma_list <- b_list <- Cov <- A_supp_list <- list()
  ZL <- Zmj3 <- Zm3 <- Zm <- Zmnj <- list()
  Delta <- matrix(0,qA_u,B)
  
  if(method=="type1"){
    for (j in 1:M) {
      I2_list[[j]] <- SMat_A_supp_u%*%Omega2(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      I3_list[[j]] <- SMat_A_supp_u%*%Omega3(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      Sigma_list[[j]] <- SMat_A_supp_u%*%Sigmahat(betahatOARFISH,lambda = lambda_series[j])%*%t(SMat_A_supp_u)
      b_list[[j]] <- SMat_A_supp_u%*%bhat(betahatOARFISH,lambda = lambda_series[j])
    }
    if(Omega1_est=="b2"){
      for (j in 1:M) {
        Omega1_list[[j]] <- SMat_A_supp_u%*%Omega1_b2(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      }
    } else {
      for (j in 1:M) {
        Omega1_list[[j]] <- SMat_A_supp_u%*%Omega1_b1y(betahatOARFISH,X_list[[j]],Y_list[[j]])%*%t(SMat_A_supp_u)
      }
    }
    for (j in 1:M) {
      if(Omega1_est!="b2"){
        I2_u <- sqrt(tcrossprod(diag(Omega1_list[[j]]),diag(I3_list[[j]])))
        I2j_trim <- abs(I2_list[[j]])<=I2_u
        for (m1 in 1:dim(I2j_trim)[1]) {
          for (m2 in 1:dim(I2j_trim)[2]) {
            I2j_trim[m1,m2] <- ifelse(I2j_trim[m1,m2],I2_list[[j]][m1,m2],sign(I2_list[[j]][m1,m2])*I2_u[m1,m2])
          }
        }
        Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2j_trim),cbind(t(I2j_trim),I3_list[[j]]))
        Cov.eig <- eigen(Cov.tmp)
        if(sum(Cov.eig$values<=0)>0){
          if(Omega1_est=="b1y_whole"){
            Cov.tmp <- Cov.tmp + diag(abs(min(Cov.eig$values)),dim(Cov.tmp)[1])
            #eigen(Cov.tmp)$values
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            I3j.eig <- eigen(I3_list[[j]])
            warning(paste0("correction ratio: ",round(abs(min(Cov.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(Cov.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(Cov.eig$values))/max(I3j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(Cov.eig$values))-min(I3j.eig$values[I3j.eig$values>0]),3)))
          } else if(Omega1_est=="b1y_schurI3"){
            schur_complement <- I3_list[[j]]-t(I2j_trim)%*%solve(Omega1_list[[j]])%*%I2j_trim
            schur.eig <- eigen(schur_complement)
            I3j.eig <- eigen(I3_list[[j]])
            Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2j_trim),cbind(t(I2j_trim),I3_list[[j]]+diag(abs(min(schur.eig$values)),dim(I3_list[[j]])[1])))
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            warning(paste0("correction ratio: ",round(abs(min(schur.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(schur.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(schur.eig$values))/max(I3j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(schur.eig$values))-min(I3j.eig$values[I3j.eig$values>0]),3)))
          } else if(Omega1_est=="b1y_schurO1"){
            schur_complement <- tryCatch(Omega1_list[[j]]-I2j_trim%*%solve(I3_list[[j]])%*%t(I2j_trim),error=function(e)print(e))
            if(!is.matrix(schur_complement)){schur_complement <- Omega1_list[[j]]-I2j_trim%*%solve(I3_list[[j]]+diag(1e-8,dim(I3_list[[j]])[1]))%*%t(I2j_trim)}
            schur.eig <- eigen(schur_complement)
            O1j.eig <- eigen(Omega1_list[[j]])
            Cov.tmp <- rbind(cbind(Omega1_list[[j]]+diag(abs(min(schur.eig$values)),dim(I3_list[[j]])[1]),I2j_trim),cbind(t(I2j_trim),I3_list[[j]]))
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            warning(paste0("correction ratio: ",round(abs(min(schur.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(schur.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(schur.eig$values))/max(O1j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(schur.eig$values))-min(O1j.eig$values[O1j.eig$values>0]),3)))
          }
        }
        Cov[[j]] <- Cov.tmp
        if(update==T){
          I2_list[[j]] <- I2j_trim
          I3_list[[j]] <- Cov[[j]][(qA_u+1):(2*qA_u),(qA_u+1):(2*qA_u)]
          Omega1_list[[j]] <- Cov[[j]][1:qA_u,1:qA_u]
        }
      } else {
        Cov[[j]] <- rbind(cbind(Omega1_list[[j]],I2_list[[j]]),cbind(I2_list[[j]],I3_list[[j]]))
      }
      I1inv_list[[j]] <- tryCatch(solve(Omega1_list[[j]] + Sigma_list[[j]]),error=function(e)print(e))
      if(!is.matrix(I1inv_list[[j]])){I1inv_list[[j]] <- solve(Omega1_list[[j]] + Sigma_list[[j]] + diag(1e-8,dim(Omega1_list[[j]])[1]))}
    }

    # bias matrix F
    Fhat <- mapply(function(X,y){X%*%y},I1inv_list,b_list)
    
    for (b in 1:B) {
      for (j in 1:M) {
        ZL[[j]] <- MASS::mvrnorm(Kfold,rep(0,2*qA_u),Cov[[j]])
        Zmj3[[j]] <- matrix(ZL[[j]][,(qA_u+1):(2*qA_u)],ncol=qA_u)
        Zm3[[j]] <- apply(Zmj3[[j]],2,sum)/sqrt(Kfold)
        Zm[[j]] <- apply(matrix(ZL[[j]][,1:qA_u],ncol=qA_u),2,sum)/sqrt(Kfold)
        zmnj_tmp <- matrix(0,Kfold,qA_u)
        for (i in 1:Kfold) {
          zmnj_tmp[i,] <- apply(ZL[[j]][-i,1:qA_u],2,sum)/sqrt(Kfold-1)
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
  } else if(method=="type3"){
    for (j in 1:M) {
      A_supp_list[[j]] <- matrix(diag(A_supp[,j])[A_supp[,j],],nrow=qA[j])
      I2_list[[j]] <- SMat_A_supp_u%*%Omega2(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      I3_list[[j]] <- SMat_A_supp_u%*%Omega3(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      Sigma_list[[j]] <- A_supp_list[[j]]%*%Sigmahat(betahatOARFISH,lambda = lambda_series[j])%*%t(A_supp_list[[j]])
      b_list[[j]] <- A_supp_list[[j]]%*%bhat(betahatOARFISH,lambda = lambda_series[j])
    }
    if(Omega1_est=="b2"){
      for (j in 1:M) {
        Omega1_list[[j]] <- A_supp_list[[j]]%*%Omega1_b2(betahatOARFISH,X_list[[j]])%*%t(A_supp_list[[j]])
      }
    } else {
      for (j in 1:M) {
        Omega1_list[[j]] <- A_supp_list[[j]]%*%Omega1_b1y(betahatOARFISH,X_list[[j]],Y_list[[j]])%*%t(A_supp_list[[j]])
      }
    }
    
    for (j in 1:M) {
      if(Omega1_est!="b2"){
        I2_u <- sqrt(tcrossprod(diag(Omega1_list[[j]]),diag(I3_list[[j]])))
        I2j_trim <- abs(A_supp_list[[j]]%*%t(SMat_A_supp_u)%*%I2_list[[j]])<=I2_u
        for (m1 in 1:dim(I2j_trim)[1]) {
          for (m2 in 1:dim(I2j_trim)[2]) {
            I2j_trim[m1,m2] <- ifelse(I2j_trim[m1,m2],(A_supp_list[[j]]%*%t(SMat_A_supp_u)%*%I2_list[[j]])[m1,m2],
                                      sign((A_supp_list[[j]]%*%t(SMat_A_supp_u)%*%I2_list[[j]])[m1,m2])*I2_u[m1,m2])
          }
        }
        Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2j_trim),cbind(t(I2j_trim),I3_list[[j]]))
        Cov.eig <- eigen(Cov.tmp)
        if(sum(Cov.eig$values<=0)>0){
          if(Omega1_est=="b1y_whole"){
            Cov.tmp <- Cov.tmp + diag(abs(min(Cov.eig$values)),dim(Cov.tmp)[1])
            #eigen(Cov.tmp)$values
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            I3j.eig <- eigen(I3_list[[j]])
            warning(paste0("correction ratio: ",round(abs(min(Cov.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(Cov.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(Cov.eig$values))/max(I3j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(Cov.eig$values))-min(I3j.eig$values[I3j.eig$values>0]),3)))
          } else if(Omega1_est=="b1y_schurI3"){
            schur_complement <- I3_list[[j]]-t(I2j_trim)%*%solve(Omega1_list[[j]])%*%I2j_trim
            schur.eig <- eigen(schur_complement)
            I3j.eig <- eigen(I3_list[[j]])
            Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2j_trim),cbind(t(I2j_trim),I3_list[[j]]+diag(abs(min(schur.eig$values)),dim(I3_list[[j]])[1])))
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            warning(paste0("correction ratio: ",round(abs(min(schur.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(schur.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(schur.eig$values))/max(I3j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(schur.eig$values))-min(I3j.eig$values[I3j.eig$values>0]),3)))
          } else if(Omega1_est=="b1y_schurO1"){
            schur_complement <- tryCatch(Omega1_list[[j]]-I2j_trim%*%solve(I3_list[[j]])%*%t(I2j_trim),error=function(e)print(e))
            if(!is.matrix(schur_complement)){schur_complement <- Omega1_list[[j]]-I2j_trim%*%solve(I3_list[[j]]+diag(1e-8,dim(I3_list[[j]])[1]))%*%t(I2j_trim)}
            schur.eig <- eigen(schur_complement)
            O1j.eig <- eigen(Omega1_list[[j]])
            Cov.tmp <- rbind(cbind(Omega1_list[[j]]+diag(abs(min(schur.eig$values)),dim(Omega1_list[[j]])[1]),I2j_trim),cbind(t(I2j_trim),I3_list[[j]]))
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            warning(paste0("correction ratio: ",round(abs(min(schur.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(schur.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(schur.eig$values))/max(O1j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(schur.eig$values))-min(O1j.eig$values[O1j.eig$values>0]),3)))
          }
        }
        Cov[[j]] <- Cov.tmp
        if(update==T){
          I2j_trim_r <- SMat_A_supp_u%*%t(A_supp_list[[j]])%*%I2j_trim
          I2_list[[j]][I2j_trim_r!=0&I2_list[[j]]!=I2j_trim_r] <- I2j_trim_r[I2j_trim_r!=0&I2_list[[j]]!=I2j_trim_r]
          I2_list[[j]][t(I2j_trim_r)!=0&I2_list[[j]]!=t(I2j_trim_r)] <- t(I2j_trim_r)[t(I2j_trim_r)!=0&I2_list[[j]]!=t(I2j_trim_r)]
          
          I3_list[[j]] <- Cov[[j]][(qA[j]+1):(qA[j]+qA_u),(qA[j]+1):(qA[j]+qA_u)]
          Omega1_list[[j]] <- Cov[[j]][1:qA[j],1:qA[j]]
        }
      } else {
        Cov[[j]] <- rbind(cbind(Omega1_list[[j]],A_supp_list[[j]]%*%t(SMat_A_supp_u)%*%I2_list[[j]]),
                          cbind(I2_list[[j]]%*%SMat_A_supp_u%*%t(A_supp_list[[j]]),I3_list[[j]]))
      }
      I1inv_list[[j]] <- tryCatch(solve(Omega1_list[[j]] + Sigma_list[[j]]),error=function(e)print(e))
      if(!is.matrix(I1inv_list[[j]])){I1inv_list[[j]] <- solve(Omega1_list[[j]] + Sigma_list[[j]] + diag(1e-8,dim(Omega1_list[[j]])[1]))}
    }

    # bias matrix F
    Fhat <- mapply(function(A,X,y){SMat_A_supp_u%*%t(A)%*%X%*%y},A_supp_list,I1inv_list,b_list)
    
    for (b in 1:B) {
      for (j in 1:M) {
        ZL[[j]] <- MASS::mvrnorm(Kfold,rep(0,qA[j]+qA_u),Cov[[j]])
        Zmj3[[j]] <- matrix(ZL[[j]][,(qA[j]+1):(qA[j]+qA_u)],ncol=qA_u)%*%SMat_A_supp_u%*%t(A_supp_list[[j]])
        Zm3[[j]] <- apply(matrix(ZL[[j]][,(qA[j]+1):(qA[j]+qA_u)],ncol=qA_u),2,sum)/sqrt(Kfold)
        Zm[[j]] <- apply(matrix(ZL[[j]][,1:qA[j]],ncol=qA[j]),2,sum)/sqrt(Kfold)
        zmnj_tmp <- matrix(0,Kfold,qA[j])
        for (i in 1:Kfold) {
          zmnj_tmp[i,] <- apply(matrix(ZL[[j]][-i,1:qA[j]],ncol=qA[j]),2,sum)/sqrt(Kfold-1)
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
              if(l!=r){AMMl <- AMMl + dim_n_p[1,l]/dim_n_p[1,r]*t(Zm[[r]])%*%I1inv_list[[r]]%*%
                A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%I2_list[[l]]%*%
                SMat_A_supp_u%*%t(A_supp_list[[r]])%*%I1inv_list[[r]]%*%Zm[[r]]}
              if(l==r){AMMl <- AMMl + sum(apply(Zmnj[[r]],1,function(x){t(x)%*%I1inv_list[[r]]%*%
                  A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%I2_list[[r]]%*%
                  SMat_A_supp_u%*%t(A_supp_list[[r]])%*%I1inv_list[[r]]%*%x}))/(Kfold-1)}
            }
            AMM[r,t] <- AMMl
          } else {
            AMMl <- 0
            for (l in 1:M) {
              if(l==r){
                AMMl <- AMMl + sqrt(dim_n_p[1,r]/dim_n_p[1,t])*t(apply(Zmnj[[r]],2,sum)/sqrt(Kfold*(Kfold-1)))%*%
                  I1inv_list[[r]]%*%A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%
                  I2_list[[r]]%*%SMat_A_supp_u%*%t(A_supp_list[[t]])%*%I1inv_list[[t]]%*%Zm[[t]]
              } else if(l==t){
                AMMl <- AMMl + sqrt(dim_n_p[1,t]/dim_n_p[1,r])*t(Zm[[r]])%*%
                  I1inv_list[[r]]%*%A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%
                  I2_list[[t]]%*%SMat_A_supp_u%*%t(A_supp_list[[t]])%*%
                  I1inv_list[[t]]%*%apply(Zmnj[[t]],2,sum)/sqrt(Kfold*(Kfold-1))
              } else {
                AMMl <- AMMl + dim_n_p[1,l]/sqrt(dim_n_p[1,t]*dim_n_p[1,r])*t(Zm[[r]])%*%
                  I1inv_list[[r]]%*%A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%
                  I2_list[[l]]%*%SMat_A_supp_u%*%t(A_supp_list[[t]])%*%I1inv_list[[t]]%*%Zm[[t]]
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
            aMl <- aMl + sqrt(dim_n_p[1,l]/dim_n_p[1,r])*t(Zm3[[l]])%*%
              SMat_A_supp_u%*%t(A_supp_list[[r]])%*%
              I1inv_list[[r]]%*%Zm[[r]]
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
        delta <- delta + sqrt(nmax/dim_n_p[1,j])*w1[j]*SMat_A_supp_u%*%t(A_supp_list[[j]])%*%I1inv_list[[j]]%*%Zm[[j]]
      }
      Delta[,b] <- delta
    } 
  } else if(method=="type2"){
    length(I1inv_list) <- M*M
    dim(I1inv_list) <- c(M,M)
    I2o_list <- list()
    for (j in 1:M) {
      A_supp_list[[j]] <- matrix(diag(A_supp[,j])[A_supp[,j],],nrow=qA[j])
      I2_list[[j]] <- SMat_A_supp_u%*%Omega2(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      I2o_list[[j]] <- A_supp_list[[j]]%*%Omega2(betahatOARFISH,X_list[[j]])%*%t(A_supp_list[[j]])
      I3_list[[j]] <- SMat_A_supp_u%*%Omega3(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      Sigma_list[[j]] <- SMat_A_supp_u%*%Sigmahat(betahatOARFISH,lambda = lambda_series[j])%*%t(SMat_A_supp_u)
      b_list[[j]] <- A_supp_list[[j]]%*%bhat(betahatOARFISH,lambda = lambda_series[j])
    }
    if(Omega1_est=="b2"){
      for (j in 1:M) {
        Omega1_list[[j]] <- SMat_A_supp_u%*%Omega1_b2(betahatOARFISH,X_list[[j]])%*%t(SMat_A_supp_u)
      }
    } else {
      for (j in 1:M) {
        Omega1_list[[j]] <- SMat_A_supp_u%*%Omega1_b1y(betahatOARFISH,X_list[[j]],Y_list[[j]])%*%t(SMat_A_supp_u)
      }
    }
    
    for (j in 1:M) {
      if(Omega1_est!="b2"){
        I2_u <- sqrt(tcrossprod(diag(Omega1_list[[j]]),diag(I3_list[[j]])))
        I2j_trim <- abs(I2_list[[j]])<=I2_u
        for (m1 in 1:dim(I2j_trim)[1]) {
          for (m2 in 1:dim(I2j_trim)[2]) {
            I2j_trim[m1,m2] <- ifelse(I2j_trim[m1,m2],I2_list[[j]][m1,m2],sign(I2_list[[j]][m1,m2])*I2_u[m1,m2])
          }
        }
        Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2j_trim),cbind(t(I2j_trim),I3_list[[j]]))
        Cov.eig <- eigen(Cov.tmp)
        if(sum(Cov.eig$values<=0)>0){
          if(Omega1_est=="b1y_whole"){
            Cov.tmp <- Cov.tmp + diag(abs(min(Cov.eig$values)),dim(Cov.tmp)[1])
            #eigen(Cov.tmp)$values
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            I3j.eig <- eigen(I3_list[[j]])
            warning(paste0("correction ratio: ",round(abs(min(Cov.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(Cov.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(Cov.eig$values))/max(I3j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(Cov.eig$values))-min(I3j.eig$values[I3j.eig$values>0]),3)))
          } else if(Omega1_est=="b1y_schurI3"){
            schur_complement <- I3_list[[j]]-t(I2j_trim)%*%solve(Omega1_list[[j]])%*%I2j_trim
            schur.eig <- eigen(schur_complement)
            I3j.eig <- eigen(I3_list[[j]])
            Cov.tmp <- rbind(cbind(Omega1_list[[j]],I2j_trim),cbind(t(I2j_trim),I3_list[[j]]+diag(abs(min(schur.eig$values)),dim(I3_list[[j]])[1])))
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            warning(paste0("correction ratio: ",round(abs(min(schur.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(schur.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(schur.eig$values))/max(I3j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(schur.eig$values))-min(I3j.eig$values[I3j.eig$values>0]),3)))
          } else if(Omega1_est=="b1y_schurO1"){
            schur_complement <- tryCatch(Omega1_list[[j]]-I2j_trim%*%solve(I3_list[[j]])%*%t(I2j_trim),error=function(e)print(e))
            if(!is.matrix(schur_complement)){schur_complement <- Omega1_list[[j]]-I2j_trim%*%solve(I3_list[[j]]+diag(1e-8,dim(I3_list[[j]])[1]))%*%t(I2j_trim)}
            schur.eig <- eigen(schur_complement)
            O1j.eig <- eigen(Omega1_list[[j]])
            Cov.tmp <- rbind(cbind(Omega1_list[[j]]+diag(abs(min(schur.eig$values)),dim(Omega1_list[[j]])[1]),I2j_trim),cbind(t(I2j_trim),I3_list[[j]]))
            Cov.tmp <- Cov.tmp + diag(1e-8,dim(Cov.tmp)[1])
            warning(paste0("correction ratio: ",round(abs(min(schur.eig$values))/max(Cov.eig$values),3)))
            warning(paste0("correction magnitude: ",round(abs(min(schur.eig$values))-min(Cov.eig$values[Cov.eig$values>0]),3)))
            warning(paste0("correction ratio for I3_",j,": ",round(abs(min(schur.eig$values))/max(O1j.eig$values),3)))
            warning(paste0("correction magnitude for I3_",j,": ",round(abs(min(schur.eig$values))-min(O1j.eig$values[O1j.eig$values>0]),3)))
          }
        }
        Cov[[j]] <- Cov.tmp
        if(update==T){
          I2_list[[j]] <- I2j_trim
          I3_list[[j]] <- Cov[[j]][(qA_u+1):(2*qA_u),(qA_u+1):(2*qA_u)]
          Omega1_list[[j]] <- Cov[[j]][1:qA_u,1:qA_u]
        }
      } else {
        Cov[[j]] <- rbind(cbind(Omega1_list[[j]],I2_list[[j]]),
                          cbind(I2_list[[j]],I3_list[[j]]))
      }
      I1_list[[j]] <- Omega1_list[[j]] + Sigma_list[[j]]
      for (k in 1:M) {
        I1inv_list[[j,k]] <- tryCatch(solve(A_supp_list[[k]]%*%t(SMat_A_supp_u)%*%I1_list[[j]]%*%SMat_A_supp_u%*%t(A_supp_list[[k]])),error=function(e)print(e))
        if(!is.matrix(I1inv_list[[j,k]])){
          I1inv_list[[j,k]] <- solve(A_supp_list[[k]]%*%t(SMat_A_supp_u)%*%I1_list[[j]]%*%SMat_A_supp_u%*%t(A_supp_list[[k]])+diag(1e-8,dim(A_supp_list[[k]])[1]))
        }
      }
    }

    # bias matrix F
    Fhat <- mapply(function(A,X,y){SMat_A_supp_u%*%t(A)%*%A%*%t(SMat_A_supp_u)%*%X%*%SMat_A_supp_u%*%t(A)%*%y},A_supp_list,I1_list,b_list)
    
    for (b in 1:B) {
      for (j in 1:M) {
        ZL[[j]] <- MASS::mvrnorm(Kfold,rep(0,2*qA_u),Cov[[j]])
        Zmj3[[j]] <- matrix(ZL[[j]][,(qA_u+1):(2*qA_u)],ncol=qA_u)%*%SMat_A_supp_u%*%t(A_supp_list[[j]])
        Zm3[[j]] <- apply(matrix(ZL[[j]][,(qA_u+1):(2*qA_u)],ncol=qA_u),2,sum)/sqrt(Kfold)
        Zm[[j]] <- apply(matrix(ZL[[j]][,1:qA_u],ncol=qA_u),2,sum)/sqrt(Kfold)
        zmnj_tmp <- matrix(0,Kfold,qA_u)
        for (i in 1:Kfold) {
          zmnj_tmp[i,] <- apply(ZL[[j]][-i,1:qA_u],2,sum)/sqrt(Kfold-1)
        }
        Zmnj[[j]] <- zmnj_tmp%*%SMat_A_supp_u%*%t(A_supp_list[[j]])
      }
      
      AMM <- matrix(0,M,M)
      aM <- vector(length = M)
      for (r in 1:M) {
        for (t in 1:M) {
          if(t==r){
            AMMl <- 0
            for (l in 1:M) {
              if(l!=r){AMMl <- AMMl + dim_n_p[1,l]/dim_n_p[1,r]*t(Zm[[r]])%*%SMat_A_supp_u%*%t(A_supp_list[[l]])%*%I1inv_list[[r,l]]%*%
                I2o_list[[l]]%*%
                I1inv_list[[r,l]]%*%A_supp_list[[l]]%*%t(SMat_A_supp_u)%*%Zm[[r]]}
              if(l==r){AMMl <- AMMl + sum(apply(Zmnj[[r]],1,function(x){t(x)%*%I1inv_list[[r,r]]%*%
                  I2o_list[[r]]%*%
                  I1inv_list[[r,r]]%*%x}))/(Kfold-1)}
            }
            AMM[r,t] <- AMMl
          } else {
            AMMl <- 0
            for (l in 1:M) {
              if(l==r){
                AMMl <- AMMl + sqrt(dim_n_p[1,r]/dim_n_p[1,t])*t(apply(Zmnj[[r]],2,sum)/sqrt(Kfold*(Kfold-1)))%*%
                  I1inv_list[[r,r]]%*%
                  I2o_list[[r]]%*%I1inv_list[[t,r]]%*%A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%Zm[[t]]
              } else if(l==t){
                AMMl <- AMMl + sqrt(dim_n_p[1,t]/dim_n_p[1,r])*t(Zm[[r]])%*%SMat_A_supp_u%*%t(A_supp_list[[t]])%*%
                  I1inv_list[[r,t]]%*%
                  I2o_list[[t]]%*%
                  I1inv_list[[t,t]]%*%apply(Zmnj[[t]],2,sum)/sqrt(Kfold*(Kfold-1))
              } else {
                AMMl <- AMMl + dim_n_p[1,l]/sqrt(dim_n_p[1,t]*dim_n_p[1,r])*t(Zm[[r]])%*%SMat_A_supp_u%*%t(A_supp_list[[l]])%*%
                  I1inv_list[[r,l]]%*%
                  I2o_list[[l]]%*%I1inv_list[[t,l]]%*%A_supp_list[[l]]%*%t(SMat_A_supp_u)%*%Zm[[t]]
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
              tmp <- tmp + t(Zmj3[[r]][j,])%*%I1inv_list[[r,r]]%*%Zmnj[[r]][j,]
            }
            aMl <- aMl + tmp/sqrt(Kfold-1)
          } else {
            aMl <- aMl + sqrt(dim_n_p[1,l]/dim_n_p[1,r])*t(Zm3[[l]])%*%
              SMat_A_supp_u%*%t(A_supp_list[[r]])%*%
              I1inv_list[[r,r]]%*%A_supp_list[[r]]%*%t(SMat_A_supp_u)%*%Zm[[r]]
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
        delta <- delta + sqrt(nmax/dim_n_p[1,j])*w1[j]*SMat_A_supp_u%*%t(A_supp_list[[j]])%*%I1inv_list[[j,j]]%*%A_supp_list[[j]]%*%t(SMat_A_supp_u)%*%Zm[[j]]
      }
      Delta[,b] <- delta
    } 
  }
  qb_l <- apply(Delta,1,quantile,prob=first_err/2)
  qb_u <- apply(Delta,1,quantile,prob=1-first_err/2)
  ci_l <- betahatOARFISH[A_supp_u] + Fhat%*%wOARFISH - qb_u/sqrt(nmax)
  ci_u <- betahatOARFISH[A_supp_u] + Fhat%*%wOARFISH - qb_l/sqrt(nmax)
  return(list(ci_l=ci_l,ci_u=ci_u))
}

