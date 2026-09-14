###############################
## Arguments:
## theta/beta: coefficients, d x 1 vector
## lambda: penalty hyper-parameter, scalar
## a: penalty hyper-parameter for SCAD, scalar
## Y: response, n x 1 vector
## X: design, n x d matrix
## Y_list: response list in which n x 1 vector is stored, list
## X_list: design list in which n x d matrix is stored, list with the same dimension as Y_list
## eta: the difference between local gradient and global gradient, d x 1 vector
## alpha: regularizer of proximal point, scalar
## theta_t: initial coefficient estimate, d x 1 vector
## learning_rate: the step size at each iteration of gradient descent, scalar
## maxiter: the maximum of iteration of the algorithm, scalar
## tol: tolerance of accuracy, scalar
## t: communication rounds, scalar
## penalty: penalty function used in loss function, character
## trunconst: the truncated value for compressing the small coefficient estimate to 0, for 'SCAD' and 'LASSO' only, scalar
## rho: hyper-parameter in ADMM algorithm, scalar
## init_replacement: (DANE only) whether to use 'theta_t' as the output and be incorporated in the averaging step
##                  if 'minimize_logistic_prox_cvx' faces an error on theta_t, Boolean
## first_error: (DANE/CEASE1/CSL1 only) the confidence on the interval is given by 1 - first_error, scalar

library(CVXR)

# LASSO penalty
p_lambda_lasso<-function(theta,lambda){
  theta <- as.matrix(theta)
  return(lambda*sum(abs(theta)))
}

# SCAD penalty
p_lambda_scad<-function(theta,lambda,a=3.7){
  theta <- as.matrix(theta)
  p <- dim(theta)[1]
  p_lam <- vector(length = p)
  for (i in 1:p) {
    if(abs(theta[i])<lambda){
      p_lam[i] <- lambda*abs(theta[i])
    } else if(lambda<=abs(theta[i]) & abs(theta[i])<a*lambda){
      p_lam[i] <- (-lambda^2+2*a*lambda*abs(theta[i])-theta[i]^2)/(2*(a-1))
    } else {
      p_lam[i] <- (a+1)*lambda^2/2
    }
  }
  return(sum(as.matrix(p_lam)))
}

# first order derivative of SCAD penalty
p1_lambda_scad<-function(theta,lambda,a=3.7){
  theta <- as.matrix(theta)
  p <- dim(theta)[1]
  p1_lam <- vector(length = p)
  for (i in 1:p) {
    if(abs(theta[i])<lambda){
      p1_lam[i] <- lambda
    } else if(lambda<=abs(theta[i]) & abs(theta[i])<a*lambda){
      p1_lam[i] <- (a*lambda-abs(theta[i]))/(a-1)
    } else {
      p1_lam[i] <- 0
    }
  }
  return(as.matrix(p1_lam))
}

# second order derivative of SCAD penalty
p2_lambda_scad<-function(theta,lambda,a=3.7){
  theta <- as.matrix(theta)
  p <- dim(theta)[1]
  p2_lam <- vector(length = p)
  for (i in 1:p) {
    if(lambda<=abs(theta[i]) & abs(theta[i])<a*lambda){
      p2_lam[i] <- -sign(theta[i])/(a-1)
    } else {
      p2_lam[i] <- 0
    }
  }
  return(as.matrix(p2_lam))
}

link <- function(x){
  # x is n-by-1 array
  return( 1/(1+exp(-x)))
}

gradient <- function(beta,Y,X){
  return(t(X)%*%(link(X%*%beta)-Y)/length(Y))
}

hessian <- function(beta,X){
  t(X)%*%diag(c(exp(-X%*%beta)/(1+exp(-X%*%beta))^2))%*%X/dim(X)[1]
}

nlogLik <- function(beta,Y,X){
  return((-t(Y)%*%X%*%beta+sum(log(1+exp(X%*%beta))))/length(Y))
}

# negative log likelihood: -y_i*log(pi_i/(1-pi_i))+log(1-pi_i)
#                          -y_i*x^T\beta+log(1+exp(x^T\beta))
minimize_logistic_prox <- function(X, Y, eta, alpha, theta_t, learning_rate=0.01, maxiter=1000, tol=0.001){
  n <- dim(X)[1]
  p <- dim(X)[2]
  Y <- matrix(Y,n,1)
  theta_s <- theta_t
  for (s in 1:maxiter) {
    gradLk = t(X)%*%(link(X%*%theta_s)-Y)/n
    g_s = gradLk + eta + alpha * (theta_s - theta_t)
    if(s>1 & norm(g_s,type = "2")<tol)
      break
    theta_s <- theta_s - learning_rate * g_s
  }
  theta_s <- matrix(theta_s,p)
  obj_val <- nlogLik(theta_s,Y,X) + t(eta)%*%theta_s  + .5*alpha*sum((theta_s - theta_t)^2)
  theta <- list(coefs=theta_s,iter=s,obj_val=obj_val)
  return(theta)
}

minimize_logistic_loss <- function(X, Y, learning_rate=0.01, maxiter=1000, tol=0.001){
  p = dim(X)[2]
  return(minimize_logistic_prox(X, Y, eta=rep(0,p), alpha=0, theta_t=rep(0,p),
                                learning_rate=learning_rate, maxiter=maxiter, tol=tol))
}

minimize_logistic_prox_cvx <- function(X, Y, eta= rep(0,dim(X)[2]), alpha=.05*dim(X)[2]/dim(X)[1], 
                                       theta_t= rep(0.1,dim(X)[2]), lambda, penalty=c("null","lasso","scad"), 
                                       trunconst=0.001, maxiter=50, tol=(1e-4*dim(X)[2])^2){
  n <- dim(X)[1]
  p <- dim(X)[2]
  Y <- matrix(Y,n,1)
  eta <- as.matrix(eta)
  penalty <- match.arg(penalty)
  if(penalty=="scad" | penalty=="lasso"){maxiter=min(maxiter,20)}
  for (s in 1:maxiter) {
    theta_t <- as.matrix(theta_t)
    theta_s <- CVXR::Variable(rows = p,cols = 1)
    if(penalty=="lasso"){
      penalty_fun <- p_lambda_lasso
      Penalty <- function(theta_s,lambda)lambda*p_norm(theta_s,1)
    } else if(penalty=="scad"){
      penalty_fun <- p_lambda_scad
      Sigma_lam <- c(p1_lambda_scad(theta_t,lambda,a=3.7)/abs(theta_t))
      Sigma_lam[which(is.infinite(Sigma_lam)==1)] <- 0 # if betahat=0, p1_lambda_scad=0 already and do not need to be approximated
      Sigma_lam[which(Sigma_lam>1e5)] <- 1e5 # avoid p1_lambda_scad too large
      Penalty <- function(theta_s,lambda).5*quad_form(theta_s,diag(Sigma_lam+1e-4,p))
    } else {
      penalty_fun <- function(theta_s,lambda) 0
      Penalty <- function(theta_s,lambda) 0
    }
    Ltilde <- (- t(Y)%*%X%*%theta_s + sum(CVXR::logistic(X%*%theta_s)))/n + Penalty(theta_s,lambda) + t(eta)%*%theta_s  + .5*alpha*sum((theta_s - theta_t)^2)
    prob <- CVXR::Problem(Minimize(Ltilde))
    result <- try(CVXR::solve(prob,solver="ECOS"))
    if(grepl("solver_error",result$status)){
      result <- try(CVXR::solve(prob,solver="SCS"))
      if(grepl("solver_error",result$status)|grepl("unbounded",result$status)|grepl("infeasible",result$status)){
        if(s==1){
          obj_val = nlogLik(theta_t,Y,X) + penalty_fun(theta_t,lambda) + t(eta)%*%theta_t
        }
        theta=list(coefs=theta_t,iter=s-1,obj_val=obj_val);return(theta)
      }
    }
    theta_s <- result$getValue(theta_s)
    obj_val <- result$value
    g_s <- gradient(theta_s,Y,X) + eta + alpha * (theta_s - theta_t)
    if(penalty=="scad"){
      theta_s[abs(theta_s)<trunconst] <- 0
      if(s>1){
        theta_s[theta_t==0] <- 0
      }
    }
    if(penalty=="lasso"){
      theta_s[abs(theta_s)<trunconst] <- 0
    }
    if(s>1 & penalty=="null" & norm(g_s,type = "2")<tol)
      break;
    if(s>1 & penalty=="lasso" & norm(g_s+sum(lambda*sign(theta_s)),type = "2")<tol)
      break;
    if(s>1 & penalty=="scad" & norm(g_s+n*sum(p1_lambda_scad(theta_s,lambda)*sign(theta_s)),type = "2")<tol)
      break;
    if(norm(theta_s-theta_t,type = "2")<tol)
      break;
    theta_t <- theta_s
  }
  theta <- list(coefs=theta_s,iter=s,obj_val=obj_val)
  return(theta)
}

# DANE is an averaging version of CSL and CEASE
# 'X_list' should be standardized in terms of the common features of all sites
logistic_DANE <- function(X_list, Y_list, t, alpha, theta_initial, lambda, penalty=c("null","lasso","scad"), 
                          trunconst=0.001, learning_rate=0.01, maxiter=1000, tol=0.001, init_replacement=T, first_error=0.05, ci=T){
  m=length(X_list)
  dim_n_p <- sapply(X_list,dim)
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])
  
  k1 <- which.min(abs(dim_n_p[1,] - median(dim_n_p[1,])))[1]
  nk1 <- dim_n_p[1,k1]
  
  zt <- theta_initial
  coef_DANE <- matrix(0,p,t)
  sandwitch1_DANE <- array(0,c(p,p,t))
  sandwitch2_DANE <- array(0,c(p,p,t))
  ci1_u <- matrix(0,p,t)
  ci1_l <- matrix(0,p,t)
  ci2_u <- matrix(0,p,t)
  ci2_l <- matrix(0,p,t)
  
  for (iter in 1:t) {
    print(paste('Algorithm DANE: iteration', iter))
    Z <- matrix(0,p,m)
    s <- vector(length = m)
    gradLk_list <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
    gradL <- rowSums(gradLk_list%*%dim_n_p[1,])/N
    if(penalty=="null"){
      Xk1 = X_list[[k1]]
      Yk1 = Y_list[[k1]]
      gradLk1 <- gradient(zt,Yk1,Xk1)
      etak1 <- matrix(- gradLk1 + gradL,p,1)
      prox_last <- alpha*zt
    }
    for (k in 1:m) {
      Xk = X_list[[k]]
      Yk = Y_list[[k]]
      gradLk <- gradient(zt,Yk,Xk)
      etak <- matrix(- gradLk + gradL,p,1)
      if(penalty=="null"){
        result <- minimize_logistic_prox(X=Xk, Y=Yk, eta=etak, alpha=alpha, theta_t=zt,
                                         learning_rate = learning_rate, maxiter=maxiter, tol=tol)
        Z[,k] <- result$coefs
        s[k] <- result$iter
      } else if(penalty=="lasso"){
        result <- minimize_logistic_prox_cvx(X=Xk, Y=Yk, eta=etak, alpha=alpha, theta_t=zt, lambda=lambda, penalty = "lasso",
                                             trunconst = trunconst, maxiter=maxiter, tol=tol)
        Z[,k] <- result$coefs
        s[k] <- result$iter
      } else {
        result <- minimize_logistic_prox_cvx(X=Xk, Y=Yk, eta=etak, alpha=alpha, theta_t=zt, lambda=lambda, penalty = "scad",
                                             trunconst = trunconst, maxiter=maxiter, tol=tol)
        Z[,k] <- result$coefs
        s[k] <- result$iter
      }
    }
    s <- !s==0
    if(init_replacement==T){
      zt <- rowMeans(Z)
    } else {
      zt <- rowMeans(Z[,s])
    }
    coef_DANE[1:p,iter] = zt
    if(penalty=="null"&ci==T){
      filling1 <- t(Xk1)%*%diag(c((link(Xk1%*%zt)-Yk1)^2))%*%Xk1/nk1 - gradient(zt,Yk1,Xk1)%*%t(etak1+alpha*zt-prox_last) -
        (etak1+alpha*zt-prox_last)%*%t(gradient(zt,Yk1,Xk1)) + (etak1+alpha*zt-prox_last)%*%t(etak1+alpha*zt-prox_last)
      bread1 <- solve(hessian(zt,Xk1)+alpha*diag(nrow = p))
      sandwitch1_DANE[,,iter] <- bread1%*%filling1%*%bread1
      
      gradLk_list_new <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
      filling2 <- Reduce('+',apply(gradLk_list_new,2,function(x){x%*%t(x)},simplify = FALSE))/m*mean(dim_n_p[1,])
      sandwitch2_DANE[,,iter] <- bread1%*%filling2%*%bread1
      
      ci1_l[,iter] <- zt - qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch1_DANE[,,iter]))/sqrt(N)
      ci1_u[,iter] <- zt + qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch1_DANE[,,iter]))/sqrt(N)
      ci2_l[,iter] <- zt - qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch2_DANE[,,iter]))/sqrt(N)
      ci2_u[,iter] <- zt + qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch2_DANE[,,iter]))/sqrt(N)
    } else {
      sandwitch2_DANE <- NULL
      sandwitch1_DANE <- NULL
      ci1_l=ci1_u=ci2_l=ci2_u=NULL
    }
  }
  return(list(coefs=coef_DANE,Sigma1=sandwitch1_DANE,Sigma2=sandwitch2_DANE,ci1_l=ci1_l,ci1_u=ci1_u,
              ci2_l=ci2_l,ci2_u=ci2_u))
}

logistic_CEASE1 <- function(X_list, Y_list, t, alpha, theta_initial, lambda, penalty=c("null","lasso","scad"), 
                            trunconst=0.001, learning_rate=0.01, maxiter=1000, tol=0.001, first_error=0.05, ci=T){
  m=length(X_list)
  dim_n_p <- sapply(X_list,dim)
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])
  k <- which.min(abs(dim_n_p[1,] - median(dim_n_p[1,])))[1]
  nk <- dim_n_p[1,k]
  
  zt <- theta_initial
  coef_CEASE1 <- matrix(0,p,t)
  sandwitch1_CEASE1 <- array(0,c(p,p,t))
  sandwitch2_CEASE1 <- array(0,c(p,p,t))
  ci1_u <- matrix(0,p,t)
  ci1_l <- matrix(0,p,t)
  ci2_u <- matrix(0,p,t)
  ci2_l <- matrix(0,p,t)
  
  for (iter in 1:t) {
    print(paste('Algorithm CEASE1: iteration', iter))
    gradLk_list <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
    gradL <- rowSums(gradLk_list%*%dim_n_p[1,])/N

    Xk = X_list[[k]]
    Yk = Y_list[[k]]
    gradLk <- gradient(zt,Yk,Xk)
    etak <- matrix(- gradLk + gradL,p,1)
    prox_last <- alpha*zt
    if(penalty=="null"){
      zt <- minimize_logistic_prox(X=Xk, Y=Yk, eta=etak, alpha=alpha, theta_t=zt,
                                      learning_rate = learning_rate, maxiter=maxiter, tol=tol)$coefs
    } else if(penalty=="lasso"){
      zt <- minimize_logistic_prox_cvx(X=Xk, Y=Yk, eta=etak, alpha=alpha, theta_t=zt, lambda=lambda, penalty = "lasso",
                                          trunconst = trunconst, maxiter=maxiter, tol=tol)$coefs
    } else {
      zt <- minimize_logistic_prox_cvx(X=Xk, Y=Yk, eta=etak, alpha=alpha, theta_t=zt, lambda=lambda, penalty = "scad",
                                          trunconst = trunconst, maxiter=maxiter, tol=tol)$coefs
    }
    coef_CEASE1[1:p,iter] = zt
    if(penalty=="null"&ci==T){
      filling1 <- t(Xk)%*%diag(c((link(Xk%*%zt)-Yk)^2))%*%Xk/nk - gradient(zt,Yk,Xk)%*%t(etak+alpha*zt-prox_last) -
        (etak+alpha*zt-prox_last)%*%t(gradient(zt,Yk,Xk)) + (etak+alpha*zt-prox_last)%*%t(etak+alpha*zt-prox_last)
      bread1 <- solve(hessian(zt,Xk)+alpha*diag(nrow = p))
      sandwitch1_CEASE1[,,iter] <- bread1%*%filling1%*%bread1
      
      gradLk_list_new <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
      filling2 <- Reduce('+',apply(gradLk_list_new,2,function(x){x%*%t(x)},simplify = FALSE))/m*mean(dim_n_p[1,])
      sandwitch2_CEASE1[,,iter] <- bread1%*%filling2%*%bread1
      
      ci1_l[,iter] <- zt - qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch1_CEASE1[,,iter]))/sqrt(N)
      ci1_u[,iter] <- zt + qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch1_CEASE1[,,iter]))/sqrt(N)
      ci2_l[,iter] <- zt - qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch2_CEASE1[,,iter]))/sqrt(N)
      ci2_u[,iter] <- zt + qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch2_CEASE1[,,iter]))/sqrt(N)
    } else {
      sandwitch2_CEASE1 <- NULL
      sandwitch1_CEASE1 <- NULL
      ci1_l=ci1_u=ci2_l=ci2_u=NULL
    }
  }
  return(list(coefs=coef_CEASE1,Sigma1=sandwitch1_CEASE1,Sigma2=sandwitch2_CEASE1,ci1_l=ci1_l,ci1_u=ci1_u,
              ci2_l=ci2_l,ci2_u=ci2_u))
}

logistic_CSL1 <- function(X_list, Y_list, t, theta_initial, lambda, penalty=c("null","lasso","scad"), 
                          trunconst=0.001, learning_rate=0.01, maxiter=1000, tol=0.001, first_error=0.05, ci=T){
  m=length(X_list)
  dim_n_p <- sapply(X_list,dim)
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])
  k <- which.min(abs(dim_n_p[1,] - median(dim_n_p[1,])))[1]
  nk <- dim_n_p[1,k]
  
  zt <- theta_initial
  coef_CSL1 <- matrix(0,p,t)
  sandwitch1_CSL1 <- array(0,c(p,p,t))
  sandwitch2_CSL1 <- array(0,c(p,p,t))
  ci1_u <- matrix(0,p,t)
  ci1_l <- matrix(0,p,t)
  ci2_u <- matrix(0,p,t)
  ci2_l <- matrix(0,p,t)
  
  for (iter in 1:t) {
    print(paste('Algorithm CSL1: iteration', iter))
    gradLk_list <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
    gradL <- rowSums(gradLk_list%*%dim_n_p[1,])/N
    
    Xk = X_list[[k]]
    Yk = Y_list[[k]]
    gradLk <- gradient(zt,Yk,Xk)
    etak <- matrix(- gradLk + gradL,p,1)
    if(penalty=="null"){
      zt <- minimize_logistic_prox(X=Xk, Y=Yk, eta=etak, alpha=0, theta_t=zt,
                                   learning_rate = learning_rate, maxiter=maxiter, tol=tol)$coefs
    } else if(penalty=="lasso"){
      zt <- minimize_logistic_prox_cvx(X=Xk, Y=Yk, eta=etak, alpha=0, theta_t=zt, lambda=lambda, penalty = "lasso",
                                       trunconst = trunconst, maxiter=maxiter, tol=tol)$coefs
    } else {
      zt <- minimize_logistic_prox_cvx(X=Xk, Y=Yk, eta=etak, alpha=0, theta_t=zt, lambda=lambda, penalty = "scad",
                                       trunconst = trunconst, maxiter=maxiter, tol=tol)$coefs
    }
    coef_CSL1[1:p,iter] = zt
    if(penalty=="null"&ci==T){
      filling1 <- t(Xk)%*%diag(c((link(Xk%*%zt)-Yk)^2))%*%Xk/nk - gradient(zt,Yk,Xk)%*%t(etak) -
        etak%*%t(gradient(zt,Yk,Xk)) + etak%*%t(etak)
      bread1 <- solve(hessian(zt,Xk))
      sandwitch1_CSL1[,,iter] <- bread1%*%filling1%*%bread1
      
      gradLk_list_new <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
      filling2 <- Reduce('+',apply(gradLk_list_new,2,function(x){x%*%t(x)},simplify = FALSE))/m*mean(dim_n_p[1,])
      sandwitch2_CSL1[,,iter] <- bread1%*%filling2%*%bread1
      
      ci1_l[,iter] <- zt - qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch1_CSL1[,,iter]))/sqrt(N)
      ci1_u[,iter] <- zt + qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch1_CSL1[,,iter]))/sqrt(N)
      ci2_l[,iter] <- zt - qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch2_CSL1[,,iter]))/sqrt(N)
      ci2_u[,iter] <- zt + qnorm(first_error/2,lower.tail = FALSE)*sqrt(diag(sandwitch2_CSL1[,,iter]))/sqrt(N)
    } else {
      sandwitch2_CSL1 <- NULL
      sandwitch1_CSL1 <- NULL
      ci1_l=ci1_u=ci2_l=ci2_u=NULL
    }
  }
  return(list(coefs=coef_CSL1,Sigma1=sandwitch1_CSL1,Sigma2=sandwitch2_CSL1,ci1_l=ci1_l,ci1_u=ci1_u,
              ci2_l=ci2_l,ci2_u=ci2_u))
}

logistic_ADMM <- function(X_list, Y_list, t, rho, theta_initial, 
                          learning_rate=0.01, maxiter=1000, tol=0.001){
  m=length(X_list)
  dim_n_p <- sapply(X_list,dim)
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])

  zt <- theta_initial
  coef_ADMM <- matrix(0,p,t)
  Z = matrix(0,p,m)
  W = matrix(0,p,m)
  
  for (iter in 1:t) {
    print(paste('Algorithm ADMM: iteration', iter))
    # update Z
    for (k in 1:m) {
      Xk = X_list[[k]]
      Yk = Y_list[[k]]        
      wk = W[, k]
      Z[, k] = minimize_logistic_prox(X=Xk, Y=Yk, eta=wk, alpha=rho, theta_t=zt, learning_rate = learning_rate, maxiter=maxiter, tol=tol)$coefs
    }
    zt = rowMeans(Z)
    # update W
    for (k in 1:m){
      W[, k] = W[, k] + rho * (Z[, k] - zt)
    }
    coef_ADMM[1:p,iter] = zt  
  }
  return(coef_ADMM)
}

logistic_GIANT <- function(X_list, Y_list, t, theta_initial){
  m=length(X_list)
  dim_n_p <- sapply(X_list,dim)
  N <- sum(dim_n_p[1,])
  p <- min(dim_n_p[2,])
  
  zt <- theta_initial
  coef_GIANT <- matrix(0,p,t)
  
  for (iter in 1:t) {
    print(paste('Algorithm GIANT: iteration', iter))
    Z = matrix(0,p,m)
    gradLk_list <- mapply(gradient,Y=Y_list,X=X_list,MoreArgs = list(beta=zt))
    gradL <- rowSums(gradLk_list%*%dim_n_p[1,])/N
    for (k in 1:m) {
      Xk = X_list[[k]]
      xi = Xk %*% zt
      nk = dim_n_p[1,k]
      bk = rep(0,nk)
      for (j in 1:nk) {
        tmp = xi[j]
        if(abs(tmp)>30) {
          bk[j]=0
        }
        bk[j] = 1/(exp(tmp) + exp(-tmp) + 2)
      }
      Hk = t(Xk)%*%diag(c(bk),nk)%*%Xk/nk + 1e-4*diag(nrow = p)
      Z[,k] = qr.coef(qr(Hk),gradL)
    }
    tmp = rowMeans(Z)
    print(norm(tmp,type = "2"))
    zt = zt-tmp
    coef_GIANT[,iter] <- zt
  }
  return(coef_GIANT)
}

logistic_AGD <- function(X, Y, t, theta_initial, learning_rate){
  N <- dim(X)[1]
  p <- dim(X)[2]
  coef_AGD <- matrix(0,p,t)
  
  x_t = theta_initial
  y_t = theta_initial
  for (iter in 1:t) {
    print(paste('Algorithm AGD: iteration', iter))
    tmp = x_t
    grad_t = gradient(y_t,Y,X)
    x_t = y_t - learning_rate * grad_t
    y_t = x_t + (t+1)/(t+4)*(x_t - tmp)
    coef_AGD[,iter] = x_t
  }
  return(coef_AGD)
}


















  
  