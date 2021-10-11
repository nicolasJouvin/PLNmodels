#include "RcppArmadillo.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]
// [[Rcpp::plugins(cpp11)]]

// _____________________________________________________________________________

// [[Rcpp::export]]
double objective_M(arma::vec param,
                    const arma::mat &Y,
                    const arma::mat &X,
                    const arma::mat &O,
                    const arma::mat &S,
                    const arma::mat &Theta,
                    const arma::mat &Omega) {

  arma::uword n = Y.n_rows;
  arma::uword p = Y.n_cols;
  arma::mat M(&param[0], n, p, true);

  const arma::mat X_Theta = X * Theta; // (n,p)
  arma::mat M_X_Theta_Omega = (M - X_Theta) * Omega; // (n,p)
  arma::mat A = exp(O + 0.5 * S % S + M); // (n,p)

  double objective = - accu(Y % M - A) + 0.5 * accu(M_X_Theta_Omega * (M - X_Theta).t());
  return objective;

}

// [[Rcpp::export]]
arma::vec grad_M(arma::vec param,
                    const arma::mat &Y,
                    const arma::mat &X,
                    const arma::mat &O,
                    const arma::mat &S,
                    const arma::mat &Theta,
                    const arma::mat &Omega) {

  arma::uword n = Y.n_rows;
  arma::uword p = Y.n_cols;
  arma::mat M(&param[0], n, p, true);
  arma::mat A = exp(O + M + 0.5 * S % S);                       // (n,p)
  arma::mat M_X_Theta_Omega = (M - X * Theta) * Omega; // (n,p)

  arma::vec grad = arma::vectorise( A - Y + M_X_Theta_Omega);

  return grad ;
};

// [[Rcpp::export]]
double objective_S(arma::vec param,
                    const arma::mat &O,
                    const arma::mat &M,
                    const arma::mat &Theta,
                    const arma::mat &Omega) {

  arma::uword n = O.n_rows;
  arma::uword p = O.n_cols;
  arma::mat S(&param[0], n, p, true);

  const arma::mat O_M = O + M;
  const arma::vec diag_Omega = diagvec(Omega);
  arma::mat A = exp(O_M + 0.5 * S % S); // (n,p)

  // trace(1^T log(S)) == accu(log(S)).
  // S_bar = diag(sum(S, 0)). trace(Omega * S_bar) = dot(diagvec(Omega), sum(S2, 0))
  double objective = trace(A + 0.5 * dot(diag_Omega, sum(S % S, 0))) - 0.5 * accu(log(S % S));

  return objective;

}

// [[Rcpp::export]]
arma::vec grad_S(arma::vec param,
                    const arma::mat &O,
                    const arma::mat &M,
                    const arma::mat &Theta,
                    const arma::mat &Omega) {

  arma::uword n = O.n_rows;
  arma::uword p = O.n_cols;
  arma::mat S(&param[0], n, p, true);

  const arma::mat O_M = O + M;
  const arma::vec diag_Omega = diagvec(Omega);
  arma::mat A = exp(O_M + 0.5 * S % S); // (n,p)

  arma::vec grad = arma::vectorise(S.each_row() % diagvec(Omega).t() + S % A - pow(S, -1.) );

  return grad ;
};

/*** R
library(nloptr)
library(optimx)
library(purrr)
library(mvtnorm)
n <- 100; p <- 5; d <- 1
corr <- toeplitz(0.5^(1:p - 1))
sigma2 <- 2; rho <- 0.25
Sigma <- sigma2 * (rho * corr + (1 - rho) * diag(1, p, p))
Z <- rmvnorm(n, sigma = corr)
O <- matrix(log(round(runif(n, 10, 100))),n,p)
Y <- matrix(rpois(n* p, exp(O + Z)), n, p)
X <- matrix(1,n,d)

LMs <- lapply(1:p, function(j) lm.fit(X, log(1 + Y[,j]), offset =  O[,j]) )
M     <- t(do.call(rbind, lapply(LMs, residuals)))
Theta <- t(do.call(rbind, lapply(LMs, coefficients)))
Omega <- solve(cov(M))

## M

param <- rep(0, n*p)
S <- sqrt(matrix(sqrt(.1), n, p))

print(objective_M(param, Y, X, O, S, Theta, Omega))
print(grad_M(param, Y, X, O, S, Theta, Omega))

objective_M_R<- function(.x) {
  objective_M(.x, Y, X, O, S, Theta, Omega)
}

grad_M_R<- function(.x) {
  grad_M(.x, Y, X, O, S, Theta, Omega)
}
check <- nloptr::check.derivatives(.x = param, objective_M_R, grad_M_R, check_derivatives_print = 'errors')


## S
param <- rep(sqrt(0.1), n*p)

print(objective_S(param, O, M, Theta, Omega))
print(grad_S(param, O, M, Theta, Omega))

objective_S_R<- function(.x) {
  objective_S(.x, O, M, Theta, Omega)
}

grad_S_R<- function(.x) {
  grad_S(.x, O, M, Theta, Omega)
}
check <- nloptr::check.derivatives(.x = param, objective_S_R, grad_S_R, check_derivatives_print = 'errors')

optim(param, objective_S_R, grad_S_R, method = "BFGS")

*/
