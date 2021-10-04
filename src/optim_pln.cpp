#include <RcppArmadillo.h>

#include "nlopt_wrapper.h"
#include "packing.h"
#include "utils.h"

// [[Rcpp::export]]
arma::vec cpp_optimize_pln_vloglik(
    const arma::mat & Y,      // responses (n,p)
    const arma::mat & X,      // covariates (n,d)
    const arma::mat & O,      // offsets (n,p)
    const arma::mat & Omega,  // (p,p)
    const arma::mat & Theta,  // (d,p)
    const arma::mat & M,      // (n,p)
    const arma::mat & S       // (n,p)
) {
    const arma::uword n = Y.n_rows;
    const arma::uword p = Y.n_cols;

    const arma::mat S2 = S % S ;
    const arma::mat A = exp(O + M + .5 * S2) ;
    const arma::mat M_X_Theta = M - X * Theta ;
    return (
        0.5 * real(log_det(Omega)) + ki(Y)
        + sum(Y % (O + M) - A + 0.5 * log(S2)
            - 0.5 * ((M_X_Theta * Omega) % M_X_Theta + S2 * diagmat(Omega)), 1)
    ) ;

}

// [[Rcpp::export]]
arma::mat cpp_optimize_pln_Omega_full(
    const arma::mat & M,     // (n,p)
    const arma::mat & X,     // (n,d)
    const arma::mat & Theta, // (d,p)
    const arma::mat & S      // (n,p)
) {
    const arma::uword n = M.n_rows;
    arma::mat M_X_Theta = M - X * Theta;
    return (double(n) * inv_sympd(M_X_Theta.t() * M_X_Theta + diagmat(sum(S % S, 0))));
}

// [[Rcpp::export]]
arma::mat cpp_optimize_pln_Omega_spherical(
    const arma::mat & M,     // (n,p)
    const arma::mat & X,     // (n,d)
    const arma::mat & Theta, // (d,p)
    const arma::mat & S      //  (n,p)
) {
    const arma::uword n = M.n_rows;
    const arma::uword p = M.n_cols;
    double sigma2 = accu( pow(M - X * Theta, 2) + S % S ) / double(n * p) ;
    return arma::diagmat(arma::ones(p)/sigma2) ;
}

// [[Rcpp::export]]
arma::mat cpp_optimize_pln_Omega_diagonal(
    const arma::mat & M,     // (n,p)
    const arma::mat & X,     // (n,d)
    const arma::mat & Theta, // (d,p)
    const arma::mat & S      // (n,p)
) {
    const arma::uword n = M.n_rows;
    const arma::uword p = M.n_cols;
    return arma::diagmat(double(n) / sum( pow(M - X * Theta, 2) + S % S, 0)) ;
}

// ---------------------------------------------------------------------------------------

// [[Rcpp::export]]
arma::mat cpp_optimize_pln_Theta(
    const arma::mat & M, // (n,p)
    const arma::mat & X  // (n,d)
) {
    // X^T X is sympd, provide this indications to solve()
    return solve(X.t() * X, X.t() * M, arma::solve_opts::likely_sympd);
}

// ---------------------------------------------------------------------------------------
// Step e, optimizes M

// [[Rcpp::export]]
Rcpp::List cpp_optimize_pln_M(
    const arma::mat & init_M,        // (n,p)
    const arma::mat & Y,             // responses (n,p)
    const arma::mat & X,             // covariates (n,d)
    const arma::mat & O,             // offsets (n, p)
    const arma::mat & S,             // (n,p)
    const arma::mat & Theta,         // (d,p)
    const arma::mat & Omega,         // (p,p)
    const Rcpp::List & configuration // List of config values ; xtol_abs is M only (double or mat)
) {
    const auto metadata = tuple_metadata(init_M);
    enum { M_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<M_ID>(parameters.data()) = init_M;

    auto optimizer = new_nlopt_optimizer(configuration, parameters.size());
    if(configuration.containsElementNamed("xtol_abs")) {
        SEXP value = configuration["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto packed = std::vector<double>(metadata.packed_size);
            metadata.map<M_ID>(packed.data()) = Rcpp::as<arma::mat>(value);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    const arma::mat X_Theta = X * Theta; // (n,p)
    const arma::mat O_S2 = O + 0.5 * S % S; // (n,p)

    // Optimize
    auto objective_and_grad =
        [&metadata, &Y, &X, &O_S2, &X_Theta, &Omega](const double * params, double * grad) -> double {
        const arma::mat M = metadata.map<M_ID>(params);

        arma::mat A = exp(O_S2 + M);                       // (n,p)
        arma::mat M_X_Theta_Omega = (M - X_Theta) * Omega; // (n,p)

        double objective = - trace(Y % M - A) + 0.5 * trace(M_X_Theta_Omega * (M - X_Theta).t());
        metadata.map<M_ID>(grad) = M_X_Theta_Omega + A - Y;
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    arma::mat M = metadata.copy<M_ID>(parameters.data());
    return Rcpp::List::create(
        Rcpp::Named("status") = static_cast<int>(result.status),
        Rcpp::Named("iterations") = result.nb_iterations,
        Rcpp::Named("M") = M);
}

// ---------------------------------------------------------------------------------------
// Step f, optimizes S

// [[Rcpp::export]]
Rcpp::List cpp_optimize_pln_S(
    const arma::mat & init_S,        // (n,p)
    const arma::mat & O,             // offsets (n, p)
    const arma::mat & M,             // (n,p)
    const arma::mat & Theta,         // (d,p)
    const arma::vec & diag_Omega,    // (p,1)
    const Rcpp::List & configuration // List of config values ; xtol_abs is S2 only (double or mat)
) {
    const auto metadata = tuple_metadata(init_S);
    enum { S_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<S_ID>(parameters.data()) = init_S;

    auto optimizer = new_nlopt_optimizer(configuration, parameters.size());
    if(configuration.containsElementNamed("xtol_abs")) {
        SEXP value = configuration["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto packed = std::vector<double>(metadata.packed_size);
            metadata.map<S_ID>(packed.data()) = Rcpp::as<arma::mat>(value);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    const arma::mat O_M = O + M;

    // Optimize
    auto objective_and_grad = [&metadata, &O_M, &diag_Omega](const double * params, double * grad) -> double {
        const arma::mat S = metadata.map<S_ID>(params);

        arma::uword n = S.n_rows;
        arma::mat A = exp(O_M + 0.5 * S % S); // (n,p)

        // trace(1^T log(S)) == accu(log(S)).
        // S_bar = diag(sum(S, 0)). trace(Omega * S_bar) = dot(diagvec(Omega), sum(S2, 0))
        double objective = trace(A + 0.5 * dot(diag_Omega, sum(S % S, 0))) - 0.5 * accu(log(S % S));
        // S2^\emptyset interpreted as pow(S2, -1.) as that makes the most sense (gradient component for log(S2))
        // 1_n Diag(Omega)^T is n rows of diag(omega) values
        metadata.map<S_ID>(grad) = S.each_row() % diag_Omega.t() + S % A - pow(S, -1.) ;
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    arma::mat S = metadata.copy<S_ID>(parameters.data());
    return Rcpp::List::create(
        Rcpp::Named("status") = static_cast<int>(result.status),
        Rcpp::Named("iterations") = result.nb_iterations,
        Rcpp::Named("S") = S);
}
