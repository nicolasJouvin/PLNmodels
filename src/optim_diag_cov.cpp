#include "RcppArmadillo.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]
// [[Rcpp::plugins(cpp11)]]

#include "nlopt_wrapper.h"
#include "packing.h"
#include "utils.h"

// ---------------------------------------------------------------------------------------
// Diagonal covariance

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_diagonal(
    const Rcpp::List & init_parameters, // List(B, M, S)
    const arma::mat & Y,                // responses (n,p)
    const arma::mat & X,                // covariates (n,d)
    const arma::mat & O,                // offsets (n,p)
    const arma::vec & w,                // weights (n)
    const Rcpp::List & configuration    // List of config values
) {
    // Conversion from R, prepare optimization
    const auto init_B = Rcpp::as<arma::mat>(init_parameters["B"]); // (p,d)
    const auto init_M = Rcpp::as<arma::mat>(init_parameters["M"]); // (n,p)
    const auto init_S = Rcpp::as<arma::mat>(init_parameters["S"]); // (n,p)

    const auto metadata = tuple_metadata(init_B, init_M, init_S);
    enum { B_ID, M_ID, S_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<B_ID>(parameters.data()) = init_B;
    metadata.map<M_ID>(parameters.data()) = init_M;
    metadata.map<S_ID>(parameters.data()) = init_S;

    auto optimizer = new_nlopt_optimizer(configuration, parameters.size());
    if(configuration.containsElementNamed("xtol_abs")) {
        SEXP value = configuration["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<B_ID>(packed.data()), per_param_list["B"]);
            set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_param_list["M"]);
            set_from_r_sexp(metadata.map<S_ID>(packed.data()), per_param_list["S"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    const double w_bar = accu(w);

    // Optimize
    auto objective_and_grad = [&metadata, &O, &X, &Y, &w, &w_bar](const double * params, double * grad) -> double {
        const arma::mat B = metadata.map<B_ID>(params);
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat S = metadata.map<S_ID>(params);

        arma::mat S2 = S % S;
        arma::mat Z = O + X * B.t() + M;
        arma::mat A = exp(Z + 0.5 * S2);
        arma::rowvec diag_sigma = w.t() * (M % M + S2) / w_bar;
        double objective = accu(diagmat(w) * (A - Y % Z - 0.5 * log(S2))) + 0.5 * w_bar * accu(log(diag_sigma));

        metadata.map<B_ID>(grad) = (A - Y).t() * (X.each_col() % w);
        metadata.map<M_ID>(grad) = diagmat(w) * ((M.each_row() / diag_sigma) + A - Y);
        metadata.map<S_ID>(grad) = diagmat(w) * (S.each_row() % pow(diag_sigma, -1) + S % A - pow(S, -1));
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Variational parameters
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat S = metadata.copy<S_ID>(parameters.data());
    arma::mat S2 = S % S;
    // Regression parameters
    arma::mat B = metadata.copy<B_ID>(parameters.data());
    // Variance parameters
    arma::rowvec sigma2 = w.t() * (M % M + S2) / w_bar;
    arma::vec omega2 = pow(sigma2.t(), -1);
    arma::sp_mat Sigma(Y.n_cols, Y.n_cols);
    Sigma.diag() = sigma2;
    arma::sp_mat Omega(Y.n_cols, Y.n_cols);
    Omega.diag() = omega2;
    // Element-wise log-likelihood
    arma::mat Z = O + X * B.t() + M;
    arma::mat A = exp(Z + 0.5 * S2);
    arma::mat loglik =
        sum(Y % Z - A + 0.5 * log(S2), 1) - 0.5 * (pow(M, 2) + S2) * omega2 + 0.5 * sum(log(omega2)) + ki(Y);

    Rcpp::NumericVector Ji = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik));
    Ji.attr("weights") = w;
    return Rcpp::List::create(
        Rcpp::Named("B", B),
        Rcpp::Named("Sigma", Sigma),
        Rcpp::Named("Omega", Omega),
        Rcpp::Named("M", M),
        Rcpp::Named("S", S),
        Rcpp::Named("Z", Z),
        Rcpp::Named("A", A),
        Rcpp::Named("Ji", Ji),
        Rcpp::Named("monitoring", Rcpp::List::create(
            Rcpp::Named("status", static_cast<int>(result.status)),
            Rcpp::Named("backend", "nlopt"),
            Rcpp::Named("iterations", result.nb_iterations)
        ))
    );
}

// ---------------------------------------------------------------------------------------
// VE diagonal

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_vestep_diagonal(
    const Rcpp::List & init_parameters, // List(M, S)
    const arma::mat & Y,        // responses (n,p)
    const arma::mat & X,        // covariates (n,d)
    const arma::mat & O,        // offsets (n,p)
    const arma::vec & w,        // weights (n)
    const arma::mat & B,        // (p,d)
    const arma::mat & Omega,    // (p,p)
    const Rcpp::List & configuration    // List of config values
) {
    // Conversion from R, prepare optimization
    const auto init_M = Rcpp::as<arma::mat>(init_parameters["M"]); // (n,p)
    const auto init_S = Rcpp::as<arma::mat>(init_parameters["S"]); // (n,p)

    const auto metadata = tuple_metadata(init_M, init_S);
    enum { M_ID, S_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<M_ID>(parameters.data()) = init_M;
    metadata.map<S_ID>(parameters.data()) = init_S;

    auto optimizer = new_nlopt_optimizer(configuration, parameters.size());
    if(configuration.containsElementNamed("xtol_abs")) {
        SEXP value = configuration["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_param_list["M"]);
            set_from_r_sexp(metadata.map<S_ID>(packed.data()), per_param_list["S"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    // Optimize
    auto objective_and_grad = [&metadata, &O, &X, &Y, &w, &B, &Omega](const double * params, double * grad) -> double {
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat S = metadata.map<S_ID>(params);

        arma::mat S2 = S % S;
        arma::mat Z = O + X * B.t() + M;
        arma::mat A = exp(Z + 0.5 * S2);
        arma::vec omega2 = arma::diagvec(Omega);
        double objective =
            accu(w.t() * (A - Y % Z - 0.5 * log(S2))) + 0.5 * as_scalar(w.t() * (pow(M, 2) + S2) * omega2) ;

        metadata.map<M_ID>(grad) = diagmat(w) * (M * arma::diagmat(omega2) + A - Y);
        metadata.map<S_ID>(grad) = diagmat(w) * (S.each_row() % omega2.t() + S % A - pow(S, -1));
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Model and variational parameters
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat S = metadata.copy<S_ID>(parameters.data());
    arma::mat S2 = S % S;
    arma::vec omega2 = Omega.diag();
    // Element-wise log-likelihood
    arma::mat Z = O + X * B.t() + M;
    arma::mat A = exp(Z + 0.5 * S2);
    arma::mat loglik =
      sum(Y % Z - A + 0.5 * log(S2), 1) - 0.5 * (pow(M, 2) + S2) * omega2 + 0.5 * sum(log(omega2)) + ki(Y);

    Rcpp::NumericVector Ji = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik));
    Ji.attr("weights") = w;
    return Rcpp::List::create(
      Rcpp::Named("M") = M,
      Rcpp::Named("S") = S,
      Rcpp::Named("Ji") = Ji);
}
