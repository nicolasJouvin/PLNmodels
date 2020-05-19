#include <RcppArmadillo.h> // Includes R stuff
#include <nlopt.h>

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>       // packer system
#include <type_traits> // remove_pointer
#include <utility>     // move, forward

// ---------------------------------------------------------------------------------------
// Misc

inline arma::vec logfact(arma::mat y) {
    y.replace(0., 1.);
    return sum(y % log(y) - y + log(8 * pow(y, 3) + 4 * pow(y, 2) + y + 1. / 30.) / 6. + std::log(M_PI) / 2., 1);
}

// Returns true if the matrix or vector value only contains ones
template <typename Expr> inline bool all_ones(Expr && expr) {
    return all(vectorise(std::forward<Expr>(expr)) == 1.);
}

// ---------------------------------------------------------------------------------------
// Algorithm naming

// List of all supported algorithms and associated names
struct AlgorithmNameAssociation {
    const char * name;
    nlopt_algorithm enum_value;
};
static const AlgorithmNameAssociation supported_algorithms[] = {
    {"LBFGS_NOCEDAL", NLOPT_LD_LBFGS_NOCEDAL},
    {"LBFGS", NLOPT_LD_LBFGS},
    {"VAR1", NLOPT_LD_VAR1},
    {"VAR2", NLOPT_LD_VAR2},
    {"TNEWTON", NLOPT_LD_TNEWTON},
    {"TNEWTON_RESTART", NLOPT_LD_TNEWTON_RESTART},
    {"TNEWTON_PRECOND", NLOPT_LD_TNEWTON_PRECOND},
    {"TNEWTON_PRECOND_RESTART", NLOPT_LD_TNEWTON_PRECOND_RESTART},
    {"MMA", NLOPT_LD_MMA},
    {"CCSAQ", NLOPT_LD_CCSAQ},
};

// Retrieve the algorithm enum value associated to 'name', or throw an error
nlopt_algorithm algorithm_from_name(const std::string & name) {
    for(const AlgorithmNameAssociation & association : supported_algorithms) {
        if(name == association.name) {
            return association.enum_value;
        }
    }
    // None matched
    std::string msg;
    msg += "Unsupported algorithm name: \"";
    msg += name;
    msg += "\"\nSupported:";
    for(const AlgorithmNameAssociation & association : supported_algorithms) {
        msg += " ";
        msg += association.name;
    }
    throw Rcpp::exception(msg.c_str());
}

// ---------------------------------------------------------------------------------------
// nlopt wrapper

// Required configuration values for using an optimizer
struct OptimizerConfiguration {
    nlopt_algorithm algorithm; // Must be from the supported algorithm list.

    arma::vec xtol_abs; // of size nb_parameters
    double xtol_rel;
    arma::vec lower_bounds; // of size nb_parameters

    double ftol_abs;
    double ftol_rel;

    int maxeval;
    double maxtime;

    // Build from R list (with named elements).
    // xtol_abs and lower_bounds field should contain a sublist with values for each parameter.
    // arma::vec list_by_parameter_to_packed(Rcpp::List) : extract list values and pack them like parameters.
    template <typename F>
    static OptimizerConfiguration from_r_list(const Rcpp::List & list, F list_by_parameter_to_packed) {
        return {
            algorithm_from_name(Rcpp::as<std::string>(list["algorithm"])),

            list_by_parameter_to_packed(Rcpp::as<Rcpp::List>(list["xtol_abs"])),
            Rcpp::as<double>(list["xtol_rel"]),
            list_by_parameter_to_packed(Rcpp::as<Rcpp::List>(list["lower_bounds"])),

            Rcpp::as<double>(list["ftol_abs"]),
            Rcpp::as<double>(list["ftol_rel"]),

            Rcpp::as<int>(list["maxeval"]),
            Rcpp::as<double>(list["maxtime"]),
        };
    }
};

struct OptimizerResult {
    nlopt_result status;
    double objective;
};

// nlopt uses an opaque pointer to a declared-but-not-defined struct.
// Retrieve the struct type to use it in a std::unique_ptr
using Optimizer = std::remove_pointer<nlopt_opt>::type;

// Deleter = cleanup function called by unique_ptr, just forward to nlopt_destroy
struct OptimizerDeleter {
    void operator()(Optimizer * optimizer) const { nlopt_destroy(optimizer); }
};

// Create and configure an optimizer with everything except the optimized function.
// This function is split from templated optimize() to enable some code reuse in the compiler.
// The optimizer is stored in a unique_ptr, so it will automatically be destroyed when out of scope.
std::unique_ptr<Optimizer, OptimizerDeleter> create_configured_optimizer(
    arma::uword nb_parameters, const OptimizerConfiguration & config) {
    // Check bounds
    if(!(config.xtol_abs.n_elem == nb_parameters)) {
        throw Rcpp::exception("config.xtol_abs size");
    }
    if(!(config.lower_bounds.n_elem == nb_parameters)) {
        throw Rcpp::exception("config.lower_bounds size");
    }
    // Create
    auto optimizer = std::unique_ptr<Optimizer, OptimizerDeleter>(nlopt_create(config.algorithm, nb_parameters));
    if(!optimizer) {
        throw Rcpp::exception("nlopt_create");
    }
    // Set config
    auto check = [](nlopt_result r, const char * reason) {
        if(r != NLOPT_SUCCESS) {
            throw Rcpp::exception(reason);
        }
    };
    check(nlopt_set_xtol_abs(optimizer.get(), config.xtol_abs.memptr()), "nlopt_set_xtol_abs");
    check(nlopt_set_xtol_rel(optimizer.get(), config.xtol_rel), "nlopt_set_xtol_rel");
    check(nlopt_set_lower_bounds(optimizer.get(), config.lower_bounds.memptr()), "nlopt_set_lower_bounds");
    check(nlopt_set_ftol_abs(optimizer.get(), config.ftol_abs), "nlopt_set_ftol_abs");
    check(nlopt_set_ftol_rel(optimizer.get(), config.ftol_rel), "nlopt_set_ftol_rel");
    check(nlopt_set_maxeval(optimizer.get(), config.maxeval), "nlopt_set_maxeval");
    check(nlopt_set_maxtime(optimizer.get(), config.maxtime), "nlopt_set_maxtime");
    return optimizer;
}

// Find parameters minimizing the given objective function, under the given configuration.
template <typename ObjectiveAndGradFn> OptimizerResult minimize_objective_on_parameters(
    // Parameters are modified in place
    arma::vec & parameters,
    const OptimizerConfiguration & config,
    // The function should be a closure (stateful lambda) with the given prototype:
    // double objective_and_grad(const arma::vec & parameters, arma::vec & grad_storage);
    // It should compute and return the objective value, and store computed gradients in grad_storage.
    // Both vectors are of size nb_parameters.
    const ObjectiveAndGradFn & objective_and_grad_fn //
) {
    // Setup optimizer
    auto optimizer = create_configured_optimizer(parameters.n_elem, config);

    // A closure (stateful lambda, functor) is a struct instance with an operator() method.
    // nlopt requires a pair of function pointer and custom data pointer for the objective function.
    // The closure can fit this interface by passing the closure pointer as custom data.
    // A raw function 'optim_fn' (stateless lambda) is used to cast back the data pointer to the closure struct type.
    // The closure can then be called. Additionnally raw arrays are wrapped as arma::vec.
    auto optim_fn = [](unsigned n, const double * x, double * grad, void * data) -> double {
        // Wrap raw C arrays from nlopt into arma::vec
        const auto parameters = arma::vec(const_cast<double *>(x), n, false, true);
        auto grad_storage = arma::vec(grad, n, false, true);
        // Restore objective_and_grad_fn and call it
        const ObjectiveAndGradFn & objective_and_grad_fn = *static_cast<const ObjectiveAndGradFn *>(data);
        return objective_and_grad_fn(parameters, grad_storage);
    };
    if(nlopt_set_min_objective(
           optimizer.get(),
           optim_fn,
           const_cast<ObjectiveAndGradFn *>(&objective_and_grad_fn) // Make non-const for cast to void*
           ) != NLOPT_SUCCESS) {
        throw Rcpp::exception("nlopt_set_min_objective");
    }

    double objective = 0.;
    nlopt_result status = nlopt_optimize(optimizer.get(), parameters.memptr(), &objective);
    return {status, objective};
}

// ---------------------------------------------------------------------------------------
// Packing / unpacking utils

// Stores type, dimensions and offset for a single T object
// Must be specialised ; see specialisations for arma::vec/arma::mat below
//
// Required API when specialised:
// Constructor(const T & reference_object, arma::uword & current_offset);
// "T-like type" unpack(const arma::vec & packed_storage);
// void pack(arma::vec & packed_storage, "T-like type" object);
template <typename T> struct PackedInfo;

template <> struct PackedInfo<arma::vec> {
    arma::uword offset;
    arma::uword size;

    PackedInfo(const arma::vec & v, arma::uword & current_offset) {
        offset = current_offset;
        size = v.n_elem;
        current_offset += size;
    }

    arma::span span() const { return arma::span(offset, offset + size - 1); }

    auto unpack(const arma::vec & packed) const -> decltype(packed.subvec(span())) { return packed.subvec(span()); }

    template <typename Expr> void pack(arma::vec & packed, Expr && expr) const {
        packed.subvec(span()) = std::forward<Expr>(expr);
    }
};

template <> struct PackedInfo<arma::mat> {
    arma::uword offset;
    arma::uword rows;
    arma::uword cols;

    PackedInfo(const arma::mat & m, arma::uword & current_offset) {
        offset = current_offset;
        rows = m.n_rows;
        cols = m.n_cols;
        current_offset += rows * cols;
    }

    arma::span span() const { return arma::span(offset, offset + rows * cols - 1); }

    auto unpack(const arma::vec & packed) const
        -> decltype(arma::reshape(packed.subvec(span()), arma::size(rows, cols))) {
        return arma::reshape(packed.subvec(span()), arma::size(rows, cols));
    }

    template <typename Expr> void pack(arma::vec & packed, Expr && expr) const {
        packed.subvec(span()) = arma::vectorise(std::forward<Expr>(expr));
    }
};

// Stores packing information for multiple objects of types T1,T2,...,TN.
// Created (with type deduction) using make_packer(T1, ..., TN) below.
template <typename... Types> struct Packer {
    std::tuple<PackedInfo<Types>...> elements; // Packing info for each element (offset, type, dimensions)
    arma::uword size;                          // Total number of packed elements

    template <std::size_t Index> auto unpack(const arma::vec & packed) const
        -> decltype(std::get<Index>(elements).unpack(packed)) {
        return std::get<Index>(elements).unpack(packed);
    }
    template <std::size_t Index, typename Expr> void pack(arma::vec & packed, Expr && expr) const {
        std::get<Index>(elements).pack(packed, std::forward<Expr>(expr));
    }
};

template <typename... Types> Packer<Types...> make_packer(const Types &... values) {
    // Initialize Packer<Types...> using brace init, which guarantees evaluation order (required here !).
    // Will call each PackedInfo<T> constructor in order, increasing offset.
    // Then the final offset value will be copied into the size field.
    arma::uword current_offset = 0;
    return {
        {PackedInfo<Types>(values, current_offset)...}, // Increases offset sequentially for each value
        current_offset,                                 // Final value
    };
}

// ---------------------------------------------------------------------------------------
// Fully parametrized covariance

struct OptimizeFullParameters {
    arma::mat theta; // (p,d)
    arma::mat m;     // (n,p)
    arma::mat s;     // (n,p)

    explicit OptimizeFullParameters(const Rcpp::List & list)
        : theta(Rcpp::as<arma::mat>(list["Theta"])),
          m(Rcpp::as<arma::mat>(list["M"])),
          s(Rcpp::as<arma::mat>(list["S"])) {}
};

// [[Rcpp::export]]
Rcpp::List cpp_optimize_full(
    const Rcpp::List & init_parameters, // OptimizeFullParameters
    const arma::mat & y,                // responses (n,p)
    const arma::mat & x,                // covariates (n,d)
    const arma::mat & o,                // offsets (n,p)
    const arma::vec & w,                // weights (n)
    const Rcpp::List & configuration    // OptimizerConfiguration
) {
    // Conversion from R, prepare optimization
    const auto init = OptimizeFullParameters(init_parameters);
    const auto packer = make_packer(init.theta, init.m, init.s);
    enum { THETA, M, S }; // Nice names for packer indexes
    auto parameters = arma::vec(packer.size);
    packer.pack<THETA>(parameters, init.theta);
    packer.pack<M>(parameters, init.m);
    packer.pack<S>(parameters, init.s);

    const auto config = OptimizerConfiguration::from_r_list(configuration, [&packer](Rcpp::List list) {
        auto values = OptimizeFullParameters(list);
        auto packed = arma::vec(packer.size);
        packer.pack<THETA>(packed, values.theta);
        packer.pack<M>(packed, values.m);
        packer.pack<S>(packed, values.s);
        return packed;
    });

    const double w_bar = accu(w);
    int nb_iterations = 0;

    // Optimize
    OptimizerResult result;
    if(all_ones(w)) {
        // Version with manual simplifications for w = {1,...,1}
        auto objective_and_grad =
            [&packer, &o, &x, &y, &nb_iterations](const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            auto m = packer.unpack<M>(parameters);
            auto s = packer.unpack<S>(parameters);

            const arma::uword n = y.n_rows;
            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z + 0.5 * s);
            arma::mat omega = double(n) * inv_sympd(m.t() * m + diagmat(sum(s, 0)));
            double objective = accu(a - y % z - 0.5 * log(s)) - 0.5 * double(n) * real(log_det(omega));

            packer.pack<THETA>(grad_storage, (a - y).t() * x);
            packer.pack<M>(grad_storage, m * omega + a - y);
            packer.pack<S>(grad_storage, 0.5 * (arma::ones(n) * diagvec(omega).t() + a - 1. / s));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    } else {
        auto objective_and_grad = [&packer, &y, &x, &o, &w, &w_bar, &nb_iterations](
                                      const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            arma::mat m = packer.unpack<M>(parameters);
            arma::mat s = packer.unpack<S>(parameters);

            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z + 0.5 * s);
            arma::mat omega = w_bar * inv_sympd(m.t() * (m.each_col() % w) + diagmat(sum(s.each_col() % w, 0)));
            double objective = accu(diagmat(w) * (a - y % z - 0.5 * log(s))) - 0.5 * w_bar * real(log_det(omega));

            packer.pack<THETA>(grad_storage, (a - y).t() * (x.each_col() % w));
            packer.pack<M>(grad_storage, diagmat(w) * (m * omega + a - y));
            packer.pack<S>(grad_storage, 0.5 * (w * diagvec(omega).t() + diagmat(w) * (a - 1. / s)));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    }

    // Post process
    arma::mat theta = packer.unpack<THETA>(parameters);
    arma::mat m = packer.unpack<M>(parameters);
    arma::mat s = packer.unpack<S>(parameters);
    arma::mat z = o + x * theta.t() + m;
    arma::mat sigma = (1. / w_bar) * (m.t() * (m.each_col() % w) + diagmat(sum(s.each_col() % w, 0)));

    const arma::uword p = y.n_cols;
    arma::mat omega = inv_sympd(sigma);
    arma::mat a = exp(z + 0.5 * s);
    arma::vec loglik = sum(y % z - a + 0.5 * log(s) - 0.5 * ((m * omega) % m + s * diagmat(omega)), 0) +
                       0.5 * real(log_det(omega)) - logfact(y) + 0.5 * double(p);

    return Rcpp::List::create(
        Rcpp::Named("status", static_cast<int>(result.status)),
        Rcpp::Named("iterations", nb_iterations),
        Rcpp::Named("Theta", theta),
        Rcpp::Named("M", m),
        Rcpp::Named("S", s),
        Rcpp::Named("Z", z),
        Rcpp::Named("A", a),
        Rcpp::Named("Sigma", sigma),
        Rcpp::Named("loglik", loglik));
}

// ---------------------------------------------------------------------------------------
// Spherical covariance

struct OptimizeSphericalParameters {
    arma::mat theta; // (p,d)
    arma::mat m;     // (n,p)
    arma::vec s;     // (n)

    explicit OptimizeSphericalParameters(const Rcpp::List & list)
        : theta(Rcpp::as<arma::mat>(list["Theta"])),
          m(Rcpp::as<arma::mat>(list["M"])),
          s(Rcpp::as<arma::vec>(list["S"])) {}
};

// [[Rcpp::export]]
Rcpp::List cpp_optimize_spherical(
    const Rcpp::List & init_parameters, // OptimizeSphericalParameters
    const arma::mat & y,                // responses (n,p)
    const arma::mat & x,                // covariates (n,d)
    const arma::mat & o,                // offsets (n,p)
    const arma::vec & w,                // weights (n)
    const Rcpp::List & configuration    // OptimizerConfiguration
) {
    // Conversion from R, prepare optimization
    const auto init = OptimizeSphericalParameters(init_parameters);
    const auto packer = make_packer(init.theta, init.m, init.s);
    enum { THETA, M, S }; // Nice names for packer indexes
    auto parameters = arma::vec(packer.size);
    packer.pack<THETA>(parameters, init.theta);
    packer.pack<M>(parameters, init.m);
    packer.pack<S>(parameters, init.s);

    const auto config = OptimizerConfiguration::from_r_list(configuration, [&packer](Rcpp::List list) {
        auto values = OptimizeSphericalParameters(list);
        auto packed = arma::vec(packer.size);
        packer.pack<THETA>(packed, values.theta);
        packer.pack<M>(packed, values.m);
        packer.pack<S>(packed, values.s);
        return packed;
    });

    const double w_bar = accu(w);
    int nb_iterations = 0;

    // Optimize
    OptimizerResult result;
    if(all_ones(w)) {
        // Version with manual simplifications for w = {1,...,1}
        auto objective_and_grad =
            [&packer, &o, &x, &y, &nb_iterations](const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            auto m = packer.unpack<M>(parameters);
            auto s = packer.unpack<S>(parameters);

            const arma::uword n = y.n_rows;
            const arma::uword p = y.n_cols;
            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z.each_col() + 0.5 * s);
            double sigma2 = arma::as_scalar(accu(m % m) / double(n * p) + accu(s) / double(n));
            double objective =
                accu(a - y % z) - 0.5 * double(p) * accu(log(s)) + 0.5 * double(n * p) * std::log(sigma2);

            packer.pack<THETA>(grad_storage, (a - y).t() * x);
            packer.pack<M>(grad_storage, m / sigma2 + a - y);
            packer.pack<S>(grad_storage, 0.5 * (sum(a, 1) - double(p) * (1. / s) - double(p) / sigma2));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    } else {
        auto objective_and_grad = [&packer, &o, &x, &y, &w, &w_bar, &nb_iterations](
                                      const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            arma::mat m = packer.unpack<M>(parameters);
            auto s = packer.unpack<S>(parameters);

            const arma::uword p = y.n_cols;
            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z.each_col() + 0.5 * s);
            double sigma2 = arma::as_scalar(accu(m % (m.each_col() % w)) / (w_bar * double(p)) + accu(w % s) / w_bar);
            double objective = accu(diagmat(w) * (a - y % z)) - 0.5 * double(p) * accu(w % log(s)) +
                               0.5 * w_bar * double(p) * log(sigma2);

            packer.pack<THETA>(grad_storage, (a - y).t() * (x.each_col() % w));
            packer.pack<M>(grad_storage, diagmat(w) * (m / sigma2 + a - y));
            packer.pack<S>(grad_storage, w % (0.5 * (sum(a, 1) - double(p) * (1. / s) - double(p) / sigma2)));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    }

    // Post process
    arma::mat theta = packer.unpack<THETA>(parameters);
    arma::mat m = packer.unpack<M>(parameters);
    arma::mat s = packer.unpack<S>(parameters); // vec(n) -> mat(n, 1)
    arma::mat z = o + x * theta.t() + m;
    const arma::uword p = y.n_cols;
    double sigma2 = arma::as_scalar(accu(m % m) / (w_bar * double(p)) + accu(s) / w_bar);
    arma::mat sigma = arma::eye(p, p) * sigma2;

    arma::mat a = exp(z.each_col() + 0.5 * s);
    arma::mat loglik = sum(y % z - a, 1) + 0.5 * double(p) * log(s) - (0.5 / sigma2) * (sum(m % m, 1) + double(p) * s) -
                       0.5 * double(p) * log(sigma2) - logfact(y) + 0.5 * double(p);

    return Rcpp::List::create(
        Rcpp::Named("status", static_cast<int>(result.status)),
        Rcpp::Named("iterations", nb_iterations),
        Rcpp::Named("Theta", theta),
        Rcpp::Named("M", m),
        Rcpp::Named("S", s),
        Rcpp::Named("Z", z),
        Rcpp::Named("A", a),
        Rcpp::Named("Sigma", sigma),
        Rcpp::Named("loglik", loglik));
}

// ---------------------------------------------------------------------------------------
// Diagonal covariance

struct OptimizeDiagonalParameters {
    arma::mat theta; // (p,d)
    arma::mat m;     // (n,p)
    arma::mat s;     // (n,p)

    explicit OptimizeDiagonalParameters(const Rcpp::List & list)
        : theta(Rcpp::as<arma::mat>(list["Theta"])),
          m(Rcpp::as<arma::mat>(list["M"])),
          s(Rcpp::as<arma::mat>(list["S"])) {}
};

// [[Rcpp::export]]
Rcpp::List cpp_optimize_diagonal(
    const Rcpp::List & init_parameters, // OptimizeDiagonalParameters
    const arma::mat & y,                // responses (n,p)
    const arma::mat & x,                // covariates (n,d)
    const arma::mat & o,                // offsets (n,p)
    const arma::vec & w,                // weights (n)
    const Rcpp::List & configuration    // OptimizerConfiguration
) {
    // Conversion from R, prepare optimization
    const auto init = OptimizeDiagonalParameters(init_parameters);
    const auto packer = make_packer(init.theta, init.m, init.s);
    enum { THETA, M, S }; // Nice names for packer indexes
    auto parameters = arma::vec(packer.size);
    packer.pack<THETA>(parameters, init.theta);
    packer.pack<M>(parameters, init.m);
    packer.pack<S>(parameters, init.s);

    const auto config = OptimizerConfiguration::from_r_list(configuration, [&packer](Rcpp::List list) {
        auto values = OptimizeDiagonalParameters(list);
        auto packed = arma::vec(packer.size);
        packer.pack<THETA>(packed, values.theta);
        packer.pack<M>(packed, values.m);
        packer.pack<S>(packed, values.s);
        return packed;
    });

    const double w_bar = accu(w);
    int nb_iterations = 0;

    // Optimize
    OptimizerResult result;
    if(all_ones(w)) {
        // Version with manual simplifications for w = {1,...,1}
        auto objective_and_grad =
            [&packer, &o, &x, &y, &nb_iterations](const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            arma::mat m = packer.unpack<M>(parameters);
            auto s = packer.unpack<S>(parameters);

            const arma::uword n = y.n_rows;
            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z + 0.5 * s);
            arma::rowvec diag_sigma = sum(m % m + s, 0) / double(n);
            double objective = accu(a - y % z - 0.5 * log(s)) + 0.5 * double(n) * accu(log(diag_sigma));

            packer.pack<THETA>(grad_storage, (a - y).t() * x);
            packer.pack<M>(grad_storage, (m.each_row() / diag_sigma) + a - y);
            packer.pack<S>(grad_storage, 0.5 * (arma::ones(n) * (1. / diag_sigma) + a - 1. / s));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    } else {
        auto objective_and_grad = [&packer, &o, &x, &y, &w, &w_bar, &nb_iterations](
                                      const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            arma::mat m = packer.unpack<M>(parameters);
            arma::mat s = packer.unpack<S>(parameters);

            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z + 0.5 * s);
            arma::rowvec diag_sigma = sum(m % (m.each_col() % w) + (s.each_col() % w), 0) / w_bar;
            double objective = accu(diagmat(w) * (a - y % z - 0.5 * log(s))) + 0.5 * w_bar * accu(log(diag_sigma));

            packer.pack<THETA>(grad_storage, (a - y).t() * (x.each_col() % w));
            packer.pack<M>(grad_storage, diagmat(w) * ((m.each_row() / diag_sigma) + a - y));
            packer.pack<S>(grad_storage, 0.5 * (w * (1. / diag_sigma) + diagmat(w) * (a - 1. / s)));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    }

    // Post process
    arma::mat theta = packer.unpack<THETA>(parameters);
    arma::mat m = packer.unpack<M>(parameters);
    arma::mat s = packer.unpack<S>(parameters);
    arma::mat z = o + x * theta.t() + m;
    arma::rowvec diag_sigma = sum(m % (m.each_col() % w) + (s.each_col() % w), 0) / w_bar;
    arma::mat sigma = diagmat(diag_sigma);

    arma::mat a = exp(z + 0.5 * s);
    const arma::uword p = y.n_cols;
    arma::mat loglik =
        sum(y % z - a + 0.5 * log(s) - 0.5 * ((m.each_row() / diag_sigma) % m + (s.each_row() / diag_sigma)), 1) -
        0.5 * accu(log(diag_sigma)) - logfact(y) + 0.5 * double(p);

    return Rcpp::List::create(
        Rcpp::Named("status", static_cast<int>(result.status)),
        Rcpp::Named("iterations", nb_iterations),
        Rcpp::Named("Theta", theta),
        Rcpp::Named("M", m),
        Rcpp::Named("S", s),
        Rcpp::Named("Z", z),
        Rcpp::Named("A", a),
        Rcpp::Named("Sigma", sigma),
        Rcpp::Named("loglik", loglik));
}

// ---------------------------------------------------------------------------------------
// Rank-constrained covariance

// Rank (q) has been removed from config ; it is already determined by param dimensions

struct OptimizeRankParameters {
    arma::mat theta; // (p,d)
    arma::mat b;     // (p,q)
    arma::mat m;     // (n,q)
    arma::mat s;     // (n,q)

    explicit OptimizeRankParameters(const Rcpp::List & list)
        : theta(Rcpp::as<arma::mat>(list["Theta"])),
          b(Rcpp::as<arma::mat>(list["B"])),
          m(Rcpp::as<arma::mat>(list["M"])),
          s(Rcpp::as<arma::mat>(list["S"])) {}
};

// [[Rcpp::export]]
Rcpp::List cpp_optimize_rank(
    const Rcpp::List & init_parameters, // OptimizeRankParameters
    const arma::mat & y,                // responses (n,p)
    const arma::mat & x,                // covariates (n,d)
    const arma::mat & o,                // offsets (n,p)
    const arma::vec & w,                // weights (n)
    const Rcpp::List & configuration    // OptimizerConfiguration
) {
    // Conversion from R, prepare optimization
    const auto init = OptimizeRankParameters(init_parameters);
    const auto packer = make_packer(init.theta, init.b, init.m, init.s);
    enum { THETA, B, M, S }; // Nice names for packer indexes
    auto parameters = arma::vec(packer.size);
    packer.pack<THETA>(parameters, init.theta);
    packer.pack<B>(parameters, init.b);
    packer.pack<M>(parameters, init.m);
    packer.pack<S>(parameters, init.s);

    const auto config = OptimizerConfiguration::from_r_list(configuration, [&packer](Rcpp::List list) {
        auto values = OptimizeRankParameters(list);
        auto packed = arma::vec(packer.size);
        packer.pack<THETA>(packed, values.theta);
        packer.pack<B>(packed, values.b);
        packer.pack<M>(packed, values.m);
        packer.pack<S>(packed, values.s);
        return packed;
    });

    int nb_iterations = 0;

    // Optimize
    OptimizerResult result;
    if(all_ones(w)) {
        // Version with manual simplifications for w = {1,...,1}
        auto objective_and_grad =
            [&packer, &o, &x, &y, &nb_iterations](const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            auto b = packer.unpack<B>(parameters);
            auto m = packer.unpack<M>(parameters);
            auto s = packer.unpack<S>(parameters);

            arma::mat z = o + x * theta.t() + m * b.t();
            arma::mat a = exp(z + 0.5 * s * (b % b).t());
            double objective = accu(a - y % z) + 0.5 * accu(m % m + s - log(s) - 1);

            packer.pack<THETA>(grad_storage, (a - y).t() * x);
            packer.pack<B>(grad_storage, (a - y).t() * m + (a.t() * s) % b);
            packer.pack<M>(grad_storage, (a - y) * b + m);
            packer.pack<S>(grad_storage, .5 * (1. - 1. / s + a * (b % b)));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    } else {
        auto objective_and_grad = [&packer, &o, &x, &y, &w, &nb_iterations](
                                      const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            auto b = packer.unpack<B>(parameters);
            auto m = packer.unpack<M>(parameters);
            arma::mat s = packer.unpack<S>(parameters);

            arma::mat z = o + x * theta.t() + m * b.t();
            arma::mat a = exp(z + 0.5 * s * (b % b).t());
            double objective = accu(diagmat(w) * (a - y % z)) + 0.5 * accu(diagmat(w) * (m % m + s - log(s) - 1.));

            packer.pack<THETA>(grad_storage, (a - y).t() * (x.each_col() % w));
            packer.pack<B>(grad_storage, (diagmat(w) * (a - y)).t() * m + (a.t() * (s.each_col() % w)) % b);
            packer.pack<M>(grad_storage, diagmat(w) * ((a - y) * b + m));
            packer.pack<S>(grad_storage, .5 * diagmat(w) * (1. - 1. / s + a * (b % b)));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    }

    // Post process
    arma::mat theta = packer.unpack<THETA>(parameters);
    arma::mat b = packer.unpack<B>(parameters);
    arma::mat m = packer.unpack<M>(parameters);
    arma::mat s = packer.unpack<S>(parameters);
    arma::mat z = o + x * theta.t() + m * b.t();
    const arma::uword n = y.n_rows;
    arma::mat sigma = b * (m.t() * m + diagmat(sum(s, 0))) * b.t() / double(n);

    arma::mat a = exp(z + 0.5 * s * (b % b).t());
    arma::mat loglik = arma::sum(y % z - a, 1) - 0.5 * sum(m % m + s - log(s) - 1., 1) - logfact(y);

    return Rcpp::List::create(
        Rcpp::Named("status", static_cast<int>(result.status)),
        Rcpp::Named("iterations", nb_iterations),
        Rcpp::Named("Theta", theta),
        Rcpp::Named("B", b),
        Rcpp::Named("M", m),
        Rcpp::Named("S", s),
        Rcpp::Named("Z", z),
        Rcpp::Named("A", a),
        Rcpp::Named("Sigma", sigma),
        Rcpp::Named("loglik", loglik));
}

// ---------------------------------------------------------------------------------------
// Sparse inverse covariance

struct OptimizeSparseParameters {
    arma::mat theta; // (p,d)
    arma::mat m;     // (n,p)
    arma::mat s;     // (n,p)

    explicit OptimizeSparseParameters(const Rcpp::List & list)
        : theta(Rcpp::as<arma::mat>(list["Theta"])),
          m(Rcpp::as<arma::mat>(list["M"])),
          s(Rcpp::as<arma::mat>(list["S"])) {}
};

// [[Rcpp::export]]
Rcpp::List cpp_optimize_sparse(
    const Rcpp::List & init_parameters, // OptimizeSparseParameters
    const arma::mat & y,                // responses (n,p)
    const arma::mat & x,                // covariates (n,d)
    const arma::mat & o,                // offsets (n,p)
    const arma::vec & w,                // weights (n)
    const arma::mat & omega,            // covinv (p,p)
    const Rcpp::List & configuration    // OptimizerConfiguration
) {
    // Conversion from R, prepare optimization
    const auto init = OptimizeSparseParameters(init_parameters);
    const auto packer = make_packer(init.theta, init.m, init.s);
    enum { THETA, M, S }; // Nice names for packer indexes
    auto parameters = arma::vec(packer.size);
    packer.pack<THETA>(parameters, init.theta);
    packer.pack<M>(parameters, init.m);
    packer.pack<S>(parameters, init.s);

    const auto config = OptimizerConfiguration::from_r_list(configuration, [&packer](Rcpp::List list) {
        auto values = OptimizeSparseParameters(list);
        auto packed = arma::vec(packer.size);
        packer.pack<THETA>(packed, values.theta);
        packer.pack<M>(packed, values.m);
        packer.pack<S>(packed, values.s);
        return packed;
    });

    const double log_det_omega = real(log_det(omega));
    const double w_bar = accu(w);
    int nb_iterations = 0;

    // Optimize
    OptimizerResult result;
    if(all_ones(w)) {
        // Version with manual simplifications for w = {1,...,1}
        auto objective_and_grad = [&packer, &o, &x, &y, &omega, &log_det_omega, &nb_iterations](
                                      const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            auto m = packer.unpack<M>(parameters);
            auto s = packer.unpack<S>(parameters);

            const arma::uword n = y.n_rows;
            const arma::uword p = y.n_cols;
            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z + 0.5 * s);
            arma::mat nSigma = m.t() * m + diagmat(sum(s, 0));
            double objective = accu(a - y % z - 0.5 * log(s)) -
                               0.5 * (double(n) * log_det_omega + double(n * p) - trace(omega * nSigma));

            packer.pack<THETA>(grad_storage, (a - y).t() * x);
            packer.pack<M>(grad_storage, m * omega + a - y);
            packer.pack<S>(grad_storage, 0.5 * (arma::ones(n) * diagvec(omega).t() + a - 1. / s));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    } else {
        auto objective_and_grad = [&packer, &o, &x, &y, &w, &w_bar, &omega, &log_det_omega, &nb_iterations](
                                      const arma::vec & parameters, arma::vec & grad_storage) -> double {
            auto theta = packer.unpack<THETA>(parameters);
            arma::mat m = packer.unpack<M>(parameters);
            arma::mat s = packer.unpack<S>(parameters);

            const arma::uword p = y.n_cols;
            arma::mat z = o + x * theta.t() + m;
            arma::mat a = exp(z + 0.5 * s);
            arma::mat nSigma = m.t() * (m.each_col() % w) + diagmat(sum(s.each_col() % w, 0));
            double objective = accu(diagmat(w) * (a - y % z - 0.5 * log(s))) -
                               0.5 * (w_bar * log_det_omega + w_bar * double(p) - trace(omega * nSigma));

            packer.pack<THETA>(grad_storage, (a - y).t() * (x.each_col() % w));
            packer.pack<M>(grad_storage, diagmat(w) * (m * omega + a - y));
            packer.pack<S>(grad_storage, 0.5 * (w * diagvec(omega).t() + diagmat(w) * (a - 1. / s)));
            nb_iterations += 1;
            return objective;
        };
        result = minimize_objective_on_parameters(parameters, config, objective_and_grad);
    }

    // Post process
    arma::mat theta = packer.unpack<THETA>(parameters);
    arma::mat m = packer.unpack<M>(parameters);
    arma::mat s = packer.unpack<S>(parameters);
    arma::mat z = o + x * theta.t() + m;
    const arma::uword n = y.n_rows;
    const arma::uword p = y.n_cols;
    arma::mat sigma = (m.t() * (m.each_col() % w) + diagmat(sum(s.each_col() % w, 0))) / accu(w);

    arma::mat a = exp(z + 0.5 * s);
    arma::mat loglik = sum(y % z - a + 0.5 * log(s) - 0.5 * ((m * omega) % m + s * diagmat(omega)), 1) +
                       0.5 * log_det_omega - logfact(y) + 0.5 * double(p);

    return Rcpp::List::create(
        Rcpp::Named("status", static_cast<int>(result.status)),
        Rcpp::Named("iterations", nb_iterations),
        Rcpp::Named("Theta", theta),
        Rcpp::Named("M", m),
        Rcpp::Named("S", s),
        Rcpp::Named("Z", z),
        Rcpp::Named("A", a),
        Rcpp::Named("Sigma", sigma),
        Rcpp::Named("loglik", loglik));
}

// TODO

// VE step thing

// New model