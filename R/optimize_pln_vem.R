# ZI optimization:
#
# init_parameters: named list(Theta, Theta0, Omega, M, S2, Pi)
# Y, X, O: observed data, constant
# configuration: named list(algorithm, xtol_rel, xtol_abs, ftol_rel, ftol_abs, maxeval, maxtime)
#
# Returns named list(parameters: named list, nb_iter: int, stop_reason: str)
#
# Configuration :
# Forwarded to nloptr iterative steps (c,e,f).
# Except for algorithm, the keys are optional termination criteria. Not all of them are needed.
# A missing key or a key set to a negative number means disabled.
# xtol_abs: double for every parameter, or named list with values for each parameter.
#
# xtol_abs, xtol_rel, maxeval are also used for overall loop.
# ftol_abs and ftol_rel could maybe be used if defined on J(params), but I'm not sure.
#
# Dimensions are checked only on C++ side.
optimize_vem <- function(init_parameters, Y, X, O, configuration) {

    config_optim_M <- config_for("M", configuration)
    config_optim_S <- config_for("S", configuration)

    # Link to the approximate function to optimize Omega ,depending on the target structure
    cpp_optimize_pln_Omega <- switch(
      configuration$covariance,
      "spherical" = cpp_optimize_pln_Omega_spherical,
      "diagonal"  = cpp_optimize_pln_Omega_diagonal,
      "full"      = cpp_optimize_pln_Omega_full
      )

    maxit_out <- if("maxit_out" %in% names(configuration)) { configuration$maxit_out } else {50}

    # Main loop
    nb_iter <- 0
    parameters <- init_parameters
    criterion <- vector("numeric", maxit_out)
    vloglik <- -Inf; objective <- Inf
    repeat {
        # Check maxeval
        if(maxit_out >= 0 && nb_iter >= maxit_out) {
            return(list(
                parameters = parameters,
                nb_iter = nb_iter,
                stop_reason = "maximum number of iterations reached",
                criterion = criterion[1:nb_iter],
                vloglik = vloglik
            ))
        }

        # Steps

        new_Omega <- cpp_optimize_pln_Omega(M = parameters$M, X = X, Theta = parameters$Theta, S = parameters$S)
        new_Theta <- cpp_optimize_pln_Theta(M = parameters$M, X = X)
        # new_VE <- cpp_optimize_pln_VE(init_M = parameters$M, init_S = parameters$S,
        #                               Y =Y, O = O, X = X,
        #                               new_Theta, Omega = new_Omega, configuration)
        # new_M <- new_VE$M; new_S <- new_VE$S
        new_M <- cpp_optimize_pln_M(
            init_M = parameters$M,
            Y = Y, X = X, O = O, S = parameters$S, Theta = new_Theta, Omega = new_Omega,
            configuration = config_optim_M
        )$M
        new_S <- cpp_optimize_pln_S(
            init_S = parameters$S,
            O = O, M = new_M, Theta = new_Theta, diag_Omega = diag(new_Omega),
            configuration = config_optim_S
        )$S

        # Check convergence
        new_parameters <- list(Omega = new_Omega, Theta = new_Theta, M = new_M, S = new_S)
        nb_iter <- nb_iter + 1

        vloglik <- cpp_optimize_pln_vloglik(Y, X, O, new_Omega, new_Theta, new_M, new_S)

        criterion[nb_iter] <- new_objective <- -sum(vloglik)

        objective_converged <-
            abs(objective - new_objective) < configuration$ftol_out |
            abs(objective - new_objective)/abs(new_objective) < configuration$ftol_out

        parameters_converged <- parameter_list_converged(
            parameters, new_parameters,
            xtol_abs = configuration$xtol_abs, xtol_rel = configuration$xtol_rel
        )

        if (parameters_converged | objective_converged) {
            return(list(
                parameters = parameters,
                nb_iter = nb_iter,
                stop_reason = "converged",
                criterion = criterion[1:nb_iter],
                vloglik = vloglik
            ))
        }

        parameters <- new_parameters
        objective  <- new_objective
    }
}

