def box_qp_control(max_iters=1000, eps_abs=0.001, eps_rel=0.001, check_termination=1,
                   rho=1, verbose=False, scaling_iter=0, aa_iter=0, reduce='max',
                   unroll=False, backward='fixed_point'):
    control = {"max_iters": max_iters,
               "eps_abs": eps_abs,
               "eps_rel": eps_rel,
               "check_terimnation": check_termination,
               "rho": rho,
               "verbose": verbose,
               "scaling_iter": scaling_iter,
               "aa_iter": aa_iter,
               "reduce": reduce,
               "unroll": unroll,
               "backward": backward
               }
    return control


def optnet_control(max_iters=10, tol=0.001, check_termination=1, verbose=False, reduce='max', int_reg=10 ** -6):
    control = {"max_iters": max_iters,
               "tol": tol,
               "check_terimnation": check_termination,
               "verbose": verbose,
               "reduce": reduce,
               "int_reg": int_reg
               }
    return control


def scs_control(use_indirect=False,
                mkl=False,
                gpu=False,
                verbose=False,
                normalize=True,
                max_iters=int(1e5),
                scale=0.1,
                adaptive_scale=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                eps_infeas=1e-7,
                alpha=1.5,
                rho_x=1e-6,
                acceleration_lookback=10,
                acceleration_interval=10,
                time_limit_secs=0,
                write_data_filename=None,
                log_csv_filename=None):
    control = {"use_indirect": use_indirect,
               "mkl": mkl,
               "gpu": gpu,
               "verbose": verbose,
               "normalize": normalize,
               "max_iters": max_iters,
               "scale": scale,
               "adaptive_scale": adaptive_scale,
               "eps_abs": eps_abs,
               "eps_rel": eps_rel,
               "eps_infeas": eps_infeas,
               "alpha": alpha,
               "rho_x": rho_x,
               "acceleration_lookback": acceleration_lookback,
               "acceleration_interval": acceleration_interval,
               "time_limit_secs": time_limit_secs,
               "write_data_filename": write_data_filename,
               "log_csv_filename": log_csv_filename}

    return control
