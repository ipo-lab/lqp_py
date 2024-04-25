def box_qp_control(max_iters=10_000, eps_abs=1e-3, eps_rel=1e-3, check_solved=None,
                   rho=None, rho_min=1e-6, rho_max=1e6, adaptive_rho=True, adaptive_rho_tol=10,
                   adaptive_rho_iter=100, adaptive_rho_max_iter=1000, adaptive_rho_threshold=1e-5,
                   verbose=False, scale=True, beta=None, unroll=False, backward='fixed_point', **kwargs):
    control = {"max_iters": max_iters,
               "eps_abs": eps_abs,
               "eps_rel": eps_rel,
               "check_terimnation": check_solved,
               "rho": rho,
               "rho_min": rho_min,
               "rho_max": rho_max,
               "adaptive_rho": adaptive_rho,
               "adaptive_rho_tol": adaptive_rho_tol,
               "adaptive_rho_iter": adaptive_rho_iter,
               "adaptive_rho_max_iter": adaptive_rho_max_iter,
               "adaptive_rho_threshold": adaptive_rho_threshold,
               "verbose": verbose,
               "scale": scale,
               "unroll": unroll,
               "beta": beta,
               "backward": backward
               }
    control.update(**kwargs)
    return control


def optnet_control(max_iters=10, tol=1e-3, check_solved=1, verbose=False, reduce='max', int_reg=1e-6,  **kwargs):
    control = {"max_iters": max_iters,
               "tol": tol,
               "check_terimnation": check_solved,
               "verbose": verbose,
               "reduce": reduce,
               "int_reg": int_reg
               }
    control.update(**kwargs)
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
                log_csv_filename=None,
                **kwargs):
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
    control.update(**kwargs)
    return control
