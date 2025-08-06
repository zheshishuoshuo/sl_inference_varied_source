"""Simple test module."""


def run():
def main():
    mock_lens_data, mock_observed_data = run_mock_simulation(n_samples=1)
    logMh_grid = np.linspace(11.5, 14.0, 100)

    logMstar_list, detJ_list = build_interp_list_for_lenses(
        mock_observed_data, logMh_grid, zl=0.3, zs=2.0
    )

    from likelihood import log_posterior, initializer_for_pool

    sampler = run_mcmc(
        data_df=mock_observed_data,
        logMstar_interp_list=logMstar_list,
        detJ_interp_list=detJ_list,
        use_interp=True,
        log_posterior_func=log_posterior,
        backend_file="mcmc_chain100.h5",
        nwalkers=8,
        nsteps=10,
        ndim=6,
        initial_guess=np.array([12.5, 2.0, 0.0, 0.3, 0.05, 0.05]),
        processes=4
        )

if __name__ == "__main__":
    run()
