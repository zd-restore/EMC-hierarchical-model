# -*- coding: utf-8 -*-

"""
Bayesian single lumped EMC model.

Input file:
    lumped_model_data.csv

Required columns:
    Event index
    log EMC
    Total rainfall
    Event duration
    Peak rainfall
    ADD
    season

"""

import numpy as np
import pandas as pd
import emcee
from pathlib import Path


# ----------------------------------------------------
# overall setting
# change path when needed  

DATA_FILE = Path(
    r"C:\Users\zdong262\OneDrive - The University of Western Ontario"
    r"\Desktop\WQ\EMC\revision\lumped_model_data.csv"
)

BACKEND_FILE = Path(
    r"C:\Users\zdong262\OneDrive - The University of Western Ontario"
    r"\Desktop\WQ\EMC\revision\lumped_model_chain.h5"
)

N_WALKERS = 16
N_STEPS = 20000
RANDOM_SEED = 11

RESET_BACKEND = True # start a new chain


# ----------------------------------------------------
# data

data = pd.read_csv(DATA_FILE)

# log-transformed EMC
emc_o = data["log EMC"].to_numpy(dtype=float)

# standardized predictors
predictor_names = [
    "Total rainfall",
    "Event duration",
    "Peak rainfall",
    "ADD",
]

variables = data[predictor_names].to_numpy(dtype=float)

# categorical season: summer = 0; fall = 1
season_mapping = {
    "summer": 0,
    "fall": 1,
}

season_labels = (
    data["season"]
    .astype(str)
    .str.strip()
    .str.lower()
)

event_seasons = season_labels.map(season_mapping)

if event_seasons.isna().any():
    raise ValueError(
        "Season must be either 'summer' or 'fall'."
    )

event_seasons = event_seasons.to_numpy(dtype=int)

n_events = len(data)


# ----------------------------------------------------
# parameter setup

n_landuses = 1
n_seasons = 2
variable_n = len(predictor_names)
n_sigmas = 1

PARAM_DIM = (
    n_landuses
    + n_seasons * n_landuses
    + variable_n * n_landuses
    + n_sigmas
)

parameter_names = [
    "global",
    "summer",
    "fall",
    "beta_total_rainfall",
    "beta_event_duration",
    "beta_peak_rainfall",
    "beta_ADD",
    "sigma_res",
]


# ----------------------------------------------------
# parameter split

def unpack_p(params):

    params = np.asarray(params, dtype=float)

    base_len = n_landuses
    season_len = n_seasons * n_landuses
    variable_len = variable_n * n_landuses

    if len(params) != PARAM_DIM:
        raise ValueError(
            f"Parameter vector has length {len(params)}, "
            f"but {PARAM_DIM} parameters are expected."
        )

    global_emc = params[:base_len]

    seasonal_devs = params[
        base_len:
        base_len + season_len
    ]

    offset = base_len + season_len

    variable_coefs = params[
        offset:
        offset + variable_len
    ]

    sigma_res = params[-1]

    return (
        global_emc,
        seasonal_devs,
        variable_coefs,
        sigma_res,
    )


# ----------------------------------------------------
# prior setup

def log_prior(params):

    params = np.asarray(params, dtype=float)

    if len(params) != PARAM_DIM:
        return -np.inf

    if not np.all(np.isfinite(params)):
        return -np.inf

    try:
        (
            global_emc,
            seasonal_devs,
            variable_coefs,
            sigma_res,
        ) = unpack_p(params)

    except ValueError:
        return -np.inf

    if sigma_res <= 0:
        return -np.inf

    logp = 0.0

    # global parameter ~ Normal(5, 0.5^2)
    global_mean = 5.0
    global_sd = 0.5

    logp += np.sum(
        -0.5 * (
            (global_emc - global_mean)
            / global_sd
        ) ** 2
        - np.log(global_sd)
        - 0.5 * np.log(2.0 * np.pi)
    )

    # seasonal parameters ~ Normal(0, 0.5^2)
    seasonal_sd = 0.5

    logp += np.sum(
        -0.5 * (
            seasonal_devs
            / seasonal_sd
        ) ** 2
        - np.log(seasonal_sd)
        - 0.5 * np.log(2.0 * np.pi)
    )

    # coefficients ~ Normal(0, 0.5^2)
    coefficient_sd = 0.5

    logp += np.sum(
        -0.5 * (
            variable_coefs
            / coefficient_sd
        ) ** 2
        - np.log(coefficient_sd)
        - 0.5 * np.log(2.0 * np.pi)
    )

    # residual SD ~ HalfNormal(1)
    sigma_scale = 1.0

    logp += (
        np.log(2.0)
        - np.log(sigma_scale)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * (
            sigma_res
            / sigma_scale
        ) ** 2
    )

    if not np.isfinite(logp):
        return -np.inf

    return float(logp)


# ----------------------------------------------------
# likelihood setup

def log_likelihood(
    params,
    variables,
    emc_o,
    event_seasons,
):

    (
        global_emc,
        seasonal_devs,
        variable_coefs,
        sigma_res,
    ) = unpack_p(params)

    if sigma_res <= 0:
        return np.full(
            len(emc_o),
            -np.inf,
        )

    variables_array = np.asarray(
        variables,
        dtype=float,
    )

    emc_o = np.asarray(
        emc_o,
        dtype=float,
    )

    event_seasons = np.asarray(
        event_seasons,
        dtype=int,
    )

    season_para = seasonal_devs[event_seasons]

    event_para = (
        variables_array
        @ variable_coefs
    )

    predicted_log_emc = (
        global_emc[0]
        + season_para
        + event_para
    )

    residual = (
        emc_o
        - predicted_log_emc
    )

    pointwise_log_like = (
        -0.5 * (
            residual
            / sigma_res
        ) ** 2
        - np.log(sigma_res)
        - 0.5 * np.log(
            2.0 * np.pi
        )
    )

    return pointwise_log_like


# ----------------------------------------------------
# posterior setup

def log_posterior(
    params,
    variables,
    emc_o,
    event_seasons,
):

    lp = log_prior(params)

    if not np.isfinite(lp):
        return (
            -np.inf,
            np.full(
                len(emc_o),
                np.nan,
            ),
            np.nan,
        )

    pointwise_log_like = log_likelihood(
        params,
        variables,
        emc_o,
        event_seasons,
    )

    if not np.all(
        np.isfinite(
            pointwise_log_like
        )
    ):
        return (
            -np.inf,
            np.full(
                len(emc_o),
                np.nan,
            ),
            lp,
        )

    total_log_like = np.sum(
        pointwise_log_like
    )

    log_post = (
        lp
        + total_log_like
    )

    return (
        log_post,
        pointwise_log_like,
        lp,
    )


# ----------------------------------------------------
# initial walkers

def initialize_walkers(rng):

    initial_center = np.array(
        [
            5.0,  # global
            0.0,  # summer
            0.0,  # fall
            0.0,  # beta_total_rainfall
            0.0,  # beta_event_duration
            0.0,  # beta_peak_rainfall
            0.0,  # beta_ADD
            1.0,  # sigma_res
        ],
        dtype=float,
    )

    walkers = np.zeros(
        (
            N_WALKERS,
            PARAM_DIM,
        ),
        dtype=float,
    )

    for walker_i in range(N_WALKERS):

        proposal = (
            initial_center
            + rng.normal(
                loc=0.0,
                scale=0.02,
                size=PARAM_DIM,
            )
        )

        proposal[-1] = max(
            proposal[-1],
            0.05,
        )

        walkers[walker_i] = proposal

    return walkers


# ----------------------------------------------------
# run model

def run_mcmc():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    initial_state = initialize_walkers(
        rng
    )

    blobs_dtype = [
        (
            "log_likelihood",
            float,
            (n_events,),
        ),
        (
            "log_prior",
            float,
        ),
    ]

    backend = emcee.backends.HDFBackend(
        str(BACKEND_FILE)
    )

    if RESET_BACKEND:
        backend.reset(
            N_WALKERS,
            PARAM_DIM,
        )

    sampler = emcee.EnsembleSampler(
        N_WALKERS,
        PARAM_DIM,
        log_posterior,
        args=(
            variables,
            emc_o,
            event_seasons,
        ),
        blobs_dtype=blobs_dtype,
        backend=backend,
    )

    sampler.run_mcmc(
        initial_state,
        N_STEPS,
        progress=True,
    )

    return sampler


sampler = run_mcmc()

# ----------------------------------------------------
# read posterior samples

BURN_IN = 1000

posterior_samples = sampler.get_chain(
    discard=BURN_IN,
    flat=True
)

print(
    "Posterior sample shape:",
    posterior_samples.shape
)