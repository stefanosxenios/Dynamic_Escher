#!/usr/bin/env python3
import json
import sys
import numpy as np
import pandas as pd

#!/usr/bin/env python3
import json
import sys
import numpy as np
import pandas as pd

from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, load_concentrations
from skimpy.analysis.ode.utils import make_flux_fun
from skimpy.utils.namespace import QSSA
from skimpy.utils.tabdict import TabDict
from skimpy.core.parameters import load_parameter_population


# ------------------- CONSTANTS -----------------------
NCPU = 4
TOUT = np.logspace(-8, 2, 1000)
CONCENTRATION_SCALING = 1e6
TIME_SCALING = 1
DENSITY = 1521
GDW_GWW_RATIO = 0.3

DEFAULT_SAMPLE_ID = 144

path_to_kmodel  = './skimpy_runner/models/kmodel3_last.yml'
path_to_tmodel  = './skimpy_runner/models/tmodel_last.json'
path_to_samples = './skimpy_runner/sample1.csv'
path_to_params  = './skimpy_runner/pruned_parameters_2032025.hdf5'
# -----------------------------------------------------


def run_simulation_multi(perturbations):
    """
    perturbations = {
        "PFK": 0.5,
        "PGI": 1.2,
        "ENO": 0.8
    }

    This function:
      1) Loads models
      2) Loads parameters
      3) Applies ALL vmax changes BEFORE solving the ODE
      4) Runs single ODE simulation
      5) Computes fluxes
      6) Applies the SAME post-scaling logic you require
      7) Returns perturbed + reference flux dicts
    """

    # ------------- Load models ------------------------
    tmodel = load_json_model(path_to_tmodel)
    kmodel = load_yaml_model(path_to_kmodel)

    # ------------- Load parameter population ----------
    parameter_population = load_parameter_population(path_to_params)
    ix = '0,3496'    # your choice to always use this one
    kmodel.parameters = parameter_population[ix]

    # ------------- Load TFA sample --------------------
    samples = pd.read_csv(path_to_samples, index_col=0)
    sample = samples.loc[DEFAULT_SAMPLE_ID]

    # ------------- ODE model setup --------------------
    kmodel.prepare(mca=False)
    kmodel.compile_ode(sim_type=QSSA, ncpu=NCPU)

    concentrations = load_concentrations(
        sample, tmodel, kmodel,
        concentration_scaling=CONCENTRATION_SCALING
    )
    kmodel.initial_conditions = TabDict([(k, v) for k, v in concentrations.items()])

    # steady-state reference
    fluxes_steady = load_fluxes(
        sample, tmodel, kmodel,
        density=DENSITY,
        ratio_gdw_gww=GDW_GWW_RATIO,
        concentration_scaling=CONCENTRATION_SCALING,
        time_scaling=TIME_SCALING
    )

    f = pd.DataFrame(data=fluxes_steady / 456000, columns=['Steady'])

    # ----------------------------------------------------------
    # 🔥 APPLY ALL ENZYME PERTURBATIONS TO vmax BEFORE ODE SOLVE
    # ----------------------------------------------------------
    for enzyme_id, fold_change in perturbations.items():
        if enzyme_id not in kmodel.reactions:
            raise ValueError(f"Reaction '{enzyme_id}' not found in kinetic model.")

        param_name = str(kmodel.reactions[enzyme_id].parameters.vmax_forward.symbol)
        kmodel.parameters[param_name].value *= fold_change

    # ----------------------------------------------------------
    # 🔥 SINGLE ODE SOLVE AFTER ALL PERTURBATIONS
    # ----------------------------------------------------------
    sol = kmodel.solve_ode(
        TOUT,
        solver_type='cvode',
        rtol=1e-6,
        atol=1e-6,
        max_steps=1e9,
    )

    cons = sol.concentrations.iloc[-1]
    flux_fun = make_flux_fun(kmodel, sim_type=QSSA)

    # raw simulated fluxes
    flux = flux_fun.__call__(cons, parameter_population[ix])

    # ----------------------------------------------------------
    # 🔥 POST-SCALING EXACTLY AS REQUIRED
    # ----------------------------------------------------------
    # For every perturbed enzyme, scale its flux by the fold change
    for enzyme_id, fold_change in perturbations.items():
        flux[enzyme_id] = flux[enzyme_id] * fold_change

    # store into dataframe
    f[ix] = flux.values()
    f[ix] = f[ix] / 456300

    # ----------------------------------------------------------
    # Convert to dicts for backend → frontend
    # ----------------------------------------------------------
    perturbed_fluxes = {rid: float(val) for rid, val in zip(f.index, f[ix])}
    reference_fluxes = {rid: float(val) for rid, val in zip(f.index, f['Steady'])}

    return {
        "fluxes": perturbed_fluxes,
        "reference": reference_fluxes,
        "perturbations": perturbations,
        "ix": ix
    }


# =====================================================================
# CLI ENTRY (supports BOTH single and multi)
# =====================================================================
if __name__ == "__main__":
    payload = json.loads(sys.argv[1])

    # OLD (single enzyme):
    # { "enzyme_id": "PFK", "fold_change": 0.3 }

    # NEW (multi):
    # { "perturbations": { "PFK": 0.3, "PGI": 2.0 } }

    if "perturbations" in payload:
        perturb = payload["perturbations"]
    else:
        # backward compatibility
        perturb = {payload["enzyme_id"]: payload["fold_change"]}

    result = run_simulation_multi(perturb)
    print(json.dumps(result))



