from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import cobra
from cobra.io.json import load_json_model
import pandas as pd
from pathlib import Path
import subprocess

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "backend/data"

# Hard-coded single model for now
MODEL_REGISTRY = {
    "ecoli_core": {
        "model_file": f"{DATA_PATH}/e_coli_core.json",
        "map_file": f"{DATA_PATH}/e_coli_core_map.json",
        "baseline_fluxes_file": f"{DATA_PATH}/baseline_fluxes.csv",
    }
}

# ------------------------------------------------------
# Load reference fluxes (same as before)
# ------------------------------------------------------
def compute_reference_fluxes(model_path):
    model = load_json_model(model_path)
    sol = model.optimize()
    if sol.status != "optimal":
        raise RuntimeError("FBA solution not optimal")
    return sol.fluxes.to_dict()


def load_baseline_fluxes(csv_path: str, model_path: str | None = None):
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        raise RuntimeError(f"Baseline flux CSV not found at '{csv_path}'")

    if df.empty:
        raise RuntimeError(f"Baseline flux CSV at '{csv_path}' is empty")

    df = df.iloc[:, :2]
    df.columns = ["reaction", "flux"]
    df = df.dropna(subset=["reaction", "flux"])
    df["reaction"] = df["reaction"].astype(str)
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df = df.dropna(subset=["flux"])
    baseline = dict(zip(df["reaction"], df["flux"]))

    if model_path is not None:
        model = load_json_model(model_path)
        reaction_ids = {rxn.id for rxn in model.reactions}
        baseline = {rid: v for rid, v in baseline.items() if rid in reaction_ids}

    return baseline


# ------------------------------------------------------
# INITIAL VIEW ENDPOINT
# ------------------------------------------------------
@app.get("/models/{model_id}/initial_view")
def initial_view(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")

    paths = MODEL_REGISTRY[model_id]

    with open(paths["map_file"], "r") as f:
        map_json = json.load(f)

    with open(paths["model_file"], "r") as f:
        model_json = json.load(f)

    baseline_path = paths.get("baseline_fluxes_file")
    if baseline_path:
        try:
            fluxes = load_baseline_fluxes(baseline_path, paths["model_file"])
        except RuntimeError:
            fluxes = compute_reference_fluxes(paths["model_file"])
    else:
        fluxes = compute_reference_fluxes(paths["model_file"])

    return {
        "map_json": map_json,
        "model_json": model_json,
        "reference": {"fluxes": fluxes},
    }


# ------------------------------------------------------
# SKIMPY SIMULATION ENDPOINT (UPDATED)
# ------------------------------------------------------

# Path to skimpy env interpreter
SKIMPY_PYTHON = "/opt/anaconda3/envs/skimpy-env/bin/python"

# Path to simulate.py
SIM_SCRIPT = str(Path(__file__).resolve().parents[1] / "skimpy_runner" / "simulate.py")


@app.post("/simulate_fluxes")
def simulate_fluxes(payload: dict):
    """
    Accepts BOTH:
      - Single perturbation:
            {"enzyme_id": "PFK", "fold_change": 0.5}
      - Multi perturbation:
            {"perturbations": {"PFK": 0.5, "PGI": 2.0}}
    """

    # ------------------------------------------------------
    # Normalize payload into multi-perturbation format
    # ------------------------------------------------------
    if "perturbations" in payload:
        perturbations = payload["perturbations"]
    else:
        # Backward compatible mode
        if "enzyme_id" not in payload:
            raise HTTPException(status_code=400, detail="Missing enzyme_id")
        if "fold_change" not in payload:
            raise HTTPException(status_code=400, detail="Missing fold_change")

        perturbations = {payload["enzyme_id"]: payload["fold_change"]}

    # ------------------------------------------------------
    # Run skimpy simulation
    # ------------------------------------------------------
    cmd = [
        SKIMPY_PYTHON,
        SIM_SCRIPT,
        json.dumps({"perturbations": perturbations})
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed:\n{e.stderr}"
        )







