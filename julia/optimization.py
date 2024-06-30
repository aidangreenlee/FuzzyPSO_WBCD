import optuna
from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval('include("main.jl")')
jl.eval('clusters, DATA = loadData(3)')

def objective(trial):
    phi1 = trial.suggest_float('phi1', 0, 2)
    phi2 = trial.suggest_float('phi2', 0, 2)
    W    = trial.suggest_float('W', 0, 4)
    return 1 - jl.eval("WBCD_algorithm(clusters, DATA, 20, 3.0, {phi1:.2f}, {phi2:.2f}, {W:.2f}, debug=false)".format(W=W,phi1=phi1,phi2=phi2))

print("evaluating")
study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="WBCD"
)
study.optimize(objective, n_trials=1000)

study.best_params  # E.g. {'x': 2.002108042}
print(study.best_params)