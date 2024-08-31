import optuna
import sys
from julia.api import Julia

ncluster = sys.argv[1]
jl = Julia(compiled_modules=False)
jl.eval('include("main.jl")')
jl.eval('clusters, DATA = loadData({ncluster}, 1)'.format(ncluster=ncluster))
jl.eval('output_data = output(100, {ncluster}, 2*size(clusters[1].c,1)+1, false)'.format(ncluster=ncluster))

def objective(trial):
    phi1 = trial.suggest_float('phi1', 0, 1)
    phi2 = trial.suggest_float('phi2', 0, 3)
    W    = trial.suggest_float('W', 0, 4)
    Vmax = trial.suggest_float('Vmax', 0, 4)
    iteration = trial.number
    return jl.eval("WBCD_algorithm(clusters, DATA, 20, {Vmax:.8f}, {W:.8f}, {phi1:.8f}, {phi2:.8f}, alpha=0.4, beta=0.3, gamma=0.3,debug=output_data, iteration={iteration:d})".format(Vmax=Vmax,W=W,phi1=phi1,phi2=phi2,iteration=iteration))

print("evaluating")
study = optuna.create_study(
    storage="sqlite:///norep_db.sqlite3",
    study_name="WBCD_{ncluster}clusters_0.4_0.3_0.3".format(ncluster=ncluster),
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=500)

study.best_params  # E.g. {'x': 2.002108042}
print(study.best_params)