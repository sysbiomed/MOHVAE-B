import argparse
import torch
import numpy as np
import pandas as pd
import optuna
import os 
from optuna.samplers import TPESampler
from src.tools.core_utils import train
from src.tools.utils import read_data,save_stratified_splits

original_space = { # For second stage
    'hidden_dim': [
        [512, 256],
        [512, 256, 128],
    ],
    'central_dim': [
        [512, 256],
        [512, 256, 128],
    ],
    'classifier_dim': [
        [128, 64],
        [256,128],
    ],
    'latent_dim': [32, 64, 128, 256]
}

optuna_params = { # Found on 1st stage
    'dropout': 0.20513212535950442,
    'batch_size': 128,
    'beta': 0.46577623207178603,
    'lr': 0.0013652973945860643,
    'lambda_classif': 17.737276799099185,
    'n_epochs': 18
}

param_space = {
    'dropout': (0.1, 0.5),
    'batch_size': [32, 64, 128], 
    'beta': (0.1, 2.0),
    'lr': (1e-5, 1e-2),
    'lambda_classif': (1, 20),
    'n_epochs': [10, 14, 18, 20, 24] 
}

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cohorts', help='list of cohorts to process', type=str)
parser.add_argument('-dv', '--device', help='torch device in which the computations will be done', type=str, default='cpu')
parser.add_argument('-dr', '--data_directory', help='folder in which the data are stored', type=str, default='../TCGA/')
parser.add_argument('-res', '--result_directory', help='folder in which the results should be stored', type=str, default='results/')
parser.add_argument('-t', '--task', help='task to perform', type=str, choices=['classification', 'survival'], default='classification')
parser.add_argument('-src', '--sources', help='list of sources to integrate', type=str, default='CNV,RNAseq,methyl')
parser.add_argument('-hd', '--hidden_dim', help='list of neurones for the hidden layers of the intermediate autoencoders', type=str, default='1024,512,256')
parser.add_argument('-ct', '--central_dim', help='list of neurones for the hidden layers of the central autoencoder', type=str, default='2048,1024,512,256')
parser.add_argument('-ch', '--classifier_dim', help='list of neurones for the classifier hidden layers', type=str, default='256,128')
parser.add_argument('-sh', '--survival_dim', help='list of neurones for the survival hidden layers', type=str, default='64,32')
parser.add_argument('-n', '--n_trials', help='number of Bayesian optimization trials', type=int, default=30)

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sources = args.sources.split(',')

omics_df, clinical_df, lt_samples = read_data()
clinical_df['Tumor'], _ = pd.factorize(clinical_df['Tumor'])

save_split = True
n_splits = 5
if save_split:
    save_stratified_splits(clinical_df, "PANCAN", n_splits)

def objective(trial): # For stage 2, need adpatation to run for phase 1 again. Change params and fill the params with original space values
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', original_space['hidden_dim']),
        'central_dim': trial.suggest_categorical('central_dim', original_space['central_dim']),
        'classifier_dim': trial.suggest_categorical('classifier_dim', original_space['classifier_dim']),
        'latent_dim': trial.suggest_categorical('latent_dim', original_space['latent_dim']),
    }
    
    print(f"Trial {trial.number} with params: {params}")

    metrics = train(
        args.task, "PANCAN", sources, device, omics_df, clinical_df, lt_samples,
        batch_size=optuna_params['batch_size'], n_epochs=optuna_params['n_epochs'],
        beta=optuna_params['beta'], lr=optuna_params['lr'],
        hidden_dim=params['hidden_dim'], central_dim=params['central_dim'],
        latent_dim=params['latent_dim'],
        dropout=optuna_params['dropout'],
        classifier_dim=params['classifier_dim'], survival_dim=[64, 32],
        lambda_classif=optuna_params['lambda_classif'],
        lambda_survival=0, explain=False, explained_source='Her2', explained_class='RNAseq'
    )
    
    auc_list = [m['AUC'] for m in metrics]
    mean_auc = np.mean(auc_list)
    
    # Store other metrics for analysis
    accuracy_list = [m['Accuracy'] for m in metrics]
    f1_list = [m['F1-score'] for m in metrics]
    precision_list = [m['Precision'] for m in metrics]
    recall_list = [m['Recall'] for m in metrics]
    
    # Report to Optuna
    trial.set_user_attr('mean_accuracy', np.mean(accuracy_list))
    trial.set_user_attr('mean_f1', np.mean(f1_list))
    trial.set_user_attr('mean_precision', np.mean(precision_list))
    trial.set_user_attr('mean_recall', np.mean(recall_list))
    
    return mean_auc

sampler = TPESampler()  
study = optuna.create_study(
    direction='maximize',  # We want to maximize AUC
    sampler=sampler,
    study_name='customics_hyperparameter_optimization'
)

study.optimize(objective, n_trials=args.n_trials)

# Print and save results
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
best_trial = study.best_trial
print("  Value (AUC): ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

results = []
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:  
        result_row = {
            **trial.params,
            'mean_auc': trial.value,
            'mean_accuracy': trial.user_attrs['mean_accuracy'],
            'mean_f1': trial.user_attrs['mean_f1'],
            'mean_precision': trial.user_attrs['mean_precision'],
            'mean_recall': trial.user_attrs['mean_recall'],
            'trial_number': trial.number
        }
        results.append(result_row)

df = pd.DataFrame(results)
output_dir = '../outputs/Optimization/'
os.makedirs(output_dir, exist_ok=True)
df.to_csv("layers_size_metrics.csv", index=False)
print("Summary CSV written: layers_size_metrics.csv")   