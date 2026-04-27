import numpy as np
import pandas as pd
from src.tools.utils import read_data
from src.network.customics import CustOMICS
from src.tools.utils import save_stratified_splits, get_splits, get_sub_omics_df
from torch.utils.data import DataLoader


def train(task, cohorts, sources, device, omics_df, clinical_df, lt_samples,
          batch_size=64, n_epochs=10, beta=1, lr=1e-3,
          hidden_dim=[512, 256], central_dim=[512, 256], latent_dim=128, dropout=0.2,
          classifier_dim=[128, 64], survival_dim=[64, 32], lambda_classif=5,
          lambda_survival=5, explain=False, explained_source='RNAseq', explained_class='Her2'):

    n_splits = 5
    lt_metrics = []
    for split in range(1, n_splits + 1):

        if cohorts == 'PANCAN':
            label = "Tumor"
            event = 'Tumor'
            surv_time = 'Tumor'
        else:
            label = 'pathology_T_stage'
            event = 'status'
            surv_time = 'overall_survival'
        
        samples_train, samples_val, samples_test = get_splits(cohorts, split)

        omics_train = get_sub_omics_df(omics_df, samples_train)
        omics_val = get_sub_omics_df(omics_df, samples_val)
        omics_test = get_sub_omics_df(omics_df, samples_test)

        x_dim = [omics_df[omic_source].shape[1] for omic_source in omics_df.keys()]

        num_classes = len(np.unique(clinical_df[label].values))

        rep_dim = latent_dim

        source_params = {}
        central_params = {'hidden_dim': central_dim, 'latent_dim': latent_dim,
                        'norm': True, 'dropout': dropout, 'beta': beta}
        classif_params = {'n_class': num_classes, 'lambda': lambda_classif,
                        'hidden_layers': classifier_dim, 'dropout': dropout}
        surv_params = {'lambda': lambda_survival, 'dims': survival_dim,
                    'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout': dropout}
        for i, source in enumerate(sources):
            source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim,
                                    'latent_dim': rep_dim, 'norm': True, 'dropout': dropout}
        train_params = {'switch': n_epochs//2, 'lr': lr}

        model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                        surv_params=surv_params, train_params=train_params, device=device).to(device)
        model.get_number_parameters()
        model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=False)
        metric = model.evaluate(omics_test=omics_test, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                                task=task, batch_size=batch_size, plot_roc=False)
        
        #model.plot_loss()
        model.plot_representation(omics_train, clinical_df, label,
                                'plot_representation', 'Representation of the latent space', show=False, method='tsne')
        if cohorts != 'PANCAN':
            model.explain(lt_samples, omics_df, clinical_df,
                        'RNAseq', 'Her2', label, device, False, 1)
        if task == 'survival':
            print(model.predict_survival(omics_test))
            model.stratify(omics_df=omics_train, clinical_df=clinical_df,
                        event='status', surv_time='overall_survival')
            
        #print(metric)
        lt_metrics.append(metric[0])
    return lt_metrics
