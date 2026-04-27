# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the CustOMICS module.
"""
import numpy as np
import pandas as pd

from src.loss.survival_loss import CoxLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from torch.optim import Adam

import shap

from src.datasets.multi_omics_dataset import MultiOmicsDataset
from src.models.autoencoder import AutoEncoder
from src.encoders.encoder import Encoder
from src.decoders.decoder import Decoder
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.tasks.classification import MultiClassifier
from src.tasks.survival import SurvivalNet
from src.models.vae import VAE
from src.loss.classification_loss import classification_loss
from src.loss.consensus_loss import consensus_loss
from src.metrics.classification import multi_classification_evaluation, plot_roc_multiclass
from src.metrics.survival import CIndex_lifeline, cox_log_rank
from src.tools.utils import save_plot_score
from src.tools.utils import get_common_samples
from src.ex_vae.shap_vae import processPhenotypeDataForSamples, randomTrainingSample, splitExprandSample, ModelWrapper, addToTensor, sample_background_and_tumor
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})


class CustOMICS(nn.Module):
    """
    The main CustOMICS object that represents the main network for dealing with multi-source integration and multi-task learning
    """
    def __init__(self, source_params, central_params, classif_params, surv_params, train_params, device):
        """
        Construct the whole architecture with intermediate autoencoders, central layer and eventually downstream predictors
        Parameters:
            source_params (dict)      -- parameters related to the different sources to integrate
            central_params (dict)     -- parameters of the central autoencoder
            classif_params (dict)     -- classifier parameters
            surv_params (dict)        -- parameters of the survival network
            train_params (dict)       -- training hyperparameters
            device (pytorch)          -- the device in which the computation is done
        """
        super(CustOMICS, self).__init__()
        self.n_source = len(list(source_params.keys()))
        self.device = device
        self.lt_encoders = [Encoder(input_dim=source_params[source]['input_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             latent_dim=source_params[source]['latent_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.lt_decoders = [Decoder(latent_dim=source_params[source]['latent_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             output_dim=source_params[source]['input_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.rep_dim = sum([source_params[source]['latent_dim'] for source in source_params])
        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim, hidden_dim=central_params['hidden_dim'], 
                                                    latent_dim=central_params['latent_dim'], norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'], hidden_dim=central_params['hidden_dim'], 
                                                    output_dim=self.rep_dim, norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.beta = central_params['beta']
        self.num_classes = classif_params['n_class']
        self.lambda_classif = classif_params['lambda']
        self.classifier =  MultiClassifier(n_class=self.num_classes, latent_dim=central_params['latent_dim'], dropout=classif_params['dropout'],
            class_dim = classif_params['hidden_layers']).to(self.device)
        self.lambda_survival = surv_params['lambda']
        surv_param = {'drop': surv_params['dropout'], 'norm': surv_params['norm'], 'dims': [central_params['latent_dim']] + surv_params['dims'] + [1], 
                    'activation': surv_params['activation'], 'l2_reg': surv_params['l2_reg'], 'device': self.device}
        self.survival_predictor = SurvivalNet(surv_param)
        self.phase = 1
        self.switch_epoch = train_params['switch']
        self.lr = train_params['lr']
        self.autoencoders = []
        self.central_layer = None
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.optimizer = self._get_optimizer(self.lr)
        self.vae_history = []
        self.survival_history = []
        self.label_encoder = None
        self.one_hot_encoder = None

    def _get_optimizer(self, lr):
        """
        Initilizes the optimizer
        Parameters:
            lr (float)      -- learning rate for the CustOmics network
        """
        lt_params = []
        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())
        lt_params += list(self.central_layer.parameters())
        lt_params += list(self.survival_predictor.parameters())
        lt_params += list(self.classifier.parameters())        
        optimizer = Adam(lt_params, lr=lr)
        return optimizer

    def _set_autoencoders(self):
        """
        Initializes the autoencoders
        """
        for i in range(self.n_source):
            self.autoencoders.append(AutoEncoder(self.lt_encoders[i], self.lt_decoders[i], self.device))

    def _set_central_layer(self):
        """
        Initializes the central variational autoencoder
        """
        self.central_layer = VAE(self.central_encoder, self.central_decoder, self.device)

    def _relocate(self):
        """
        Relocates the network to specified device
        """
        for i in range(self.n_source):
            self.autoencoders[i].to(self.device)
        self.central_layer.to(self.device)

    def _switch_phase(self, epoch):
        """
        Switches phases during training
        Parameters:
            epoch (int)      -- epoch starting which to switch phases
        """
        if epoch < self.switch_epoch:
            self.phase = 1
        else:
            self.phase = 2

    def _compute_baseline(self, clinical_df, lt_samples, event, surv_time):
        kmf = KaplanMeierFitter()
        kmf.fit(clinical_df.loc[lt_samples, surv_time], clinical_df.loc[lt_samples, event])
        return kmf.survival_function_


    def per_source_forward(self, x):
        lt_forward = []
        for i in range(self.n_source):
            lt_forward.append(self.autoencoders[i](x[i]))
        return lt_forward

    def get_per_source_representation(self, x):
        lt_rep = []
        for i in range(self.n_source):
            lt_rep.append(self.autoencoders[i](x[i])[1])
        return lt_rep

    def decode_per_source_representation(self, lt_rep):
        lt_hat = []
        for i in range(self.n_source):
            lt_hat.append(self.autoencoders[i].decode(lt_rep[i]))
        return lt_hat



    def forward(self, x):
        lt_forward = self.per_source_forward(x)
        lt_hat = [element[0] for element in lt_forward]
        lt_rep = [element[1] for element in lt_forward]
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = self.central_encoder(central_concat)
        return lt_hat, lt_rep ,mean

    def _compute_loss(self, x):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            return lt_rep, loss
        elif self.phase == 2:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            central_concat = torch.cat(lt_rep, dim=1)
            loss += self.central_layer.loss(central_concat, self.beta)
            mean, logvar = self.central_encoder(central_concat)
            z = mean
            return z, loss

    def _train_loop(self, x, labels, os_time, os_event):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        # If we don't disjoint the autoencoder's architecture 
        loss = 0
        self.optimizer.zero_grad()
        if self.phase == 1:
            lt_rep, loss = self._compute_loss(x)
            for z in lt_rep:
                #hazard_pred = self.survival_predictor(z)
                #survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                #loss += self.lambda_survival * survival_loss
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels)
                loss += self.lambda_classif * classification

        elif self.phase == 2:
            z, loss = self._compute_loss(x)
            #hazard_pred = self.survival_predictor(z)
            #survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
            #loss += self.lambda_survival * survival_loss
            y_pred_proba = self.classifier(z)
            classification = classification_loss('CE', y_pred_proba, labels)
            loss += self.lambda_classif * classification

        return loss


    def fit(self, omics_train, clinical_df, label, event, surv_time, omics_val=None, batch_size=32, n_epochs=30, verbose=False):
        self.label_encoder = LabelEncoder()
        y_raw = clinical_df[label].values
        y_int = self.label_encoder.fit_transform(y_raw)  

        clinical_df = clinical_df.copy()
        clinical_df[label] = y_int

        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(
            y_int.reshape(-1,1)
        )
        #self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(clinical_df.loc[:, label].values.reshape(-1,1))

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])
        #self.baseline = self._compute_baseline(clinical_df, lt_samples_train, event, surv_time)
        dataset_train = MultiOmicsDataset(omics_df=omics_train, clinical_df=clinical_df, lt_samples=lt_samples_train,
                                            label=label, event=event, surv_time=surv_time)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)
        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            dataset_val = MultiOmicsDataset(omics_df=omics_val, clinical_df=clinical_df, lt_samples=lt_samples_val,
                                            label=label, event=event, surv_time=surv_time)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

        self.history = []
        for epoch in range(n_epochs):
            overall_loss = 0
            self._switch_phase(epoch)
            for batch_idx, (x,labels,os_time,os_event) in enumerate(train_loader):
                self.train_all()
                loss_train = self._train_loop(x, labels, os_time,os_event)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx+1)*batch_size)
            overall_loss = 0
            if omics_val != None:
                for batch_idx, (x,labels, os_time,os_event) in enumerate(val_loader):
                    self.eval_all()
                    loss_val = self._train_loop(x, labels, os_time,os_event)
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / ((batch_idx+1)*batch_size)

                self.history.append((average_loss_train, average_loss_val))
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train, "\tAverage Loss Val : ", average_loss_val)
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train)


    def get_latent_representation(self, omics_df, tensor=False, batch_size=256):
        self.eval_all()
        latent_representations = []
        num_samples = len(next(iter(omics_df.values())))
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            if not tensor:
                batch_x = [torch.Tensor(omics_df[source].values[start:end]) for source in omics_df.keys()]
            else:
                batch_x = [omics[start:end] for omics in omics_df]
            with torch.no_grad():
                for i in range(len(batch_x)):
                    batch_x[i] = batch_x[i].to(self.device)
                z, _ = self._compute_loss(batch_x)
            latent_representations.append(z.cpu().detach().numpy())
        return np.concatenate(latent_representations, axis=0)

    def reconstruct(self, x):
        x = torch.Tensor(x)
        z = self.autoencoders[0](x)[1]
        return self.autoencoders[0].decode(z).cpu().detach().numpy()


    def plot_representation(self, omics_df, clinical_df, labels, filename, title, show=False, method='tsne'):
        labels_df = clinical_df.loc[:, labels]
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df=omics_df)
        save_plot_score(filename, z, labels_df[lt_samples].values, title, show, method)


    #def source_predict(self, expr_df, source):
    #    #tensor_expr = torch.Tensor(expr_df.values)
    #    tensor_expr = expr_df
    #    if source == 'CNV' or source == 'protein' or source == 'cnv2':
    #        z = self.lt_encoders[0](tensor_expr)
    #    elif source == 'RNAseq' or source == 'gene_exp' or source == 'ge2' or source == 'ge':
    #        z = self.lt_encoders[1](tensor_expr)
    #    elif source == 'methyl' or source == 'pe2':
    #        z = self.lt_encoders[2](tensor_expr)
    #    y_pred_proba = self.classifier(z)
    #    return y_pred_proba

    def source_predict(self, expr_df, source):
        #tensor_expr = torch.Tensor(expr_df.values)
        tensor_expr = expr_df
        if source == 'RNAseq' or source == 'gene_exp' or source == 'ge2' or source == 'ge':
            z = self.lt_encoders[0](tensor_expr)
        elif source == 'methyl' or source == 'pe2':
            z = self.lt_encoders[1](tensor_expr)
        y_pred_proba = self.classifier(z)
        return y_pred_proba

    def predict_risk(self, omics_df):
        z = self.get_latent_representation(omics_df)
        return self.survival_predictor(torch.Tensor(z))

    def predict_survival(self, omics_df, t=None):
        lt_samples = get_common_samples([df for df in omics_df.values()])
        dt_surv = {}
        risk_score = self.predict_risk(omics_df).cpu().detach().numpy()
        for sample, risk in zip(lt_samples, risk_score):
            dt_surv[sample] = self.baseline*np.exp(risk[0])
        return dt_surv

    def evaluate_latent(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32, plot_roc=False):
        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
                                            label=label, event=event, surv_time=surv_time)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels, os_time, os_event) in enumerate(test_loader):
                z, loss  = self._compute_loss(x)
                if task == 'survival':
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
                    return np.mean(c_index)
                elif task == 'classification':
                    svc_model = SVC()
                    z = self.get_latent_representation(x, tensor=True)
                    svc_model.fit(z.cpu().detach().numpy(), labels.cpu().detach().numpy())
                    y_pred_proba = svc_model.predict_proba()
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
                    if plot_roc:
                        plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename='test', n_classes=self.num_classes,
                                            var_names=np.unique(clinical_df.loc[:, label].values.tolist()))

                
                    return classif_metrics




    def evaluate(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32, plot_roc=False):

        #encoded_clinical_df = clinical_df.copy()
        #encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_test = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        
        #lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        #dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
        #                                    label=label, event=event, surv_time=surv_time)
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=clinical_df, lt_samples=lt_samples_test,
                                            label=label, event=event, surv_time=surv_time)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = {}
        c_index = []
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        with torch.no_grad():
            for batch_idx, (x, labels, os_time, os_event) in enumerate(test_loader):
                z, loss  = self._compute_loss(x)
                if task == 'survival':
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
                    return np.mean(c_index)
                elif task == 'classification':  
                    #y_pred_proba = self.classifier(z)
                    #y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    #y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    #y_true = labels.cpu().detach().numpy()
                    y_pred_proba_batch = self.classifier(z)
                    y_pred_batch = torch.argmax(y_pred_proba_batch, dim=1)
                    all_y_true.append(labels.cpu().detach().numpy())
                    all_y_pred.append(y_pred_batch.cpu().detach().numpy())
                    all_y_pred_proba.append(y_pred_proba_batch.cpu().detach().numpy())

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_pred_proba = np.concatenate(all_y_pred_proba)
                    
        classif_metrics = multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder)

        labels = self.label_encoder.classes_ 
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        class_names = list(self.one_hot_encoder.categories_[0])


        if plot_roc:
            plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename='test', n_classes=self.num_classes,
                                            var_names=np.unique(clinical_df.loc[:, label].values.tolist()))
        
        return classif_metrics, cm, class_names



    def stratify(self, omics_df, clinical_df, event, surv_time, treshold=0.5, 
                    save_plot=False, plot_title="", filename=''):
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df)
        hazard_pred = self.survival_predictor(torch.Tensor(z)).cpu().detach().numpy()
        dt_strat = {'high': [], 'low': []}
        for i in range(len(lt_samples)):
            if hazard_pred[i] <= np.mean(hazard_pred):
                dt_strat['low'].append(lt_samples[i])
            else:
                dt_strat['high'].append(lt_samples[i])
        kmf_low = KaplanMeierFitter(label='low risk')
        kmf_high = KaplanMeierFitter(label='high risk')
        kmf_low.fit(clinical_df.loc[dt_strat['low'], surv_time], clinical_df.loc[dt_strat['low'], event])
        kmf_high.fit(clinical_df.loc[dt_strat['high'], surv_time], clinical_df.loc[dt_strat['high'], event])
        p_value = cox_log_rank(hazard_pred.reshape(1,-1)[0], np.array(clinical_df.loc[lt_samples, event].values, dtype=float), np.array(clinical_df.loc[lt_samples, surv_time].values, dtype=float))

        kmf_low.plot()
        kmf_high.plot()
        plt.title(plot_title + " (p-value = {:.3g})".format(p_value))
        plt.xlim((0,2500))
        if save_plot:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()


    def explain(self, sample_id, omics_df, clinical_df, source, subtype, label='', device='cpu', show=False, mode=1):
        """
        :param sample_id: List of samplesid to consider for explanation.
        :param vae_model: The CustOmics model to explain, the output should be a 1xn_class tensor.
        :param expr_df: DataFrame of data to explain, the input has to match the source selected for the explanation.
        :param clinical_df: DataFrame containing the clinical data.
        :param source: Omic source to explain.
        :param subtype: Subtype that needs explanation, shap values will be computed for this subtypes against the others.
        :param le: LabelEncoder to revert back from numerical encoding to original names.
        :return:None, plots of the top 10 genes and their shap values
        """
        #This class has combined the different analysis' of the Deep SHAP values we conducted.
        #SHAP reference: Lundberg et al., 2017: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

        self.eval_all()
        expr_df = omics_df[source]

        sample_id = list(set(sample_id).intersection(set(expr_df.index)))
        phenotype = processPhenotypeDataForSamples(clinical_df, sample_id)

        conditionaltumour = phenotype[label] == subtype

        sample_number = 30
        if subtype == 5:
            sample_number = 200

        background_sample, tumor_sample = sample_background_and_tumor(expr_df, conditionaltumour, n_samples=sample_number)
    
        # put on device as correct datatype
        background = torch.Tensor(background_sample.values.astype('float32')).to(device)
        tumour_expr_tensor = torch.Tensor(tumor_sample.values.astype('float32')).to(device)

        e = shap.DeepExplainer(ModelWrapper(self, source=source), background)
        shap_values_female = e.shap_values(tumour_expr_tensor, ranked_outputs=None, check_additivity=False)
        shap_values_class = shap_values_female[:, :, subtype]

        if mode == 2:
            return shap_values_class, tumor_sample


        #can also use bar
        shap.summary_plot(shap_values_class,features=tumor_sample,feature_names=list(tumor_sample.columns), show=False, plot_type="violin", max_display=10, plot_size=[10,6])
        plt.savefig('shap_{}_{}.png'.format(source, subtype), bbox_inches='tight')
        if show:
            plt.show()

    def plot_loss(self):
        n_epochs = len(self.history)
        plt.title('Evolution of the loss function with respect to the epochs')
        plt.vlines(x=self.switch_epoch, ymin=0, ymax=2.5, colors='purple', ls='--', lw=2, label='phase 2 switch')
        plt.plot(range(0, n_epochs), [loss[0] for loss in self.history], label = 'train loss')
        plt.plot(range(0, n_epochs), [loss[1] for loss in self.history], label = 'val loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def print_parameters(self):
        lt_params = []
        lt_names = []
        for autoencoder in self.autoencoders:
            for name, param in autoencoder.named_parameters():
                lt_params.append(param.data)
                lt_names.append(name)
        for name, param in self.central_layer.named_parameters():
            lt_params.append(param.data)
            lt_names.append(name)
        print(len(lt_params))

    def get_number_parameters(self):
        sum_params = 0
        for autoencoder in self.autoencoders:
            sum_params += sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        sum_params += sum(p.numel() for p in self.central_layer.parameters() if p.requires_grad)
        return sum_params


    def train_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.train()
            decoder.train()
        for autoencoder in self.autoencoders:
            autoencoder.train()
        self.central_layer.train()
        if self.survival_predictor:
            self.survival_predictor.train()
        if self.classifier:
            self.classifier.train()
    def eval_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.eval()
            decoder.eval()
        for autoencoder in self.autoencoders:
            autoencoder.eval()
        self.central_layer.eval()
        if self.survival_predictor:
            self.survival_predictor.eval()
        if self.classifier:
            self.classifier.eval()