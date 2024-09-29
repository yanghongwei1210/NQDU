import os
import copy
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
import torch
from models import *
from dataset import DVRLDataset, GraphSourceTargetDataset

from HAN import HAN_Trainer
from HAN import utils
class DVRL_HAN(object):
    def __init__(self, args, pred_model: HAN_Trainer, flags):
        self.args = args
        self.source_target_dataset = GraphSourceTargetDataset(args)
        self.x_train, self.y_train = self.source_target_dataset.get_source_data()
        self.x_valid, self.y_valid = self.source_target_dataset.get_target_data()

        self.problem = args.problem
        self.device = args.device

        # One-hot encoded labels

        self.y_train_onehot = \
            np.eye(len(np.unique(self.y_train)))[self.y_train.astype(int)]
        self.y_valid_onehot = \
            np.eye(len(np.unique(self.y_train)))[self.y_valid.astype(int)]
        
        # Network parameters
        self.data_dim = len(self.x_train[0, :])
        self.label_dim = len(self.y_train_onehot[0, :])
        self.hidden_dim = args.hidden_dim
        self.comb_dim = args.comb_dim
        self.outer_iterations = args.outer_iterations
        self.act_fn = args.act_fn
        self.layer_number = args.layer_number
        self.batch_size = np.min([args.batch_size, len(self.x_train[:, 0])])
        self.learning_rate = args.learning_rate

        # Basic parameters
        self.epsilon = args.epsilon  # Adds to the log to avoid overflow
        self.threshold = args.threshold  # Encourages exploration

        # Flags
        self.flag_sgd = flags['sgd']
        self.flag_pretrain = flags['pretrain']

        # If the pred_model uses stochastic gradient descent (SGD) for training
        # if self.flag_sgd:
        #     self.inner_iterations = args.inner_iterations
        #     self.batch_size_predictor = np.min([args.batch_size_predictor, len(self.x_valid[:, 0])])

        self.inner_iterations = args.epochs
        self.batch_size_predictor = args.inner_batch_size

        # Checkpoint file name
        self.checkpoint_file_name = args.checkpoint_file_name

        # Pred model (Note that any model architecture can be used as the predictor
        # model, either randomly initialized or pre-trained with the training data.
        # The condition for predictor model to have fit (e.g. using certain number
        # of back-propagation iterations) and predict functions as its subfunctions.
        self.pred_model = pred_model

        # Final model
        self.final_model = self.pred_model.copy()

        # With randomly initialized predictor
        # if (not self.flag_pretrain) & self.flag_sgd:
        #     if not os.path.exists('tmp'):
        #         os.makedirs('tmp')
        #     # Saves initial randomization
        #     self.pred_model.save_model('tmp/pred_model.pth')
            # With pre-trained model, pre-trained model should be saved as
            # 'tmp/pred_model.h5'

        # Baseline model
        print('--start ori_model.fit--')
        self.ori_model = self.pred_model.copy()
        self.ori_model.fit(is_train_dataset=True, batch_size=self.batch_size_predictor, epochs=self.inner_iterations)
        print('--finish ori_model.fit--')

        # Valid baseline model
        print('--start val_model.fit--')
        self.val_model = self.pred_model.copy()
        self.val_model.fit(is_train_dataset=False, batch_size=self.batch_size_predictor, epochs=self.inner_iterations)
        print('--finish val_model.fit--')

    def train_dvrl(self, perf_metric,num):
        """Trains DVRL based on the specified objective function.
        Args:
        perf_metric: 'auc', 'accuracy', 'log-loss' for classification
                    'mae', 'mse', 'rmspe' for regression
        """

        # Generates selected probability
        est_data_value_model = DataValueEstimator(self.data_dim, self.label_dim, self.hidden_dim, self.layer_number, self.act_fn, self.comb_dim)
        est_data_value_model.to(self.device)

        # Generator loss (REINFORCE algorithm)
        dev_loss = DVELoss(self.epsilon, self.threshold).to(self.device)

        # Solver
        dve_solver = torch.optim.Adam(est_data_value_model.parameters(), self.learning_rate)

        # Baseline performance
        print('--start Baseline performance(ori_model.predict)--')
        if self.flag_sgd:
            y_valid_hat,_,_ = self.ori_model.predict(is_train_dataset=False, batch_size=self.batch_size_predictor, proba=False)
        else:
            y_valid_hat,_,_ = self.ori_model.predict(is_train_dataset=False, batch_size=self.batch_size_predictor, proba=True)
        if perf_metric == 'auc':
            valid_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
        elif perf_metric == 'accuracy':
            valid_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                    axis=1))
        elif perf_metric == 'mi-f1':
            valid_perf,_ = utils.f1_scores(y_valid_hat, self.y_valid)

        print('--finish Baseline performance(ori_model.predict)--')
        # elif perf_metric == 'log_loss':
        #     valid_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
        # elif perf_metric == 'rmspe':
        #     valid_perf = dvrl_metrics.rmspe(self.y_valid, y_valid_hat)
        # elif perf_metric == 'mae':
        #     valid_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
        # elif perf_metric == 'mse':
        #     valid_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

        # Prediction differences
        print('--start Prediction differences(val_model.predict)--')
        if self.flag_sgd:
            y_train_valid_pred,_,_ = self.val_model.predict(is_train_dataset=True, batch_size=self.batch_size_predictor, proba=False)
        else:
            y_train_valid_pred,_,_ = self.val_model.predict(is_train_dataset=True, batch_size=self.batch_size_predictor, proba=True)

        y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred)
        print('--finish Prediction differences(val_model.predict)--')

        # Training
        print('--start Training DVE--')
        with open('DVE loss result.txt', 'a', encoding='utf-8') as f:
            f.write('Train DVE outer_iterations loss: \n')
            for i in range(self.outer_iterations):
                print('---------Train DVE outer_iterations: {}---------'.format(i))
                # Batch selection
                batch_idx = np.random.choice(self.x_train.shape[0], self.batch_size, replace=False)

                x_batch = self.x_train[batch_idx, :]
                y_batch_onehot = self.y_train_onehot[batch_idx]
                y_batch = self.y_train[batch_idx]
                y_hat_batch = y_pred_diff[batch_idx]

                x_batch_tensor = torch.from_numpy(x_batch).float().to(self.device)
                y_batch_onehot_tensor = torch.from_numpy(y_batch_onehot).float().to(self.device)
                y_batch_tensor = torch.from_numpy(y_batch).float().to(self.device)
                y_hat_batch_tensor = torch.from_numpy(y_hat_batch).float().to(self.device)

                # Generates selection probability
                est_data_value_model.eval()
                with torch.no_grad():
                    est_dv_curr = est_data_value_model(x_batch_tensor, y_batch_onehot_tensor, y_hat_batch_tensor)

                est_dv_curr = est_dv_curr.cpu().detach().numpy()

                # Samples the selection probability
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

                # Exception (When selection probability is 0)
                if np.sum(sel_prob_curr) == 0:
                    est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                    sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

                # Trains predictor
                # If the predictor is neural network

                new_model = self.pred_model
                print('---start new_model.fit---')
                new_model.fit(is_train_dataset=True, batch_size=self.batch_size_predictor, epochs=self.inner_iterations, sample_weights=sel_prob_curr[:, 0], sample_indices=batch_idx)
                print('---finish new_model.fit---')




                # Prediction
                print('---start new_model.predict---')
                y_valid_hat,_,_ = new_model.predict(is_train_dataset=False, batch_size=self.batch_size_predictor, proba=True)
                print('---finish new_model.predict---')
                # Reward computation
                if perf_metric == 'auc':
                    dvrl_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
                elif perf_metric == 'accuracy':
                    dvrl_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                            axis=1))
                elif perf_metric == 'mi-f1':

                    dvrl_perf, _ = utils.f1_scores(y_valid_hat, self.y_valid)


                if self.problem == 'classification':
                    reward_curr = dvrl_perf - valid_perf
                elif self.problem == 'regression':
                    reward_curr = valid_perf - dvrl_perf


                # Trains the generator
                est_data_value_model.train()
                est_data_value_model.zero_grad()

                est_data_value = est_data_value_model(x_batch_tensor, y_batch_onehot_tensor, y_hat_batch_tensor)
                # est_data_value.to('cpu')
                s_input = torch.from_numpy(sel_prob_curr).float().to(self.device)
                # s_input = torch.from_numpy(sel_prob_curr).float()
                reward_curr=np.array(reward_curr)
                reward_input = torch.from_numpy(reward_curr).float().to(self.device)
                # reward_input = torch.from_numpy(reward_curr).float()
                loss = dev_loss(est_data_value, s_input, reward_input)
                print('--Train DVE outer_iterations: {}\t Loss: {:.6f}---------'.format(i, loss.item()))
                f.write(str(i) + '\t' + str(loss.item()) + '\n')
                loss.backward()
                dve_solver.step()
        f.close()
        print('--finish Training DVE--')
        # Saves trained model
        print('--save est_data_value_model save--')
        save_dve_model_path = "dvemodel/" + str(num) + "_dve_model.ckpt"
        torch.save(est_data_value_model.state_dict(), save_dve_model_path)
        torch.save(est_data_value_model.state_dict(), self.checkpoint_file_name)
        print('--finish est_data_value_model save--')

        # Trains DVRL predictor
        # Generate data values
        est_data_value_model.eval()
        final_data_value = []
        # final_dataset = DVRLDataset(self.x_train, self.y_train_onehot, y_pred_diff)
        # final_dataloader = torch.utils.data.DataLoader(final_dataset, batch_size=self.batch_size, shuffle=False)
        # with torch.no_grad():
        #     for batch in final_dataloader:
        #         x_input_batch, y_input_batch, y_hat_input_batch = batch
        #         x_input_batch = x_input_batch.float().to(self.device)
        #         y_input_batch = y_input_batch.float().to(self.device)
        #         y_hat_input_batch = y_hat_input_batch.float().to(self.device)
        #         batch_out = est_data_value_model(x_input_batch, y_input_batch, y_hat_input_batch)
        #         batch_out = batch_out.cpu().detach().numpy()
        #         final_data_value.append(batch_out)
        #
        # final_data_value = np.concatenate(final_data_value, 0)
        # final_data_value = final_data_value[:, 0]

        with torch.no_grad():
            x_input_batch = torch.from_numpy(self.x_train).float().to(self.device)
            y_input_batch = torch.from_numpy(self.y_train_onehot).float().to(self.device)
            y_hat_input_batch = torch.from_numpy(y_pred_diff).float().to(self.device)
            batch_out = est_data_value_model(x_input_batch, y_input_batch, y_hat_input_batch)
            # batch_out = est_data_value_model(self.x_train, self.y_train_onehot, y_pred_diff)
            batch_out = batch_out.cpu().detach().numpy()
        final_data_value = batch_out[:, 0]


        # Trains final model
        # If the final model is neural network

        # self.final_model.fit(is_train_dataset=True, batch_size=self.batch_size_predictor, epochs=self.outer_iterations, sample_weight=final_data_value, sample_indices=None)
        print('  start Trains final model  ')
        self.final_model.fit(is_train_dataset=True, batch_size=self.batch_size_predictor, epochs=self.inner_iterations,sample_weights=final_data_value, sample_indices=None,Flag='final')
        save_model_path = "finalmodel/" + str(num) +"_final_model.ckpt"
        self.final_model.save_model(save_model_path)
        print('  finish Trains final model  ')
        print('  start final model evalate  ')
        y_predict,_,_ = self.final_model.predict(is_train_dataset=False, batch_size=self.batch_size_predictor, proba=True)
        acc = metrics.accuracy_score(self.y_valid, np.argmax(y_predict,axis=1))
        micro = f1_score(self.y_valid, np.argmax(y_predict, axis=1), average='micro')
        macro = f1_score(self.y_valid, np.argmax(y_predict, axis=1), average='macro')
        print(
            '*Final_model performance: num: {} \t accuracy: {:.6f}\t  micro-f1: {:.6f}\t  macro-f1: {:.6f} \t  modelpath: {}'.format(num,
                acc, micro, macro, save_model_path))
        print('  finish final model evalate  ')
        string='*Final_model performance: num: '+ str(num)+' accuracy: '+str(acc) +'  micro-f1: '+str(micro)+' macro-f1: '+str(macro)+' path: '+save_model_path +'\n'
        with open('DVE final model result.txt', 'a', encoding='utf-8') as f:
            f.write(string)
        f.close()
    def data_valuator(self, x_train, y_train):
        """Returns data values using the data valuator model.

        Args:
        x_train: training features
        y_train: training labels
        Returns:
        final_dat_value: final data values of the training samples
        """

        # One-hot encoded labels
        # if self.problem == 'classification':
        #     y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]
        #     y_train_valid_pred = self.val_model.predict_proba(x_train)
        # elif self.problem == 'regression':
        #     y_train_onehot = np.reshape(y_train, [len(y_train), 1])
        #     y_train_valid_pred = np.reshape(self.val_model.predict(x_train),
        #                                 [-1, 1])
        y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]
        y_train_valid_pred,_,_ = self.val_model.predict(is_train_dataset=True, batch_size=self.batch_size_predictor,
                                                    proba=True)

        # Generates y_train_hat
        if self.problem == 'classification':
            y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)
        # elif self.problem == 'regression':
        #     y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)/y_train_onehot

        # Restores the saved model
        est_data_value_model = DataValueEstimator(self.data_dim, self.label_dim, self.hidden_dim, self.layer_number, self.act_fn, self.comb_dim)
        est_data_value_model.to(self.device)

        est_data_value_model.load_state_dict(torch.load(self.checkpoint_file_name))
        est_data_value_model.eval()

        # final_data_value = []
        # final_dataset = DVRLDataset(self.x_train, self.y_train_onehot, y_train_hat)
        # final_dataloader = torch.utils.data.DataLoader(final_dataset, batch_size=self.batch_size, shuffle=False)
        # with torch.no_grad():
        #     for batch in final_dataloader:
        #         x_input_batch, y_input_batch, y_hat_input_batch = batch
        #         x_input_batch = x_input_batch.float().to(self.device)
        #         y_input_batch = y_input_batch.float().to(self.device)
        #         y_hat_input_batch = y_hat_input_batch.float().to(self.device)
        #         batch_out = est_data_value_model(x_input_batch, y_input_batch, y_hat_input_batch)
        #         batch_out = batch_out.cpu().detach().numpy()
        #         final_data_value.append(batch_out)
        #
        # final_data_value = np.concatenate(final_data_value, 0)
        # final_data_value = final_data_value[:, 0]

        with torch.no_grad():
            x_input_batch = torch.from_numpy(self.x_train).float().to(self.device)
            y_input_batch = torch.from_numpy(self.y_train_onehot).float().to(self.device)
            y_hat_input_batch = torch.from_numpy(y_train_hat).float().to(self.device)
            batch_out = est_data_value_model(x_input_batch, y_input_batch, y_hat_input_batch)
            # batch_out = est_data_value_model(self.x_train, self.y_train_onehot, y_pred_diff)
            batch_out = batch_out.cpu().detach().numpy()
        final_data_value = batch_out[:, 0]

        return final_data_value

    def dvrl_predictor(self, x_test):
        """Returns predictions using the predictor model.
        Args:
        x_test: testing features
        Returns:
        y_test_hat: predictions of the predictive model with DVRL
        """

        if self.flag_sgd:
            y_test_hat,homo_outT,new_hsT = self.final_model.predict(x_test)
        else:
            y_test_hat,homo_outT,new_hsT = self.final_model.predict(is_train_dataset=False, batch_size=self.batch_size_predictor, sample_weight=None, sample_indices=None)

        return y_test_hat,homo_outT,new_hsT