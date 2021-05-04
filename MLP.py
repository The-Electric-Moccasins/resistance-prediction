# import needed libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from hyper_params import HyperParams
from tqdm import tqdm

# create params object
params = HyperParams()
# set PyTorch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create Loaders
def create_loaders(dataset, num_y, batch_size, balance = True):
    # split the data into train, validation, and test sets
    if num_y == 1:
        x_train, x_test, y_train, y_test = train_test_split(dataset[:,:-num_y], dataset[:,-num_y], test_size=params.test_set_fraction)
    else:
        x_train, x_test, y_train, y_test = train_test_split(dataset[:,:-num_y], dataset[:,-num_y:], test_size=params.test_set_fraction)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=params.validation_set_fraction / (params.validation_set_fraction + params.train_set_fraction))
    
    # convert the NumPy arrays into Pytorch tensors
    x_train = torch.from_numpy(x_train).type(torch.float)
    x_val = torch.from_numpy(x_val).type(torch.float)
    x_test = torch.from_numpy(x_test).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.float)
    
    # Create datasets from the tensors
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    if balance:
        class_sample_count = np.unique(y_train, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[y_train.type(torch.int8)]

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = DataLoader(train_dataset, batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)
    
    return train_loader, val_loader, test_loader

# create MLP model class with 2 hidden layers, relu activation, and sigmoid output activation
class MLP(nn.Module):
    def __init__(self, nodes, p, num_in, num_out, multilabel):
        super(MLP, self).__init__()
        self.nodes = nodes
        self.p = p
        self.input_nodes = num_in
        self.output_nodes = num_out
        self.multilabel = multilabel
        if len(self.nodes) == 2:
            self.fc1 = nn.Linear(self.input_nodes, self.nodes[0])
            self.dropout1 = nn.Dropout(p = self.p)
            self.bn1 = nn.BatchNorm1d(self.nodes[0])
            self.fc2 = nn.Linear(self.nodes[0], self.nodes[1])
            self.dropout2 = nn.Dropout(p = self.p)
            self.bn2 = nn.BatchNorm1d(self.nodes[1])
            self.fc3 = nn.Linear(self.nodes[1], self.output_nodes)
            if self.multilabel:
                self.out = nn.Sigmoid()
        elif len(self.nodes) == 4:
            self.fc1 = nn.Linear(self.input_nodes, self.nodes[0])
            self.dropout1 = nn.Dropout(p = self.p)
            self.bn1 = nn.BatchNorm1d(self.nodes[0])
            self.fc2 = nn.Linear(self.nodes[0], self.nodes[1])
            self.dropout2 = nn.Dropout(p = self.p)
            self.bn2 = nn.BatchNorm1d(self.nodes[1])
            self.fc3 = nn.Linear(self.nodes[1], self.nodes[2])
            self.dropout3 = nn.Dropout(p = self.p)
            self.bn3 = nn.BatchNorm1d(self.nodes[2])
            self.fc4 = nn.Linear(self.nodes[2], self.nodes[3])
            self.dropout4 = nn.Dropout(p = self.p)
            self.bn4 = nn.BatchNorm1d(self.nodes[3])
            self.fc5 = nn.Linear(self.nodes[3], self.output_nodes)
            if self.multilabel:
                self.out = nn.Sigmoid()
        elif len(self.nodes) == 6:
            self.fc1 = nn.Linear(self.input_nodes, self.nodes[0])
            self.dropout1 = nn.Dropout(p = self.p)
            self.bn1 = nn.BatchNorm1d(self.nodes[0])
            self.fc2 = nn.Linear(self.nodes[0], self.nodes[1])
            self.dropout2 = nn.Dropout(p = self.p)
            self.bn2 = nn.BatchNorm1d(self.nodes[1])
            self.fc3 = nn.Linear(self.nodes[1], self.nodes[2])
            self.dropout3 = nn.Dropout(p = self.p)
            self.bn3 = nn.BatchNorm1d(self.nodes[2])
            self.fc4 = nn.Linear(self.nodes[2], self.nodes[3])
            self.dropout4 = nn.Dropout(p = self.p)
            self.bn4 = nn.BatchNorm1d(self.nodes[3])
            self.fc5 = nn.Linear(self.nodes[3], self.nodes[4])
            self.dropout5 = nn.Dropout(p = self.p)
            self.bn5 = nn.BatchNorm1d(self.nodes[4])
            self.fc6 = nn.Linear(self.nodes[4], self.nodes[5])
            self.dropout6 = nn.Dropout(p = self.p)
            self.bn6 = nn.BatchNorm1d(self.nodes[5])
            self.fc7 = nn.Linear(self.nodes[5], self.output_nodes)
            if self.multilabel:
                self.out = nn.Sigmoid()
        else:
            raise

    def forward(self, x):
        if len(self.nodes) == 2:
            x = self.fc1(x)
            x = self.dropout1(x)
            x = nn.functional.elu(x)
            x = self.bn1(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            x = nn.functional.elu(x)
            x = self.bn2(x)
            x = self.fc3(x)
            if self.multilabel:
                x = self.out(x)
            return x
        elif len(self.nodes) == 4:
            x = self.fc1(x)
            x = self.dropout1(x)
            x = nn.functional.elu(x)
            x = self.bn1(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            x = nn.functional.elu(x)
            x = self.bn2(x)
            x = self.fc3(x)
            x = self.dropout3(x)
            x = nn.functional.elu(x)
            x = self.bn3(x)
            x = self.fc4(x)
            x = self.dropout4(x)
            x = nn.functional.elu(x)
            x = self.bn4(x)
            x = self.fc5(x)
            if self.multilabel:
                x = self.out(x)
            return x
        elif len(self.nodes) == 6:
            x = self.fc1(x)
            x = self.dropout1(x)
            x = nn.functional.elu(x)
            x = self.bn1(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            x = nn.functional.elu(x)
            x = self.bn2(x)
            x = self.fc3(x)
            x = self.dropout3(x)
            x = nn.functional.elu(x)
            x = self.bn3(x)
            x = self.fc4(x)
            x = self.dropout4(x)
            x = nn.functional.elu(x)
            x = self.bn4(x)
            x = self.fc5(x)
            x = self.dropout5(x)
            x = nn.functional.elu(x)
            x = self.bn5(x)
            x = self.fc6(x)
            x = self.dropout6(x)
            x = nn.functional.elu(x)
            x = self.bn6(x)
            x = self.fc7(x)
            if self.multilabel:
                x = self.out(x)
            return x
        else:
            raise

def plot_loss(n_epochs, train_loss, val_loss, upper_y_lim):
    plt.plot(list(range(1, n_epochs+1)), train_loss, color='blue')
    plt.plot(list(range(1, n_epochs+1)), val_loss, color='orange')
    plt.ylim((0.0, upper_y_lim))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training (Blue) and Validation (Orange) Loss')
    plt.show()
    return

def classify(model, loader, multilabel):
    model.eval()
    y_true = torch.LongTensor()
    y_pred = torch.LongTensor()
    for data in loader:
        x, y = data[0].to(device), data[1].to(device)
        y_hat = model(x)
        if multilabel:
            num_pos_labels = y_hat.shape[1] // 2
            y_hat = torch.max(y_hat[:,-num_pos_labels:], 1).values
            y = torch.max(y[:,-num_pos_labels:], 1).values
        else:
            y_hat = torch.sigmoid(y_hat)
            y_hat = y_hat.view(y_hat.shape[0])
        y_hat = torch.where(y_hat >= 0.5, torch.ones_like(y_hat), torch.zeros_like(y_hat))
        y_true = torch.cat((y_true, y.to('cpu').long()), dim=0)
        y_pred = torch.cat((y_pred,  y_hat.to('cpu').long()), dim=0)
    return y_true, y_pred

def predict(model, loader, multilabel):
    model.eval()
    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()
    for data in loader:
        x, y = data[0].to(device), data[1].to(device)
        y_hat = model(x)
        if multilabel:
            num_pos_labels = y_hat.shape[1] // 2
            y_hat = torch.max(y_hat[:,-num_pos_labels:], 1).values
            y = torch.max(y[:,-num_pos_labels:], 1).values
        else:
            y_hat = torch.sigmoid(y_hat)
            y_hat = y_hat.view(y_hat.shape[0])
        y_true = torch.cat((y_true, y.to('cpu').float()), dim=0)
        y_pred = torch.cat((y_pred,  y_hat.to('cpu').float()), dim=0)
    return y_true, y_pred

# ROC_AUC curve
def plt_roc_auc_curve(model, loader, model_name, multilabel):
    # predict probabilities
    y_test, model_probs = predict(model, loader, multilabel)
    # convert from Torch to Numpy
    y_test, model_probs = y_test.detach().numpy(), model_probs.detach().numpy()
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    model_auc = roc_auc_score(y_test, model_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print(model_name + ': ROC AUC=%.3f' % (model_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

def plt_precision_recall_curve(model, loader, model_name, multilabel):
    # predict probabilities
    y_test, model_probs = predict(model, loader, multilabel)
    # convert from Torch to Numpy
    y_test, model_probs = y_test.detach().numpy(), model_probs.detach().numpy()
    # predict class values
    _, y_pred = classify(model, loader, multilabel)
    # convert from Torch to Numpy
    y_pred = y_pred.detach().numpy()
    model_precision, model_recall, _ = precision_recall_curve(y_test, model_probs)
    model_f1, model_auc = f1_score(y_test, y_pred), auc(model_recall, model_precision)
    # summarize scores
    print(model_name + ': f1=%.3f auc=%.3f' % (model_f1, model_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(model_recall, model_precision, marker='.', label=model_name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

def evaluate(model, test_loader, multilabel):
    # Prediction
    y_true, y_pred = classify(model, test_loader, multilabel)

    # Classification report (recall, preccision, f-score, accuracy)
    print(classification_report(y_true, y_pred, digits=4))
    print()
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    print('TN:',tn, 'FP:',fp, 'FN:',fn, 'TP:',tp )

    # ROC_AUC curve
    model_name='MLP'
    print()
    plt_roc_auc_curve(model, test_loader, model_name, multilabel)
    # Precision_Recall curve
    print()
    plt_precision_recall_curve(model, test_loader, model_name, multilabel)
    return

#-------------------------------------Binary-------------------------------------
def binary(dataset, n_epochs, nodes, batch_size = 32, upper_y_lim = 1, p = 0.5):
    train_loader, val_loader, test_loader = create_loaders(dataset, 1, batch_size, balance = True)

    # create a training function that will output the model and its metrics for given nodes
    def train(dataset, n_epochs, nodes, p):
        num_in = dataset.shape[1] - 1
        num_out = 1
        model = MLP(nodes, p, num_in, num_out, multilabel=False).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loss = []
        val_loss = []
        for epoch in tqdm(range(n_epochs)):
            train_loss_epoch = 0
            val_loss_epoch = 0
            model.train()
            for data in train_loader:
                x, y = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                y_hat = y_hat.view(y_hat.shape[0])
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            model.eval()
            for data in val_loader:
                x, y = data[0].to(device), data[1].to(device)
                y_hat = model(x)
                y_hat = y_hat.view(y_hat.shape[0])
                loss = criterion(y_hat, y)
                val_loss_epoch += loss.item()
            train_loss.append(train_loss_epoch / len(train_loader))
            val_loss.append(val_loss_epoch / len(val_loader))
        return model, train_loss, val_loss

    model, train_loss, val_loss = train(dataset, n_epochs, nodes, p)
    plot_loss(n_epochs, train_loss, val_loss, upper_y_lim)
    evaluate(model, test_loader, multilabel = False)
    
    return model

#-----------------------------------Mulitlabel-----------------------------------
def multilabel(dataset, num_y, n_epochs, nodes, batch_size = 32, upper_y_lim = 1, p = 0.5):
    train_loader, val_loader, test_loader = create_loaders(dataset, num_y, batch_size, balance = False)
    
    # create a training function that will output the model and its metrics for given nodes
    def train(dataset, num_y, n_epochs, nodes, p):
        num_in = dataset.shape[1] - num_y
        num_out = num_y
        model = MLP(nodes, p, num_in, num_out, multilabel=True).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loss = []
        val_loss = []
        for epoch in tqdm(range(n_epochs)):
            train_loss_epoch = 0
            val_loss_epoch = 0
            model.train()
            for data in train_loader:
                x, y = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            model.eval()
            for data in val_loader:
                x, y = data[0].to(device), data[1].to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss_epoch += loss.item()
            train_loss.append(train_loss_epoch / len(train_loader))
            val_loss.append(val_loss_epoch / len(val_loader))
        return model, train_loss, val_loss
    
    model, train_loss, val_loss = train(dataset, num_y, n_epochs, nodes, p)
    plot_loss(n_epochs, train_loss, val_loss, upper_y_lim)
    evaluate(model, test_loader, multilabel = True)
    
    return model