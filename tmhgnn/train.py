# import torch
# import torch.nn
# import torch.nn.functional as F
# from tqdm import tqdm
# import argparse
# import pandas as pd
# import numpy as np
# import sys
# import os

# import sklearn
# from sklearn import *
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import PrecisionRecallDisplay
# from sklearn.metrics import precision_recall_curve, auc
# from sklearn import metrics
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
# from conf import *

# import net as networks
# import util


# def get_optimizer(name, model, lr, wd):
#     if name == 'Adam':
#         return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
#     elif name == 'AdamW':
#         return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
#     else:
#         raise ValueError(f'Optimizer {name} is not supported yet.')

# def get_lr(optimizer):
#     for pg in list(optimizer.param_groups)[::-1]:
#         return pg['lr']
        
# # Function to save the model 
# def saveModel(): 
#     path = args.output

#     dir_name = os.path.dirname(path)
#     if dir_name != "" and not os.path.exists(dir_name):
#         os.makedirs(dir_name, exist_ok=True)

#     torch.save(model.state_dict(), path) 
    
# # Save Last Results into DataFrame
# def saveResult(pred_epoch, true_epoch):
#     pre_cat=np.concatenate((pred_epoch),axis=0).squeeze()
#     true_cat=np.concatenate((true_epoch), axis=0).squeeze()
#     result_df = pd.DataFrame({'pred':pre_cat.tolist(), 'label':true_cat.tolist()})
#     path = args.exp_name + '.pkl'
#     result_df.to_pickle(path)
    
# def eval(model, loader, device):
#     running_accuracy = 0
#     running_vall_loss = 0
#     pred_epoch = []
#     true_epoch = []
#     with torch.no_grad():
#         model.eval()
#         for data in loader:
#             data = data.to(device)
#             # indicies for taxonomy hyperedges' edge index
#             out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)  # Perform a single forward pass.

#             logits = out.view(-1)                  # (B,)
#             y_true = data.y.view(-1).float()       # (B,)

#             val_loss = criterion(logits, y_true)

#             probs = torch.sigmoid(logits)          # (B,)
#             pred = (probs >= 0.5).long()           # 0 또는 1

#             pred_np = probs.detach().cpu().numpy()
#             true_np = y_true.detach().cpu().numpy()
#             pred_epoch.append(pred_np)
#             true_epoch.append(true_np)

#             running_vall_loss += val_loss.item()
#             running_accuracy += int((pred == y_true.long()).sum())

#     # Calculate validation loss value
#     val_loss_value = running_vall_loss/len(loader.dataset)

#     accuracy = 100 * running_accuracy/len(loader.dataset)
#     # Get AUPRC, AUROC
#     true_values = np.concatenate((true_epoch), axis=0).squeeze().tolist()
#     predicted_values = np.concatenate((pred_epoch),axis=0).squeeze().tolist()
#     auprc = average_precision_score(true_values, predicted_values)
#     auroc = roc_auc_score(true_values, predicted_values)

#     return val_loss_value, accuracy, auprc, auroc, pred_epoch, true_epoch


# # Training Function
# def train(num_epochs, model, train_loader, val_loader,  test_loader, criterion, device):
    
#     # for best summary
#     best_accuracy = 0.0
#     best_loss = 100000
#     best_AUPRC = 0.0
#     best_AUROC = 0.0    
    
#     pred_list = []
     
#     print("Begin training...") 
#     step = 1
#     for epoch in range(1, num_epochs+1): 
#         running_train_loss = 0.0 
#         running_train_correct = 0
#         total = 0 
         
#         # Training Loop 
#         print('training epoch',epoch)
#         model.train()
#         for data in tqdm(train_loader): 
#             data = data.to(device)
#             optimizer.zero_grad()   # zero the parameter gradients    
            
#             out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)  # Perform a single forward pass.
#             # loss = criterion(out.float(), data.y.unsqueeze(1).float()) # Compute the loss.
        
#             logits = out.view(-1)                 # (B,)
#             y_true = data.y.view(-1).float()      # (B,)
#             loss = criterion(logits, y_true)   

#             loss.backward()  # Derive gradients.
#             optimizer.step()  # Update parameters based on gradients.
            
#             running_train_loss += loss.item()  # track the loss value 

#             probs = torch.sigmoid(logits)         # (B,)
#             pred = (probs >= 0.5).long()          # 0 또는 1          
#             running_train_accuracy = int((pred == data.y).sum())
            
#             step += 1
 
#         # Calculate training loss value 
#         train_loss_value = running_train_loss/len(train_loader.dataset) 
#         train_acc_value = 100.0 * running_train_correct / len(train_loader.dataset)
        
#         val_loss, val_acc, val_auprc, val_auroc, _, _ = eval(model, val_loader, device)
        
#         if val_acc > best_accuracy:
#             best_accuracy = val_acc
        
#         # get best loss
#         if val_loss < best_loss:
#             best_loss = val_loss
        
#         # get best AUPRC
#         # Save the model if the AUPRC is the best        
#         if val_auprc > best_AUPRC:
#             saveModel()             
#             best_AUPRC = val_auprc
            
#             # test model if the AUPRC is the best
#             _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(model, test_loader, device)
#             print('Epoch:',epoch,'========Test acc: %.4f,' %test_acc, 'Test auproc: %.4f,' %test_auprc, 'Test auroc: %.4f.' %test_auroc,'========')            
        
#         # get best AUROC
#         if val_auroc > best_AUROC:
#             best_AUROC = val_auroc
        
#         # Print the statistics of the epoch 
#         print('Completed Epoch:', epoch, ', Training Loss : %.4f' %train_loss_value, ', Training Accuracy : %.4f %%' % train_acc_value, ', Validation Loss : %.4f' %val_loss, ', Validation Accuracy : %.4f %%' %val_acc)
        
#     print("========End Training========")
#     _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(model, test_loader, device)
#     print('Epoch:',epoch,'========Val acc: %.4f,' %val_acc, 'Val auproc: %.4f,' %val_auprc, 'Val auroc: %.4f.' %val_auroc,'========')            
#     print('Epoch:',epoch,'========Test acc: %.4f,' %test_acc, 'Test auproc: %.4f,' %test_auprc, 'Test auroc: %.4f.' %test_auroc,'========')            
        
#     return test_pred_epoch, test_true_epoch

# if __name__ == '__main__':
#     user_config()
#     args = parse_arguments()    # from conf.py

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
    
#     # Multi Hyperedge
#     if args.dload == 'multi_hyper':
#         # train.py 파일 위치 기준으로 ../graph_construction 를 sys.path에 추가
#         here = os.path.dirname(os.path.abspath(__file__))
#         graph_dir = os.path.join(here, '..', 'graph_construction')
#         sys.path.append(graph_dir)

#         from LoadData import *  # LoadPaitentData 불러오기
#         print('LoadData => LoadPaitentData imported')
#     else:
#         raise ValueError(f'Unknown dataloader: {args.dload}')
        
#     util.seed_everything(args.seed)
#     loader = LoadPaitentData(name=args.task, type='cutoff', tokenizer=args.tokenizer, data_type=args.dtype) 
#     train_loader, val_loader, test_loader, n_class = loader.get_train_test(batch_size=args.bsz, seed=args.seed)

#     if args.tokenizer == 'clinicalbert':
#         in_dim = 4 + 768  
#     elif args.tokenizer == 'word2vec':
#         in_dim = 4 + 100   

#     args.node_features = in_dim
    
#     # Define model            
#     model = getattr(networks, args.model)(num_features=args.node_features, hidden_channels=args.hidden_channels*args.heads_1)
#     model.to(device)
    
#     # Optimization settings
#     if args.loss == 'BCEWithLogitsLoss':
#         criterion = torch.nn.BCEWithLogitsLoss()
#     else:
#         raise ValueError(f'Unknown loss name: {args.loss}')
        
#     optimizer = get_optimizer(args.optimizer, model, args.init_lr, args.weight_decay) 

#     # train model
#     predicted_last, true_last = train(args.num_epochs, model, train_loader, val_loader, test_loader, criterion, device)

#     # save last best results (from test data)
#     saveResult(predicted_last, true_last)

import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

import sklearn
from sklearn import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, auc
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from conf import *

import net as networks
import util

def save_loss_curve(history, args):
    lr = args.init_lr
    decay = args.weight_decay

    fname = f"lr{lr}_decay{decay}_plot.png"
    out_path = os.path.join(os.path.dirname(args.exp_name), fname)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Loss curve saved to {out_path}")

def save_metrics_log(history, args):
    lr = args.init_lr
    decay = args.weight_decay

    fname = f"lr{lr}_decay{decay}_log.txt"
    out_path = os.path.join(os.path.dirname(args.exp_name), fname)

    best = history["best"]
    final_val  = history["final_val"]
    final_test = history["final_test"]

    with open(out_path, "w") as f:
        f.write("=== Best on Validation (across epochs) ===\n")
        f.write(f"best_val_acc   : {best['val_acc']:.4f}\n")
        f.write(f"best_val_loss  : {best['val_loss']:.4f}\n")
        f.write(f"best_val_auprc : {best['val_auprc']:.4f}\n")
        f.write(f"best_val_auroc : {best['val_auroc']:.4f}\n\n")

        f.write("=== Final Epoch Validation ===\n")
        f.write(f"val_acc   : {final_val['val_acc']:.4f}\n")
        f.write(f"val_auprc : {final_val['val_auprc']:.4f}\n")
        f.write(f"val_auroc : {final_val['val_auroc']:.4f}\n\n")

        f.write("=== Final Test ===\n")
        f.write(f"test_acc   : {final_test['test_acc']:.4f}\n")
        f.write(f"test_auprc : {final_test['test_auprc']:.4f}\n")
        f.write(f"test_auroc : {final_test['test_auroc']:.4f}\n")

    print(f"[INFO] Metrics log saved to {out_path}")



def get_optimizer(name, model, lr, wd):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Optimizer {name} is not supported yet.')

def get_lr(optimizer):
    for pg in list(optimizer.param_groups)[::-1]:
        return pg['lr']
        
# Function to save the model 
def saveModel(): 
    path = args.output

    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    torch.save(model.state_dict(), path) 
    
# Save Last Results into DataFrame
def saveResult(pred_epoch, true_epoch):
    pre_cat=np.concatenate((pred_epoch),axis=0).squeeze()
    true_cat=np.concatenate((true_epoch), axis=0).squeeze()
    result_df = pd.DataFrame({'pred':pre_cat.tolist(), 'label':true_cat.tolist()})
    path = args.exp_name + '.pkl'
    result_df.to_pickle(path)
    
def eval(model, loader, device):
    running_accuracy = 0
    running_vall_loss = 0
    pred_epoch = []
    true_epoch = []
    with torch.no_grad():
        model.eval()
        for data in loader:
            data = data.to(device)
            # indicies for taxonomy hyperedges' edge index
            out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)  # Perform a single forward pass.

            logits = out.view(-1)                  # (B,)
            y_true = data.y.view(-1).float()       # (B,)

            val_loss = criterion(logits, y_true)

            probs = torch.sigmoid(logits)          # (B,)
            pred = (probs >= 0.5).long()           # 0 또는 1

            pred_np = probs.detach().cpu().numpy()
            true_np = y_true.detach().cpu().numpy()
            pred_epoch.append(pred_np)
            true_epoch.append(true_np)

            running_vall_loss += val_loss.item()
            running_accuracy += int((pred == y_true.long()).sum())

    # Calculate validation loss value
    val_loss_value = running_vall_loss/len(loader.dataset)

    accuracy = 100 * running_accuracy/len(loader.dataset)
    # Get AUPRC, AUROC
    true_values = np.concatenate((true_epoch), axis=0).squeeze().tolist()
    predicted_values = np.concatenate((pred_epoch),axis=0).squeeze().tolist()
    auprc = average_precision_score(true_values, predicted_values)
    auroc = roc_auc_score(true_values, predicted_values)

    return val_loss_value, accuracy, auprc, auroc, pred_epoch, true_epoch


# Training Function
def train(num_epochs, model, train_loader, val_loader,  test_loader, criterion, device):
    
    # for best summary
    best_accuracy = 0.0
    best_loss = 100000
    best_AUPRC = 0.0
    best_AUROC = 0.0    
    
    pred_list = []

    # 히스토리 기록용
    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []
    val_auprcs   = []
    val_aurocs   = []
     
    print("Begin training...") 
    step = 1
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        running_train_correct = 0
        total = 0 
         
        # Training Loop 
        print('training epoch',epoch)
        model.train()
        for data in tqdm(train_loader): 
            data = data.to(device)
            optimizer.zero_grad()   # zero the parameter gradients    
            
            out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)  # Perform a single forward pass.
            # loss = criterion(out.float(), data.y.unsqueeze(1).float()) # Compute the loss.
        
            logits = out.view(-1)                 # (B,)
            y_true = data.y.view(-1).float()      # (B,)
            loss = criterion(logits, y_true)   

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            
            running_train_loss += loss.item()  # track the loss value 

            probs = torch.sigmoid(logits)         # (B,)
            pred = (probs >= 0.5).long()          # 0 또는 1  
            
            running_train_correct += int((pred == y_true.long()).sum())
            total += y_true.size(0)
                    
            step += 1
 
        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader.dataset) 
        train_acc_value = 100.0 * running_train_correct / len(train_loader.dataset)
        
        val_loss, val_acc, val_auprc, val_auroc, _, _ = eval(model, val_loader, device)

        train_losses.append(train_loss_value)
        val_losses.append(val_loss)
        train_accs.append(train_acc_value)
        val_accs.append(val_acc)
        val_auprcs.append(val_auprc)
        val_aurocs.append(val_auroc)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
        
        # get best loss
        if val_loss < best_loss:
            best_loss = val_loss
        
        # get best AUPRC
        # Save the model if the AUPRC is the best        
        if val_auprc > best_AUPRC:
            saveModel()             
            best_AUPRC = val_auprc
            
            # test model if the AUPRC is the best
            _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(model, test_loader, device)
            print('Epoch:',epoch,'========Test acc: %.4f,' %test_acc, 'Test auproc: %.4f,' %test_auprc, 'Test auroc: %.4f.' %test_auroc,'========')            
        
        # get best AUROC
        if val_auroc > best_AUROC:
            best_AUROC = val_auroc
        
        # Print the statistics of the epoch 
        print('Completed Epoch:', epoch, ', Training Loss : %.4f' %train_loss_value, ', Training Accuracy : %.4f %%' % train_acc_value, ', Validation Loss : %.4f' %val_loss, ', Validation Accuracy : %.4f %%' %val_acc)
        
    print("========End Training========")
    _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(model, test_loader, device)
    print('Epoch:',epoch,'========Val acc: %.4f,' %val_acc, 'Val auproc: %.4f,' %val_auprc, 'Val auroc: %.4f.' %val_auroc,'========')            
    print('Epoch:',epoch,'========Test acc: %.4f,' %test_acc, 'Test auproc: %.4f,' %test_auprc, 'Test auroc: %.4f.' %test_auroc,'========')     

    history = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_acc":  train_accs,
        "val_acc":    val_accs,
        "val_auprc":  val_auprcs,
        "val_auroc":  val_aurocs,
        "best": {
            "val_acc":   best_accuracy,
            "val_loss":  best_loss,
            "val_auprc": best_AUPRC,
            "val_auroc": best_AUROC,
        },
        "final_val": {
            "val_acc":   val_acc,
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
        },
        "final_test": {
            "test_acc":   test_acc,
            "test_auprc": test_auprc,
            "test_auroc": test_auroc,
        },
    }       
        
    return test_pred_epoch, test_true_epoch, history

if __name__ == '__main__':
    user_config()
    args = parse_arguments()    # from conf.py

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Multi Hyperedge
    if args.dload == 'multi_hyper':
        # train.py 파일 위치 기준으로 ../graph_construction 를 sys.path에 추가
        here = os.path.dirname(os.path.abspath(__file__))
        graph_dir = os.path.join(here, '..', 'graph_construction')
        sys.path.append(graph_dir)

        from LoadData import *  # LoadPaitentData 불러오기
        print('LoadData => LoadPaitentData imported')
    else:
        raise ValueError(f'Unknown dataloader: {args.dload}')
        
    util.seed_everything(args.seed)
    loader = LoadPaitentData(name=args.task, type='cutoff', tokenizer=args.tokenizer, data_type=args.dtype) 
    train_loader, val_loader, test_loader, n_class = loader.get_train_test(batch_size=args.bsz, seed=args.seed)

    if args.tokenizer == 'clinicalbert':
        in_dim = 4 + 768  
    elif args.tokenizer == 'word2vec':
        in_dim = 4 + 100   

    args.node_features = in_dim
    
    # Define model            
    model = getattr(networks, args.model)(num_features=args.node_features, hidden_channels=args.hidden_channels*args.heads_1)
    model.to(device)
    
    # Optimization settings
    if args.loss == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unknown loss name: {args.loss}')
        
    optimizer = get_optimizer(args.optimizer, model, args.init_lr, args.weight_decay) 

    # train model
    predicted_last, true_last, history = train(args.num_epochs, model, train_loader, val_loader, test_loader, criterion, device)

    # save last best results (from test data)
    # saveResult(predicted_last, true_last)
    save_loss_curve(history, args)
    save_metrics_log(history, args)