from func_dataset.data_utils import get_train_test_excl_lab, loady_dataset
from models.my_models import all_classifiers
from func_dataset.data_utils import build_dataset
from settings.config import config_data_classifiers
from settings.config import config_GCOOD_network
from binary.binary_utils import get_train_test_for_network
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import time
import os
from sklearn.metrics import f1_score

# Choose Adam or SGD optimizer.
def choose_optimizer(choose_optimizer, model, learning_rate):
    if(choose_optimizer == 'adam'):
        # Create the Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(choose_optimizer == 'sgd'):
        # Create the SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
        
    else:
        raise ValueError(f'Invalid choose_optmizer({choose_optimizer})')
        
    return optimizer


#This function receives data from cifar 10 through the build_dataset function of the func_dataset.data_utils package and trains model_resnet152 which is in the models.my_models package.
def train_classifier(device, model, config, train_loader, valid_loader):
    
    # Defines the Cost Function Based on Cross Entropy
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer chosen
    optimizer = choose_optimizer(config['optimizer'], model, config['learning_rate'])
    
    
    # A gamma value decays with each step epoch
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=config['step_size'], gamma=config['gamma'])
    
    # Loop for each epoch
    for epoch in range(config['init_count_epoch'], config['epochs']+1):
    
        # Start counting the start of the execution time of each epoch
        init_time = time.time()
    
        # Initialize Error Every Epoch
        train_loss = 0.0
        valid_loss = 0.0
    
        # Create Model Training Object
        model.train()
        
        # starts error counter to calculate training accuracy
        running_corrects_train = 0
        
        # For Each Data Batch Run Training
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            _, preds_train = torch.max(output, 1)
            running_corrects_train += torch.sum(preds_train == target)
        exp_lr_scheduler.step
            
        # Creates Model Assessment Object    
        model.eval()
    
        # starts the error counter to calculate the validation accuracy
        running_corrects_valid = 0
        
        # For Each Data Batch Run the Assessment
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data) 
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            _, preds_valid = torch.max(output, 1)
            running_corrects_valid += torch.sum(preds_valid == target)
        
        # Calculates Error in Epoch
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        # Accuracy Calculation
        acc_train = running_corrects_train.double() / len(train_loader.sampler)
        acc_valid = running_corrects_valid.double() / len(valid_loader.sampler)

        if epoch % config['frequency_save_model'] == 0:
            torch.save(model.state_dict(), '{}/salved_epoch{}.pt'.format(config['path_folder_save_weight'], epoch))
        
        # Time taken to run an epoch
        time_elapsed = time.time() - init_time
    
        # Print
        print(f'| Epoch: {epoch:02} | Error_train: {train_loss:.7f} | Acc_train {acc_train:.7f} | Error_val: {valid_loss:.7f} | Acc_val {acc_valid:.7f} | Time_spent: {int(time_elapsed//60):}m {int(time_elapsed % 60)}s|')


# Test the initial model
def test_classifier(device, model, test_loader):
    
    model.eval()

    # Defines the Cost Function Based on Cross Entropy
    criterion = nn.CrossEntropyLoss()
    
    # initiate the error
    test_loss = 0.0
    
    # starts the error counter to calculate the test accuracy
    running_corrects_test = 0
        
    # Run the model on some test examples
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data) 
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, preds_test = torch.max(output, 1)
            running_corrects_test += torch.sum(preds_test == target)
        

        # Calculates Error in Epoch
        test_loss = test_loss/len(test_loader.sampler)
        
        # Accuracy Calculation
        acc_test = running_corrects_test.double() / len(test_loader.sampler)

        print(f'| Error_test: {test_loss:.7f} | Acc_test {acc_test:.7f} |')
        
        return acc_test.item()
    
# This function takes the configuration data, the list of label that will be deleted, and whether the data will be processed in cpu or gpu, and trains the data by storing the model, and tests the result.
def pipeline_excl_lab_classifier(device, labels):
    
    # Reconfiguring the path where weights will be saved
    try:
        old_path = config_data_classifiers['path_folder_save_weight']
        new_path = old_path.split('models')[0]+'models/'+'_'.join(map(str, labels))
        config_data_classifiers['path_folder_save_weight'] = new_path
        os.mkdir(new_path)
    except FileNotFoundError:
        raise FileNotFoundError('The {} directory cannot be created. Check the path entered.'.format(path_save))
    except FileExistsError:
        config_data_classifiers['path_folder_save_weight'] = new_path
    
    #Datasets
    train_loader, valid_loader, test_loader, num_class = get_train_test_excl_lab(device, labels, batch_size=config_data_classifiers['batch_size'])

    # Model
    model = all_classifiers(num_class, config_data_classifiers['model'])
    model.to(device)
    
    # Training the model
    train_classifier(device, model, config_data_classifiers, train_loader, valid_loader)

    # Testing model performance
    test_classifier(device, model, test_loader)
    
# This function takes the data from the last layer of a pre-processed classifier, and trains that data.
def train_GCOOD(device, model, config, train_loader, valid_loader):
    
    # Defines the Cost Function Based on Cross Entropy. 
    # Default value of pos_weight0 = 5,875
    pos_weight0 = 5.875
    pos_weight = torch.as_tensor([pos_weight0, 1 / pos_weight0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Chosen optimizer
    optimizer = choose_optimizer(config['optimizer'], model, config['learning_rate'])
    
    #Loop for each epoch
    for epoch in range(config['init_count_epoch'], config['epochs']+1):
    
        # Start counting the start of the execution time of each epoch
        init_time = time.time()
    
        # Initialize Error Every Epoch
        train_loss = 0.0
        valid_loss = 0.0
    
        # Create Model Training Object
        model.train()
        
        #Start error counter to calculate training accuracy
        running_corrects_train = 0
        
        # For Each Data Batch Run Training
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(0).transpose(1,0).type(torch.float32) #transform target
            data, target = data.to(device), target.to(device)
            targ = target.tolist()
        
            optimizer.zero_grad()
            logits,probab = model(data.unsqueeze(0).transpose(0,1)) # It is in the format to use the network 
            target_for_criterion = torch.stack((1 - target, target), dim=1)
            target_for_criterion = torch.squeeze(target_for_criterion)
            loss = criterion(logits, target_for_criterion)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            _, preds_train = torch.max(logits, 1)
            
            pred = preds_train.tolist()
            F1_score_train = f1_score(targ, pred)
                
            running_corrects_train += torch.tensor(torch.sum(preds_train == target,dim=0)[0], dtype=torch.float32)

            
        # reates Model Assessment Object    
        model.eval()
    
        #Starts error counter to calculate validation accuracy
        running_corrects_valid = 0
        
        # For Each Data Batch Run the Assessment
        for batch_idx, (data, target) in enumerate(valid_loader):
            target = target.unsqueeze(0).transpose(1,0).type(torch.float32)
            data, target = data.to(device), target.to(device)
            targ_v = target.tolist()
            
            logits, probab = model(data.unsqueeze(0).transpose(0,1)) # It is in the format to use the network 
            target_for_criterion = torch.stack((1 - target, target), dim=1)
            target_for_criterion = torch.squeeze(target_for_criterion)
            loss = criterion(logits, target_for_criterion)
            valid_loss += loss.item()*data.size(0)
            _, preds_valid = torch.max(logits, 1)
            running_corrects_valid += torch.tensor(torch.sum(preds_valid == target,dim=0)[0], dtype=torch.float32)
    
            pred_v = preds_valid.tolist()
            
            F1_score_valid = f1_score(targ_v, pred_v)
        
        # Calculates Error in Epoch
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        # Accuracy Calculation
        acc_train = running_corrects_train.double() / len(train_loader.sampler)
        acc_valid = running_corrects_valid.double() / len(valid_loader.sampler)
        

        #Time taken to run an epoch
        time_elapsed = time.time() - init_time
        
        # Print
        print(f'| Epoch: {epoch:02} | Error_train: {train_loss:.7f} | Acc_train {acc_train:.7f} | F1_Score {F1_score_train:.7f} | Error_val: {valid_loss:.7f} | Acc_val {acc_valid:.7f} | F1_Score {F1_score_valid:.7f} |') 
        print('\n')
        
        
# This function takes the data from the last layer of resnet152 and after processing, tests the network.
def test_GCOOD(device, model, test_loader):
    
    model.eval()

    # Defines the Cost Function Based on Cross Entropy
    # Default value of pos_weight0 = 5,875
    pos_weight0 = 5.875
    pos_weight = torch.as_tensor([pos_weight0, 1 / pos_weight0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    #initiate the error
    test_loss = 0.0
            
    #starts the error counter to calculate the test accuracy
    running_corrects_test = 0
        
    # Run the model on some test examples
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target.unsqueeze(0).transpose(1,0).type(torch.float32)
            data, target = data.to(device), target.to(device)
            
            targ_test = target.tolist()
            logits,probab = model(data.unsqueeze(0).transpose(0,1)) # It is in the format to use the network
            target_for_criterion = torch.stack((1 - target, target), dim=1)
            target_for_criterion = torch.squeeze(target_for_criterion)
            loss = criterion(logits, target_for_criterion)
            test_loss += loss.item()*data.size(0)
            _, preds_test = torch.max(logits, 1)
            running_corrects_test += torch.tensor(torch.sum(preds_test == target,dim=0)[0], dtype=torch.float32)
            
            pred_test = preds_test.tolist()
            
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(targ_test, pred_test).ravel()
        
            
            F1_score_test = f1_score(targ_test, pred_test)
        

        # Calculates Error in Epoch
        test_loss = test_loss/len(test_loader.sampler)
        
        # Accuracy Calculation
        acc_test = running_corrects_test.double() / len(test_loader.sampler)

        print(f'| Error_test: {test_loss:.7f} | Acc_test {acc_test:.7f} | F1_Score {F1_score_test:.7f} |')
        
        
# This function is intended to take the data and train and test the model.
def GCOOD_pipeline(device, model):
    
    #Datasets
    train_loader, valid_loader, test_loader = get_train_test_for_network()
    
    # Training the model
    train_GCOOD(device, model, config_GCOOD_network, train_loader, valid_loader)

    # Testing model performance
    test_GCOOD(device, model, test_loader)