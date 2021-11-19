from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

def all_classifiers(num_class, type_model, network_pretrained = True, path_weight = None, num_epoch = None):
    
    if(type_model == 'resnet'):
        
        if(network_pretrained):
            model = models.resnet152(pretrained=True)
            # Changing the model head
            model.fc = nn.Linear(model.fc.in_features, num_class)
        elif(not network_pretrained):
            model = models.resnet152(pretrained=False)
            # Changing the model head
            model.fc = nn.Linear(model.fc.in_features, num_class)
            # If path_weight has been entered, the network will use these weights
            try:
                model.load_state_dict(torch.load('{}/salved_epoch{}.pt'.format(path_weight, num_epoch), map_location='cuda:0'))
            except RuntimeError:
                raise RuntimeError("CUDA out of memory")
            except FileNotFoundError:
                raise FileNotFoundError("File not found")
            except:
                raise ValueError('Unexpected problem')
                
                
    elif(type_model == 'squeezenet'):
        
        if(network_pretrained):
            model = models.squeezenet1_0(pretrained=True)                         
            # Changing the model head
            model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1,1), stride=(1,1))
        elif(not network_pretrained):
            model = models.squeezenet1_0(pretrained=False)
            # Changing the model head
            model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1,1), stride=(1,1))
            # If path_weight has been entered, the network will use these weights
            try:
                model.load_state_dict(torch.load('{}/salved_epoch{}.pt'.format(path_weight, num_epoch), map_location='cuda:0'))
            except RuntimeError:
                raise RuntimeError("CUDA out of memory")
            except FileNotFoundError:
                raise FileNotFoundError("File not found")
            except:
                raise ValueError('Unexpected problem')
                
    elif(type_model == 'vgg'):
        
        if(network_pretrained):
            model = models.vgg16(pretrained=True) 
            # Changing the model head
            model.classifier[6] = nn.Linear(4096,num_class)
        elif(not network_pretrained):
            model = models.vgg16(pretrained=False)
            # Changing the model head
            model.classifier[6] = nn.Linear(4096,num_class)
            # If path_weight has been entered, the network will use these weights
            try:
                model.load_state_dict(torch.load('{}/salved_epoch{}.pt'.format(path_weight, num_epoch), map_location='cuda:0'))
            except RuntimeError:
                raise RuntimeError("CUDA out of memory")
            except FileNotFoundError:
                raise FileNotFoundError("File not found")
            except:
                raise ValueError('Unexpected problem')
                
    elif(type_model == 'densenet'):
        
        if(network_pretrained):
            model = models.densenet201(pretrained=True)
            # Changing the model head
            model.classifier = nn.Linear(1920, num_class)
        elif(not network_pretrained):
            model = models.densenet201(pretrained=False)
            # Changing the model head
            model.classifier = nn.Linear(1920, num_class)
            # If path_weight has been entered, the network will use these weights
            try:
                model.load_state_dict(torch.load('{}/salved_epoch{}.pt'.format(path_weight, num_epoch), map_location='cuda:0'))
            except RuntimeError:
                raise RuntimeError("CUDA out of memory")
            except FileNotFoundError:
                raise FileNotFoundError("File not found")
            except:
                raise ValueError('Unexpected problem')
                
    return model

#This network receives a vector of probabilities of a data and analyzes whether this data is OOD or not.
class GCOOD_network(nn.Module):
    def __init__(self, num_class, input_size, config, norm=True):
        super(GCOOD_network, self).__init__()
        
        feature_extraction_depth = config['feature_extraction_depth']
        feature_extraction_out_channels = config['groups']
        feature_extraction_activation = config['activ_function']
        classification_depth = config['classification_depth']
        classification_dropout = config['dropout']
        
        if(norm):
            layers = [nn.Sequential(
                nn.Conv1d(in_channels=1, 
                          out_channels=input_size*feature_extraction_out_channels, 
                          kernel_size=1, 
                          stride=1),
                nn.BatchNorm1d(input_size*feature_extraction_out_channels),
                choose_activ_function(feature_extraction_activation))]
            for i in range(feature_extraction_depth-1):
                layers.append(nn.Sequential(
                    nn.Conv1d(in_channels=input_size*feature_extraction_out_channels, 
                              out_channels=input_size*feature_extraction_out_channels, 
                              kernel_size=1, 
                              stride=1,
                              groups=feature_extraction_out_channels),
                    nn.BatchNorm1d(input_size*feature_extraction_out_channels),
                    choose_activ_function(feature_extraction_activation)))
            self.features = nn.Sequential(*layers)
           
            layers = [nn.Flatten()]
        
            for _ in range(classification_depth-1):
                layers.append(nn.Sequential(
                    nn.Linear(in_features=input_size*(input_size*feature_extraction_out_channels), 
                              out_features=input_size*(input_size*feature_extraction_out_channels)),
                    nn.BatchNorm1d(input_size*feature_extraction_out_channels),
                    nn.Dropout(p=classification_dropout)))
                
        elif(not norm):
            layers = [nn.Sequential(
                nn.Conv1d(in_channels=1, 
                          out_channels=input_size*feature_extraction_out_channels, 
                          kernel_size=1, 
                          stride=1),
                choose_activ_function(feature_extraction_activation))]
            for i in range(feature_extraction_depth-1):
                layers.append(nn.Sequential(
                    nn.Conv1d(in_channels=input_size*feature_extraction_out_channels, 
                              out_channels=input_size*feature_extraction_out_channels, 
                              kernel_size=1, 
                              stride=1,
                              groups=feature_extraction_out_channels),
                    choose_activ_function(feature_extraction_activation)))
            self.features = nn.Sequential(*layers)
           
            layers = [nn.Flatten()]
        
            for _ in range(classification_depth-1):
                layers.append(nn.Sequential(
                    nn.Linear(in_features=input_size*(input_size*feature_extraction_out_channels), 
                              out_features=input_size*(input_size*feature_extraction_out_channels)),
                    nn.Dropout(p=classification_dropout)))
                
        layers.append(nn.Linear(in_features=input_size*(input_size*feature_extraction_out_channels),
                                out_features=2))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        probab = F.sigmoid(logits)
        return logits, probab
    

# Choose activation function.    
def choose_activ_function(choose_activ):
    if(choose_activ == 'relu'):
        return nn.ReLU()
    if(choose_activ == 'softmax'):
        return nn.Softmax()
    if(choose_activ == 'sigmoid'):
        return nn.Sigmoid()
    if(choose_activ == 'tanh'):
        return nn.Tanh()
    else:
        raise ValueError(f'Invalid choose_activ({choose_activ})')