#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
# Completed using Udacity Knowledge and Udacity GPT

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets 
import argparse
import os

from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    #print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}, Testing Loss: {total_loss}")
    #hook.set_mode(smd.modes.PREDICT)
    

def train(model, train_loader, validation_loader, criterion, optimizer,
          device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=2
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                #hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                #hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                #if running_samples>(0.2*len(image_dataset[phase].dataset)):
                #    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
                    
            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                           epoch_loss,
                                                                           epoch_acc,
                                                                           best_loss))
        if loss_counter==1:
            break
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    print("successfully set data paths")
    print(f"Train data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")
    print(f"Validation data path: {validation_data_path}")
    
    # Define transformations for the training, validation, and test sets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the datasets
    #train_data = datasets.ImageFolder(os.path.join(data, 'train'), transform=train_transforms)
    #val_data = datasets.ImageFolder(os.path.join(data, 'val'), transform=val_transforms)
    #test_data = datasets.ImageFolder(os.path.join(data, 'test'), transform=test_transforms)
    
    # Create data loaders
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transforms)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=val_transforms)
    val_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    print(f"Successfully created data loaders")
    
    return train_loader, test_loader, val_loader

def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #os.environ["SM_CHANNEL_TRAIN"]
    #os.environ["SM_CHANNEL_VALID"]
    #os.environ["SM_CHANNEL_TEST"]
    print(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
                  
    print(f'Data Paths: {args.data}')             
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Instantiated model and device")
    print(f"Running on Device {device}")
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    
    '''
    TODO: Create debug hook
    '''
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader, validation_loader = create_data_loaders(args.data,
                                                                       args.batch_size)
    model = model.to(device)
    
                  
    print("Model training beginning")             
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    print("Model testing beggining")
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--batch-size',
                        type=int,
                        default=32)
    parser.add_argument('--early-stopping-rounds',
                        type=int,
                        default=10)
    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
                  
    args=parser.parse_args()
    
    main(args)