import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

def train_discrete(model, iterator, opt, args):
    model.train()

    loss_hist = []

    # Do one pass over the data accessed by the training iterator
    # Upload the data in each batch to the GPU (if applicable)
    # Zero the accumulated gradient in the optimizer 
    # Compute the cross_entropy loss with and without weights  
    # Compute the derivatives of the loss w.r.t. network parameters
    # Take a step in the approximate gradient direction using the optimizer opt  
    
    # calcuate the inverse class frequencies to use as weights for CrossEntropyLoss()
    # initialize the weights of all classes to 0.
    wts = torch.zeros([args.n_steering_classes])
    # iterate over all the batches.
    for batch in iterator:
        y = batch['cmd']
        # iterate over all the target commands in each batch.
        for label in y:
            # increase the counter for the class by 1 on occurence.
            wts[label] = wts[label] + 1.
    # get the inverse frequencies which correspond to the class weights.
    wts = 1./wts 

    for i_batch, batch in enumerate(iterator):

        #
        # YOUR CODE GOES HERE
        # load the images and their corresponding commands into x and y.
        x = batch['image']
        y = batch['cmd']

        # load x and y into devices. please change device to GPU in utils to run on GPU.
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # before each iteration zero the accumulated gradient in the optimizer from the previous iteration
        opt.zero_grad()

        # predict the output for the x's
        y_pred = model(x)

        # following line is for weighted CrossEntropyLoss, please uncomment it to get answer to Question 4 of assignment. 
        criterion = nn.CrossEntropyLoss(weight=wts)
        
        # following line for non-weighted CrossEntropyLoss. please comment it when using weighted CrossEntropyLoss.
        # criterion = nn.CrossEntropyLoss()
        
        # calculate the loss with the target y's and the predicted y's.
        loss = criterion(y_pred, y)

        # propagate the loss backward.
        loss.backward()

        # take a step in the approximate gradient direction.
        opt.step()
        #
        
        loss = loss.detach().cpu().numpy()
        # append the loss in this step to the list of losses.
        loss_hist.append(loss)
        
        PRINT_INTERVAL = int(len(iterator) / 3)        
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print ('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0),
                time.time() - args.start_time,
            ))


def accuracy(y_pred, y_true):
    "y_true is (batch_size) and y_pred is (batch_size, K)"
    _, y_max_pred = y_pred.max(1)
    correct = ((y_true == y_max_pred).float()).mean() 
    acc = correct * 100
    return acc


def test_discrete(model, iterator, opt, args):
    model.train()
    
    acc_hist = []
    
    for i_batch, batch in enumerate(iterator):
        x = batch['image']
        y = batch['cmd']

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        logits = model(x)
        y_pred = F.softmax(logits, 1)

        acc = accuracy(y_pred, y)
        acc = acc.detach().cpu().numpy()
        acc_hist.append(acc)
        
    avg_acc = np.asarray(acc_hist).mean()
    
    print ('\tVal: \tAcc: {}  Time: {:10.3f}'.format(
        avg_acc,
        time.time() - args.start_time,
    ))
    
    return avg_acc
    
def get_class_distribution(iterator, args):
    class_dist = np.zeros((args.n_steering_classes,), dtype=np.float32)
    for i_batch, batch in enumerate(iterator):
        y = batch['cmd'].detach().numpy().astype(np.int32)
        class_dist[y] += 1
        
    return (class_dist / sum(class_dist))

    
def main(args):
    
    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])
    
    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)
    
    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)
    
    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    
    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()
    
    print (driving_policy)
    print (opt)
    print (args)

    args.class_dist = get_class_distribution(training_iterator, args)
    best_val_accuracy = 0 
    for epoch in range(args.n_epochs):
        print ('EPOCH ', epoch)

        #
        # YOUR CODE GOES HERE
        
        # train model on the batches generator by the iterator using train_discrete function.
        train_discrete(driving_policy, training_iterator, opt, args)

        # Evaluate the driving policy on the validation set
        # If the accuracy on the validation set is a new high then save the network weights 
        avg_acc = test_discrete(driving_policy, validation_iterator, opt, args)
        if avg_acc > best_val_accuracy:
            best_val_accuracy = avg_acc
            torch.save(driving_policy.state_dict(), args.weights_out_file)
        
 
    return driving_policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights",
                        required=True)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)
    
    args = parser.parse_args()

    main(args)
        
    
