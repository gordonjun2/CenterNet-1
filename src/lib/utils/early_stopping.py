import numpy as np
import torch

from lib.models.model import create_model, load_model, save_model

import sys
sys.path.append("../") # Adds higher directory to python modules path. 
import os 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch, arch, task, lr_step):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            dir = self.save_checkpoint(val_loss, model, epoch, arch, task, lr_step)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            dir = self.save_checkpoint(val_loss, model, epoch, arch, task, lr_step)
            self.counter = 0

        return dir

    def save_checkpoint(self, val_loss, model, epoch, arch, task, lr_step, es = True):
        '''Saves model when validation loss decrease.'''
        if self.verbose and es:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint.pt')

        if not es:
            print(f'Best validation loss ({val_loss:.6f}).  Saving model ...')
        
        dirPath = "../exp/" + str(task) + "/" + str(arch) + "/"
        
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        
        fileList = os.listdir(dirPath)
        
        for fileName in fileList:           # To remove all other files in the cache directory that are saved during training (Save space)
            if fileName.endswith('.pth'):
                os.remove(dirPath + fileName)

        if epoch not in lr_step:
            best_model_dir = dirPath + 'model_' + str(epoch) + '_' + str(val_loss) + '.pth'
        else:
            best_model_dir = dirPath + 'model_lr_step_' + str(epoch) + '_' + str(val_loss) + '.pth'
        
        save_model(best_model_dir, epoch, model)

        if es:
            self.val_loss_min = val_loss

        return best_model_dir