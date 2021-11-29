import os
import time
from numpy.random.mtrand import sample
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )

        ### YOUR CODE HERE
        # define cross entropy loss and optimizer

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.solver = torch.optim.SGD(self.network.parameters(), 0.1, 0.9, 2e-4)
        self.variable_lr = torch.optim.lr_scheduler.StepLR(self.solver, 50, 0.1)
        
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs

            avg_loss = 0.0
            ### YOUR CODE HERE
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
          
                batch_X = [parse_record(curr_x_train[i*self.config.batch_size + j], True) for j in range(0,self.config.batch_size)]
                
                batch_y = curr_y_train[i*self.config.batch_size : i*self.config.batch_size + self.config.batch_size]
                
                batch_X = np.array(batch_X)
                batch_y = np.array(batch_y)
                ### YOUR CODE HERE
                
                batch_X = torch.from_numpy(batch_X).to(dtype=torch.float)/1.0
                batch_y = torch.from_numpy(batch_y)
                
                # Forward pass
                
                
                self.solver.zero_grad()
                y_hat = self.network(batch_X) 
                loss = self.cross_entropy_loss(y_hat, batch_y) 
                avg_loss += loss.item()
                loss.backward()
                self.solver.step()
                print('\rBatch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='', flush=True)
            
            self.variable_lr.step() 
            duration = time.time() - start_time
            print(' - Epoch {:d} Loss {:.6f} Average_Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, avg_loss/num_batches, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        folder_path = os.path.join('./','models','0', 'model_v2')
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(folder_path, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            self = self.float()
            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                sample_in = torch.from_numpy(parse_record(x[i], False).reshape(1,3,32,32)).float()
                sample_pred = self.network(sample_in)
                preds.append(torch.argmax(sample_pred))
                ### END CODE HERE
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))

    def predict(self, x, checkpoint_num_list):
        self.network.eval()
        folder_path = os.path.join('./','models','0', 'model_v2')
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(folder_path, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            self = self.float()
            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                sample_in = torch.from_numpy(parse_record(x[i].reshape(32,32,3).transpose(2,0,1).reshape(3072), False).reshape(1,3,32,32)).float()
                sample_pred = self.network(sample_in)
                probs = nn.Softmax(dim = 1)
                sample_pred = probs(sample_pred)
                sample_pred = sample_pred.detach().numpy()
                preds.append(sample_pred.reshape(-1))
                ### END CODE HERE

            return np.array(preds)
           
    
    def save(self, epoch):
        folder_path = os.path.join('./','models','0',self.config.modeldir)
        checkpoint_path = os.path.join(folder_path, 'model-%d.ckpt'%(epoch))
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path) 
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))