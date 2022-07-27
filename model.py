# The self. modules() method returns an iterable to the many layers or 
# “modules” defined in the model class. 

import torch
import torch.nn as nn
import torch.optim as optim # implementing various optimization algorithms
import torch.nn.functional as F # for different torch functions
import os # for saving the model

class Linear_QNet(nn.Module): #Inherits from nn.Module. It is a base class for all neural network module.
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # calling super initializer from the game
        self.linear1 = nn.Linear(input_size, hidden_size) # It is used to apply a linear transformation to the incoming data: y=xA^T+b
        self.linear2 = nn.Linear(hidden_size, output_size) #  creates single layer feed forward network with n inputs and m output.

    def forward(self, x):
        x = F.relu(self.linear1(x)) # activation function from functional module
        x = self.linear2(x) # applying second layer dont need activation here
        return x

    def save(self, file_name='model.pth'): # database file, saves every different details
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # saving the model using save dictionary
        # a Python dictionary object that maps each layer to its parameter tensor.

class QTrainer: 
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # adam optimizer
        # Adam optimizer requires minimum memory space or efficiently works with large problems
        # which contain large data. Used for stochastic gradient descent
         
        self.criterion = nn.MSELoss() # loss function - mean squared error

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1: # only one dimension so we need to change
            # need to keep it in the format (1, x)
            state = torch.unsqueeze(state, 0) 
            # torch.unsqueeze returns a new tensor with a dimension of size one inserted at the specified position
            
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # tuple with only one value

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone() # cloning to get free/unused values copies doing pytorch tensor

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            # preds[argmax(action)] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()

        self.optimizer.zero_grad() # to empty the gradient for pytorch
        # It is beneficial to zero out gradients when building a neural network. 
        # This is because by default, gradients are accumulated in buffers (i.e, not overwritten) whenever . backward() is called.
        loss = self.criterion(target, pred) # computes the cross entropy loss between input and target
        loss.backward()
        # compute gradient of loss w.r.t all the parameters in loss that
        # have requires_grad = True
        
        self.optimizer.step()
        # performs a parameter update based on the current gradient 



