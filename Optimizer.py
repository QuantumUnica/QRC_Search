import torch, utils
from torch import optim

class Optimizer():

    def __init__(self, loss_function, initial_guess, dev):
        
        self.weights = torch.tensor(initial_guess, requires_grad=True, device=dev)
        self.loss_fn = loss_function         
        self.optimizer = optim.Adam([self.weights], lr=0.3)

    
    def optimize(self, epochs):

        # Optimization loop
        for _ in range(epochs):  # Number of epochs
            def closure():
                self.optimizer.zero_grad()
                if not utils.are_valid_angles(self.weights):
                    #print("Not all elements in range [0, 2π]")
                    self.weights = self.weights % (2*torch.pi)
                    
                loss = self.loss_fn(self.weights) # Check the inequality
                loss.backward()    
                return loss
            self.optimizer.step(closure)