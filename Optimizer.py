import torch, utils
from torch import optim

class Optimizer_Constr_Mod():

    def __init__(self, loss_function, initial_guess, dev):
        
        self.weights = torch.tensor(initial_guess, requires_grad=True, device=dev)
        self.loss_fn = loss_function         
        self.optimizer = optim.Adam([self.weights], lr=0.3)

   
    def optimize(self, epochs):

        # Optimization loop
        for _ in range(epochs):  # Number of epochs
            def closure():
                self.optimizer.zero_grad()
                    
                loss = self.loss_fn(self.weights) # Check the inequality
                loss.backward()    
                return loss
            self.optimizer.step(closure)

        if not utils.are_valid_angles(self.weights):
            self.weights = self.weights % (2*torch.pi)


class Optimizer_Constr_Clamp():

    def __init__(self, loss_function, initial_guess, dev):
        
        self.weights = torch.tensor(initial_guess, requires_grad=True, device=dev)
        self.loss_fn = loss_function         
        self.optimizer = optim.Adam([self.weights], lr=0.3)

    
    def optimize(self, epochs):

        # Optimization loop
        for _ in range(epochs):  # Number of epochs
            def closure():
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.weights)
                loss.backward()
                return loss
            self.optimizer.step(closure)
            with torch.no_grad():
                self.weights.clamp_(0, 2 * torch.pi)   # Ensures parameters constraints


            