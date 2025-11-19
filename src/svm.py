"""import torch

def decorrelation_pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return torch.sum(x * y) / (torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + 1e-8)

def svm_decorrelation_loss(outputs, labels, bookmaker_probs, reg_strength, decorrel_weight):
    # Standard hinge loss
    hinge = torch.mean(torch.clamp(1 - labels * outputs, min=0))
    # Optional L2
    reg = reg_strength * torch.sum(outputs ** 2)
    # Decorrelation (can use .detach() if you do not want gradients wrt bookmaker_probs)
    decor = torch.abs(decorrelation_pearson(outputs, bookmaker_probs))
    total_loss = hinge + reg + decorrel_weight * decor
    return total_loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

df = pd.read_csv("data/cleaned_games.csv")
#but thats bookmaker prob :((
target_H_prob = (1/df['Odds_H'].values)/((1/df['Odds_H'].values)+(1/df['Odds_D'].values)+(1/df['Odds_A'].values))
target_A_prob = (1/df['Odds_'].values)/((1/df['Odds_H'].values)+(1/df['Odds_D'].values)+(1/df['Odds_A'].values))
target_D_prob = (1/df['Odds_D'].values)/((1/df['Odds_H'].values)+(1/df['Odds_D'].values)+(1/df['Odds_A'].values))

iris = datasets.load_iris()
X = iris.data
y = iris.target



class LinearSVR(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(feature_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x @ self.w + self.b

def epsilon_insensitive_loss(outputs, targets, epsilon):
    diff = torch.abs(outputs - targets) - epsilon
    return torch.mean(torch.clamp(diff, min=0))

def decorrelation_pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return torch.sum(x * y) / (torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + 1e-8)

def svr_decorrelation_loss(outputs, targets, bookmaker_probs, reg_strength, decorrel_weight, epsilon):
    # Epsilon-insensitive loss for regression
    loss = epsilon_insensitive_loss(outputs, targets, epsilon)
    # Regularization on weights (L2 norm)
    reg = reg_strength * torch.sum(outputs ** 2)
    # Decorrelation penalty with bookmaker probabilities
    decor = torch.abs(decorrelation_pearson(outputs, bookmaker_probs))
    total_loss = loss + reg + decorrel_weight * decor
    return total_loss

# Example usage:
# X = feature matrix (torch tensor)
# y = target continuous values
# bookmaker_probs = bookmaker predicted probabilities (torch tensor)
# Initialize model, optimizer, etc.

# Assume model, optimizer, and loss function svr_decorrelation_loss are defined (from previous code)

epochs = 200
lr = 0.01

optimizer = torch.optim.SGD(LinearSVR.parameters(), lr=lr)

for epoch in range(epochs):
    optimizer.zero_grad()                  # Reset gradients from previous step
    outputs = LinearSVR(X)                  # Forward pass: predict outputs for input batch
    loss = svr_decorrelation_loss(outputs, y, bookmaker_probs, reg_strength=0.01, decorrel_weight=1.0, epsilon=0.1)
    loss.backward()                       # Compute gradients for all model parameters
    optimizer.step()                      # Update model parameters by taking a step in gradient direction
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
