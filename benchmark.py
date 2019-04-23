import torch.nn as nn
import torch
import numpy as np

class BenchMark(nn.Module):
    
    
    
    def __init__(self, input_dim, output_dim, device):

        super(BenchMark, self).__init__()
        
        self.device = device
        
        self.classification = nn.Sequential(
            nn.Linear(2*input_dim, 512),
            nn.Linear(512, 512),
            nn.Linear(512, output_dim)
            )
        
    def forward(self, x1, len_x1, x2, len_x2):
        
        input1 = torch.sum(x1, dim=0)/len_x2.unsqueeze(1).repeat(1,300).type(torch.FloatTensor).to(self.device)
        input2 = torch.sum(x2, dim=0)/len_x2.unsqueeze(1).repeat(1,300).type(torch.FloatTensor).to(self.device)
        
        input_classifier = torch.cat((input1, input2), 1)
        
        out = self.classification(input_classifier)
        
        return out
