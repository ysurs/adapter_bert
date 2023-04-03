import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from config import *

# This module has been checked and is good to use
class adapter(nn.Module):
    '''
    Adapter module implementation
    '''
    
    def __init__(self,num_feedforward_nodes):
      super().__init__()
      self.linear_down_projection = nn.Linear(num_feedforward_nodes,bottleneck_size)
      self.linear_up_projection = nn.Linear(bottleneck_size,num_feedforward_nodes)
      self.activation=nn.ReLU()
      


    def forward(self,input):
      
      # Created copy of the input tensor, this will be used in skip connection.
      original_input=input.clone()
      output_from_non_linearity=self.activation(self.linear_down_projection(input))
      output_before_skip_connection=self.linear_up_projection(output_from_non_linearity)
      return output_before_skip_connection+original_input #This acts as a skip connection 


