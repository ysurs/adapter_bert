import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

# This module has been checked and is good to use
class adapter(nn.Module):
    '''
    Adapter module implementation
    '''
    
    def __init__(self,num_projection_nodes,num_bottleneck_nodes):
      super().__init__()
      self.num_projection_nodes = num_projection_nodes
      self.num_bottleneck_nodes = num_bottleneck_nodes
      self.linear_down_projection = nn.Linear(self.num_projection_nodes,self.num_bottleneck_nodes)
      self.linear_up_projection = nn.Linear(self.num_bottleneck_nodes,self.num_projection_nodes)
      self.activation=nn.ReLU()
      


    def forward(self,input):
      
      # Created copy of the input tensor, this will be used in skip connection.
      original_input=input.clone()
      output_from_non_linearity=self.activation(self.linear_down_projection(input))
      output_before_skip_connection=self.linear_up_projection(output_from_non_linearity)
      return output_before_skip_connection+original_input #This acts as a skip connection 


class adapter_bert(nn.Module):
    
    def __init__(self):
        pass


    def forward(self, input, output):
        pass