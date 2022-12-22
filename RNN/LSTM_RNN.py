import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNNLSTM_Model(nn.Module):
    """RNNLSTM_Model"""
    def __init__(self, input_size, hidden_dim, num_layers, output_dim ,dropout_prob):
        super(RNNLSTM_Model, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, 
                                num_layers=num_layers, batch_first=True, dropout=dropout_prob )
        
        # Fully Connected layers 
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initializing hidden state for first input with zeros 
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad = True)

        # Initializing cell state for first input with zeros
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad = True) 

        # We need to detach as we are doing truncated backpropagation through time (BPTT) 
        # If we don't, we'll backprop all the way to the start even after going through another batch 
        # Forward propagation by passing in the input, hidden state, and cell state into the model 

        output, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach())) # lstm with input, hidden and internal state 

        hn = hn.view(-1, self.hidden_dim) # Reshaping for hidden layer next 
        out = self.fc(out)

        return out 

lstm_model = RNNLSTM_Model(input_size=128, hidden_dim=32, num_layers=2, output_dim=1, dropout_prob=0.5) 
print(lstm_model)