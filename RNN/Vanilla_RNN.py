import torch 
import torch.nn as nn 

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super().__init__()

        #Defining number of hidden layers and the nodes in each layer 
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim 

        #RNN layer 
        self.rnn = nn.RNN(
                input_dim, hidden_dim, layer_dim, batch_first=True, dropout= dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        # Initializing hidden state for first input with zeros 
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True)

        # Forward propagation by passing in the input and hidden state into the model 
        out, h0 = self.rnn(x, h0.detach())

        #Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size) 
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim) 
        out = self.fc(out) 
        return out 


model = RNNModel(10,1,2,2,0.5)
print(model)
