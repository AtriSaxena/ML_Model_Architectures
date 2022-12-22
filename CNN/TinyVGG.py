import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TinyVGG(nn.Module):
    """Some Information about TinyVGG"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(TinyVGG, self).__init__()

        self.block1 = nn.Sequential(
                            nn.Conv2d(in_channels=input_shape, 
                                        out_channels= hidden_units, 
                                        kernel_size=3,
                                        stride=1, #Default 
                                        padding=1), #options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
                            nn.ReLU(),
                            nn.Conv2d(in_channels=hidden_units,
                                    out_channels=hidden_units,
                                    kernel_size=3,
                                    stride = 1,
                                    padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.block2 = nn.Sequential(
                            nn.Conv2d(in_channels=hidden_units,
                                        out_channels=hidden_units,
                                        kernel_size=3,
                                        padding=1),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=hidden_units, 
                                        out_channels=hidden_units, 
                                        kernel_size= 3, 
                                        stride=1,
                                        padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
                                nn.Flatten(),
                                # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
                                nn.Linear(in_features=hidden_units * 7 * 7,
                                            out_features=output_shape))

    def forward(self, x):
        x = self.block1(x) 

        x = self.block2(x) 

        x = self.classifier(x)

        return x 

class_count = 5
model = TinyVGG(input_shape=1,hidden_units=10, output_shape=class_count)
print(model) 


