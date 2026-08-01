import torch
import torch.nn as nn
import torchinfo
from torchinfo import summary


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.conv1 = self.get_conv(3, 32)  ##112*112
        self.conv2 = self.get_conv(32, 64) ##56*56
        self.conv3 = self.get_conv(64, 128) ## 28*28
        self.conv4 = self.get_conv(128, 256) ## 14*14
        self.conv5 = self.get_conv(256, 512) ## 7*7
        
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512*7*7, 512, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout),
                                        nn.Linear(512, 128, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout),
                                        nn.Linear(128, num_classes)
                                       )
                                
  
    
        
    def get_conv(self, n_filters_in, n_filters_out):
                  
        conv_block = nn.Sequential(
            nn.Conv2d(n_filters_in, n_filters_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters_out),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.005)
        )
           
        return conv_block
        
    def forward(self, x):
                                        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)

        return x


    def test_model(self):
        x = torch.ones(3, 3, 224, 224)
        print(x.shape)
        out = self(x) #MyModel()(x)
        print(f'Output from model is of dimension: {out.shape}')
        num_param = sum([p.numel() for p in self.parameters()])
        print(f'Number of Model Parameters : {num_param}')
        print(f'Displaying Model Parameters summary:\n {summary(MyModel(), input_size=(3,3,224,224))}')
        return out, out.shape, num_param

   # def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
    #    return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
