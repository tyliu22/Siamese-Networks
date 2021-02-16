import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        # Forward pass
        # output = self.cnn1(x)
        # output = output.view(output.size()[0], -1)
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


def get_Input_Data(BASE_PATH):
    Test_Example_Label = 1.0
    DATA_PATH = BASE_PATH+'/Test_Example_Data.pt'
    Test_Example_Data = torch.load(DATA_PATH)

    return Test_Example_Data

def main(BASE_PATH, threshold):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(BASE_PATH))

    # Tensor : [1, 100]
    data0 = torch.zeros(1, 100)
    data1 = torch.ones(1, 100)

    output1, output2 = model(data0, data1)
    eucledian_distance = F.pairwise_distance(output1, output2)

    label = 0
    if eucledian_distance > threshold:
        label = 1

    # DataType: float
    Label = float(label)

    print(Label)

if __name__ == "__main__":
    # File address
    BASE_PATH = '/home/tianyliu/Data/Siamese-Networks/content/model.pt'
    # can be adjusted according to your task
    threshold = 1
    main(BASE_PATH, threshold)