# Check out the code in work at https://www.kaggle.com/hsankesara/prototypical-net/
# Check out the blog at <COMING SOON>


import numpy as np  # linear algebra
# from ProtoNetDrift import PrototypicalNet, train_step, test_step, load_weights
import torch.optim as optim
from preprocessing import LoadDriftData

import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader




# 创建子类
class subDataset(Dataset.Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.IntTensor(self.Label[index])
        return data, label


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


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive




def main():

    Data_Vector_Length = 200
    DATA_FILE = 'drift-200-25'
    ModelSelect = 'FCN' # 'RNN', 'FCN', 'Seq2Seq'

    # Reading the data
    print('Reading the data')
    all_data_frame = LoadDriftData(Data_Vector_Length, DATA_FILE)
    Drift_data_array = all_data_frame.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))

    np.random.shuffle(Drift_data_array)
    Drift_data_tensor = torch.tensor(Drift_data_array)

    train_size = int(0.8 * len(Drift_data_tensor))
    test_size = int(len(Drift_data_tensor) - train_size)
    train_drift_data = Drift_data_array[0:train_size, :]
    test_drift_data = Drift_data_array[train_size:int(len(Drift_data_tensor)), :]
    Drift_train_x, Drift_train_y = torch.tensor(train_drift_data).split(Data_Vector_Length, 1)
    Drift_test_x, Drift_test_y = torch.tensor(test_drift_data).split(Data_Vector_Length, 1)

    Drift_train_x, Drift_train_y = Drift_train_x.numpy(), Drift_train_y.numpy()
    Drift_test_x, Drift_test_y = Drift_test_x.numpy(), Drift_test_y.numpy()


    train_dataset = subDataset(Drift_train_x, Drift_train_y)
    test_dataset = subDataset(Drift_test_x, Drift_test_y)


    # 创建DataLoader迭代器
    # 创建DataLoader，batch_size设置为2，shuffle=False不打乱数据顺序，num_workers= 4使用4个子进程：
    train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=40, shuffle=False, num_workers=4)
    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=4)
    # 使用enumerate访问可遍历的数组对象：
    # for i, item in enumerate(train_dataloader):
    #     print('i:', i)
    #     data, label = item
    #     print('data:', data)
    #     print('label:', label)

    # Check whether you have GPU is loaded or not
    if torch.cuda.is_available():
        print('Yes')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Declare Siamese Network
    net = SiameseNetwork().to(device)
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

    def train():
        counter = []
        loss_history = []
        iteration_number = 0

        for epoch in range(0, 50):
            for i, item in enumerate(train_dataloader, 0):
                data, label = item
                # data0, data1 = data.split(100, 100)
                data0 = data[:, 0:100]
                data1 = data[:,100:200]
                data0, data1, label = data0.cuda(), data1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = net(data0, data1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()
                if i % 50 == 0:
                    print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
        return net


    # Train the model
    model = train()
    torch.save(model.state_dict(), "/home/tianyliu/Data/Siamese-Networks/content/model.pt")
    print("Model Saved Successfully")

    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("/home/tianyliu/Data/Siamese-Networks/content/model.pt"))


    # Load the test dataset
    # Print the sample outputs to view its dissimilarity
    # counter = 0
    # list_0 = torch.FloatTensor([[0]])
    # list_1 = torch.FloatTensor([[1]])
    # for i, item in enumerate(test_dataloader, 0):
    #     data, label = item
    #     data0 = data[:, 0:100]
    #     data1 = data[:, 100:200]
    #     concatenated = torch.cat((data0, data1), 0)
    #     output1, output2 = model(data0.to(device), data1.to(device))
    #     eucledian_distance = F.pairwise_distance(output1, output2)
    #     # if label == list_0:
    #     #     label = "Orginial"
    #     # else:
    #     #     label = "Forged"
    #     counter = counter + 1
    #     if counter == 20:
    #         break

    # test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
    accuracy = 0
    counter = 0
    correct = 0
    for i, item in enumerate(test_dataloader, 0):
        data, label = item
        data0 = data[:, 0:100]
        data1 = data[:, 100:200]
        # onehsot applies in the output of 128 dense vectors which is then converted to 2 dense vectors
        output1, output2 = model(data0.to(device), data1.to(device))
        res = torch.abs(output1.cuda() - output2.cuda())
        label = label[0].tolist()
        label = int(label[0])
        result = torch.max(res, 1)[1][0][0][0].data[0].tolist()
        if label == result:
            correct = correct + 1
        # counter = counter + 1
    #   if counter ==20:
    #      break

    accuracy = (correct / len(test_dataloader)) * 100
    print("Accuracy:{}%".format(accuracy))








    # Training loop
    frame_loss = 0
    frame_acc = 0
    for i in range(num_episode):
        # print(i)
        loss, acc, model_embeding = train_step(protonet, Drift_train_x, Drift_train_y, 5, 4, 5, optimizer)
        # print(acc)
        frame_loss += loss.data
        frame_acc += acc.data
        if((i+1) % frame_size == 0):
            print("Frame Number:", ((i+1) // frame_size), 'Frame Loss: ', frame_loss.data.cpu().numpy().tolist() /
                  frame_size, 'Frame Accuracy:', (frame_acc.data.cpu().numpy().tolist() * 100) / frame_size)
            frame_loss = 0
            frame_acc = 0
    PATH = '/home/tianyliu/Data/ConceptDrift/input/Model/'+ModelSelect+'/'+DATA_FILE+'/model_embeding.pkl'
    torch.save(model_embeding.state_dict(), PATH)

    # Test loop
    num_test_episode = 5000
    avg_loss = 0
    avg_acc = 0
    for _ in range(num_test_episode):
        loss, acc = test_step(protonet, Drift_test_x, Drift_test_y, 5, 4, 15)
        avg_loss += loss.data
        avg_acc += acc.data
        # print(acc)
    print('Avg Loss: ', avg_loss.data.cpu().numpy().tolist() / num_test_episode,
          'Avg Accuracy:', (avg_acc.data.cpu().numpy().tolist() * 100) / num_test_episode)

    # Using Pretrained Model
    # protonet = load_weights('./protonet.pt', protonet, use_gpu)


if __name__ == "__main__":
    main()
