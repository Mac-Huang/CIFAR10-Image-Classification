import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from model import Net
import time
import pandas as pd
import matplotlib.pyplot as plt

# Data transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading dataset
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

# Model, loss, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Prepare to track loss
epoch_losses = []

if __name__ == '__main__':
    
    # Dataframe to record the loss
    loss_records = {'Epoch': [], 'Batch': [], 'Loss': []}
    
    for epoch in range(20):
        timestart = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_records['Epoch'].append(epoch + 1)
            loss_records['Batch'].append(i + 1)
            loss_records['Loss'].append(loss.item())

            # Print and reset running loss every 500 batches
            if i % 500 == 499:
                average_loss = running_loss / 500
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, average_loss))
                running_loss = 0.0

        print('Epoch %d completed in %3f seconds.' % (epoch + 1, time.time() - timestart))
    torch.save(net.state_dict(), './Deep_Learning/EasyNet/easynet.pth')
    print('Finished Training and saving the model.')

    # Convert the dictionary into DataFrame 
    loss_df = pd.DataFrame.from_dict(loss_records)
    # Save the DataFrame to Excel file
    loss_df.to_excel('./Deep_Learning/Results/EasyNet_Results/Training_Loss.xlsx', index=False)

