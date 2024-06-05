import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model import Net
import pandas as pd
import time

def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss()

    # Dataframe to record the loss
    loss_records = {'Epoch': [], 'Batch': [], 'Loss': []}

    for epoch in range(20):  # loop over the dataset multiple times
        timestart = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_records['Epoch'].append(epoch + 1)
            loss_records['Batch'].append(i + 1)
            loss_records['Loss'].append(loss.item())

            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 500:.4f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} completed in {time.time() - timestart:.3f} seconds')

    print('Finished Training')
    torch.save(net.state_dict(), './Deep_Learning/ResMacNet/ResMacNet.pth')

    # Convert the dictionary into DataFrame 
    loss_df = pd.DataFrame.from_dict(loss_records)
    # Save the DataFrame to Excel file
    loss_df.to_excel('./Deep_Learning/Results/ResMacNet_Results/Training_Loss.xlsx', index=False)

if __name__ == '__main__':
    main()
