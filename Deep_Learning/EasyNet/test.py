import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Test on Training-Dataset
    # testset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    net = Net()
    net.load_state_dict(torch.load('./Deep_Learning/EasyNet/easynet.pth'))
    net.eval()

    results = {'Time': [], 'Accuracy': [], 'Recall': [], 'ROC_AUC': [], 'Loss': []}

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_outputs = [] 
    class_correct = [0.] * 10
    class_total = [0.] * 10
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()

    for data in testloader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
        all_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().numpy())
        
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

    # Index
    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_preds, average='macro')
    roc_auc = roc_auc_score(np.eye(10)[all_labels], np.array(all_outputs), multi_class='ovr')

    results['Time'].append(time.time() - start_time)
    results['Accuracy'].append(accuracy)
    results['Recall'].append(recall)
    results['ROC_AUC'].append(roc_auc)
    results['Loss'].append(np.mean(losses))

    # toExcel
    df = pd.DataFrame(results)
    df.to_excel('./Deep_Learning/Results/EasyNet_Results/test_results.xlsx', index=False)

    # ROC
    plt.figure(figsize=(10, 8))
    for i in range(10):
        fpr, tpr, _ = roc_curve(np.eye(10)[all_labels][:, i], np.array(all_outputs)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{testset.classes[i]} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./Deep_Learning/Results/EasyNet_Results/ROC.png')
    plt.show()

    # Accuracy for each class
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(testset.classes, class_accuracy, color='skyblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy for each class')
    ax.set_ylim([0, 100])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    plt.savefig('./Deep_Learning/Results/EasyNet_Results/Confusion_Matrix.png')
    # plt.savefig('./Deep_Learning/Results/EasyNet_Results/Accuracy_on_Training_Dataset.png')
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./Deep_Learning/Results/EasyNet_Results/EasyNet_ResultsConfusion_Matrix.png')
    # plt.savefig('./Deep_Learning/Results/EasyNet_Results/EasyNet_ResultsConfusion_Matrix_on_Training_Dataset.png')

    plt.show()
    
if __name__ == '__main__':
    main()
