# CIFAR10 Image Classification

This project demonstrates various machine learning methods for image classification based on the CIFAR10 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The CIFAR10 dataset consists of 60,000 32x32 color images in 10 different classes. This project explores several machine learning and deep learning techniques for classifying these images.

## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/CIFAR10-Image-Classification.git
cd CIFAR10-Image-Classification
```
Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Take ResMacNet as example:
Run the train.py first:
```bash
train.py
```
Evaluate the model then:
```bash
test.py
```

## Results
The following table shows the accuracy achieved by different models:
| Model       | Accuracy |
|-------------|----------|
| EasyNet     | 54.65%   |
| Res-MacNet  | 81.3%    |

![Results](Deep_Learning/Results/Compare/Accuracy%20Comparison%20between%20EasyNet%20and%20Res-MacNet.png)

![Results](Deep_Learning/Results/Compare/Training%20Loss%20for%20Mac&Easy.png)

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)
```
