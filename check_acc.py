import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='resnet18', type=str, help='model name')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get Saved Model
if args.model == 'resnet18':
    from models import ResNet18
    net = ResNet18()
elif args.model == 'resnet34':
    from models import ResNet34
    net = ResNet34()
elif args.model == 'resnet50':
    from models import ResNet50
    net = ResNet50()
elif args.model == 'resnet101':
    from models import ResNet101
    net = ResNet101()
elif args.model == 'resnet152':
    from models import ResNet152
    net = ResNet152()
else:
    print('Model not available')
    exit()

net = net.to(device)

# Load saved model state_dict
checkpoint = torch.load('./cifar10+{}.pth'.format(args.model), map_location=device)
net.load_state_dict(checkpoint['net'])

# Load in test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)


# Evaluate and save correct indices
net.eval()
correct = 0
total = 0
correct_indices = []
with torch.no_grad():
    for i, data in enumerate(test_dataset):
        images, labels = data
        outputs = net(images.unsqueeze(0))
        _, predicted = outputs.max(1)
        total += 1
        if torch.equal(predicted, torch.tensor([labels])):
            correct += 1
            correct_indices.append(i)
    
acc = 100.*correct/total
print('Accuracy of the network on the 10000 test images: %f %%' % acc)

# Write the indices to a file
with open('./cifar10+{}_correct_indices.txt'.format(args.model), 'w') as f:
    for idx in correct_indices:
        f.write(str(idx) + ',')
