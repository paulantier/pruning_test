import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Définition du modèle LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Charger les données MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Initialisation du modèle, de l'optimiseur et de la fonction de perte
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner et tester le modèle
num_epochs = 3
train_accuracy = []
test_accuracy = []

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

# Prunage du réseau
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.99)

# Geler les poids prunés en créant un masque de prune
for module, param_name in parameters_to_prune:
    prune.custom_from_mask(module, param_name, mask=module.weight_mask)

# Tester le réseau pruné
test_acc_pruned = evaluate(model, test_loader)
print(f'Test Accuracy after pruning: {test_acc_pruned:.2f}%')

# Fine-tuning du réseau pruné en bloquant les poids prunés
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
fine_tune_epochs = num_epochs
fine_tune_test_accuracy = []

for epoch in range(fine_tune_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_acc = evaluate(model, test_loader)
    fine_tune_test_accuracy.append(test_acc)
    print(f'Fine-tune Epoch [{epoch+1}/{fine_tune_epochs}], Test Accuracy: {test_acc:.2f}%')

# Afficher la courbe d'efficacité de test
plt.plot(range(1, num_epochs + 1), test_accuracy, label='Test Accuracy Before Pruning')
plt.plot(range(num_epochs + 1, num_epochs + fine_tune_epochs + 1), fine_tune_test_accuracy, label='Test Accuracy After Pruning')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
