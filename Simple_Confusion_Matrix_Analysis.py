import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the confusion matrix
confmat = torch.load('./checkpoints/smallnet_cifar10_confmat.pt')
print("Confusion Matrix Shape:", confmat.shape)
print("\nConfusion Matrix:")
print(confmat)

# CIFAR-10 class names
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Simple matplotlib heatmap
plt.figure(figsize=(10, 8))
plt.imshow(confmat.numpy(), interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(10), classes, rotation=45)
plt.yticks(range(10), classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add text annotations
for i in range(10):
    for j in range(10):
        plt.text(j, i, int(confmat[i, j]), ha='center', va='center',
                color='white' if confmat[i, j] > confmat.max()/2 else 'black')

plt.tight_layout()
plt.show()

# Find most confused pairs
print("\nMost common misclassifications:")
for i in range(10):
    for j in range(10):
        if i != j and confmat[i][j] > 50:  # More than 50 misclassifications
            print(f"{classes[i]} confused with {classes[j]}: {confmat[i][j]} times")

# Calculate per-class precision and recall
print("\nDetailed per-class metrics:")
for i in range(10):
    true_positive = confmat[i, i]
    false_positive = confmat[:, i].sum() - true_positive
    false_negative = confmat[i, :].sum() - true_positive
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    print(f"{classes[i]}: Precision={precision:.3f}, Recall={recall:.3f}")