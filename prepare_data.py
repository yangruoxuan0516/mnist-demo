import json
import torch
from torch import nn
from torchvision import datasets, transforms

# 1. 加载 MNIST
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# 2. 选一张 0-9 的样本
examples = {}
for img, label in train_ds:
    label = int(label)
    if label not in examples:
        examples[label] = img.squeeze(0)  # 28x28
    if len(examples) == 10:
        break

# 3. 定义一个简单的 MLP: 784 -> 16 -> 10
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP()

# 4. 略微训练一下（不追求极致，只要能跑通）
loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(2):  # 小训练两轮就行，主要用于演示
    for x, y in loader:
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

# 5. 导出样本
sample_list = []
for digit in range(10):
    img = examples[digit]  # 28x28，取值[0,1]
    pixels = (img * 255).round().int().view(-1).tolist()  # 0-255 转成 int
    sample_list.append({
        "label": digit,
        "pixels": pixels,
        "width": 28,
        "height": 28
    })

with open("mnist_samples.json", "w", encoding="utf-8") as f:
    json.dump(sample_list, f, ensure_ascii=False)

# 6. 导出权重
state_dict = model.state_dict()
W1 = state_dict["fc1.weight"].numpy().tolist()  # 16x784
b1 = state_dict["fc1.bias"].numpy().tolist()    # 16
W2 = state_dict["fc2.weight"].numpy().tolist()  # 10x16
b2 = state_dict["fc2.bias"].numpy().tolist()    # 10

weights = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
}

with open("mlp_weights.json", "w", encoding="utf-8") as f:
    json.dump(weights, f, ensure_ascii=False)

print("Done: mnist_samples.json & mlp_weights.json generated.")
