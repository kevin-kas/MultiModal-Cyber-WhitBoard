import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import os
import itertools


class Model_re(nn.Module):
    def __init__(self, input_shape=(32, 32, 1), num_classes=22):
        super(Model_re, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(in_features=128 * (input_shape[0] // 4) * (input_shape[1] // 4), out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])


class Data(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

        self.image_path = [i for i in self.image_path if i.endswith('.jpg')]

        self.class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                             '+': 10, '-': 11, 'times': 12, 'div': 13, '(': 14, ')': 15, '=': 16,
                             'log': 17, 'sqrt': 18, 'sin': 19, 'cos': 20,
                             'pi': 21}

    def __getitem__(self, idx):
        image_name = self.image_path[idx]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        img = transforms(Image.open(image_item_path).convert('L'))
        label = self.label_dir
        return img, torch.tensor(self.class_to_idx[label])

    def __len__(self):
        return len(self.image_path)


def load_data():
    train_root_list = ['test_data', 'train_data']
    label_root_list = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '+', '-', 'times', 'div', '(', ')', '=',
        'log', 'sqrt', 'sin', 'cos',
        'pi'
    ]

    train_list = [Data(train_root_list[1], i) for i in label_root_list]
    test_list = [Data(train_root_list[0], i) for i in label_root_list]

    all_train_data = None
    for i in train_list:
        if all_train_data is None:
            all_train_data = i
        else:
            all_train_data += i

    all_test_data = None
    for i in test_list:
        if all_test_data is None:
            all_test_data = i
        else:
            all_test_data += i

    return all_train_data, all_test_data


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs=25, verbose=True):
    model.to(device)
    criterion.to(device)
    best_accuracy = 0.0
    best_model = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []

        for batch_idx, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)

            output = model(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录每个批次的损失
            train_losses.append(loss.item())

            # 输出每个批次的损失
            if verbose:
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # 计算平均训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {epoch + 1}/{epochs}, Average Train Loss: {avg_train_loss:.4f}')

        # 评估阶段
        model.eval()
        total_accuracy = 0.0
        test_losses = []

        with torch.no_grad():
            for img, target in test_loader:
                img, target = img.to(device), target.to(device)

                output = model(img)
                loss = criterion(output, target)
                test_losses.append(loss.item())

                accuracy = (output.argmax(1) == target).sum().item()
                total_accuracy += accuracy

        # 计算平均测试损失和准确率
        avg_test_loss = sum(test_losses) / len(test_losses)
        accuracy = total_accuracy / len(test_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Average Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict().copy()
            print(f'Epoch {epoch + 1}: Best model saved with accuracy: {best_accuracy:.4f}')

    return best_accuracy, best_model


def grid_search(param_grid, train_data, test_data, device, epochs=25):
    # 生成所有参数组合
    param_combinations = list(itertools.product(*param_grid.values()))

    results = []

    for params in param_combinations:
        # 创建参数字典
        param_dict = dict(zip(param_grid.keys(), params))

        print(f"\nTraining with parameters: {param_dict}")

        # 创建模型
        model = Model_re()

        # 设置优化器
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=param_dict['learning_rate'],
            momentum=0.9
        )

        # 设置损失函数
        criterion = nn.CrossEntropyLoss()

        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=param_dict['batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=param_dict['batch_size'], shuffle=False)

        # 训练和评估模型
        accuracy, best_model = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs
        )

        # 记录结果
        result = {
            'params': param_dict,
            'accuracy': accuracy,
            'model_state': best_model
        }
        results.append(result)

        print(f"Final accuracy for parameters {param_dict}: {accuracy:.4f}")
        print("-" * 50)

    # 找到最佳参数组合
    best_result = max(results, key=lambda x: x['accuracy'])

    print(f"\nBest parameters: {best_result['params']}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")

    return best_result


if __name__ == "__main__":
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_data, test_data = load_data()

    # 定义要搜索的参数空间
    param_grid = {
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64]
    }

    # 执行Grid Search
    best_result = grid_search(param_grid, train_data, test_data, device, epochs=25)

    # 保存最佳模型
    if not os.path.exists('models'):
        os.makedirs('models')

    torch.save(best_result['model_state'], 'models/best_model.pth')
    print("Best model saved!")