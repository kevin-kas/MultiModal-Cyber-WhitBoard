import torch
import torch.nn as nn
from torchview import draw_graph
class CVAE(nn.Module):
    def __init__(self, num_classes, latent_dim=20):
        super(CVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.embed_c = nn.Embedding(num_classes, 128)
        self.fc1 = nn.Linear(256 * 2 * 2 + 128, 256)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc2 = nn.Linear(latent_dim + 128, 1024)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def encoder(self, x, c):
        c = c.long()
        c_embed = self.embed_c(c)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c_embed], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)

    def decoder(self, z, c):
        c = c.long()
        c_embed = self.embed_c(c)
        z = torch.cat([z, c_embed], dim=1)
        z = torch.relu(self.fc2(z))
        z = z.view(z.size(0), 256, 2, 2)
        z = torch.relu(self.deconv1(z))
        z = torch.relu(self.deconv2(z))
        z = torch.relu(self.deconv3(z))
        z = torch.sigmoid(self.deconv4(z))
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar


# 生成网络结构图（需指定输入示例）
model = CVAE(num_classes=10, latent_dim=20)
# 输入示例：x (batch_size=1, channels=1, height=32, width=32), c (batch_size=1)
x = torch.randn(1, 1, 32, 32)  # 假设输入图像尺寸为32x32
c = torch.tensor([0])  # 类别标签示例
graph = draw_graph(model, input_data=(x, c), save_graph=True, format="svg")