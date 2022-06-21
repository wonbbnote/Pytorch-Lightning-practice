import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

# 파이토치 라이트닝 상속
class LitAutoEncoder(pl.LightningModule):
    #신경망 모델 -> Linear, BatchNorm1d, ReLU 순서로 Layer를 쌓기
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    # 모델의 예측 결과를 제공
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    # 학습 루프의 body 부분 (loss 계산)
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    # Adam optimzer 사용
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# 데이터셋 불러오고 전처리 및 트레이닝/테스트 분리
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

# 학습돌리기
epochs = 2
autoencoder = LitAutoEncoder()
trainer = pl.Trainer(max_epochs=epochs)
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))