
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.logging.wandb import WandbLogger

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataGenerator

import sklearn.metrics as sm
import scipy.stats as ss
import numpy as np


def out_shape(model, input_dim):
    return model(torch.rand(1, *(input_dim))).data.size()


def Encoder_Block(in_channels, out_channels):
    return  nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU()
    )


class MRI_LSTM(LightningModule):
    

    def __init__(self, args):
        super(MRI_LSTM, self).__init__()

        wandb.init(project="sipam-lstm-final")        
        
        self.data_col = args.path_col
        self.input_dim = {
            "9dof_1mm_vol": (1, 218, 182),
            "9dof_2mm_vol": (1, 109, 91),
            "9dof_4mm_vol": (1, 55, 46),
            "9dof_8mm_vol": (1, 27, 23)
        }[self.data_col]
        
        self.feat_embed_dim = args.feat_embed_dim
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.init_lr

        self.train_data_path = args.train_data
        self.valid_data_path = args.valid_data
        self.test_data_path  = args.test_data
        
        # Build Encoder
        encoder_blocks = []
        encoder_blocks.append(Encoder_Block(1, 32))
        encoder_blocks.append(Encoder_Block(32, 64))
        encoder_blocks.append(Encoder_Block(64, 128))
        encoder_blocks.append(Encoder_Block(128, 256))
        encoder_blocks.append(Encoder_Block(256, 256))
        self.encoder = nn.Sequential(*encoder_blocks)
        
        # Post processing
        self.post_proc = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d([3,2]),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, self.feat_embed_dim, 1)
        )
        
        # Connect w/ LSTM
        self.n_layers = 1
        self.lstm = nn.LSTM(self.feat_embed_dim, self.latent_dim, self.n_layers, batch_first=True)
        
        # Build regressor
        self.lstm_post = nn.Linear(self.latent_dim, 64)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.init_weights()
    

    def init_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and "regressor" in k:
                m.bias.data.fill_(62.68)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def init_hidden(self, batch):
        weight = next(self.parameters()).data 
        h_0 = weight.new(self.n_layers, batch.size(0), self.latent_dim).zero_()
        c_0 = weight.new(self.n_layers, batch.size(0), self.latent_dim).zero_()
        h_0.requires_grad=True
        c_0.requires_grad=True
        return h_0, c_0


    def encode(self, x, h_t, c_t):
        B, C, H, W, D = x.size()
        for i in range(H):
            out = self.encoder(x[:, :, i, :, :])
            out = self.post_proc(out)
            out = out.view(B, 1, self.feat_embed_dim)
            h_t = h_t.view(1, B, self.latent_dim)
            c_t = c_t.view(1, B, self.latent_dim)
            h_t, (_, c_t) = self.lstm(out, (h_t, c_t))
        encoding = h_t.view(B, self.latent_dim)
        return encoding


    def forward(self, batch, h_0, c_0):
        x, y_true = batch
        x, y_true = x.cuda(), y_true.cuda()  # ???
        embedding = self.encode(x, h_0, c_0)
        post = self.lstm_post(embedding)
        y_pred = self.regressor(post)
        return y_pred, embedding


    def calc_loss(self, y_true, y_pred):
        return F.mse_loss(y_pred, y_true)
    
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        h_0, c_0 = self.init_hidden(x)
        y_pred, embedding = self(batch, h_0, c_0)
        loss = self.calc_loss(y_true, y_pred)

        logs = {"train_loss": loss, "epoch": self.current_epoch}
        wandb.log(logs)

        return {"loss": loss, "embed": embedding, "y_pred": y_pred, "y_true": y_true}
    

    def training_epoch_end(self, outputs):
        y_true = []
        y_pred = []
        loss = 0

        for out in outputs:
            y_true.append(out["y_true"].cpu().detach().numpy())
            y_pred.append(out["y_pred"].cpu().detach().numpy())
            loss += out["loss"]

        y_pred = np.concatenate(y_pred).ravel()
        y_true = np.concatenate(y_true).ravel()

        logs = {"lr": self.optimizer.param_groups[0]["lr"]}
        logs["rmse"]  = sm.mean_squared_error(y_true, y_pred)**0.5
        logs["mae"]   = sm.mean_absolute_error(y_true, y_pred)
        logs["corr"]  = ss.pearsonr(y_true, y_pred)[0]
        wandb.log(logs)
        
        return {"train_loss":loss}
    
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        h_0, c_0 = self.init_hidden(x)
        y_pred, embedding = self(batch, h_0, c_0)
        loss = self.calc_loss(y_true, y_pred)
        
        logs = {"valid_loss": loss}
        wandb.log(logs)

        return {"val_loss": loss, "embed": embedding, "y_pred": y_pred, "y_true": y_true}


    def validation_epoch_end(self, outputs):
        print("\nEpoch: {}/{} - LR: {:.6f}".format(
            self.current_epoch, self.epochs, self.optimizer.param_groups[0]["lr"]))
        
        y_true = []
        y_pred = []
        loss = []

        for out in outputs:
            y_true.append(out["y_true"].cpu().detach().numpy())
            y_pred.append(out["y_pred"].cpu().detach().numpy())
            loss.append(out["val_loss"].float().item())

        y_pred = np.concatenate(y_pred).ravel()
        y_true = np.concatenate(y_true).ravel()

        logs = {"avg_val_loss": np.mean(loss), "lr": self.optimizer.param_groups[0]["lr"]}
        logs["valid_rmse"]  = sm.mean_squared_error(y_true, y_pred)**0.5
        logs["valid_mae"]   = sm.mean_absolute_error(y_true, y_pred)
        logs["valid_corr"]  = ss.pearsonr(y_true, y_pred)[0]
        
        wandb.log(logs)

        return {"val_loss": np.mean(loss)}
    

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        h_0, c_0 = self.init_hidden(x)
        y_pred, embedding = self(batch, h_0, c_0)

        return {"y_pred": y_pred, "y_true": y_true, "embed": embedding}
    

    def test_epoch_end(self, outputs):
        y_true = []
        y_pred = []
        
        for out in outputs:
            y_true.append(out["y_true"].cpu().detach().numpy())
            y_pred.append(out["y_pred"].cpu().detach().numpy())
        
        y_pred = np.concatenate(y_pred).ravel()
        y_true = np.concatenate(y_true).ravel()

        stats = {}
        stats["test_rmse"]  = [sm.mean_squared_error(y_true, y_pred)**0.5]
        stats["test_mae"]   = [sm.mean_absolute_error(y_true, y_pred)]
        stats["test_corr"]  = [ss.pearsonr(y_true, y_pred)[0]]
        stats["test_rcorr"] = [ss.pearsonr(y_true, y_pred - y_true)[0]]

        results_table = wandb.Table(columns=["RMSE", "MAE", "corr", "rcorr"])
        results_table.add_data(stats["test_rmse"], stats["test_mae"], stats["test_corr"], stats["test_rcorr"])
        wandb.log({"results": results_table})
    

    def configure_optimizers(self):
        trainable_param = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(trainable_param, lr=self.lr, eps=1e-8)

        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.8, patience=8, 
            min_lr=1e-8, verbose=True)
        
        scheduler = {
            "scheduler": lr_schedule,
            "interval": "epoch", # The unit of the scheduler's step size
            "frequency": 1, # The frequency of the scheduler
            "reduce_on_plateau": True,  # For ReduceLROnPlateau scheduler
            "monitor": "val_loss", # Metric to monitor
            "name": "ReduceLROnPlateau"
        }

        return [self.optimizer], [scheduler]
    

    def train_dataloader(self):
        train_set = DataGenerator(self.train_data_path, data_col=self.data_col)
        
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
        return loader


    def val_dataloader(self):
        valid_set = DataGenerator(self.valid_data_path, data_col=self.data_col)
        
        loader = torch.utils.data.DataLoader(
            valid_set, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return loader
    
    
    def test_dataloader(self):
        test_set = DataGenerator(self.test_data_path, data_col=self.data_col)
    
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return loader
