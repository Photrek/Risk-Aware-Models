import cv2 as cv
import lightning as pl
import numpy as np
import os
from pathlib import Path
from pdb import set_trace
import shutil
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torchmetrics import Accuracy


#ref: 
#  github pl-mnist
#  collab mnist-hello-world.ipynb 
#  VAE model: closely follows SashaMalysheva's Pytorch-VAE (GitHub project)

torch.set_float32_matmul_precision('medium')  # bfloat16

class cfg:
    BASE_DIR=Path('.')
    TB_LOG_DIR = 'tb_log'
    RUN_NAME = 'tempRunName'  # subdir name under TB_LOG_DIR
    RUN_VER_NAME = 'testTrainMnist'   # subdir name under RUN_NAME
    SAVED_MODEL_DIR = BASE_DIR / 'saved_models' / RUN_NAME
    SAVE_TOP_K = 1
    NUM_WORKERS = os.cpu_count()
    class vae:
        LEARNING_RATE=1e-3
        MAX_EPOCHS = 1<<6   # 7=128
        BATCH_SIZE = 128  # CONFIRMED: >128 doesn't fit in GPU-RAM memory
        BOOL_TRAIN=True
        DELETE_LAST_CHECKPOINTS = True   # Only actiates if BOOL_TRAIN = True
    class vae_rnn:
        SEQ_LEN = 8   # number of images used in a sequence
        LEARNING_RATE=1e-3
        MAX_EPOCHS = 20
        BATCH_SIZE = 16  # maxes memory when BS=16, SL=8
        DELETE_LAST_CHECKPOINTS = True   


class RandRotTransform:
    def __init__(self, maxAngle=90):
        self.maxAngle = maxAngle
    def __call__(self, image):
        input_img = np.asarray(image)
        theta = np.random.uniform(0, self.maxAngle)
        center_px_loc = [int(s/2) for s in input_img.shape]
        #rm = cv.getRotationMatrix2D( image.size, theta, 1)
        #rm = cv.getRotationMatrix2D( input_img.shape, theta, 1)
        rm = cv.getRotationMatrix2D( center_px_loc, theta, 1)
        #rot_img = cv.warpAffine(image, rm, (image.size[1], image.size[0]))
        rot_img = cv.warpAffine(
            input_img, rm, (input_img.shape[1], input_img.shape[0])
        ).astype(np.float32)
        #rot_img = np.clip(rot_img, 0.0, 255.0).astype(np.int8)
        rot_img = np.clip(rot_img, 0.0, 255.0).astype(np.uint8)
        #return np.clip(rot_img, 0.0, 255.0).astype(np.int8)  # remove rotation-based interpolation artefacts
        #print(f"RandRotTransform: rot_img min/max = {rot_img.min():.1f}, {rot_img.max():.1f}")
        #return np.clip(rot_img, 0.0, 255.0)/255.0  # remove rotation-based interpolation artefacts
        return rot_img

class LitMNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "data"):
        super().__init__()
        self.data_dir = data_dir
        randRotXform = RandRotTransform(9)
        self.transform = transforms.Compose([
            randRotXform,
            transforms.ToTensor(),  # NOTE: this also rescales [0, 255] to [0, 1]
            #transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(ds, [int(.8*len(ds)),int(.2*len(ds))])

    def prepare_data(self):
        print("*"*6 + " PREPARE_DATA " +"*"*6)
        MNIST(self.data_dir, train=True,download=True)
        MNIST(self.data_dir, train=False,download=True)
        
    def setup(self, stage:str = "None"):
        print("*"*6 + " SETUP " +"*"*6)
        #self.mnist_test = MNIST(self.data_dir, train=False)
        # Following are instances of torch Dataset

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=cfg.vae.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=cfg.vae.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS)

######################################################################
######################################################################

class RnnDataset(Dataset):
    def __init__(self, ds, seq_len=8, max_rot_rate=10):
        super().__init__()
        self.ds = ds   # this will be either train or val dataset from MNIST
        self.seq_len = seq_len
        self.max_rot_rate = max_rot_rate

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Note: following is where category is removed from original dataset.
        img, categ = self.ds[idx]   # PIL.Image.Image, int
        #print(f"************ len={len(img)}, type={type(img)}")
        #print(f"************ type={type(img)}")
        # FINDING: with no transform applied, is of type PIL.Image.Image
        #print(f"img type: {type(img)}, categ type: {type(categ)}")
        rot_rate = np.random.uniform(0, self.max_rot_rate)
        #img_seq = []
        unrot_img = np.array(img)
        #print(f"***************** min={unrot_img.min()},   max={unrot_img.max()}")
        #unrot_img = (unrot_img/255.0 - .1307)/.3081  # to mirror ToTensor & Normalize
        unrot_img = unrot_img/255.0  # to mirror ToTensor
        img_seq = torch.zeros(self.seq_len, unrot_img.shape[0], unrot_img.shape[1])
        center_px_loc = [int(s/2) for s in unrot_img.shape]
        for i_seq in range(self.seq_len):
            rot_angle = i_seq*rot_rate
            #rm = cv.getRotationMatrix2D(int(unrot_img.shape/2), rot_angle, 1) # ctr, angle, scale
            rm = cv.getRotationMatrix2D(center_px_loc, rot_angle, 1) # ctr, angle, scale
            #img_seq.append(cv.warpAffine(unrot_img, rm, (unrot_img.shape[1], unrot_img.shape[0])).astype(np.float32))
            img_seq[i_seq] = torch.tensor(cv.warpAffine(
                unrot_img, rm, (unrot_img.shape[1], unrot_img.shape[0])
            ).astype(np.float32))

        #return img_seq, self.ds[idx][1]
        return img_seq

class LitRnnDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        #ds = MNIST(self.data_dir, train=True, transform=self.transform)
        ds = MNIST(self.data_dir, train=True)
        # Following are instances of torch Dataset
        self.mnist_train, self.mnist_val = random_split(ds, [int(.8*len(ds)),int(.2*len(ds))])
        self.rnn_mnist_train = RnnDataset(self.mnist_train, seq_len=cfg.vae_rnn.SEQ_LEN)
        self.rnn_mnist_val = RnnDataset(self.mnist_val, seq_len=cfg.vae_rnn.SEQ_LEN)
        print("LitRnnDataModule: ")
        print(f"  # rnn_mnist_train sequences: {len(self.rnn_mnist_train)}")
        print(f"  # rnn_mnist_val sequences: {len(self.rnn_mnist_val)}")
        
    def train_dataloader(self):
        return DataLoader(
            self.rnn_mnist_train,
            batch_size = cfg.vae_rnn.BATCH_SIZE,
            num_workers = cfg.NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(
            self.rnn_mnist_val,
            batch_size = cfg.vae_rnn.BATCH_SIZE,
            num_workers = cfg.NUM_WORKERS)

######################################################################
######################################################################

class LitVaeModel(pl.LightningModule):
    
    #  Closely follows SashaMalysheva's Pytorch-VAE (GitHub project)

    def __init__(self,learning_rate:float=2e-4):
        super().__init__()
        self.num_classes = 10
        self.z_size = 200
        self.kernel_num = 64  # number of feature layers at most latent conv layer

        self.learning_rate = learning_rate 
        self.encoder = nn.Sequential(
            self._conv(1, self.kernel_num//2),
            self._conv(self.kernel_num//2, self.kernel_num),
        )
        # Following is the size (N x N) of the 2D features obtained after encoding.
        self.feature_size = 7   # HARDCODE; TODO: make dynamically-determined 
        self.feature_volume = self.kernel_num * self.feature_size * self.feature_size
        self.q_mean = self._linear(self.feature_volume, self.z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, self.z_size, relu=False)
        self.project = self._linear(self.z_size, self.feature_volume, relu=False)
        self.decoder = nn.Sequential(
            self._deconv(self.kernel_num, self.kernel_num//2),
            nn.ReLU(),
            self._deconv(self.kernel_num//2, 1),
            #nn.ReLU(),
            #nn.ReLU(),
            #nn.Tanh()    # causes some kind of CUDA error
            #self.MySigmoid()
            nn.Sigmoid()
        )

    class MySigmoid(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return 3.2457*torch.sigmoid(x)  - 0.4242

    def forward(self,x):
        #FINDING: data has already been through transform before provided here.
        #NOTE: the ToTensor transform maps [0, 256] to [0, 1]; after Normalize(), min/max: -0.424/2.82
        #return F.relu(self.L1(x.view(x.size(0),-1)))
        #return F.log_softmax(self.L1(x.view(x.size(0),-1)),dim=1)
        #NOTE: finding is from specification in mnist-hello-world.ipynb (collab)
        #NOTE: x.size() = [<batch_size>, 1 (monochrome), 28, 28]

        encoded = self.encoder(x) # size = [batch_size, kernel_size, 7, 7]
        # Both mean & logvar have size = [<batch_size>, <z_size>]
        mean, logvar = self.q(encoded)  
        # Following executes epsilon-sampling.
        z = self.z(mean,logvar)
        z_projected = self.project(z).view(
            -1, 
            self.kernel_num,
            self.feature_size,
            self.feature_size,
        )  # size = 128, 32, 7, 7
        x_reconstructed = self.decoder(z_projected)
        #print(x_reconstructed.min(), x_reconstructed.max())
        return (mean,logvar), x_reconstructed

    def loss_func(self, x):

        #    y_pred: torch.Tensor,    # .size() = [<batch_size>, 10]
        #    y_true: torch.Tensor,    # .size() = [<batch_size>]
        #) -> torch.Tensor:
        # Functional role for this function: training_step & 
        # validation_step both point to the same common computation.
        #
        # Recall: for log_softmax in self.forward, y_pred is log-probs, and y_true is categorical index
        #loss = F.nll_loss(y_pred, y_true)
        #
        (mean, logvar), x_reconstructed = self(x)
        # Following is a scalar
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss

        #loss = F.cross_entropy(y_pred, y_true)
        #return loss
        return total_loss, reconstruction_loss, kl_divergence_loss

    def training_step(self, batch, batch_idx):
        #print("*"*6 + " TRANING_STEP " +"*"*6)
        x, y = batch  # x shape: [BATCH_SIZE, 1, 28, 28], y.shape: [128]
        #loss = F.cross_entropy(self(x),y)
        #loss = F.nll_loss(self(x),y)
        #loss = self.loss_func(self(x), y)  # self(x).shape: [128, 10]
        #y_pred = torch.argmax(self(x), dim=1)
        #y_pred = self(x)  # log-probabilities
        #loss = self.loss_func(self(x), y)  # self(x).shape: [128, 10]
        #
        loss,reconstruction_loss,kl_divergence_loss = self.loss_func(x)  # 

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('recon_loss', reconstruction_loss, 
            on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('kldiv_loss', kl_divergence_loss, 
            on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #print("*"*6 + " VALIDATION_STEP " +"*"*6)
        x, y = batch
        #loss = F.cross_entropy(self(x),y)
        #loss = F.nll_loss(self(x),y)

        # Recall: dim=1 means argmax-aggregation performed for all cols in a row.
        #y_pred = torch.argmax(self(x), dim=1) 
        #loss = self.loss_func(y_pred, y)
        #y_pred_logprob = self(x)
        #loss = self.loss_func(y_pred_logprob, y)

        #loss = self.loss_func(x)
        loss,reconstruction_loss,kl_divergence_loss = self.loss_func(x)  # all 3 are scalars
        #self.val_accuracy.update(torch.argmax(y_pred_logprob,dim=1), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('recon_loss', reconstruction_loss, 
        #    on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('kldiv_loss', kl_divergence_loss, 
        #    on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #print("*"*6 + " END: VALIDATION_STEP " +"*"*6)
        return {'loss': loss}

    def configure_optimizers(self):
        print("*"*6 + " CONFIGURE_OPTIMIZERS " +"*"*6)
        #return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        training_steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        one_cycle_sched = torch.optim.lr_scheduler.OneCycleLR(opt,
            max_lr = self.learning_rate,
            anneal_strategy='cos',
            epochs = cfg.vae.MAX_EPOCHS,
            steps_per_epoch = training_steps_per_epoch,
        )
        scheduler = {
            'scheduler': one_cycle_sched, 
            'name': 'One Cycle Scheduler', 
            'interval': 'step'}
        return {"optimizer":opt, "lr_scheduler":scheduler, 'monitor':'val_loss'}

    def on_train_epoch_end(self):
        #print("*"*6 + " ON_TRAIN_EPOCH_END " +"*"*6)
        print("")  # makes progress bar not disappear between epochs.

    #def on_validation_epoch_end(self):
    #    #print("*"*6 + " ON_VALIDATION_EPOCH_END " +"*"*6)
    #    print("")  # makes progress bar not disappear between epochs.
        
    def teardown(self,stage):
        print("*"*6 + " TEARDOWN " +"*"*6)
        #stage: TrainerFn.FITTING (on both tuning and fitting)
        #print(stage)

    ###################################################
    # Utils
    ###################################################
    def _conv(self, input_channel_size, output_channel_size):  
        # Taken from GitHub Pytorch-VAE (SashaMalysheva)
        return nn.Sequential(
            nn.Conv2d(
                input_channel_size, output_channel_size, 
                kernel_size=2, stride=2, padding=0,
            ),
            nn.BatchNorm2d(output_channel_size),
            nn.ReLU(),
        )

    def _deconv(self, input_channel_size, output_channel_size):  
        # Taken from GitHub Pytorch-VAE (SashaMalysheva)
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channel_size, output_channel_size, 
                kernel_size=2, stride=2, padding=0,
            ),
            nn.BatchNorm2d(output_channel_size),
            #nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size), 
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

    def q(self, encoded):
        # Result of following: size = [<batch_size>, <product_of_remaining_dimensions>].
        unrolled = encoded.view(-1, self.feature_volume)   
        return self.q_mean(unrolled), self.q_logvar(unrolled)
    
    def z(self, mean, logvar):
        "Samples a z from mean and logvar"
        std = logvar.mul(0.5).exp_()
        # TODO: determine how to avoid need for cuda in following.
        eps = torch.randn(std.size()).cuda()
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        #return nn.BCELoss(size_average=False)(x_reconstructed,x) / x.size(0)
        batch_size = x.size(0)
        return nn.BCELoss(reduction='sum')(x_reconstructed, x) / batch_size
        #return nn.BCELoss(reduction='sum')(x_reconstructed,(x+.4242)/3.2457) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

######################################################################
######################################################################

class LitVaeRnnModel(pl.LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
    
        self.num_layers = 1
        self.neurons_per_layer = 10
        self.neurons_readout = 5

        #self.images_at_once = 1 
        # Following takes mus, sigmas at one time instant, moves to next instant
        #self.lstm = nn.LSTM(self.images_at_once, self.neurons_per_layer)
        if 0:  # Following assumes LSTM temporally migrates the mean & logvar
            self.lstm = nn.LSTM(2*self.vae.z_size, 2*self.vae.z_size)
        elif 1:  # Following assumes LSTM temporally migrates the (flattened) z-vector
            LSTM_FTR_SIZE = 2048
            self.lstm = nn.LSTM(self.vae.feature_volume, LSTM_FTR_SIZE)
            self.lstm_readout = nn.Linear(LSTM_FTR_SIZE, self.vae.feature_volume)

    def forward(self, batch):
        img_seq = batch 
        seq_len = img_seq.size(1)
        #set_trace()
        imgseq_reconstructed = torch.zeros(
            img_seq.size(0), seq_len-1, img_seq.size(2), img_seq.size(3),
            device = img_seq.device)
        new_meanseq = torch.zeros(
            img_seq.size(0), seq_len-1, self.vae.z_size,
            device = img_seq.device)
        new_logvarseq = torch.zeros(
            img_seq.size(0), seq_len-1, self.vae.z_size,
            device = img_seq.device)
        # Note: following uses teacher-forcing method for RNN training.
        for i_seq in range(seq_len-1):
            # Note: in following, unsqueeze goes from [32, 28, 28] --> [32, 1, 28, 28]
            encoded = self.vae.encoder(img_seq[:,i_seq,:,:].unsqueeze(1))
            # Note: encoded.size = [bs, 32, 7, 7] 

            if 0:  # This assumes LSTM migrates the mean and logvar temporally.
                mean, logvar = self.vae.q(encoded)
                # Note: mean.size = [bs, vae.z_size]
                new_mean_and_logvar, _ = self.lstm(torch.cat((mean,logvar),dim=1))
                #new_meanseq.append(new_mean_and_logvar[:self.vae.z_size])
                #new_logvarseq.append(new_mean_and_logvar[self.vae.z_size:])
                new_meanseq[:,i_seq] = new_mean_and_logvar[:,:self.vae.z_size]
                new_logvarseq[:,i_seq] = new_mean_and_logvar[:,self.vae.z_size:]
                #new_z = self.vae.z(
                #    new_mean_and_logvar[:,:self.vae.z_size],
                #    new_mean_and_logvar[:,self.vae.z_size:])
            elif 1:  # This assumes LSTM temporally migrates the **pre-sampled** z-vector.
                unrolled = encoded.view(-1, self.vae.feature_volume)   
                # Note: unrolled.size = [bs, self.vae.feature_volume], typically [16, 1568]
                h_vec, unused_cell = self.lstm(unrolled)
                # Note: in following, readout output is in "unrolled" (or "flattened") format.
                #new_z = self.lstm_readout(h_vec).view(
                #    -1, 
                #    self.vae.kernel_num,
                #    self.vae.feature_size,
                #    self.vae.feature_size,
                #)  # resulting size = [vae_rnn.bs, 32, 7, 7]
                new_unrolled = self.lstm_readout(h_vec)
                new_meanseq[:,i_seq] = self.vae.q_mean(new_unrolled)
                new_logvarseq[:,i_seq] = self.vae.q_logvar(new_unrolled)
                #new_z = self.vae.z(new_meanseq[:,i_seq], new_logvarseq[:,i_seq])

            # Following conducts sampling.
            new_z = self.vae.z(new_meanseq[:,i_seq], new_logvarseq[:,i_seq])
            # Note: new_z.size = [bs, vae.z_size]
            new_z_projected = self.vae.project(new_z).view(
                -1, 
                self.vae.kernel_num,
                self.vae.feature_size,
                self.vae.feature_size,
            )  # size = [vae_rnn.bs, 32, 7, 7]
            #imgseq_reconstructed.append(self.vae.decoder(new_z_projected))
            recon_img = self.vae.decoder(new_z_projected)
            #imgseq_reconstructed[:,i_seq,:,:] = self.vae.decoder(new_z_projected)
            imgseq_reconstructed[:,i_seq,:,:] = recon_img[:,0]
        return (new_meanseq, new_logvarseq), imgseq_reconstructed
        
    def loss_func(self, imgseq):
        # Note: imgseq.size = [cfg.vae_rnn.BATCH_SIZE, seq_len, 28, 28]
        (meanseq, logvarseq), imgseq_reconstructed = self(imgseq)
        seq_reconstruction_loss = self.seq_reconstruction_loss(imgseq_reconstructed, imgseq)
        seq_kl_divergence_loss = self.seq_kl_divergence_loss(meanseq, logvarseq)
        total_loss = seq_reconstruction_loss + seq_kl_divergence_loss
        return total_loss, seq_reconstruction_loss, seq_kl_divergence_loss

    def training_step(self, batch, batch_idx):
        #imgseq, category_seq = batch
        imgseq = batch
        loss, reconstruction_loss, kl_divergence_loss = self.loss_func(imgseq)
        self.log('train_loss',loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('recon_loss',reconstruction_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('kldiv_loss',kl_divergence_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #imgseq, category_seq = batch
        imgseq = batch
        loss, reconstruction_loss, kl_divergence_loss = self.loss_func(imgseq)
        self.log('val_loss',loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('recon_loss', reconstruction_loss, on_step=False, 
        #    on_epoch=True, prog_bar=True, logger=True)
        #self.log('kldiv_loss',kl_divergence_loss, on_step=False, 
        #    on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        print("*"*6 + " CONFIGURE_OPTIMIZERS " +"*"*6)
        #return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        opt = torch.optim.Adam(self.parameters(), lr=cfg.vae_rnn.LEARNING_RATE)
        training_steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        one_cycle_sched = torch.optim.lr_scheduler.OneCycleLR(opt,
            max_lr = cfg.vae_rnn.LEARNING_RATE,
            anneal_strategy='cos',
            epochs = cfg.vae_rnn.MAX_EPOCHS,
            steps_per_epoch = training_steps_per_epoch,
        )
        scheduler = {
            'scheduler': one_cycle_sched, 
            'name': 'One Cycle Scheduler', 
            'interval': 'step'}
        return {"optimizer":opt, "lr_scheduler":scheduler, 'monitor':'val_loss'}

    def on_train_epoch_end(self):
        #print("*"*6 + " ON_TRAIN_EPOCH_END " +"*"*6)
        print("")  # makes progress bar not disappear between epochs.
        #p0 = [x for x in self.vae.parameters()][0]  # size = [16, 1, 2, 2]
        # FINDING: though I'm unsure of why 16 (batch size?), it locks the VAE values btwn calls
        #set_trace()

    #--------------------------------------------------
    # Utils
    #--------------------------------------------------
    def seq_reconstruction_loss(self, imgseq_reconstructed, imgseq_orig):
        # Note: imgseq_orig.size = [bs, seq_len-1, 28, 28]
        recon_loss = 0
        batch_size, seq_len = imgseq_orig.size(0), imgseq_orig.size(1)
        #print(f"orig: min/max = {imgseq_orig.min().item(), imgseq_orig.max().item()},     recon: {imgseq_reconstructed.min().item()}, {imgseq_reconstructed.max().item()}")
        #for i_seq in range(len(img_orig)-1):
        for i_seq in range(seq_len-1):
            recon_loss += nn.BCELoss(reduction='sum')(
                imgseq_reconstructed[:,i_seq,:,:], imgseq_orig[:,i_seq+1,:,:]
            )/ batch_size
            #if i_seq == 0:
                #print(f"loss increment #0: { nn.BCELoss(reduction='sum')(imgseq_reconstructed[:,i_seq,:,:], imgseq_orig[:,i_seq+1,:,:])/ batch_size}")
       
        return recon_loss/(seq_len-1) # average for all sequence lengths

    def seq_kl_divergence_loss(self, meanseq, logvarseq):
        kl_loss = 0
        seq_len = meanseq.size(1)  # This is imgseq length - 1
        for i_seq in range(seq_len):
            kl_loss +=  (
                (meanseq[:,i_seq,:]**2 + logvarseq[:,i_seq,:].exp() - 1 - logvarseq[:,i_seq,:])/2
            ).mean()
        return kl_loss/seq_len

######################################################################
######################################################################

if __name__ == "__main__":
    t_start = time.time()

    # Remove old TB logs
    #run_dir_name = cfg.RUN_NAME  # later: specialize to LR, etc.
    run_dir_name = cfg.RUN_NAME +f"_vae_maxEp{cfg.vae.MAX_EPOCHS}" # later: specialize to LR, etc.
    # Remove old tensorboard logs
    shutil.rmtree(str(cfg.BASE_DIR / cfg.TB_LOG_DIR / run_dir_name / cfg.RUN_VER_NAME),
        ignore_errors=True)

        
    if cfg.vae.BOOL_TRAIN:
        
        shutil.rmtree(str(cfg.BASE_DIR / cfg.TB_LOG_DIR / run_dir_name / cfg.RUN_VER_NAME),
            ignore_errors=True)

        if cfg.vae.DELETE_LAST_CHECKPOINTS:
            # Delete the old "*last"* saved models, if they are to be produced from this training.
            [f.unlink() for f in (cfg.BASE_DIR / "saved_models" / cfg.RUN_NAME).glob("vae_last*")]

        mnist = LitMNISTDataModule()

        vae_model = LitVaeModel(learning_rate=cfg.vae.LEARNING_RATE)

        tensorboard = pl.pytorch.loggers.TensorBoardLogger(cfg.TB_LOG_DIR,
            name=run_dir_name,
            version=cfg.RUN_VER_NAME,
            default_hp_metric=False)
        # Recall: launch tensorboard from Win-CMD py310 via 
        #   tensorboard --logdir %uhm%\test\nvidia_docker\ptl_mnist\tb_log\
                
        vae_callbacks=[]
        vae_callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        ))

        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath = cfg.SAVED_MODEL_DIR,
            filename = 'vae_{epoch:03d}-{val_loss:.10f}',
            save_top_k = cfg.SAVE_TOP_K,
            mode='min',
            save_last=True
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = "vae_last"
        vae_callbacks.append(checkpoint_callback)

        vae_trainer = pl.Trainer(
            accelerator="gpu", 
            max_epochs=cfg.vae.MAX_EPOCHS, 
            logger=tensorboard,
            callbacks=vae_callbacks,
        )
        #        auto_lr_find=True)

        #trainer.tune(model,datamodule=mnist) # obtains optimal LR
        vae_trainer.fit(vae_model, mnist)
        print(f"Total dur so far: {(time.time()-t_start)/60:.1f}m")
    else:
        print("Loading pre-trained VAE")
        vae_model = LitVaeModel.load_from_checkpoint(cfg.SAVED_MODEL_DIR / 'vae_last.ckpt')

        vae_model = LitVaeModel(learning_rate=cfg.vae.LEARNING_RATE)

    run_dir_name = cfg.RUN_NAME +f"_vaernn_maxEp{cfg.vae_rnn.MAX_EPOCHS}" # later: specialize to LR, etc.

    vaernn_mdl = LitVaeRnnModel(vae_model)

    rnn_mnist = LitRnnDataModule()

    if 0:
        x = rnn_mnist.train_dataloader()
        ix = iter(x)
        set_trace()

    tensorboard = pl.pytorch.loggers.TensorBoardLogger(cfg.TB_LOG_DIR,
        name=run_dir_name,
        version=cfg.RUN_VER_NAME,
        default_hp_metric=False)
        # Recall: launch tensorboard from Win-CMD py310 via 
        #   tensorboard --logdir %uhm%\test\nvidia_docker\ptl_mnist\tb_log\
                
    vaernn_callbacks=[]
    vaernn_callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step',
        log_momentum=True
    ))

    if cfg.vae_rnn.DELETE_LAST_CHECKPOINTS:
        # Delete the old "*last"* saved models, if they are to be produced from this training.
        [f.unlink() for f in (cfg.BASE_DIR / "saved_models" / cfg.RUN_NAME).glob("vaernn_last*")]

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath = cfg.SAVED_MODEL_DIR,
        filename = 'vaernn_{epoch:03d}-{val_loss:.10f}',
        save_top_k = cfg.SAVE_TOP_K,
        mode='min',
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "vaernn_last"
    vaernn_callbacks.append(checkpoint_callback)


    vaernn_trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=cfg.vae_rnn.MAX_EPOCHS, 
        logger=tensorboard,
        callbacks=vaernn_callbacks,
    )
    #        auto_lr_find=True)

        #trainer.tune(model,datamodule=mnist) # obtains optimal LR
    vaernn_trainer.fit(vaernn_mdl, rnn_mnist)


    print(f"Total dur: {(time.time()-t_start)/60:.1f}m")
