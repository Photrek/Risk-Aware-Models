import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pdb import set_trace
import time
from torchvision.datasets import MNIST
from torchvision import transforms

from train import cfg, LitVaeModel, LitRnnDataModule, LitMNISTDataModule

NUM_SEQUENCE_TO_PLOT = 1


vae_model = LitVaeModel.load_from_checkpoint(cfg.SAVED_MODEL_DIR / 'vae_last.ckpt')
for param in vae_model.parameters():
    param.requires_grad = False

vae_mnist = LitMNISTDataModule()
rnn_mnist = LitRnnDataModule()

dl = rnn_mnist.train_dataloader()
idl = iter(dl)
img_batch = next(idl)    # .size = [batch_size, seq_len, 28, 28]

x_reconstructed = []
for i in range(8):
    x = img_batch[0, i, :, :].unsqueeze(0).unsqueeze(0).cuda()
    encoded = vae_model.encoder(x) # size = [batch_size, kernel_size, 7, 7]
    # Both mean & logvar have size = [<batch_size>, <z_size>]
    mean, logvar = vae_model.q(encoded)  
    # Following executes epsilon-sampling.
    z = vae_model.z(mean,logvar)
    z_projected = vae_model.project(z).view(
        -1, 
        vae_model.kernel_num,
        vae_model.feature_size,
        vae_model.feature_size,
    )  # size = 128, 32, 7, 7
    x_reconstructed.append(vae_model.decoder(z_projected))

fig, ax = plt.subplots(2, cfg.vae_rnn.SEQ_LEN, figsize=(16,8))
for ia, a in enumerate(ax.reshape(-1)):
    if ia < 8:
        cb = plt.colorbar(a.imshow(img_batch[0, ia, :, :].cpu().detach().numpy(),clim=[0,1]),fraction=.046,pad=0.01)
        cb.ax.tick_params('both',length=2,which='major')
        [t.set_fontsize(7) for t in cb.ax.get_yticklabels()]
        a.set_axis_off()
    else:
        cb = plt.colorbar(a.imshow(x_reconstructed[ia-8].squeeze().cpu().detach().numpy(),clim=[0,1]),fraction=.046,pad=0.01) 
        cb.ax.tick_params('both',length=2,which='major')
        [t.set_fontsize(7) for t in cb.ax.get_yticklabels()]
        a.set_axis_off()
        #plt.axis('off')

#ax[0].imshow(im1),ax[0].set_title("Original, unrotated")
#ax[1].imshow(im2),ax[1].set_title(f"Rotated: {theta:.1f} deg")

plt.savefig('test-vae-rnn-reconst.png', dpi=300)

#----------------------------------------------------------------------

img_batch = next(iter(vae_mnist.train_dataloader()))[0]    # tuple of length 2, [0].size = [batch_size, 1, 28, 28]
IMG_TO_DISPLAY = 7
fig, ax = plt.subplots(2, IMG_TO_DISPLAY, figsize=(22,8))
for ia, a in enumerate(ax.reshape(-1)):
    #print(f"ia={ia}")
    if ia < IMG_TO_DISPLAY:
        x = img_batch[ia, 0, :, :]
        #print(f"ia={ia}  TOP,  x min/max: {x.min():.1f}, {x.max():.1f}")
        cb = plt.colorbar(a.imshow(x.cpu().detach().numpy(),clim=[0,1]),fraction=.046,pad=0.01)
        cb.ax.tick_params('both',length=2,which='major')
        [t.set_fontsize(7) for t in cb.ax.get_yticklabels()]
        a.set_axis_off()
    else:
        x = img_batch[ia-IMG_TO_DISPLAY, 0, :, :].unsqueeze(0).unsqueeze(0).cuda()
        #print(f"ia={ia}  BOTTOM,  x min/max: {x.min():.1f}, {x.max():.1f}")
        encoded = vae_model.encoder(x) # size = [batch_size, kernel_size, 7, 7]
        # Both mean & logvar have size = [<batch_size>, <z_size>]
        mean, logvar = vae_model.q(encoded)  
        # Following executes epsilon-sampling.
        z = vae_model.z(mean,logvar)
        z_projected = vae_model.project(z).view(
            -1, 
            vae_model.kernel_num,
            vae_model.feature_size,
            vae_model.feature_size,
        )  # size = 128, 32, 7, 7
        x_reconstructed = vae_model.decoder(z_projected) # x_reconstructed.size = [1, 1, 28, 28]
        #cb = plt.colorbar(a.imshow(x_reconstructed[0,0,:,:].cpu().detach().numpy()),fraction=.046,pad=0.01,vmin=0,vmax=1)
        cb = plt.colorbar(a.imshow(x_reconstructed[0,0,:,:].cpu().detach().numpy(),clim=[0,1]),fraction=.046,pad=0.01)
        cb.ax.tick_params('both',length=2,which='major')
        #set_trace()
        [t.set_fontsize(7) for t in cb.ax.get_yticklabels()]
        a.set_axis_off()

plt.savefig('test-vae-reconst.png',dpi=300)

if 0: # can't run this from docker
    time.sleep(1)
    os.system('explorer.exe test-imgseq.png')

#set_trace()

#        encoded = self.encoder(x) # size = [batch_size, kernel_size, 7, 7]
#        # Both mean & logvar have size = [<batch_size>, <z_size>]
#        mean, logvar = self.q(encoded)  
#        # Following executes epsilon-sampling.
#        z = self.z(mean,logvar)
#        z_projected = self.project(z).view(
#            -1, 
#            self.kernel_num,
#            self.feature_size,
#            self.feature_size,
#        )  # size = 128, 32, 7, 7
#        x_reconstructed = self.decoder(z_projected)
