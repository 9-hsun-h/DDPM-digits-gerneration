#%%
# Import of libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST
from PIL import Image

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"

#%%
no_train = False
fashion = False
batch_size = 128
n_epochs = 20
lr = 0.001

store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"

#%%
def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    print(f"the image tpye:{type(images)}")
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    images = np.transpose(images, (0,2,3,1)).astype(np.float32)
    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)
    
    # Populating figure with sub-plots
    idx = 0
    print(len(images))
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                print(images.shape)

                #plt.imshow(images[idx][0], cmap="gray")
                print(images[idx])
                plt.imshow(images[idx], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    print(f'ind={idx}')
    # Showing the figure
    plt.show()

#%%
import os
def show_images_and_save(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    print(f"the image tpye:{type(images)}")
    print(images.shape)
    print(images[0])
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    images = np.transpose(images, (0,2,3,1)).astype(np.float32)
    
    # reverse_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    # ])
    for i in range(10000):
        image_temp = images[i, :, :, 0] # imsave can only save (H,W) format
        #plt.imshow(reverse_transforms(image_temp))
        plt.imsave(os.path.join(os.path.dirname(__file__), f'generate/{i+1:05d}.png'), image_temp, cmap="gray")
        #reverse_transforms(image_temp).convert('RGB').save(os.path.join(os.path.dirname(__file__), f'generate/{i+1:05d}.png'))
        if i%50 ==0:
            print(f'{i} images has been stored.')

    # # Populating figure with sub-plots
    # idx = 0
    # print(len(images))
    # for r in range(rows):
    #     for c in range(cols):
    #         fig.add_subplot(rows, cols, idx + 1)

    #         if idx < len(images):
    #             print(images.shape)

    #             #plt.imshow(images[idx][0], cmap="gray")
    #             plt.imshow(images[idx], cmap="gray")
    #             idx += 1
    # fig.suptitle(title, fontsize=30)
    # print(f'ind={idx}')
    # # Showing the figure
    # plt.show()

#%%
def show_first_batch(loader):
    for batch in loader:
        print(type(batch))
        print(batch.shape)
        #show_images(batch[0], "Images in the first batch")
        show_images(batch, "Images in the first batch")
        break

#%%
# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_SIZE = 28
BATCH_SIZE = 128

class DataGenerator(Dataset):
    def __init__(self,phase,images,s = 0.5):
        self.phase = phase
        self.images = images
        self.transforms = transforms.Compose([#transforms.ToTensor(), # Scales data into [0,1] 
                                            transforms.ToPILImage(),
                                            transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                            transforms.ToTensor(), # Scales data into [0,1] 
                                            #transforms.RandomHorizontalFlip(0.5),
                                            transforms.Lambda(lambda t: (t -0.5) *2) # Scale between [-1, 1]
                                            ])


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self,i):
        x = Image.open(self.images[i])
        #x = x.convert('RGB')
        #print(x)
        #x = np.array(x)/255.0  # might not be needed
        x = np.array(x)  # might not be needed
        #print(x.shape)
        # print(f"x ====> {x}")
        #x = np.transpose(x, (2, 0, 1)).astype(np.float32) ###################
        #print(x.shape)
        #x = torch.from_numpy(x)
        x = self.transforms(x)
        # print(f"x1 after augment and to_tensor {x1}")
        # print(x1.shape)
        # x1 = self.preprocess(x1)
        # x2 = self.preprocess(x2)
        return x


# read the data.csv file and get the image paths
df = pd.read_csv('C:/CCBDA/HW3 V2/data.csv')
X = df.image_path.values # image paths
#y = df.target.values # targets
(x_train, x_valid) = train_test_split(X,
    test_size=0.10, random_state=42)
print(f"Training instances: {len(x_train)}")
print(f"Validation instances: {len(x_valid)}")

#create dataloader
train_dg = DataGenerator('train',X)
train_dl = DataLoader(train_dg,batch_size = BATCH_SIZE,drop_last=True)

# transform = Compose([
#     ToTensor(),
#     Lambda(lambda x: (x - 0.5) * 2)]
# )
# ds_fn = FashionMNIST if fashion else MNIST
# dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
# loader = DataLoader(dataset, batch_size, shuffle=True)

#%%
# Optionally, show a batch of regular images
show_first_batch(train_dl)

#%%
# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

#%%
# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

#%%
def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        #imgs = batch[0]
        imgs = batch

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

#%%
def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=7, gif_name="sampling.gif", c=1, h=28, w=28):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)
        print(f'created random noise')
        count = 0
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            if idx%50==0:
                print(f'ddpm step:{idx}')
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=1,b2=8)
                frame = frame.cpu().numpy().astype(np.uint8)

                count +=1
                # Rendering frame
                print(frame.shape)
                if count ==1:
                    allArrays = frame
                else:
                    allArrays = np.concatenate([allArrays, frame],axis=0)
                print(allArrays.shape)
                frames.append(frame)
        plt.imsave(os.path.join(os.path.dirname(__file__), f'diffusion_process.png'), allArrays[:,:,0], cmap="gray")
                

    # # Storing the gif
    # with imageio.get_writer(gif_name, mode="I") as writer:
    #     for idx, frame in enumerate(frames):
    #         writer.append_data(frame)
    #         if idx == len(frames) - 1:
    #             for _ in range(frames_per_gif // 3):
    #                 writer.append_data(frames[-1])
    return x

#%%
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

#%%
# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

#%%
# Optionally, show the diffusion (forward) process
show_forward(ddpm, train_dl, device)

#%%
# Optionally, show the denoising (backward) process
generated = generate_new_images(ddpm, gif_name="before_training.gif")
show_images(generated, "Images generated before training")

#%%
def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            #x0 = batch[0].to(device)
            x0 = batch.to(device) #torch.Size([128, 3, 28, 28])
            #print(x0.shape)
            x0 = x0[:,0:1,:,:]
            #print(x0.shape) #torch.Size([128, 1, 28, 28])
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

#%%
# Training
store_path = "./model_weight/ddpm_fashion.pt" if fashion else "./model_weight/ddpm_mnist.pt"
if not no_train:
    training_loop(ddpm, train_dl, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

# #%%
# # Loading the trained model
# best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
# best_model.load_state_dict(torch.load(store_path, map_location=device))
# best_model.eval()
# print("Model loaded: Generating new images")

# #%%
# generated = generate_new_images(
#         best_model,
#         n_samples=10000,
#         device=device,
#         gif_name="fashion.gif" if fashion else "mnist.gif"
#     )
# show_images_and_save(generated, "Final result_generated")
# #%%
# # from IPython.display import Image

# # Image(open('fashion.gif' if fashion else 'mnist.gif','rb').read())

# #%%
# !python -m pytorch_gan_metrics.calc_metrics --path ./generate --stats C:/CCBDA/HW3/mnist.npz
# # %%
# generated = generate_new_images(
#         best_model,
#         n_samples=8,
#         device=device,
#         gif_name="fashion.gif" if fashion else "mnist.gif"
#     )
# # %%
