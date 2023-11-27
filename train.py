import torch
from denoising_diffusion.ddpm import DenoiseDiffusion
from denoising_diffusion.unet import UNet
import torchvision
from torchvision.utils import save_image
import os
import argparse
from denoising_diffusion.dataset import MNISTDataset,CelebADataset

# hyperparameters, don't modify
n_channels=64
channel_multipliers=[1,2,2,4]
is_attention= [False, False, False, True]
n_steps=1000
batch_size=64
image_size=32
learning_rate=2e-5
n_samples=16


def train(epoch):
    """
    ### Train
    """
    # Iterate through the dataset
    for batch_ndx, data in enumerate(data_loader):
        data = data.to("cuda")
        optimizer.zero_grad()
        loss = diffusion.loss(data)
        loss.backward()
        optimizer.step()
        if batch_ndx % 100==0:
            print ('Epoch [{}] Step [{}/{}], Loss: {:.4f}'.format(epoch, batch_ndx, len(data_loader), loss.item()))

def sample(epoch):
    """
    ### Sample images
    """
    with torch.no_grad():
        x = torch.randn([n_samples, image_channels, image_size, image_size], device="cuda")

        # Remove noise for $T$ steps
        for t_ in range(n_steps):
            t = n_steps - t_ - 1
            x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

        # Log samples
        path = os.path.join("outputs/{}/epoch_{}".format(dataset_style, epoch))
        if not os.path.exists(path) :
            os.makedirs(path)
            
        for idx in range(n_samples):
            sample = x[idx]
            save_image(sample, path+'/sample_{}.png'.format(idx))

        print("save image to :{}".format(path))

def run():
    """
    ### Training loop
    """
    for epoch in range(epochs):
        # Train the model
        train(epoch)

                
        # Sample some images
        sample(epoch)

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--dataset', default='mnist', type=str, help='mnist or CelebA')
parser.add_argument('--epochs',  default=5, type=int, help='mnist=5, CelebA=100')
parser.add_argument('--channels',  default=1, type=int, help='mnist=1, CelebA=3',)
args = parser.parse_args()

image_channels = args.channels
epochs = args.epochs
dataset_style = args.dataset  
    
if __name__ == '__main__':
    eps_model = UNet(
                image_channels=image_channels,
                n_channels=n_channels,
                ch_mults=channel_multipliers,
                is_attn=is_attention,
            ).to("cuda")

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device ="cuda",
    )

    if dataset_style == 'mnist':
        dataset = MNISTDataset(image_size)
    else:
        dataset = CelebADataset(image_size)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=learning_rate)

    run()
                