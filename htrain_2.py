from hvae2_ab import VAE
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

writer = SummaryWriter('/content/drive/MyDrive/Colab Notebooks/VAE_CIFAR10/runs/hvae2_ab_cifar10/')

transform = transforms.Compose([
    transforms.ToTensor(), #convert to 0-1
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #convert to -1 to 1
])

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

loaded_train = DataLoader(train_data, batch_size=128, shuffle=True)
loaded_test = DataLoader(test_data, batch_size=1, shuffle=True)

lr = 1e-4
wd = 1e-5
epochs = 100
model = VAE().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

save_path = '/content/drive/MyDrive/Colab Notebooks/VAE_CIFAR10/models/hvae2_ab/'
import os
os.makedirs(save_path, exist_ok=True)
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_recon = 0
    epoch_middle = 0
    epoch_kl_z2 = 0
    beta = VAE.beta_schedule(epoch)
    alpha = VAE.alpha_schedule(epoch)
    for images, _ in loaded_train:
        batchsize = images.shape[0]
        images = images.to(device)
        pred, z1, mu1, logvar1, mu2, logvar2, dmu1, dlogvar1 = model(images)
        recon, middle, kl_z2 = VAE.compute_loss(pred, images, z1, mu1, logvar1, mu2, logvar2, dmu1, dlogvar1)
        loss = (recon - alpha * middle + beta * kl_z2) / batchsize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_recon += recon.item() / batchsize
        epoch_middle += middle.item() / batchsize
        epoch_kl_z2 += kl_z2.item() / batchsize

    loss_history.append(epoch_loss)
    writer.add_scalar("Loss/total", epoch_loss, epoch)
    writer.add_scalar("Loss/reconstruction", epoch_recon, epoch)
    writer.add_scalar("Loss/middle", epoch_middle, epoch)
    writer.add_scalar("Loss/KL_z2", epoch_kl_z2, epoch)
    writer.add_scalar("Scheduled/Beta", beta, epoch)
    writer.add_scalar("Scheduled/Alpha", alpha, epoch)

    writer.flush()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.3f}, Recon: {epoch_recon:.3f}, Middle: {epoch_middle:.3f}, KL_2z: {epoch_kl_z2:.3f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path + f'vae_epoch{epoch+1}.pth')
        with torch.no_grad():
            sample = images[:4]
            recons, _, _, _, _, _, _, _ = model(sample)
            sample_vis = (sample + 1) / 2
            recons_vis = (recons + 1) / 2
            comparison = torch.cat([sample_vis, recons_vis])
            writer.add_images('Reconstruction', comparison, epoch)
            generated = model.generate(4, device)
            generated_vis = (generated + 1) / 2
            writer.add_images('Generated', generated_vis, epoch)

torch.save(model.state_dict(), save_path + 'hvae2_ab.pth')
writer.close()
