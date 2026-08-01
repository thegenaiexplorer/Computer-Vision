"""
train_cgan.py
-------------
Reusable training function for cGAN experiments.
"""

import torch
import os


def train_cgan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    criterion,
    device,
    z_dim,
    num_classes,
    epochs=50
    ):
    generator.train()
    discriminator.train()
    os.makedirs('../pytorch-docker-env', exist_ok=True)

    def discriminator_loss(criterion, real_output, fake_output):
        real_loss = criterion(real_output, torch.ones_like(real_output)*0.9)
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss

    def generator_loss(criterion, fake_output):
        return criterion(fake_output, torch.ones_like(fake_output))

    for epoch in range(epochs):
        d_loss_epoch = 0
        g_loss_epoch = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            #Generate fake images and labels
            z = torch.randn(batch_size, z_dim, 1, 1).to(device) #using convolution
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)

            # Train Discriminator
            optimizer_D.zero_grad()
            fake_output = discriminator(gen_imgs.detach(), gen_labels)
            real_output = discriminator(imgs, labels)
            d_loss = discriminator_loss(criterion, real_output, fake_output)

            d_loss_epoch += d_loss.item() * batch_size

            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
         
            g_output = discriminator(gen_imgs, gen_labels)
            g_loss = generator_loss(criterion, g_output)
            g_loss_epoch += g_loss.item() * batch_size
            g_loss.backward()
            optimizer_G.step()

        avg_dloss = d_loss_epoch/len(dataloader.dataset)
        avg_gloss = g_loss_epoch/len(dataloader.dataset)
        print(
            f"[Epoch {epoch+1}/{epochs}] D loss: {avg_dloss:.4f} | G loss: {avg_gloss:.4f}"
        )

        # Save checkpoint at end of every epoch
        torch.save(
            generator.state_dict(), f"../pytorch-docker-env/checkpoints_gen{epoch+1}.pt"
            )

        #torch.save(
        #    discriminator.state_dict(), f"../pytorch-docker-env/checkpoints_disc{epoch+1}.pt",
        #    )
        print(f"Saved checkpoints at epoch {epoch+1}")
