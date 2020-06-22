from typing import List

import torch
import torch.nn as nn
from torch import optim

from configuration.config import logger


CHECK_DEGUG = False


ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()


def vae_loss_func(x, recon, mu, logvar, beta):
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD, MSE, KLD


def train(train_generator, dev_generator, pool_generator, task_model, vae, discriminator, args):
    num_epoch = args.epoch_num
    batch_size = args.batch_size
    device = args.device
    n_gpu = args.n_gpu
    beta = args.beta

    optim_task = optim.Adam(task_model.parameters(), lr=5e-5)
    optim_vae = optim.Adam(vae.parameters(), lr=5e-5)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-5)

    best_epoch, best_acc = 0, 0
    for e in range(num_epoch):

        if CHECK_DEGUG and e > 0: break

        task_model.train()
        vae.train()
        discriminator.train()
        for idx, (labeled_batch, unlabeld_batch) in enumerate(zip(train_generator, pool_generator)):

            if CHECK_DEGUG and idx > 0:
                break

            raw_text = labeled_batch[-1]
            labeled_batch = [_.to(device) for _ in labeled_batch[:-1]]

            X_ids, Y_ids, V_ids = labeled_batch

            ####################
            # task model step
            ####################
            preds = task_model(X_ids)
            task_loss = ce_loss(preds, Y_ids)
            if n_gpu > 0:
                task_loss = task_loss.mean()
            task_model.zero_grad()
            task_loss.backward()
            optim_task.step()

            #############
            # vae step
            #############
            recon, mu, logvar, z = vae(X_ids)
            vae_loss, mse_loss_value, kld_loss_value = vae_loss_func(V_ids, recon, mu, logvar, beta)

            if idx < 1: logger.info(f'vae loss: {vae_loss}')

            if isinstance(vae_loss, List) and n_gpu > 0:
                vae_loss = vae_loss.mean()

            un_X_ids, _, un_V_ids = [_.to(device) for _ in unlabeld_batch[:-1]]
            un_recon, un_mu, un_logvar, un_z = vae(un_X_ids)
            un_vae_loss, un_mse_loss_value, un_kld_loss_value = vae_loss_func(un_V_ids, un_recon, un_mu, un_logvar, beta)
            if isinstance(un_vae_loss, List) and n_gpu > 0:
                un_vae_loss = un_vae_loss.mean()

            labeled_pred = discriminator(mu)
            unlabeled_pred = discriminator(un_mu)

            labeled_real_target = torch.ones(X_ids.size()[0], device=device)
            unlabeled_real_target = torch.ones(un_X_ids.size()[0], device=device)

            dsc_loss_in_vae = bce_loss(labeled_pred, labeled_real_target) + bce_loss(unlabeled_pred, unlabeled_real_target)

            if idx<1: logger.info(f'dsc_loss_in_vae: {dsc_loss_in_vae}')

            total_loss = vae_loss + un_vae_loss + dsc_loss_in_vae

            vae.zero_grad()
            total_loss.backward()
            optim_vae.step()

            #####################
            # discriminate step
            #####################
            with torch.no_grad():
                # _, mu, _, _ = vae(X_ids)
                # _, un_mu, _, _ = vae(un_X_ids)

                mu_no_grad = mu.detach()
                un_mu_no_grad = un_mu.detach()

            labeled_pred = discriminator(mu_no_grad)
            unlabeled_pred = discriminator(un_mu_no_grad)

            labeled_real_target = torch.ones(X_ids.size()[0], device=device)
            unlabeled_fake_target = torch.zeros(un_X_ids.size()[0], device=device)

            dsc_loss = bce_loss(labeled_pred, labeled_real_target)+ bce_loss(unlabeled_pred, unlabeled_fake_target)
            if idx<1: logger.info(f'dsc_loss: {dsc_loss}')

            discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()

            if idx % 10 == 0 and idx != 0:
                logger.info(f'epoch: {e} - batch: {idx}/{len(train_generator)}')
                logger.info(f'task_model loss: {task_loss}')
                logger.info(f'labeled vae loss: {vae_loss}')
                logger.info(f'labeled mse loss: {mse_loss_value}')
                logger.info(f'labeled kld loss: {kld_loss_value}')
                logger.info(f'unlabeled vae loss: {un_vae_loss}')
                logger.info(f'unlabeled mse loss: {un_mse_loss_value}')
                logger.info(f'unlabeled kld loss: {un_kld_loss_value}')
                logger.info(f'dsc_loss_in_vae: {dsc_loss_in_vae}')
                logger.info(f'dsc_loss: {dsc_loss}')

        task_model.eval()

        correct = 0
        for idx, batch in enumerate(dev_generator):

            if CHECK_DEGUG and idx > 1:
                break

            raw_text = batch[-1]
            batch = [_.to(device) for _ in batch[:-1]]
            X_ids, Y_ids, _ = batch

            with torch.no_grad():
                logits = task_model(X_ids)
                logits = torch.argmax(logits, dim=-1)
                correct += logits.eq(Y_ids).sum()

        acc = correct / dev_generator.total_data_size

        if acc > best_acc:
            best_acc = acc
            best_epoch = e

        logger.info(f'epoch {e} - acc: {acc} - best_acc: {best_acc} - best_epoch: {best_epoch}')

    return best_acc





















