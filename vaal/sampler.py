import torch


CHECK_DEBUG = False


def sample(vae, discriminator, pool_generator, budget, args):
    device = args.device

    vae.eval()
    discriminator.eval()

    pred_list = []
    for idx, batch in enumerate(pool_generator):

        if CHECK_DEBUG and idx > 0:
            break

        X_ids = batch[0].to(device)
        with torch.no_grad():
            _, mu, _, _ = vae(X_ids)
            pred = discriminator(mu)
            pred_list.extend(-pred)

    _, indices = torch.topk(torch.tensor(pred_list), k=budget)
    return indices.detach().cpu().numpy()

