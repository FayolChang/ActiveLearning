import torch




def sample(vae, discriminator, pool_generator, budget, args):
    device = args.device

    vae.eval()
    discriminator.eval()

    pred_list = []
    for idx, batch in enumerate(pool_generator):

        if args.check_debug and idx > 0:
            break

        _,_,V_ids,Mask = [_.to(device) for _ in batch[:-1]]
        with torch.no_grad():
            _, mu, _, _ = vae(V_ids, Mask)
            pred = discriminator(mu)
            pred_list.extend(-pred)

    _, indices = torch.topk(torch.tensor(pred_list), k=budget)
    return indices.detach().cpu().numpy()

