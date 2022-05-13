import torch
from .utils import compute_poisson_loss


def inference(
        model,
        responses_train_neurons,
        responses_test_neurons,
        num_sample=10,
        num_iter=2000,
        learning_rate=1e-3,
    ):
    model.eval()
    y_train = responses_train_neurons
    y_test = responses_test_neurons

    # get latent samples
    latents = []
    for i in range(num_sample):
        _, _, z_, _, _ = model(y_train, z=None)
        z_opt = torch.clone(z_.detach())
        z_opt.requires_grad = True
        latents.append(z_opt)

    optimizer = torch.optim.Adam(latents, lr=learning_rate)

    for i in range(num_iter):
        optimizer.zero_grad()
        loss = 0
        losses = []
        for j in range(num_sample):
            y_train_, _, _, _, _ = model(y_train, z=latents[j])
            losses.append(compute_poisson_loss(y_train, y_train_))
            loss = loss + losses[-1]
        loss.backward()
        optimizer.step()

        if not (i % 100):
            train_loss = torch.min(torch.tensor(losses)).item()
            losses = []
            for j in range(num_sample):
                _, y_test_, _, _, _ = model(y_train, z=latents[j])
                losses.append(compute_poisson_loss(y_test, y_test_))
            print('INFERENCE: iter %s, negLLH(train): %s, negLLH(test): %s' % (
                i, train_loss, torch.min(torch.tensor(losses)).item()))

    # get best latents of all samples
    losses = []
    for j in range(num_sample):
        y_train_, _, _, _, _ = model(y_train, z=latents[j])
        losses.append(compute_poisson_loss(y_train, y_train_))
    best_latents = latents[torch.argmin(torch.tensor(losses))]
    return best_latents