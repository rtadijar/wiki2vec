import torch
import torch.functional as F
from torch.utils.data import DataLoader, RandomSampler

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def get_optimizer(model):
    sparse_params = []
    dense_params = []

    for name, param in model.named_parameters():
        if name == 'embeddings.weight':
            sparse_params.append(param)
        else:
            dense_params.append(param)
            
    opt_dense = torch.optim.Adam(dense_params, lr=1e-3)
    opt_sparse = torch.optim.SGD(sparse_params, lr=1e-3)

    optimizer = MultipleOptimizer(opt_sparse, opt_dense)

    return optimizer


def train(model, dataset, num_epochs=1, batch_size=512, num_samples=10, reg_coeff=1., device='cpu', print_period=1000):

    train_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    optimizer = get_optimizer(model)

    len_dataset = len(dataset.tensors[1])
    len_sequence = dataset.tensors[1].shape[1]

    for epoch in num_epochs:

        print('started epoch {}'.format(epoch))
        for idx, batch in enumerate(train_loader):           
            model.mh_attn.clear_agreement()

            x, y = batch[0].to(device), batch[1].to(device)
            sz = x.shape[0]

            optimizer.zero_grad()

            sample_indices = torch.FloatTensor(sz * num_samples).uniform_(0, len_dataset - 1).long()        
            sample_indices, tmp = torch.broadcast_tensors(sample_indices.unsqueeze(1), 
                                                        torch.arange(sz * num_samples * len_sequence)
                                                        .view(sz * num_samples, len_sequence))

            n = torch.gather(dataset.tensors[1], 0, sample_indices).to(device)

            x = model(x)
            y = model(y)
            n = model(n)

            target_loss = F.cosine_embedding_loss(x, y, torch.Tensor([1]).to(device), margin=0.5)

            x = torch.broadcast_tensors(x.unsqueeze(1), n.view(sz, num_samples, d_model))[0].flatten(0,1)

            noise_loss = F.cosine_embedding_loss(x, n, torch.Tensor([-1]).to(device), margin=0.5, reduction='none')
            noise_loss = noise_loss.view(sz, num_samples, 1).sum(1).mean()    

            loss = target_loss + noise_loss + reg_coeff * model.mh_attn.v_agreement

            loss.backward()
            optimizer.step()


            if (idx + 1) % print_period:
                print('loss at batch {}: {}', idx + 1, loss.item())