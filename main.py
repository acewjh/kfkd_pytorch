import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import *
from utils import *

class Flatten(nn.Module):
    def forward(self, x):
        N, _, _, _ = x.size()
        return x.view(N, -1)

class l2_msk_loss(nn.Module):
    def __init__(self):
        super(l2_msk_loss, self).__init__()
        return
    def forward(self, crds, pred_crds, masks):
        N = torch.sum(masks, dim=1)
        batch_size = crds.data.shape[0]
        loss = torch.sum(torch.sqrt(torch.sum(((crds - pred_crds) * masks) ** 2, dim=1) / N)) / batch_size
        return loss


kfkd_net1 = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    #map size: 16*48*48
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    #map size: 32*24*24
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    #map size: 64*11*11
    Flatten(),
    nn.Linear(7744, 2048),
    nn.ReLU(),
    #map size: 2048
    nn.Linear(2048, 1024),
    nn.ReLU(),
    #map size: 1024
    nn.Linear(1024, 30)
    #final: 30
)

def train(model, loss_fn, optimizer, data, crds, masks, dtype=torch.FloatTensor,
          num_epochs=1, batch_size=64, print_every=100):
    # train_data, test_data = data
    # train_crds, test_crds = crds
    # train_masks, test_masks = masks
    train_data = data
    train_crds = crds
    train_masks = masks
    N = train_data.shape[0]
    num_steps = int(N/batch_size)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        print('Epoch %d / %d ' % (epoch + 1, num_epochs))
        print('Learning rate: {}\n'.format(scheduler.get_lr()[0]))
        time_start = time.time()
        for step in tqdm(range(num_steps)):
            indx = np.arange(N) #random generate minibatch
            np.random.shuffle(indx)
            indx = indx[:batch_size]
            batch_data = train_data[indx]
            batch_crds = train_crds[indx]
            batch_masks = train_masks[indx]
            x_var = Variable(torch.from_numpy(batch_data).type(dtype).view(batch_size, 1, 96, 96))
            y_var = Variable(torch.from_numpy(batch_crds).type(dtype))
            m_var = Variable(torch.from_numpy(batch_masks).type(dtype))
            crds = model(x_var)
            loss = loss_fn(crds, y_var, m_var)
            if (step + 1)%print_every == 0 or step == (num_steps - 1):
                tqdm.write('Step: {}  Loss: {:.6f}'.format(step, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_end = time.time()
        print('training time %fs'%(time_end - time_start))
        localtime = time.localtime(time.time())
        torch.save(model, './ckpt/datetime_{}_{}_{}_{}_{}-epoch_{}-tr_loss_{:.6f}'.format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                                                                localtime.tm_min, localtime.tm_sec, epoch, loss.data[0]))
        #test(model, loss_fn, test_data, test_crds, test_masks, epoch+1, dtype)

def test(model, loss_fn, test_data, test_crds, test_masks, epoch, dtype=torch.FloatTensor):
    N = test_data.shape[0]
    x_var = Variable(torch.from_numpy(test_data).type(dtype).view(N, 1, 96, 96))
    y_var = Variable(torch.from_numpy(test_crds).type(dtype))
    m_var = Variable(torch.from_numpy(test_masks).type(dtype))
    model.eval()
    pred_crds = model(x_var)
    loss = loss_fn(pred_crds, y_var, m_var)
    localtime = time.localtime(time.time())
    print('test loss %f'%(loss.data.cpu().numpy()))
    torch.save(model, './ckpt/datetime_{}/{}_{}_{}_{}-epoch_{}-loss_{:.6f}'.format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                                                                   localtime.tm_min, localtime.tm_sec, epoch, loss.data[0]))

def predict(model, pred_data, dtype=torch.FloatTensor, cache_size=500):
    model.eval()
    N = pred_data.shape[0]
    steps = int(N/cache_size) + 1
    pred_crds = np.random.randn(1,30)
    for i in range(steps):
        if i == (steps - 1):
            pred_data_batch = pred_data[i*cache_size:]
            x_var = Variable(torch.from_numpy(pred_data_batch).type(dtype).view(pred_data_batch.shape[0], 1, 96, 96))
        else:
            pred_data_batch = pred_data[i*cache_size: (i+1)*cache_size]
            x_var = Variable(torch.from_numpy(pred_data_batch).type(dtype).view(cache_size, 1, 96, 96))
        pred_crds_batch = model(x_var)
        if i == 0:
            pred_crds = pred_crds_batch.data.cpu().numpy()
        else:
            pred_crds = np.concatenate((pred_crds, pred_crds_batch.data.cpu().numpy()))
    return pred_crds

#d = np.random.rand(1,1,96,96)
#var = Variable(torch.from_numpy(d)).type(torch.LongTensor).long()
#kfkd_net1(var)
data, crds, masks = load_data('./data/training.csv')
imgs = load_img('./data/test.csv')
# #N = data.shape[0]
#indx = np.arange(N)
# #np.random.shuffle(indx)
indx = np.load('./data/indx.npy')
#
data = (data[indx[:6500]], data[indx[6500:]])
crds = (crds[indx[:6500]], crds[indx[6500:]])
masks = (masks[indx[:6500]], masks[indx[6500:]])
dtype = torch.cuda.FloatTensor
kfkd_net1.type(dtype)
loss_fn = l2_msk_loss().type(dtype)
optimizer = torch.optim.Adam(kfkd_net1.parameters(), lr=2e-3)
train(kfkd_net1, loss_fn, optimizer, data, crds, masks, dtype, 25)
# pred_crds = predict(kfkd_net1, imgs, dtype)
# save_answers(pred_crds, './data/IdLookupTable.csv', './data/training.csv', './data/lr_2e-3-epoch_25_results.csv')
model = torch.load('./ckpt/datetime_1_1_15_16_30-epoch_24-tr_loss_1.313630')
pred_crds = predict(model, imgs, dtype)
save_answers(pred_crds, './data/IdLookupTable.csv', './data/training.csv', './data/results_2.csv')
pass