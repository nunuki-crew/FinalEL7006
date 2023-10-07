"""
Training functions
Ammi Beltrán & Fernanda Borja
"""
# imports 
import os
import dataprep as prep
import numpy as np
import torch
from torch import nn
from losses import InfoNceLoss

"""
Pretext
"""
def preloss(model, batch, criterion = nn.CrossEntropyLoss(), device = "cuda"):
    x1, x2 = batch
    x1 = x1.reshape((19,1,4000)).type(torch.FloatTensor)
    x2 = x2.reshape((19,1,4000)).type(torch.FloatTensor)
    model = model.to(device)
    x1, x2 = x1.to(device), x2.to(device)
    y1, y2 = model(x1), model(x2)
    del x1, x2
    loss = criterion(y1, y2)
    return loss

def pretrain_epoch(model, train_loader, criterion, optimizer):
    # training mode
    model.train()
    lossSum = 0
    iters = 0
    for i, batch in enumerate(train_loader):
        loss = preloss(model, batch, criterion)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Iter: {i + 1}/{len(train_loader)}, Loss:{loss}")
        lossSum += loss
        iters += 1
    trainloss = lossSum/iters
    print(f"Epoch train loss = {trainloss}")
    return trainloss

def pretext_train(model, epochs, train_loader, criterion, optimizer):
    cumulative_loss = []
    best_loss = np.inf
    for epoch in range(0, epochs):
        ep_loss = pretrain_epoch(model, train_loader, criterion, optimizer)
        cumulative_loss.append(ep_loss)
        if (best_loss>ep_loss):
            print(f"Better params found in epoch = {epoch}, saved params")
            torch.save(model.state_dict(), f'bestEncoderParams.pt')
        model.load_state_dict(torch.load(f'bestEncoderParams.pt'))
    return cumulative_loss    

"""
Downstream
"""

def downloss(model, batch, criterion = nn.CrossEntropyLoss(), device = "cuda"):
    x, y = batch
    model.train()
    model = model.to(device)
    x = x.reshape(x.shape[0], 1, 4000).type(torch.FloatTensor)
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    y_pred = (torch.argmax(y_pred, dim = 1)).type(torch.FloatTensor).to(device) # REVISAR FORMATO DE LABEL, POR AHORA ENTREGA 0 o 1, PODRIA SER [0, 1] y [1, 0]

    loss = criterion(y_pred, y)
    print(loss)
    return loss, y, y_pred

def downtrain_epoch(model, train_dataset, criterion, optimizer):
    acc = 0
    e_loss = 0
    count = 0
    for i, batch in enumerate(train_dataset):
        loss, y, y_pred = downloss(model, batch, criterion)

        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        e_loss +=loss
        acc +=(y == y_pred)
        count +=1
        print(f"Iter: {i + 1}/{len(train_dataset)}, Loss:{loss}")
    acc = acc/count
    e_loss = e_loss/count
    print(f"Epoch train loss = {e_loss}")
    return e_loss, acc

def downvalidate(model, val_dataset, criterion):
    acc = 0
    e_loss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_dataset):
            loss, y, y_pred = downloss(model, batch, criterion)
            e_loss +=loss
            acc+=(y==y_pred)
            count +=1
        acc = acc/count
        e_loss = e_loss/count
    return e_loss, acc

def downtrain(model, epochs, train_dataset, val_dataset, criterion, optimizer):
    model.train()
    best_loss = np.inf
    train_curves = [[],[]] # [loss, accuracy]
    val_curves = [[],[]]
    for epoch in range(0, epochs):
        # Entrenamos
        print(f"Epoch nro {epoch +1}/{epochs}")
        e_loss, e_acc = downtrain_epoch(model, train_dataset, criterion, optimizer)
        train_curves[0].append(e_loss)
        train_curves[1].append(e_acc)
        # Validamos
        val_loss, val_acc = downvalidate(model, val_dataset, criterion)
        val_curves[0].append(val_loss)
        val_curves[1].append(val_acc)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f'bestDownParams.pt')
        model.load_state_dict(torch.load(f'bestDownParams.pt'))

    return train_curves, val_curves