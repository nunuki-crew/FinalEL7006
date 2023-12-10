"""
Training functions
Ammi Beltrán & Fernanda Borja
"""
# imports 
import os
import pyFiles.dataprep as prep
import numpy as np
import torch
from torch import nn
from pyFiles.losses import InfoNceLoss
from pyFiles.losses import BarlowTwins


"""
Pretext
"""
def preloss(model, batch, criterion = nn.CrossEntropyLoss(), device = "cuda"):
    x1, x2 = batch
    # 
    model = model.to(device)
    x1, x2 = x1.to(device), x2.to(device)
    y1, y2 = model(x1), model(x2)
    # 
    del x1, x2
    loss = criterion(y1, y2)
    del y1, y2
    torch.cuda.empty_cache()
    return loss

def prevalidate(model, val_loader, criterion):
    e_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            loss= preloss(model, batch, criterion)
            e_loss +=loss
            #
            del loss, batch
            torch.cuda.empty_cache()
        e_loss = e_loss/(len(val_loader))
    return e_loss

def pretrain_epoch(model, train_loader, criterion, optimizer):
    # training mode
    model.train()
    # vloss = []
    # tloss = []
    lossSum = 0
    # iters = 0
    for i, batch in enumerate(train_loader):
        loss = preloss(model, batch, criterion)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Iter: {i + 1}/{len(train_loader)}, Loss:{loss}")
        lossSum += loss
        # iters += 1
        #
        del batch, loss
        torch.cuda.empty_cache()
        # if ((i+1)%5 == 0):
        #     val_loss = prevalidate(model, val_loader, criterion)
        #     model.train()
        #     print(f"Validating, Val loss = {val_loss}")
        #     vloss.append(val_loss)
        #     tloss.append(lossSum/iters)
    tloss = lossSum/len(train_loader)
    return tloss#, vloss

def pretext_train(model, epochs, train_loader, val_loader, criterion, optimizer, state = None, name = ""):
    # If trained before
    if state == None:
        state = {
            "epoch" : 0,
            "loss" : [[], []], # [train, val]
            "params" : None, 
            "bestloss" : np.inf
        }
    # If previously trained
    else:
        state = torch.load(state)
        model.encoder.load_state_dict(state["params"][0])
        model.final.load_state_dict(state["params"][1])

    best_loss = state["bestloss"]
    state_epochs = state["epoch"]
    for epoch in range(state_epochs, state_epochs + epochs):
        print(f"Epoch nro {epoch + 1}/{epochs}")
        # Train
        train_loss = pretrain_epoch(model, train_loader, criterion, optimizer)
        # Val
        val_loss = prevalidate(model, val_loader, criterion)
        # Save if better loss 
        print(f"Train loss = {train_loss}, Val loss = {val_loss}")

        if (best_loss>val_loss):
            best_loss = val_loss
            print(f"Better params found in epoch = {epoch + 1}, saved params")
            torch.save([model.encoder.state_dict(), model.final.state_dict()], f'bestEncoderParams{name}.pt')
        # Update state
        state["loss"][0].append(train_loss.item())
        state["loss"][1].append(val_loss.item())
        state["epoch"] = epoch + 1
        state["params"] = [model.encoder.state_dict(), model.final.state_dict()]
        state["bestloss"] = best_loss
        # Save last just in case, [includes loss!!!!]
        torch.save(state, f"Lastencoder{name}_{epoch + 1}.pt")
    return state["loss"]    

"""
Downstream
"""

def downloss(model, batch, criterion = nn.CrossEntropyLoss(), device = "cuda"):
    x, y = batch
    model.train()
    model = model.to(device)
    x, y = x.to(device), y.type(torch.int64).to(device)
    y_pred = model(x)
    # y_pred = torch.argmax(y_pred, dim = 1, keepdim= True).type(torch.FloatTensor).to(device) # REVISAR FORMATO DE LABEL, POR AHORA ENTREGA 0 o 1, PODRIA SER [0, 1] y [1, 0]
    # print(y.type())
    # print(y_pred.type())

    loss = criterion(y_pred, y.squeeze())
    return loss, y, y_pred

def downtrain_epoch(model, train_dataset, criterion, optimizer, device = "cuda"):
    acc = 0
    e_loss = 0
    total_pred = 0
    for i, batch in enumerate(train_dataset):
        loss, y, y_pred = downloss(model, batch, criterion, device)

        optimizer.zero_grad()
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()

        e_loss +=loss.item()
        y_pred = torch.argmax(y_pred, dim = 1,keepdim= True)
        
        acc += torch.sum(y == y_pred).item()
        total_pred +=y.shape[0]
        print(f"Iter: {i + 1}/{len(train_dataset)}, Loss:{loss}")
    acc = acc/total_pred
    e_loss = e_loss/len(train_dataset)
    # print(f"Epoch train loss = {e_loss}")
    return e_loss, acc

def downvalidate(model, val_dataset, criterion):
    acc = 0
    e_loss = 0
    total_pred = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_dataset):
            loss, y, y_pred = downloss(model, batch, criterion)
            e_loss +=loss.item()
            y_pred = torch.argmax(y_pred, dim = 1,keepdim= True)
            acc+=torch.sum(y==y_pred).item()
            total_pred +=y.shape[0]
        acc = acc/total_pred
        e_loss = e_loss/len(val_dataset)
    return e_loss, acc

def downtrain(model, epochs, train_dataset, val_dataset, criterion, optimizer, each = 50, state = None, name = ""):
    # If trained before
    if state == None:
        state = {
            "epoch" : 0,
            "loss" : [[], []], # [train, val]
            "acc" : [[], []], # [train, val]
            "params" : None, 
            "bestloss" : np.inf
        }
    # If previously trained
    else:
        state = torch.load(state)
        model.load_state_dict(state["params"])
    best_loss = state["bestloss"]
    for epoch in range(0, epochs):
        # Train
        print(f"Epoch nro {epoch +1}/{epochs}")
        e_loss, e_acc = downtrain_epoch(model, train_dataset, criterion, optimizer)
        # Validate
        val_loss, val_acc = downvalidate(model, val_dataset, criterion)
        #
        print(f"Epoch {epoch + 1}/{epochs}: Train loss = {e_loss}, Val loss = {val_loss}, Train acc = {e_acc}, Val acc = {val_acc}")
        # Save if better loss
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Better params found in epoch = {epoch + 1}, saved params")
            torch.save(model.state_dict(), f'bestDownParams{name}.pt')
        # Load best model so far to proceed
        # model.load_state_dict(torch.load(f'bestDownParams.pt'))
        # Save periodically for each
        if ((epoch + 1)%each == 0):
            print(f"Se ha guardado la época múltiplo de {each}")
            torch.save(model.state_dict(), f'eachDownParams{name}_{epoch + 1}.pt')
        # Update state
        state["loss"][0].append(e_loss)
        state["loss"][1].append(val_loss)
        state["acc"][0].append(e_acc)
        state["acc"][1].append(val_acc)
        state["epoch"] = epoch + 1
        state["params"] = model.state_dict()
        state["bestloss"] = best_loss
        # Save last just in case, [includes loss!!!!]
        torch.save(state, f"LastDown{name}_{epoch + 1}.pt")
    
    return state["loss"], state["acc"]