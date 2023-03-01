import numpy as np


def seq_train(loader, model, optimizer, device, epoch_idx,log_interval):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
        loss = model.criterion_calculation(ret_dict, label, label_lgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
    optimizer.scheduler.step()
    return loss_value