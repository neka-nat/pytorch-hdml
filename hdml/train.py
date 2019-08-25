import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from . import hdml


def triplet_train(data_streams, max_steps, lr_init, lr_gen=1.0e-2, lr_s=1.0e-3,
                  model_path='model', model_save_interval=2000,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()

    hdml_tri = hdml.TripletHDML().to(device)
    optimizer_c = optim.Adam(list(hdml_tri.classifier1.parameters()) + list(hdml_tri.classifier2.parameters()), lr=lr_init)
    optimizer_g = optim.Adam(hdml_tri.generator.parameters(), lr=lr_gen)
    optimizer_s = optim.Adam(hdml_tri.softmax_classifier.parameters(), lr=lr_s)

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    jm = 1.0e+6
    jgen = 1.0e+6
    cnt = 0

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)

            jgen, jmetric, jm, ce = hdml_tri(torch.from_numpy(x_batch).to(device),
                                             torch.from_numpy(label.astype(np.int64)).to(device),
                                             jm, jgen)

            optimizer_c.zero_grad()
            optimizer_g.zero_grad()
            optimizer_s.zero_grad()
            jmetric.backward(retain_graph=True)
            jgen.backward(retain_graph=True)
            ce.backward()
            optimizer_c.step()
            optimizer_g.step()
            optimizer_s.step()

            jm = jm.item()
            jgen = jgen.item()

            if cnt % model_save_interval == 0:
                torch.save(hdml_tri.state_dict(), os.path.join(model_path, 'model_%d.pth' % cnt))
            cnt += 1