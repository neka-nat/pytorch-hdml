import os
import copy
from collections import deque
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.manifold import TSNE
from . import hdml

def train_triplet(data_streams, viz, max_steps, n_class, lr,
                  model_path='model', model_save_interval=2000,
                  tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    win_jm = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Jm loss'))
    win_tsne = viz.scatter(X=np.array([[0.0, 0.0]]), opts=dict(title='t-SNE'))
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)

    tri = hdml.TripletBase(n_class=n_class, pretrained=pretrained).to(device)
    optimizer_c = optim.Adam(tri.parameters(), lr=lr, weight_decay=5e-3)

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)

            jm, _, embedding_z = tri(torch.from_numpy(x_batch).to(device))

            optimizer_c.zero_grad()
            jm.backward()
            optimizer_c.step()

            pbar.set_description("Jm: %f" % jm.item())

            if cnt > 0 and cnt % model_save_interval == 0:
                torch.save(tri.state_dict(), os.path.join(model_path, 'model_%d.pth' % cnt))

            if cnt > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                z_reduced = TSNE(n_components=2, random_state=0).fit_transform(np.vstack(test_data))
                viz.scatter(X=z_reduced, Y=np.vstack(test_label), win=win_tsne, opts=dict(title='t-SNE'))

            viz.line(X=np.array([cnt]), Y=np.array([jm.item()]), win=win_jm, update='append')

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            cnt += 1


def train_hdml_triplet(data_streams, viz, max_steps, n_class, lr_init,
                       lr_gen=1.0e-2, lr_s=1.0e-3,
                       model_path='model', model_save_interval=2000,
                       tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    win_jgen = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Jgen loss'))
    win_jmetric = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Jmetric loss'))
    win_jm = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Jm loss'))
    win_ce = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Cross-entropy loss'))
    win_tsne = viz.scatter(X=np.array([[0.0, 0.0]]), opts=dict(title='t-SNE'))
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)

    hdml_tri = hdml.TripletHDML(n_class=n_class, pretrained=pretrained).to(device)
    optimizer_c = optim.Adam(list(hdml_tri.classifier1.parameters()) + list(hdml_tri.classifier2.parameters()),
                             lr=lr_init, weight_decay=5e-3)
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
            pbar.set_description("Jmetric: %f, Jgen: %f, Jm: %f, CrossEntropy: %f" % (jmetric.item(), jgen, jm, ce.item()))

            if cnt % model_save_interval == 0:
                torch.save(hdml_tri.state_dict(), os.path.join(model_path, 'model_%d.pth' % cnt))

            if cnt > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                z_reduced = TSNE(n_components=2, random_state=0).fit_transform(np.vstack(test_data))
                viz.scatter(X=z_reduced, Y=np.vstack(test_label), win=win_tsne, opts=dict(title='t-SNE'))

            viz.line(X=np.array([cnt]), Y=np.array([jgen]), win=win_jgen, update='append')
            viz.line(X=np.array([cnt]), Y=np.array([jmetric.item()]), win=win_jmetric, update='append')
            viz.line(X=np.array([cnt]), Y=np.array([jm]), win=win_jm, update='append')
            viz.line(X=np.array([cnt]), Y=np.array([ce.item()]), win=win_ce, update='append')

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            cnt += 1