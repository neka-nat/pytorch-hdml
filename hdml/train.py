import os
import copy
from collections import deque
import numpy as np
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from . import hdml

def train_triplet(data_streams, writer, max_steps, n_class, lr,
                  model_path='model', model_save_interval=2000,
                  tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)
    test_img = deque(maxlen=n_test_data)

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
                writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                                     torch.from_numpy(np.stack(test_img, axis=0)),
                                     global_step=cnt, tag='embedding/train')
                writer.flush()

            writer.add_scalar('Loss/Jm/train', jm.item(), cnt)

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1


def train_hdml_triplet(data_streams, writer, max_steps, n_class, lr_init,
                       lr_gen=1.0e-2, lr_s=1.0e-3,
                       model_path='model', model_save_interval=2000,
                       tsne_test_interval=1000, n_test_data=1000, pretrained=False,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)
    test_img = deque(maxlen=n_test_data)

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

            jgen, jmetric, jm, ce, embedding_z = hdml_tri(torch.from_numpy(x_batch).to(device),
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

            jm = max(jm.item(), 1.0e-6)
            jgen = max(jgen.item(), 1.0e-6)
            pbar.set_description("Jmetric: %f, Jgen: %f, Jm: %f, CrossEntropy: %f" % (jmetric.item(), jgen, jm, ce.item()))

            if cnt > 0 and cnt % model_save_interval == 0:
                torch.save(hdml_tri.state_dict(), os.path.join(model_path, 'model_%d.pth' % cnt))

            if cnt > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                                     torch.from_numpy(np.stack(test_img, axis=0)),
                                     global_step=cnt, tag='embedding/train')
                writer.flush()

            writer.add_scalar('Loss/Jgen/train', jgen, cnt)
            writer.add_scalar('Loss/Jmetric/train', jmetric.item(), cnt)
            writer.add_scalar('Loss/Jm/train', jm, cnt)
            writer.add_scalar('Loss/cross_entropy/train', ce.item(), cnt)

            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1