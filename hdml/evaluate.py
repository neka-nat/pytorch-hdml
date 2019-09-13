import os
import copy
import numpy as np
import cv2
import torch
from tqdm import tqdm
from . import hdml


def evaluate_triplet(data_streams, writer, max_steps, n_class,
                     pretrained=False,
                     model_path='model/model_30000.pth',
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_test.get_epoch_iterator()
    test_data = []
    test_label = []
    test_img = []

    tri = hdml.TripletBase(n_class=n_class, pretrained=pretrained).to(device)
    tri.load_state_dict(torch.load(model_path))
    tri.eval()

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)
            _, _, embedding_z = tri(torch.from_numpy(x_batch).to(device))
            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break
        writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                             torch.from_numpy(np.stack(test_img, axis=0)),
                             global_step=cnt, tag='embedding/test')
        writer.flush()


def evaluate_hdml_triplet(data_streams, writer, max_steps, n_class,
                          pretrained=False,
                          model_path='model/model_30000.pth',
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_test.get_epoch_iterator()
    test_data = []
    test_label = []
    test_img = []

    hdml_tri = hdml.TripletHDML(n_class=n_class, pretrained=pretrained).to(device)
    hdml_tri.load_state_dict(torch.load(model_path))
    hdml_tri.eval()

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)
            _, _, embedding_z = hdml_tri.classifier1(torch.from_numpy(x_batch).to(device))
            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break
        writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                             torch.from_numpy(np.stack(test_img, axis=0)),
                             global_step=cnt, tag='embedding/test')
        writer.flush()