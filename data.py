import numpy as np
import pandas as pd
import torch
from torch import nn

predicate = sorted(pd.read_json('data/all_50_schemas', lines=True)['predicate'].values.tolist())
pred2idx = {p: i for i, p in enumerate(predicate)}
idx2pred = {i: p for i, p in enumerate(predicate)}


class Dataset:
    def __init__(self, df):
        self.df = df[['text', 'spo_list']]

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, item):
        text, spo_list = self.df.iloc[item]
        target = self.data_fn(text, spo_list)

        object_labels_num = len(target['object_labels'])
        # return object_labels_num
        # print(object_labels_num)
        if object_labels_num == 0:
            return None
        else:
            idx = np.random.choice(object_labels_num)
            sample = target['object_labels'][idx]
            return {
                "text": text,
                "subject_label": torch.tensor(target['subject_label'], dtype=torch.float),
                "subject_idx": torch.tensor(sample[0], dtype=torch.long),
                "object_label": torch.tensor(sample[1], dtype=torch.float),
                "spo_list": target['spo_list']
            }

    @staticmethod
    def data_fn(text, spo_list) -> dict:
        """
        生成subject 与 Predict_Object对应的index_span tensor
        :param text: str
        :param spo_list: list
        :return: dict(subject_label:numpy.array, object_labels: list[tuple])
        """
        text = text.lower()
        subject = np.zeros((len(text), 2))
        object = {}
        for spo in spo_list:
            sub_str = spo['subject'].lower()
            ob_str = spo['object'].lower()
            sub_start_idx = text.find(sub_str)
            if sub_start_idx == -1:
                continue
            sub_idx = (sub_start_idx, sub_start_idx + len(sub_str))
            subject[sub_idx[0], 0] = 1
            subject[sub_idx[1] - 1, 1] = 1

            ob_start_idx = text.find(ob_str)
            if ob_start_idx == -1:
                continue
            ob_idx = (ob_start_idx, ob_start_idx + len(ob_str))
            object_matrix = np.zeros((len(text), 50, 2))
            object_matrix[ob_idx[0], pred2idx[spo['predicate']], 0] = 1
            object_matrix[ob_idx[1] - 1, pred2idx[spo['predicate']], 1] = 1
            object[sub_idx] = object.get(sub_idx, 0) + object_matrix

        return {
            'subject_label': subject,
            'object_labels': list(object.items()),
            'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in spo_list]
        }


def label2idx(label, opt='sub'):
    """
    :param label:
    :return:
    """
    assert opt in ['sub', 'ob']

    if opt == 'sub':
        sub = []
        sub_start = torch.nonzero(label[..., 0])
        sub_end = torch.nonzero(label[..., 1])
        for i in sub_start:
            end = [j[0] for j in sub_end if j[0] >= i[0]]
            if len(end) <= 0:
                continue

            sub.append((i[0].item(), end[0].item() + 1))
        return sub

    if opt == 'ob':
        ob = []
        ob_start = torch.nonzero(label[..., 0])
        ob_end = torch.nonzero(label[..., 1])

        for i in ob_start:
            end = [j[0] for j in ob_end if j[-2] >= i[0] and j[1] == i[1]]
            if len(end) <= 0:
                continue

            predicate = idx2pred[i[1].item()]
            ob.append((predicate, (i[0].item(), end[0].item() + 1)))
        return ob


def collate_fn(list_of_data):
    data = {key: [item[key] for item in list_of_data if item is not None] for key in list_of_data[0].keys()}
    data['subject_label'] = nn.utils.rnn.pad_sequence(data['subject_label'], True)
    data['object_label'] = nn.utils.rnn.pad_sequence(data['object_label'], True)

    return data


if __name__ == '__main__':
    from model import BertBaseModel

    df = pd.read_json('./data/train_data.json', lines=True)
    data = Dataset(df)
    list_of_data = [data[0], data[1], data[2]]
    src = collate_fn(list_of_data)
    bert_path = 'C:\\Users\\yhane\\workspace\\RR\\model\\bert_model\\chinese_wwm_ext_pytorch'
    model = BertBaseModel(bert_path, classes_num=50)
    embed = model._embedding(src['text'])
    text_embed = embed['text_embed']
