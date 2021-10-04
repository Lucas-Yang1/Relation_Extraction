import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW

from data import Dataset, collate_fn
from model import BertBaseModel


class Trainer(object):
    def __init__(self, batch_size=12, Epoch=5, CUDA=True):
        self.batch_size = batch_size
        self.Epoch = Epoch
        self.CUDA = CUDA

    def train(self, model, dataset, optimizer, collate_fn=None, schedule_fn=get_linear_schedule_with_warmup):
        model.train()
        model.requires_grad_()
        if self.CUDA:
            model.cuda()
        else:
            model.cpu()
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, collate_fn=collate_fn)
        batches_num = len(dataloader) * self.Epoch
        if schedule_fn != None:
            schedule = schedule_fn(optimizer,
                                   num_warmup_steps=len(dataloader),
                                   num_training_steps=batches_num)
        for e in range(self.Epoch):
            epoch_loss = 0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                if self.CUDA:
                    batch['subject_label'] = batch['subject_label'].cuda()
                    batch['object_label'] = batch['object_label'].cuda()

                loss = model.loss_compute(batch['text'], batch['subject_idx'],
                                          batch['subject_label'], batch['object_label'])

                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

                if schedule_fn != None:
                    schedule.step()
            print(f'Epoch: [{e + 1}/{self.Epoch}], loss: {epoch_loss:.3}')

    def eval(self, model, dataset, collate_fn=collate_fn, threshold=0.5):
        dataloader = DataLoader(dataset, self.batch_size, collate_fn=collate_fn)
        model.eval()
        model.requires_grad_(False)
        if self.CUDA:
            model.cuda()
            model.CUDA = True
        else:
            model.cpu()
            model.CUDA = False
        pbar = tqdm()
        x, y, z = 1e-10, 1e-10, 1e-10
        for item in dataloader:
            res = set(model(item['text'], threshold))
            target = set()
            target.update(*item['spo_list'])
            x += len(res & target)
            y += len(res)
            z += len(target)
            f1, precision, recall = 2 * x / (y + z), x / y, x / z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )

        return f1, precision, recall
if __name__ == '__main__':
    bert_path = 'C:\\Users\\yhane\\workspace\\RR\\model\\bert_model\\chinese_wwm_ext_pytorch'
    df = pd.read_json('./data/train_data.json', lines=True)
    dataset = Dataset(df)
    eval_df = pd.read_json('./data/dev_data.json', lines=True)
    eval_dataset = Dataset(eval_df)
    model = BertBaseModel(bert_path, classes_num=50)
    model.load_state_dict(torch.load('temp/model.pth'))
    trainer = Trainer(batch_size=64)
    trainer.eval(model, eval_dataset, collate_fn, 0.7)
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': 1e-3},
    ])

    # trainer.train(model, dataset, optimizer, collate_fn, schedule_fn=get_linear_schedule_with_warmup)
