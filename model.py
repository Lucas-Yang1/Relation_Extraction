import torch
import torch.nn as nn
from transformers import BertModel

from data import label2idx
from tokenizer import Tokenizer


class BertBaseModel(nn.Module):
    def __init__(self, bert_path, classes_num, CUDA=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = Tokenizer.from_pretrained(bert_path)
        self.classes_num = classes_num
        hidden_features = self.bert.config.pooler_fc_size
        self.sub_classifier = nn.Linear(hidden_features, 2)
        self.sub_lstm = nn.LSTM(hidden_features, hidden_features, batch_first=True)
        self.ob_classifier = nn.Linear(hidden_features, classes_num * 2)
        self.layernorm = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(0.1)
        self.CUDA = CUDA
        if self.CUDA:
            self.cuda()


    def _embedding(self, text: list[str]):
        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        if self.CUDA:
            text_embed = self.bert(inputs['input_ids'].cuda(),
                                   inputs['token_type_ids'].cuda(),
                                   inputs['attention_mask'].cuda()).last_hidden_state
        else:
            text_embed = self.bert(**inputs).last_hidden_state

        return dict(
            text_embed=text_embed,
            mask=inputs['attention_mask'].cuda() if self.CUDA else inputs['attention_mask']
        )

    def _sub_forward(self, text_embed: torch.Tensor):
        sub_out = self.sub_classifier(self.dropout(text_embed))
        return sub_out

    def _sub_embed_merge(self, text_embed: torch.Tensor, sub_idx: tuple[tuple]):
        sub_embed = [t[sub_idx[i][0]:sub_idx[i][1]] for i, t in enumerate(text_embed)]
        sub_pad_embed = nn.utils.rnn.pack_sequence(sub_embed, enforce_sorted=False)
        _, (sub_embed_merge, _) = self.sub_lstm(sub_pad_embed)

        return sub_embed_merge.transpose(0,1)

    def _ob_forward(self, text_embed: torch.Tensor, sub_idx: tuple[tuple]):
        sub_embed_merge = self._sub_embed_merge(text_embed, sub_idx)
        ob_out = self.ob_classifier(self.dropout(self.layernorm(torch.add(text_embed, sub_embed_merge))))
        return ob_out

    def loss_compute(self, text, sub_idx, subject_label, object_label, loss_fn=nn.BCELoss(reduction='none')):
        text_embed, mask = self._embedding(text).values()
        sub_out = self._sub_forward(text_embed)
        ob_out = self._ob_forward(text_embed, sub_idx)
        ob_out = ob_out.view(ob_out.size(0), ob_out.size(1), self.classes_num, 2)

        sub_loss = loss_fn(sub_out.sigmoid(), subject_label)
        ob_loss = loss_fn(ob_out.sigmoid(), object_label)

        sub_loss = sub_loss * mask.unsqueeze(-1)
        ob_loss = ob_loss * mask.unsqueeze(-1).unsqueeze(-1)
        loss = (sub_loss.sum() + ob_loss.sum()) / mask.sum()
        return  loss

    def forward(self, text, threshold=0.5):

        embed = self._embedding(text)
        sub_out = self._sub_forward(embed['text_embed'])
        sub_prob = sub_out.sigmoid() * embed['mask'].unsqueeze(-1)
        SPO_list = []
        for i, prob in enumerate(sub_prob):
            prob = prob * embed['mask'][i].unsqueeze(-1)
            sub_idx = label2idx(prob>threshold, 'sub')
            for idx in sub_idx:
                ob_out = self._ob_forward(embed['text_embed'][i].unsqueeze(0), (idx,))
                P_O = label2idx(ob_out[0].view(ob_out[0].size(0), self.classes_num, 2).sigmoid()>threshold, 'ob')
                SPO_list.extend([(text[i][idx[0]:idx[1]], po[0], text[i][po[1][0]:po[1][1]]) for po in P_O])

        return SPO_list







