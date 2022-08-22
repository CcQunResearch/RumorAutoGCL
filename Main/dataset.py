# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset
from Main.utils import clean_comment


class WeiboDataset(InMemoryDataset):
    def __init__(self, root, word2vec, clean=True, transform=None, pre_transform=None, pre_filter=None):
        self.word2vec = word2vec
        self.clean = clean
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y.append(post['source']['label'])
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)
                edge_index = [row, col]
                edge_attr = torch.ones(len(row), 1)
                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) if 'label' in post[
                    'source'].keys() \
                    else Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(one_data)
        else:
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
                if 'label' in post['source'].keys():
                    y.append(post['source']['label'])
                for i, comment in enumerate(post['comment']):
                    x = torch.cat(
                        [x, self.word2vec.get_sentence_embedding(clean_comment(comment['content'])).view(1, -1)], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)
                edge_index = [row, col]
                edge_attr = torch.ones(len(row), 1)
                y = torch.LongTensor(y)
                edge_index = torch.LongTensor(edge_index)
                one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) if 'label' in post[
                    'source'].keys() \
                    else Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])
