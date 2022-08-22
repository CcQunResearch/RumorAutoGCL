# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:41
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : utils.py
# @Software: PyCharm
# @Note    :
import json
import os
import shutil
import re


def write_json(dict, path):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(dict, file_obj, indent=4, ensure_ascii=False)


def write_post(post_list, path):
    for post in post_list:
        write_json(post[1], os.path.join(path, f'{post[0]}.json'))


def dataset_makedirs(dataset_path):
    train_path = os.path.join(dataset_path, 'train', 'raw')
    val_path = os.path.join(dataset_path, 'val', 'raw')
    test_path = os.path.join(dataset_path, 'test', 'raw')

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)
    os.makedirs(os.path.join(dataset_path, 'train', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'val', 'processed'))
    os.makedirs(os.path.join(dataset_path, 'test', 'processed'))

    return train_path, val_path, test_path


def clean_comment(comment_text):
    match_res = re.match('回复@.*?:', comment_text)
    if match_res:
        return comment_text[len(match_res.group()):]
    else:
        return comment_text


def create_log_dict(args):
    log_dict = {}
    log_dict['dataset'] = args.dataset
    log_dict['unsup train size'] = args.unsup_train_size
    log_dict['runs'] = args.runs

    log_dict['batch size'] = args.batch_size
    log_dict['unsup_bs_ratio'] = args.unsup_bs_ratio
    log_dict['n layers feat'] = args.n_layers_feat
    log_dict['n layers conv'] = args.n_layers_conv
    log_dict['n layers fc'] = args.n_layers_fc
    log_dict['vector size'] = args.vector_size
    log_dict['hidden'] = args.hidden
    log_dict['global pool'] = args.global_pool
    log_dict['skip connection'] = args.skip_connection
    log_dict['res branch'] = args.res_branch
    log_dict['dropout'] = args.dropout
    log_dict['edge norm'] = args.edge_norm

    log_dict['add_mask'] = args.add_mask
    log_dict['train_mode'] = args.train_mode

    log_dict['lr'] = args.lr
    log_dict['epochs'] = args.epochs
    log_dict['weight decay'] = args.weight_decay
    log_dict['lamda'] = args.lamda
    log_dict['lamda cl unlabel'] = args.lamda_cl_unlabel
    log_dict['lamda cl label'] = args.lamda_cl_label
    log_dict['lamda sim'] = args.lamda_sim
    log_dict['lamda cls'] = args.lamda_cls

    log_dict['k'] = args.k

    log_dict['record'] = []
    return log_dict


def write_log(log, str):
    log.write(f'{str}\n')
    log.flush()
