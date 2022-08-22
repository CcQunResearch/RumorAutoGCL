# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 15:11
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main.py
# @Software: PyCharm
# @Note    :
import sys
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Main.pargs import pargs
from Main.utils import create_log_dict, write_log, write_json
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset, sort_weibo_2class_dataset
from Main.dataset import WeiboDataset
from Main.model import ResGCN, GIN_NodeWeightEncoder, ViewGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(model, view_gen1, view_gen2, unsup_train_loader, train_loader, optimizer, view_optimizer, lamdas,
          train_mode=0):
    model.train()
    view_gen1.train()
    view_gen2.train()

    unlabel_loss_all = 0
    label_loss_all = 0
    loss_all = 0

    if train_mode == 0:
        unlabel_loss_all = train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer)
        label_loss_all, sim_loss_all, cls_loss_all, cl_loss_all = \
            train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer,
                                                    view_optimizer, lamdas, label_loss_dominant=False,
                                                    use_cl_loss=False, cl_loss_sample=False)
    elif train_mode == 1:
        unlabel_loss_all = train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer)
        label_loss_all, sim_loss_all, cls_loss_all, cl_loss_all = \
            train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer,
                                                    view_optimizer, lamdas, label_loss_dominant=False,
                                                    use_cl_loss=True, cl_loss_sample=True)
    elif train_mode == 2:
        unlabel_loss_all = train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer)
        label_loss_all, sim_loss_all, cls_loss_all, cl_loss_all = \
            train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer,
                                                    view_optimizer, lamdas, label_loss_dominant=False,
                                                    use_cl_loss=True, cl_loss_sample=False)
    elif train_mode == 3:
        unlabel_loss_all = train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer)
        label_loss_all, sim_loss_all, cls_loss_all, cl_loss_all = \
            train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer,
                                                    view_optimizer, lamdas, label_loss_dominant=True,
                                                    use_cl_loss=False, cl_loss_sample=False)
    elif train_mode == 4:
        unlabel_loss_all = train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer)
        label_loss_all, sim_loss_all, cls_loss_all, cl_loss_all = \
            train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer,
                                                    view_optimizer, lamdas, label_loss_dominant=True,
                                                    use_cl_loss=True, cl_loss_sample=True)
    elif train_mode == 5:
        unlabel_loss_all = train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer)
        label_loss_all, sim_loss_all, cls_loss_all, cl_loss_all = \
            train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer,
                                                    view_optimizer, lamdas, label_loss_dominant=True,
                                                    use_cl_loss=True, cl_loss_sample=False)
    elif train_mode == 6:
        loss_all = jointly_train(model, view_gen1, view_gen2, train_loader,unsup_train_loader,optimizer, view_optimizer,lamdas)

    return unlabel_loss_all, label_loss_all, loss_all


def train_model_with_fix_vg(model, view_gen1, view_gen2, unsup_train_loader, optimizer):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    total_graphs = 0

    for unlabel_data in unsup_train_loader:
        optimizer.zero_grad()

        unlabel_data = unlabel_data.to(device)
        _, view1 = view_gen1(unlabel_data, False)
        _, view2 = view_gen2(unlabel_data, False)

        input_list = [unlabel_data, view1, view2]
        input1, input2 = random.choices(input_list, k=2)

        embedding1 = model.forward_graph_cl(input1)
        embedding2 = model.forward_graph_cl(input2)

        cl_loss = loss_cl(embedding1, embedding2)
        loss = cl_loss
        loss.backward()

        loss_all += loss.item() * unlabel_data.num_graphs
        total_graphs += unlabel_data.num_graphs

        optimizer.step()
    loss_all /= total_graphs
    return loss_all


def train_model_vg_together_with_label_data(model, view_gen1, view_gen2, train_loader, optimizer, view_optimizer,
                                            lamdas, label_loss_dominant=True, use_cl_loss=True, cl_loss_sample=True):
    model.train()
    view_gen1.train()
    view_gen2.train()

    loss_all = 0
    sim_loss_all = 0
    cls_loss_all = 0
    cl_loss_all = 0
    total_graphs = 0

    for label_data in train_loader:
        view_optimizer.zero_grad()
        optimizer.zero_grad()

        label_data = label_data.to(device)

        sample1, view1 = view_gen1(label_data, True)
        sample2, view2 = view_gen2(label_data, True)

        sim_loss = F.mse_loss(sample1, sample2)
        sim_loss = (1 - sim_loss)

        output = model(label_data)
        output1 = model(view1)
        output2 = model(view2)

        loss0 = F.binary_cross_entropy(output, label_data.y.to(torch.float32))
        loss1 = F.binary_cross_entropy(output1, label_data.y.to(torch.float32))
        loss2 = F.binary_cross_entropy(output2, label_data.y.to(torch.float32))
        if label_loss_dominant:
            cls_loss = (loss1 + loss2) / 2
        else:
            cls_loss = (loss0 + loss1 + loss2) / 3

        if use_cl_loss:
            if cl_loss_sample:
                input_list = [label_data, view1, view2]
                input1, input2 = random.choices(input_list, k=2)

                embedding1 = model.forward_graph_cl(input1)
                embedding2 = model.forward_graph_cl(input2)
                cl_loss = loss_cl(embedding1, embedding2)
            else:
                embedding1 = model.forward_graph_cl(view1)
                embedding2 = model.forward_graph_cl(view2)
                cl_loss = loss_cl(embedding1, embedding2)
            loss = sim_loss + cls_loss + cl_loss
        else:
            loss = sim_loss + cls_loss

        if label_loss_dominant:
            loss = loss0 + loss * lamdas['lamda']

        loss.backward()
        loss_all += loss.item() * label_data.num_graphs
        sim_loss_all += sim_loss.item() * label_data.num_graphs
        cls_loss_all += cls_loss.item() * label_data.num_graphs
        if use_cl_loss:
            cl_loss_all += cl_loss.item() * label_data.num_graphs

        total_graphs += label_data.num_graphs
        view_optimizer.step()
        optimizer.step()

    loss_all /= total_graphs
    sim_loss_all /= total_graphs
    cls_loss_all /= total_graphs
    cl_loss_all /= total_graphs

    return loss_all, sim_loss_all, cls_loss_all, cl_loss_all


def jointly_train(model, view_gen1, view_gen2, train_loader, unsup_train_loader, optimizer, view_optimizer, lamdas):
    model.train()
    view_gen1.train()
    view_gen2.train()

    loss_all = 0
    total_graphs = 0

    for label_data,unlabel_data in zip(train_loader,unsup_train_loader):
        view_optimizer.zero_grad()
        optimizer.zero_grad()

        label_data = label_data.to(device)
        unlabel_data = unlabel_data.to(device)

        sample1, view1 = view_gen1(label_data, True)
        sample2, view2 = view_gen2(label_data, True)
        sample3, view3 = view_gen1(unlabel_data, True)
        sample4, view4 = view_gen2(unlabel_data, True)

        sim_loss = F.mse_loss(sample1, sample2)
        sim_loss = (1 - sim_loss)

        output = model(label_data)
        output1 = model(view1)
        output2 = model(view2)

        loss0 = F.binary_cross_entropy(output, label_data.y.to(torch.float32))
        loss1 = F.binary_cross_entropy(output1, label_data.y.to(torch.float32))
        loss2 = F.binary_cross_entropy(output2, label_data.y.to(torch.float32))

        cls_loss = (loss1 + loss2) / 2

        label_input_list = [label_data, view1, view2]
        input1, input2 = random.choices(label_input_list, k=2)
        embedding1 = model.forward_graph_cl(input1)
        embedding2 = model.forward_graph_cl(input2)
        label_cl_loss = loss_cl(embedding1, embedding2)

        unlabel_input_list = [unlabel_data, view3, view4]
        input3, input4 = random.choices(unlabel_input_list, k=2)
        embedding3 = model.forward_graph_cl(input3)
        embedding4 = model.forward_graph_cl(input4)
        unlabel_cl_loss = loss_cl(embedding3, embedding4)

        loss = loss0 + lamdas['lamda_cl_unlabel'] * unlabel_cl_loss + lamdas['lamda_cl_label'] * label_cl_loss + \
               lamdas['lamda_sim'] * sim_loss + lamdas['lamda_cls'] * cls_loss
        loss.backward()
        loss_all += loss.item() * label_data.num_graphs
        total_graphs += label_data.num_graphs
        view_optimizer.step()
        optimizer.step()

    loss_all /= total_graphs
    return loss_all


def fine_tuning(model, view_gen1, view_gen2, train_loader, optimizer):
    model.train()
    view_gen1.eval()
    view_gen2.eval()

    loss_all = 0
    total_graphs = 0
    for label_data in train_loader:
        optimizer.zero_grad()

        label_data = label_data.to(device)
        output = model(label_data)

        loss = F.binary_cross_entropy(output, label_data.y.to(torch.float32))
        loss.backward()
        loss_all += loss.item() * label_data.num_graphs

        total_graphs += label_data.num_graphs
        optimizer.step()

    loss_all /= total_graphs
    return loss_all


def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss


def test(model, dataloader, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        error += F.binary_cross_entropy(pred, data.y.to(torch.float32)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(dataloader.dataset), acc, prec, rec, f1


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    log_record['val accs'].append(round(val_acc, 3))
    log_record['test accs'].append(round(test_acc, 3))
    log_record['test prec T'].append(round(test_prec[0], 3))
    log_record['test prec F'].append(round(test_prec[1], 3))
    log_record['test rec T'].append(round(test_rec[0], 3))
    log_record['test rec F'].append(round(test_rec[1], 3))
    log_record['test f1 T'].append(round(test_f1[0], 3))
    log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, log_record


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    batch_size = args.batch_size
    unsup_bs_ratio = args.unsup_bs_ratio
    weight_decay = args.weight_decay
    lamdas = {
        'lamda': args.lamda,
        'lamda_cl_unlabel': args.lamda_cl_unlabel,
        'lamda_cl_label': args.lamda_cl_label,
        'lamda_sim': args.lamda_sim,
        'lamda_cls': args.lamda_cls
    }
    epochs = args.epochs
    ft_epochs = args.ft_epochs

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', 'Weibo-unsup', 'dataset')
    model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')
    weight_path = osp.join(dirname, '..', 'Model', f'{log_name}.pt')

    log = open(log_path, 'w')
    log_dict = create_log_dict(args)

    if not osp.exists(model_path):
        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path)
        elif dataset == 'Weibo-self':
            sort_weibo_self_dataset(label_source_path, label_dataset_path, unlabel_dataset_path)
        elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
            sort_weibo_2class_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                      'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

        word2vec = Embedding(model_path)
        unlabel_dataset = WeiboDataset(unlabel_dataset_path, word2vec, clean=False)
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size * unsup_bs_ratio, shuffle=True)

        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path, k)
        elif dataset == 'Weibo-self':
            sort_weibo_self_dataset(label_source_path, label_dataset_path, unlabel_dataset_path, k)
        elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
            sort_weibo_2class_dataset(label_source_path, label_dataset_path, k)

        train_dataset = WeiboDataset(train_path, word2vec)
        val_dataset = WeiboDataset(val_path, word2vec)
        test_dataset = WeiboDataset(test_path, word2vec)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = ResGCN(dataset=unlabel_dataset, hidden=args.hidden, num_feat_layers=args.n_layers_feat,
                       num_conv_layers=args.n_layers_conv, num_fc_layers=args.n_layers_fc, gfn=False,
                       collapse=False, residual=args.skip_connection, res_branch=args.res_branch,
                       global_pool=args.global_pool, dropout=args.dropout, edge_norm=args.edge_norm).to(device)
        view_gen1 = ViewGenerator(unlabel_dataset, args.hidden, GIN_NodeWeightEncoder, args.add_mask).to(device)
        view_gen2 = ViewGenerator(unlabel_dataset, args.hidden, GIN_NodeWeightEncoder, args.add_mask).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        view_optimizer = Adam([{'params': view_gen1.parameters()}, {'params': view_gen2.parameters()}],
                              lr=args.lr, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                       device, 0, args.lr, 0, 0, log_record)

        # write_log(log, 'pretrain:')
        write_log(log, log_info)
        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']

            unlabel_loss_all, label_loss_all, loss_all = train(model, view_gen1, view_gen2, unsup_train_loader,
                                                               train_loader, optimizer, view_optimizer, lamdas,
                                                               train_mode=args.train_mode)

            train_error, train_acc, _, _, _ = test(model, train_loader, device)
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                           device, epoch, lr, train_error, train_acc,
                                                           log_record)
            write_log(log, log_info)
            scheduler.step(val_error)

        # write_log(log, '\nfine_tuning:')
        # for epoch in range(1, ft_epochs + 1):
        #     lr = scheduler.optimizer.param_groups[0]['lr']
        #
        #     loss_all = fine_tuning(model, view_gen1, view_gen2, train_loader, optimizer)
        #
        #     train_error, train_acc, _, _, _ = test(model, train_loader, device)
        #     val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
        #                                                    device, epoch, lr, train_error, train_acc,
        #                                                    log_record)
        #     write_log(log, log_info)
        #     scheduler.step(val_error)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
