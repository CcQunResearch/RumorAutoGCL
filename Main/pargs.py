import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=20000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--ft_runs', type=int, default=5)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--unsup_bs_ratio', type=int, default=1)
    parser.add_argument('--n_layers_feat', type=int, default=1)
    parser.add_argument('--n_layers_conv', type=int, default=3)
    parser.add_argument('--n_layers_fc', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--skip_connection', type=str2bool, default=True)
    parser.add_argument('--res_branch', type=str, default="BNConvReLU")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--edge_norm', type=str2bool, default=True)

    parser.add_argument('--add_mask', type=str2bool, default=False)
    parser.add_argument('--train_mode', type=int, default=6)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ft_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lamda', type=float, default=0.01)
    parser.add_argument('--lamda_cl_unlabel', type=float, default=1e-2)
    parser.add_argument('--lamda_cl_label', type=float, default=1e-2)
    parser.add_argument('--lamda_sim', type=float, default=1e-2)
    parser.add_argument('--lamda_cls', type=float, default=1e-2)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args
