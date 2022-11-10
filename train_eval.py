import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        writer = SummaryWriter()
        trainer.train(log_writer=writer)
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test()
    auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
    print("-------------------------------------------------------\n\n")


if __name__ == '__main__':
    main()
