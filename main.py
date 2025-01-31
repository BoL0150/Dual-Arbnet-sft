import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
from thop import profile
from thop import clever_format
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args)                ## data loader
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


