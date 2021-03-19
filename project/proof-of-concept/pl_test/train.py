import os
import time
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data

from data import Random_dict_Dataset
from network import graph_model, dict_model


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Process data for training')
    ap.add_argument('--network', type=str, required=False, default='dict_model')
    ap.add_argument('--out_path', type=str, required=False,
                    default='output')
    ap.add_argument('--num_gpus', type=int, required=False, default=1)
    ap.add_argument('--num_workers', type=int, required=False, default=4)
    ap.add_argument('--test_size', type=int, required=False, default=100)
    ap.add_argument('--epochs', type=int, required=False, default=25)
    ap.add_argument('--time_limit', type=int, required=False, default=7200)
    ap.add_argument('--batch_size', type=int, required=False, default=16)
    ap.add_argument('--test_percent_check', type=float, required=False, default=1.0)
    ap.add_argument('--test_seed', type=int, required=False, default=None)

    args = ap.parse_args()
    network = args.network
    out_path = args.out_path
    num_gpus = args.num_gpus
    test_size = args.test_size
    num_workers = args.num_workers
    epochs = args.epochs
    time_limit = args.time_limit
    batch_size = args.batch_size
    test_percent_check = args.test_percent_check
    test_seed = args.test_seed

    pl.utilities.seed.seed_everything(seed=test_seed)
    train_dataset = Random_dict_Dataset(test_size)
    val_dataset = Random_dict_Dataset(test_size//5)
    train_loader = data.DataLoader(train_dataset, num_workers=num_workers, batch_size=1)
    val_loader = data.DataLoader(val_dataset, num_workers=num_workers, batch_size=1)

    start_time = time.time()

    test_model = globals()[network](start_time=start_time, time_limit=time_limit)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=out_path,
        filename='sgcns-{epoch:02d}-{val_loss:.2f}',
        save_top_k=10,
        mode='min',
        save_last=True
    )
    logger = TensorBoardLogger(out_path, name="log")

    trainer = pl.Trainer(gpus=num_gpus, max_epochs=epochs,
                         accumulate_grad_batches=batch_size,
                         default_root_dir=out_path,
                         distributed_backend='ddp',
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         limit_train_batches=test_percent_check,
                         limit_val_batches=test_percent_check,
                         resume_from_checkpoint=os.path.join(out_path, 'last.ckpt')
                         )

    time1 = time.time()
    trainer.fit(test_model, train_loader, val_loader)
    time2 = time.time()
    print('{} epochs takes {} seconds using {} GPUs.'.format(epochs, time2 - time1, num_gpus))
