
import argparse

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from model import MRI_LSTM

def get_args():
    
    parser = argparse.ArgumentParser()
    
    # Model param
    parser.add_argument("--n_layers",   type=int, default=1,   help="num of layers in LSTM")
    parser.add_argument("--latent_dim", type=int, default=128, help="latent dim LSTM")
    parser.add_argument("--feat_embed_dim", type=int, default=2, help="CNN output/LSTM input dim")

    # Data param
    parser.add_argument("--train_data", type=str,  default=None, help="path to train data")
    parser.add_argument("--valid_data", type=str,  default=None, help="path to valid data")
    parser.add_argument("--test_data",  type=str,  default=None, help="path to test data")
    parser.add_argument("--path_col", type=str, default='9dof_2mm_vol', help="Path col for reading data")

    # Train param
    parser.add_argument("--batch_size",  type=int,   default=8,   help="# of scans/patch")
    parser.add_argument("--momentum",    type=float, default=0.5,  help="Nesterov momentum value")
    parser.add_argument("--epochs",      type=int,   default=400,  help="# of epochs to train for")
    parser.add_argument("--init_lr",     type=float, default=1e-4, help="start learning rate value")
    
    parser.add_argument("--lr_patience",    type=int, default=20,  help="# epochs to wait before lr")
    parser.add_argument("--train_patience", type=int, default=50, help="# epochs to wait before early stop")

    # Systm/Monitoring param
    parser.add_argument("--gpu", type=int,  default=1, help="GPU idx")
    parser.add_argument("--num_workers", type=int, default=4, help="# of subprocesses for data loading")
    
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="dir to save model checkpoints")
    parser.add_argument("--model_path", type=str, default=None, help="load model from checkpoint")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    earlystopper = EarlyStopping(monitor="val_loss", patience=100, verbose=3, mode="min")

    checkpointer = ModelCheckpoint(
        filepath=args.ckpt_dir + "/weights.{epoch:02d}-{val_loss:.8f}.h5", 
        save_top_k=args.epochs - 1, monitor="epoch")

    model = MRI_LSTM(args)
    
    trainer = Trainer(
        min_epochs=10, max_epochs=args.epochs,  gpus=[args.gpu], 
        num_nodes=1, gradient_clip_val=1,
        distributed_backend="dp", checkpoint_callback=checkpointer, 
        early_stop_callback=earlystopper)
    
    if args.model_path is not None:
        trainer = Trainer(
            resume_from_checkpoint=args.model_path, min_epochs=10, 
            max_epochs=args.epochs, 
            gpus=[args.gpu], num_nodes=1, gradient_clip_val=1,
            distributed_backend="dp", checkpoint_callback=checkpointer, 
            early_stop_callback=earlystopper)
            
    trainer.fit(model)
    trainer.test(model, ckpt_path="best")
