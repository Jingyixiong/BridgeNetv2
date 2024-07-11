import os
import hydra
import torch
import time

from omegaconf import DictConfig
from lightning.pytorch import seed_everything

from trainer.bridgenet_v2_trainer import SemanticSegmentation

@hydra.main(version_base=None, config_path="conf", config_name="bridgenet_params")
def inference(cfg: DictConfig):
    model_p = os.path.join(cfg.general.default_root_dir, cfg.general.n_training, 'lightning_logs', cfg.general.version_n, 
                           'checkpoints', cfg.general.best_model_n)
    seed_everything(cfg.general.seed, workers=True) 

    # load model
    cfg.general.data_aug_flag = False
    model = SemanticSegmentation(config=cfg).to('cuda')
    model._load_checkpoint(model_p)
    model.eval()

    # load dataset
    test_dataset  = hydra.utils.instantiate(cfg.data.test_dataset)
    test_dataloader = hydra.utils.instantiate(cfg.data.test_dataloader, dataset=test_dataset, 
                                               batch_size=cfg.general.infer_bsize)
    print('The total size of the point cloud is: {}'.format(len(test_dataset)))
    
    # load metrics
    semseg_metrics = hydra.utils.instantiate(cfg.metrics).to('cuda')
    start = time.time()
    with torch.no_grad():
        for _, (xyzs, target) in enumerate(test_dataloader):
            B, N, dim = xyzs.shape
            xyzs = xyzs.to(torch.float32).to('cuda')
            target = target.long().to('cuda')
            seg_pred = model(xyzs)
            seg_pred = seg_pred.reshape(-1, cfg.general.n_cls)       # resize the output to [n, num_part]
            target = target.reshape(-1, 1)[:, 0]                     # the correct part labe
            _, point_cls = torch.max(seg_pred, dim=-1)
            semseg_metrics.update(seg_pred, target)
    end = time.time()
    acc = semseg_metrics.compute()
    print('The accuracy is: {}'.format(acc))
    print('Total time consumptions are: {}s'.format(end-start))

if __name__ == '__main__':
    inference()