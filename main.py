import os
import hydra

from omegaconf import DictConfig
from lightning.pytorch import seed_everything

from trainer.bridgenet_v2_trainer import SemanticSegmentation
from data_utils.DataLoader_pl import BridgeDataLoader

@hydra.main(version_base=None, config_path="conf", config_name="bridgenet_params")
def train(cfg: DictConfig):
    save_dir = os.path.join(cfg.general.default_root_dir, cfg.general.n_training)
    os.makedirs(save_dir, exist_ok=True)
    
    seed_everything(cfg.general.seed, workers=True)   
    model = SemanticSegmentation(config=cfg)
    callbacks = []
    for cb in cfg.callback:
        callbacks.append(hydra.utils.instantiate(cb))
    
    seg_bridge_dataset = BridgeDataLoader(cfg)
    seg_bridge_dataset.setup(stage='fit')
    trainer = hydra.utils.instantiate(config=cfg.trainer, 
                                      callbacks=callbacks, deterministic=False, 
                                      default_root_dir=save_dir, 
                                      check_val_every_n_epoch=5)    
    trainer.fit(model=model, train_dataloaders=seg_bridge_dataset)

if __name__ == '__main__':
    train()