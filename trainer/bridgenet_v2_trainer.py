import torch
import hydra

import lightning.pytorch as pl

from models.modules.utils import random_scale_point_cloud, shift_point_cloud, rotate_point_cloud_z

class SemanticSegmentation(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        model_cfg = config.models.base[config.general.backbone]
        self.model = hydra.utils.instantiate(model_cfg)
        self.criterion = hydra.utils.instantiate(self.config.loss)

        self.data_aug_flag = self.config.general.data_aug_flag
        self.data_aug_dict = self.config.general.data_aug_dict
        self.n_cls = self.config.general.n_cls

        semseg_metrics = hydra.utils.instantiate(self.config.metrics)
        self.train_semacc = semseg_metrics.clone(prefix='train_')
        self.test_semacc = semseg_metrics.clone(prefix='test_')
    
    def forward(self, xyz):
        return self.model(xyz)
    
    def training_step(self, batch, batch_idx):
        xyz, target = batch
        B, N, _ = xyz.shape
        if len(xyz.shape) != 3:
            raise Exception('The dimension of points cloud has to be 3!')
        if self.config.trainer["precision"] == '32-true':
            xyz = xyz.to(torch.float32)
        target = target.long()
        # Data augmentation
        if self.data_aug_flag:
            xyz[:, :, 0:3] = random_scale_point_cloud(xyz[:, :, 0:3], 
                scale_low=self.data_aug_dict['scale'][0],
                scale_high=self.data_aug_dict['scale'][1])  
            xyz[:, :, 0:3] = shift_point_cloud(xyz[:, :, 0:3],
                shift_range=self.data_aug_dict['shift'])    
            xyz[:, :, 0:3] = rotate_point_cloud_z(xyz[:, :, 0:3], 
                angle=self.data_aug_dict['rotation']) 
        # Model
        try:
            seg_pred = self.model(xyz)
        except:
            raise Exception('Model is run incorrectly in training mode!')
        seg_pred = seg_pred.reshape(-1, self.n_cls)       # resize the output to [n, num_part]
        target = target.reshape(-1, 1)[:, 0]              # the correct part label
    
        # Back propagation
        try:
            loss = self.criterion(seg_pred, target)      # loss function
        except:
            print('The data shape is : {}'.format(xyz.shape))
            print('The data feature shape is : {}'.format(seg_pred.shape))
            print('The target shape is : {}'.format(xyz.shape))
            raise Exception('Losss is not returned!')
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Change the lr rate every single 20 epoch
        sch = self.lr_schedulers()
    
        # step every N batches
        if (batch_idx + 1) % 128 == 0:
            sch.step()
        
        self.log_dict(
            self.train_semacc(seg_pred, target), on_step=False, on_epoch=True
            )   # only output log in the end of the epoch
        return loss

    def validation_step(self, batch, batch_idx):
        xyz, target = batch
        B, N, _ = xyz.shape
        if len(xyz.shape) != 3:
            raise Exception('The dimension of points cloud has to be 3!')
        if self.config.trainer["precision"] == '32-true':
            xyz = xyz.to(torch.float32)
        target = target.long()
        
        # Model
        try:
            seg_pred = self.model(xyz)
        except:
            raise Exception('Model is run incorrectly in test mode!')
        seg_pred = seg_pred.reshape(-1, self.n_cls)       # resize the output to [n, num_part]
        target = target.reshape(-1, 1)[:, 0]              # the correct part label

        # Back propagation
        try:
            loss = self.criterion(seg_pred, target)       # loss function
        except:
            print('The data shape is : {}'.format(xyz.shape))
            print('The data feature shape is : {}'.format(seg_pred.shape))
            print('The target shape is : {}'.format(xyz.shape))
            raise Exception('Losss is not returned!')
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.test_semacc(seg_pred, target), on_step=False, on_epoch=True
            )   # only output log in the end of the epoch
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        
        return 0

    def configure_optimizers(self):
        if self.config:
            optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())
            lr_scheduler = hydra.utils.instantiate(self.config.scheduler.scheduler, 
                                                optimizer=optimizer)
        else:
            optimizer = hydra.utils.instantiate(self.config.optimizer, 
                                    lr=self.learning_rate, params=self.parameters())
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                }
            }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms = grad_norm(self.model, norm_type=2)
        # print(norms)
        # self.log("grad_2nd_norm", norms, on_step=True, on_epoch=True)

        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            total_norm = 0.0
        else:
            device = parameters[0].grad.device
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
        self.log("grad_2nd_norm", total_norm, on_step=True, on_epoch=True)

    def _load_checkpoint(self, ckpt_fp):
        '''
        For loading weights of the network only. The default lightmodule loading
        method requries consistencies on the parameters of the pl module.
        If a new param is defined or its name is modified. Even the structure 
        of the network is the same, the weight loading still failed.
        '''
        # load weight
        checkpoint = torch.load(ckpt_fp)
        # print(self.model)
        if self.config.general.backbone=='bridgenetv2':
            backbone_weight = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items() 
                                if k.startswith('model.')}
            # print(backbone_weight.keys())
            try:
                print('Load BridgeNetv2 weight!')
                self.model.load_state_dict(backbone_weight)
                print('Success loading backbone weights.')
            except:
                raise Exception('Failed loading weights.')
        
        else:
            raise Exception('Loading weights module has not been implemented!')
        return 0
    