import lightning.pytorch as pl

import hydra

class BridgeDataLoader(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == 'fit':
            try:
                self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
                self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
            except:
                raise Exception('Train dataset is not loaded sucessfully!')
        # elif stage == 'test':
        #     try:
        #         self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
        #     except:
        #         raise Exception('Test dataset is not loaded sucessfully!')
        else:
            raise Exception('stage should be either "train" or "test".')

    def train_dataloader(self):
        return hydra.utils.instantiate(self.config.data.train_dataloader, 
                                       dataset=self.train_dataset)
                                       
    def val_dataloader(self):
        return hydra.utils.instantiate(self.config.data.test_dataloader, 
                                       dataset=self.test_dataset)



