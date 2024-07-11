import torch

from torch import Tensor
from torchmetrics import Metric, MetricCollection

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision

class Semantic_Acc(Metric):
    '''
    It measure the overall accuracy of semantic segmentation.
    Input:
        predict: [B, N]
        target: [B, N]
    '''
    def __init__(self, ):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, predict, target):
        assert predict.shape == target.shape

        self.correct += torch.sum(predict==target)
        self.total += target.numel()

    def compute(self):
        acc = self.correct/self.total
        return acc

class Semantic_ClsAvgAcc(Metric):
    '''
    It measures the mean precision of classes based on single bridge rather than batch.
    '''
    full_state_update: bool = True
    def __init__(self, n_cls):
        super().__init__()
        self.n_cls = n_cls
        # accumulated average class accruacy of batches
        self.add_state('cls_acc_predict', default=torch.zeros(n_cls), dist_reduce_fx='sum')   
        self.add_state('cls_acc_target', default=torch.zeros(n_cls), dist_reduce_fx='sum')   
        # accumulated valid number of each class of batches
        self.add_state('valid_num_1', default=torch.zeros(n_cls), dist_reduce_fx='sum')     
    
    def update(self, predict, target):
        assert predict.shape == target.shape
        B, _ = predict.shape
        for i in range(B):
            batch_predict = predict[i, :]
            batch_target = target[i, :]

            correct_cls_ps = torch.tensor([torch.sum((batch_predict==idx)&(batch_target==idx)) 
                                           for idx in range(self.n_cls)]).to(self.device)
            target_cls_ps = torch.tensor([torch.sum(batch_target==idx) 
                                          for idx in range(self.n_cls)]).to(self.device)
            self.cls_acc_predict += correct_cls_ps
            self.cls_acc_target += target_cls_ps
            self.valid_num_1 += torch.tensor([1 if target_cls_ps[idx] != 0 else 0 for idx in range(self.n_cls)]).to(self.device) 

    def compute(self):
        valid_cls = torch.tensor([1 if self.valid_num_1[i]!= 0 else 0 for i in range(self.n_cls)]).sum()
        total_avg_cls_acc = torch.tensor([self.cls_acc_predict[i]/self.cls_acc_target[i] 
                                          if self.valid_num_1[i]!= 0 
                                          else 0 for i in range(self.n_cls)]).sum()/valid_cls
        return total_avg_cls_acc

class Semantic_ClsIOU(Metric):
    '''
    It measures the mean IOU of classes based on single bridge rather than batch.
    '''
    full_state_update: bool = True
    def __init__(self, n_cls):
        super().__init__()
        
        self.n_cls = n_cls
        self.add_state('cls_iou_inter', default=torch.zeros(n_cls), dist_reduce_fx='sum')     # the sum is performed at dim=0 for multi-processing in the synchronize stage
        self.add_state('cls_iou_union', default=torch.zeros(n_cls), dist_reduce_fx='sum')  
        self.add_state('valid_num_2', default=torch.zeros(n_cls), dist_reduce_fx='sum')     # indicate if there exists points of certain class
    
    def update(self, predict, target):
        assert predict.shape == target.shape
        B, _ = predict.shape

        intersection = torch.zeros(self.n_cls).to(self.device)
        union = torch.zeros(self.n_cls).to(self.device)
        valid_cls_num = torch.zeros(self.n_cls).to(self.device)

        for i in range(B):
            batch_predict = predict[i, :]
            batch_target = target[i, :]
            for l in range(self.n_cls):
                if (torch.sum(batch_predict == l) == 0) and (torch.sum(batch_target == l) == 0):             # part is not present, no prediction as well
                    continue
                else:
                    intersection = torch.sum((batch_predict==l)&(batch_target==l))
                    union = torch.sum((batch_predict==l)|(batch_target==l))
                    self.cls_iou_inter[l] += intersection
                    self.cls_iou_union[l] += union
                    valid_cls_num[l] += 1 
        self.valid_num_2 += valid_cls_num

    def compute(self):
        valid_cls = torch.tensor([1 if self.valid_num_2[i]!= 0 else 0 for i in range(self.n_cls)]).sum()
        total_avg_cls_iou = torch.tensor([self.cls_iou_inter[i]/self.cls_iou_union[i] 
                                          if self.valid_num_2[i]!= 0 
                                          else 0 for i in range(self.n_cls)]).sum()/valid_cls
        return total_avg_cls_iou
    
class Semantic_ClsIOU_official(Metric):
    '''
    It measures the mean IOU of classes which is re-implemented based on 
    official package on torchmetrics.
    Input: 
        predict: [N, C, ...], logit output which is transformed to label by torch.max
        target: [N, ...]
    '''
    def __init__(
        self,
        num_classes: int,
        per_class: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.per_class = per_class

        self.add_state("score", default=torch.zeros(num_classes if per_class else 1), dist_reduce_fx="mean")

    def update(self, seg_pred: Tensor, target: Tensor) -> None:
        """Update the state with the new data."""
        _, preds = torch.max(seg_pred, dim=-1, keepdim=False)          # [B*N,]
        intersection, union = self._mean_iou_update(
            preds, target, self.num_classes,
        )
        score = self._mean_iou_compute(intersection, union, per_class=self.per_class)
        self.score += score.mean(0) if self.per_class else score.mean()

    def compute(self) -> Tensor:
        """Update the state with the new data."""
        return self.score  # / self.num_batches
    
    def _mean_iou_update(
        self,
        preds: Tensor,
        target: Tensor,
        num_classes: int,
        ):
        """Update the intersection and union counts for the mean IoU computation."""
        # transfer it into one-hot labels
        # preds = torch.nn.functional.one_hot(
        #     preds, num_classes=num_classes)
        preds = torch.nn.functional.one_hot(
            preds, num_classes=num_classes
            ).movedim(-1, 1)   # [B*N, C])
        target = torch.nn.functional.one_hot(
            target, num_classes=num_classes
            ).movedim(-1, 1)   # [B*N, C]
        intersection = torch.sum(preds & target, dim=0)
        target_sum = torch.sum(target, dim=0)
        pred_sum = torch.sum(preds, dim=0)
        union = target_sum + pred_sum - intersection
        return intersection, union
    
    def _mean_iou_compute(
        self,
        intersection: Tensor,
        union: Tensor,
        per_class: bool = False,
    ) -> Tensor:
        """Compute the mean IoU metric."""
        val = self._safe_divide(intersection, union)
        return val if per_class else torch.mean(val)
    
    def _safe_divide(self, 
                     num: Tensor, 
                     denom: Tensor, 
                     zero_division: float = 0.0) -> Tensor:
        """Safe division, by preventing division by zero.

        Function will cast to float if input is not already to secure backwards compatibility.

        Args:
            num: numerator tensor
            denom: denominator tensor, which may contain zeros
            zero_division: value to replace elements divided by zero
        """
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        zero_division = torch.tensor(zero_division).float().to(num.device)
        return torch.where(denom != 0, num / denom, zero_division)

def semseg_metrics_self_defined(n_cls):
    return  MetricCollection({'acc': Semantic_Acc(),
                              'cls_acc': Semantic_ClsAvgAcc(n_cls),
                              'cls_iou': Semantic_ClsIOU(n_cls)})

def semseg_metrics(n_cls, per_class=False):
    # seg_pred: [N, C, ...], target: [N, ...]
    return  MetricCollection(
        {
        'acc': MulticlassAccuracy(
            num_classes=n_cls, 
            top_k=1,
            average='micro', 
            ),       
        'cls_acc': MulticlassPrecision(
            num_classes=n_cls, 
            top_k=1, 
            zero_division=0
            ),    # set zero_division=1 so that the missing class in target will not decrease the accuracy
        'cls_iou': Semantic_ClsIOU_official(
            num_classes=n_cls, 
            per_class=per_class
            )
        }
        )
