import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
from detectron2.layers import Conv2d, ModulatedDeformConv, DeformConv

from fvcore.nn import smooth_l1_loss


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss



def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)



class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        stride = cfg.MODEL.FCOS.FPN_STRIDES

        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.register_buffer("strides", torch.tensor(stride,dtype=torch.float32))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        def convert_locations(relative_coords,H,W):
            coords = relative_coords.permute(0,2,1).reshape(-1,H,W,2)
            x = torch.abs(coords).sum(-1).min(1)[1][:,0:1]
            y = torch.abs(coords).sum(-1).min(2)[1][:,0:1]
            h = torch.ones_like(x)*H
            w = torch.ones_like(y)*W
            return torch.cat([x,y,h,w],dim=1)
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params
        asso_mask_head_params = instances.mask_head_params2

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations 
            offset = instances.offset_pred # N,2
            
            asso_instance_locations = instances.locations  +  offset * 128

            asso_relative_coords = asso_instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            
            asso_relative_coords = asso_relative_coords.permute(0,2,1).float()
            relative_coords = relative_coords.permute(0, 2, 1).float()
            
            soi = self.sizes_of_interest.float()[instances.fpn_levels]

            asso_relative_coords = asso_relative_coords / soi.reshape(-1,1,1) #shape: N,2,HW
            relative_coords = relative_coords / soi.reshape(-1, 1, 1) 

            asso_relative_coords = asso_relative_coords.to(dtype=mask_feats.dtype)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            asso_mask_head_inputs = torch.cat([
                asso_relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        asso_mask_head_inputs = asso_mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        weights2, biases2 = parse_dynamic_params(
            asso_mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        asso_mask_logits = self.mask_heads_forward(asso_mask_head_inputs, weights2, biases2, n_inst)
        
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        asso_mask_logits = asso_mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        asso_mask_logits = aligned_bilinear(asso_mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.sigmoid(), asso_mask_logits.sigmoid()

    def find_gt_relation_masks(self, gt_instances):
        gt_relations_masks = []
        for per_im in gt_instances:
            half_length = int(len(per_im.gt_bitmasks) / 2)
            gt_relations_masks.append(per_im.gt_bitmasks[half_length:])
            gt_relations_masks.append(per_im.gt_bitmasks[:half_length])
        return torch.cat(gt_relations_masks)
    
    def get_target_iou(self, mask,gt):
        mask = (mask>0.5).float()
        mask_area = mask.sum((-1,-2))
        overlap = (mask*gt).sum((-1,-2))
        gt_area = gt.sum((-1,-2))
        return overlap / (mask_area + gt_area - overlap + 1e-7)

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_asso_masks = self.find_gt_relation_masks(gt_instances)
            gt_asso_masks = gt_asso_masks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            gt_offset = torch.cat([per_im.gt_relations for per_im in gt_instances])
            gt_offset = (gt_offset[gt_inds]  - pred_instances.locations).to(dtype=mask_feats.dtype)

            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                loss_asso_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params2.sum() * 0 
                asso_offset_losses = pred_instances.offset_pred.sum() * 0
            else:
            
                mask_scores, asso_mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                
                asso_masks_losses =  dice_coefficient(asso_mask_scores,gt_asso_masks )

                asso_offset_losses = smooth_l1_loss(pred_instances.offset_pred,gt_offset/128,0.5,reduction='mean')

                loss_mask = mask_losses.mean() 

                loss_asso_mask = asso_masks_losses.mean()

            return loss_mask.float(),loss_asso_mask.float(), asso_offset_losses
        else:
            if len(pred_instances) > 0:
                mask_scores,asso_mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_scores.float()
                pred_instances.pred_asso_global_masks = asso_mask_scores.float()
                # Maskness 20221009
                #'''
                mask_preds = mask_scores > 0.3
                sum_masks = mask_preds.sum((1, 2, 3)).float()
                seg_scores = (mask_scores * mask_preds.float()).sum((1, 2, 3)) / sum_masks
                pred_instances.scores = pred_instances.scores*seg_scores
                #'''

            '''
            from datetime import datetime
            import os
            sigmoid = nn.Sigmoid()
            import torchvision.transforms as transforms
            new_dir = './test_SSIS/' + str(datetime.now())
            os.mkdir(new_dir)
            for index, mask_feat in enumerate(mask_scores):
                main = transforms.ToPILImage()(sigmoid(mask_feat.squeeze(0)))
                asso = transforms.ToPILImage()(sigmoid(asso_mask_scores[index].squeeze(0)))
                main.save(new_dir + "/" + str(index) + "_main" + '.png')
                asso.save(new_dir + "/" + str(index) + "_asso" + '.png')
            '''

            return pred_instances
