import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
from detectron2.layers import Conv2d, ModulatedDeformConv, DeformConv


def parse_dynamic_params(params, channels, weight_nums, bias_nums, last_out_channels):
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
            weight_splits[l] = weight_splits[l].reshape(num_insts * last_out_channels, -1, 1, 1) #1
            bias_splits[l] = bias_splits[l].reshape(num_insts * last_out_channels) #1

    return weight_splits, bias_splits


def build_dynamic_track_head(cfg):
    return DynamicTrackHead(cfg)



class DynamicTrackHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicTrackHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        #self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.track_embedding_dimension = cfg.MODEL.CONDINST.TRACK_EMBEDDOMG_DIMENSION
        self.channels = int(self.track_embedding_dimension/8) #8
        self.last_out_channels = self.track_embedding_dimension
        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST

        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * self.last_out_channels) #1
                bias_nums.append(self.last_out_channels) #1
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        #self.track_embedding_main_fc1 = nn.Linear(self.last_out_channels, self.track_embedding_dimension*2)
        #self.track_embedding_main_fc2 = nn.Linear(self.track_embedding_dimension*2, self.track_embedding_dimension)
        self.track_embedding_main_seq = nn.Sequential(
            #self.track_embedding_main_fc1,
            nn.Linear(self.last_out_channels, self.track_embedding_dimension*2),
            nn.LayerNorm(self.track_embedding_dimension*2),
            nn.SELU(),
            #nn.BatchNorm1d(self.track_embedding_dimension*2),
            #nn.ReLU(),
            nn.Linear(self.track_embedding_dimension*2, self.track_embedding_dimension)
            #self.track_embedding_main_fc2
        )
        for layer in self.track_embedding_main_seq:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        #self.track_embedding_side_fc1 = nn.Linear(self.last_out_channels, self.track_embedding_dimension*2)
        #self.track_embedding_side_fc2 = nn.Linear(self.track_embedding_dimension*2, self.track_embedding_dimension)
        self.track_embedding_side_seq = nn.Sequential(
            #self.track_embedding_side_fc1,
            nn.Linear(self.last_out_channels, self.track_embedding_dimension*2),
            nn.LayerNorm(self.track_embedding_dimension*2),
            nn.SELU(),
            #nn.BatchNorm1d(self.track_embedding_dimension*2),
            #nn.ReLU(),
            nn.Linear(self.track_embedding_dimension*2, self.track_embedding_dimension)
            #self.track_embedding_side_fc2
        )
        for layer in self.track_embedding_side_seq:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
        '''
        for index in range(self.num_layers):
            if index == self.num_layers - 1:
                self.add_module('dynamic_batchnorm_obj_{}'.format(index), nn.BatchNorm2d(self.last_out_channels))
                self.add_module('dynamic_batchnorm_sha_{}'.format(index), nn.BatchNorm2d(self.last_out_channels))
            else:
                self.add_module('dynamic_batchnorm_obj_{}'.format(index), nn.BatchNorm2d(self.channels))
                self.add_module('dynamic_batchnorm_sha_{}'.format(index), nn.BatchNorm2d(self.channels))
        '''
    def track_heads_forward(self, features, weights, biases, num_insts, flag):
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
            x = F.relu(x)
            # BatchNorm
            '''
            x = x.reshape(num_insts, -1, x.shape[-2], x.shape[-1])
            if i == 0:
                x = self.dynamic_batchnorm_obj_0(x) if flag == 0 else self.dynamic_batchnorm_sha_0(x)
            elif i == 1:
                x = self.dynamic_batchnorm_obj_1(x) if flag == 0 else self.dynamic_batchnorm_sha_1(x)
            else:
                x = self.dynamic_batchnorm_obj_2(x) if flag == 0 else self.dynamic_batchnorm_sha_2(x)
            x = x.reshape(1, -1, x.shape[-2], x.shape[-1]) 
            '''
            if i < n_layers - 1:
                m = nn.MaxPool2d(kernel_size=2, stride=2)
                x = m(x)
            
        return x

    def track_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances, gt_instances
    ):
        # max pooling on mask_feature map
        '''
        m = nn.MaxPool2d(kernel_size=2, stride=2)
        mask_feats = m(mask_feats)
        mask_feat_stride*=2
       '''
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params3
        asso_mask_head_params = instances.mask_head_params4
        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            '''
            if self.training:
                gt_offset = torch.cat([per_im.gt_relations for per_im in gt_instances])
                gt_offset = (gt_offset[instances.gt_inds]  - instances.locations).to(dtype=mask_feats.dtype)
                offset = gt_offset/128
            else: 
                offset = instances.offset_pred # N,2
            '''
            offset = instances.offset_pred.detach() # N,2
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
        
        # Main for Obj, Side for Sha
        if self.training:
            labels = instances.labels
        else:
            labels = instances.pred_classes
        #mask_head_inputs[labels == 1], asso_mask_head_inputs[labels==1] = asso_mask_head_inputs[labels==1], mask_head_inputs[labels == 1]

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        asso_mask_head_inputs = asso_mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums, self.last_out_channels
        )

        weights2, biases2 = parse_dynamic_params(
            asso_mask_head_params, self.channels,
            self.weight_nums, self.bias_nums, self.last_out_channels
        )

        main_track_logits = self.track_heads_forward(mask_head_inputs, weights, biases, n_inst, flag=0)
        side_track_logits = self.track_heads_forward(asso_mask_head_inputs, weights2, biases2, n_inst, flag=1)
        H = int(H/4) # MaxPooling Two Times
        W = int(W/4)
        main_track_logits = main_track_logits.reshape(-1, self.last_out_channels, H, W)
        side_track_logits = side_track_logits.reshape(-1, self.last_out_channels, H, W)
       
        main_track_logits = torch.mean(main_track_logits, (2,3), True).reshape(-1, self.last_out_channels)
        side_track_logits = torch.mean(side_track_logits, (2,3), True).reshape(-1, self.last_out_channels)

        main_track_logits = self.track_embedding_main_seq(main_track_logits)
        side_track_logits = self.track_embedding_side_seq(side_track_logits)

        #main_track_logits[labels == 1], side_track_logits[labels==1] = side_track_logits[labels==1], main_track_logits[labels == 1] 
        return main_track_logits, side_track_logits


    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances):
        if len(pred_instances) == 0:
            return [], []
        else:
            main_track_logits, side_track_logits = self.track_heads_forward_with_coords(
                mask_feats, mask_feat_stride, pred_instances, gt_instances
            )

        return main_track_logits, side_track_logits

