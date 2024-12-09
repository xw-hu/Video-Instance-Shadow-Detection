# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .dynamic_mask_head import build_dynamic_mask_head
from .dynamic_track_head import build_dynamic_track_head
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear
import pysobatools.sobaeval as eval
from detectron2.structures import Boxes, BoxMode
import numpy as np

__all__ = ["CondInst"]

logger = logging.getLogger(__name__)


def decode(segm):
    return eval.maskUtils.decode(segm).astype('uint8')


def encode(segm):
    return eval.maskUtils.encode(segm)


@META_ARCH_REGISTRY.register()
class CondInst(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.track_head = build_dynamic_track_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        # SSIS_v1
        self.track_embedding_dimension = cfg.MODEL.CONDINST.TRACK_EMBEDDOMG_DIMENSION

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module
        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )

        self.controller2 = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )

        # SSIS_v1, track_embedding_main
        self.track_controller_main = nn.Conv2d(
            in_channels, self.track_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )

        # SSIS_v1, track_embedding_side
        self.track_controller_side = nn.Conv2d(
            in_channels, self.track_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )


        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        torch.nn.init.normal_(self.controller2.weight, std=0.01)
        torch.nn.init.constant_(self.controller2.bias, 0)

        # SSIS_v1, track_embedding_main
        torch.nn.init.normal_(self.track_controller_main.weight, std=0.01)
        torch.nn.init.constant_(self.track_controller_main.bias, 0)

        # SSIS_v1, track_embedding_side
        torch.nn.init.normal_(self.track_controller_side.weight, std=0.01)
        torch.nn.init.constant_(self.track_controller_side.bias, 0)


        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        # Visualize Feature Map
        '''
        print(mask_feats.shape)
        from datetime import datetime
        sigmoid = nn.Sigmoid()
        import torchvision.transforms as transforms
        for index, mask_feat in enumerate(mask_feats[0]):
            img1 = transforms.ToPILImage()(sigmoid(mask_feat))
            img1.save('./test_SSIS/'+str(datetime.now())+"_"+str(index)+'.png')
        '''

        # SSIS_v1, where we generate tracking embedding
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, self.controller, self.controller2, self.track_controller_main, self.track_controller_side
        )
        if self.training:
            loss_mask, loss_asso_mask, asso_offset_losses = self._forward_mask_heads_train(proposals, mask_feats,
                                                                                           gt_instances)

            track_embedding_main, track_embedding_side = self._forward_track_heads_generate(proposals, mask_feats, gt_instances)
            track_losses = self._forward_track_heads_train(proposals, gt_instances, track_embedding_main, track_embedding_side)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update({"loss_mask": loss_mask})
            losses.update({"loss_asso_mask": loss_asso_mask})
            losses.update({"asso_offset_loss": asso_offset_losses})
            # SSIS_v1, new added losses
            losses.update(track_losses)

            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)
            padded_im_h, padded_im_w = images.tensor.size()[-2:]
            processed_results = []
            processed_associations = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                with torch.no_grad():
                    instances_per_im, pred_associations = self.postprocess(
                        instances_per_im, height, width,
                        padded_im_h, padded_im_w
                    )
                #print(instances_per_im.scores)
                #print(instances_per_im.ctrness)
                processed_results.append({
                    "instances": instances_per_im
                })
                processed_associations.append({
                    "instances": pred_associations
                })

            return processed_results, processed_associations

    # SSIS_v1 Contrast SEPA, Main and Side Heads, version 11
    def _forward_track_heads_train(self, proposals, gt_instances, track_embedding_main, track_embedding_side):
        loss_center_l1 = nn.SmoothL1Loss(reduction='mean')
        loss_center_contra = nn.CrossEntropyLoss(reduction='mean')
        softmax = nn.Softmax(dim=1).float()

        if 0 <= self.max_proposals < len(proposals["instances"]):
            inds = torch.randperm(len(proposals["instances"]), device=track_embedding_main.device).long()
            gt_instances = gt_instances[:self.max_proposals]
            proposals["instances"] = proposals["instances"][inds[:self.max_proposals]]

        gt_inds_distinct = sorted(list(set(proposals["instances"].gt_inds.tolist())))
        ##### Test Object Shadow Associations
        # print(gt_inds_distinct)
        # for gt_instance in gt_instances:
        #     print(gt_instance.gt_classes)
        # for i in gt_inds_distinct:
        #     print(i, proposals["instances"].gt_inds.tolist().count(i))
        #     print(proposals["instances"].labels[proposals["instances"].gt_inds == i])

        # equation 2 in "Learning to Track Instances without Video Annotations"
        # For Object
        center_representations_all_obj_ = []
        center_representations_all_obj_index_ = []
        center_representations_all_sha_ = []
        center_representations_all_sha_index_ = []
        center_loss_sum = 0

        accumulate_gt_index = 0
        obj_match_count = 0
        sha_match_count = 0

        for gt_instance in gt_instances:
            for gt_index in range(len(gt_instance.gt_classes)):
                real_gd_index = accumulate_gt_index + gt_index
                # If there's a prediction that match ground truth (Sometime no !!!)
                if real_gd_index not in gt_inds_distinct: continue
                # For Object
                if gt_index < len(gt_instance.gt_classes)/2:
                    obj_match_count+=1
                    instance_representations_obj = track_embedding_main[proposals["instances"].
                                                                           gt_inds == real_gd_index].float()
                    center_representation_obj = torch.mean(instance_representations_obj, 0)
                    center_representations_all_obj_.append(center_representation_obj)
                    center_representations_all_obj_index_.append(real_gd_index)
                    center_representation_repeat_obj = center_representation_obj.expand(len(instance_representations_obj),
                                                                                -1).float()

                    # Add Relation Info
                    '''
                    ######################
                    relative_gd_index = real_gd_index + int(len(gt_instance.gt_classes)/2)
                    if relative_gd_index in gt_inds_distinct:
                        obj_side_match_count+=1
                        instance_representations_obj_side = track_embedding_side[proposals["instances"].
                                                                                    gt_inds == relative_gd_index].float()
                        instance_representations_obj = torch.cat(
                            (instance_representations_obj, instance_representations_obj_side), 0)
                    '''

                    center_loss_l1_obj = \
                        loss_center_l1(instance_representations_obj, center_representation_repeat_obj.detach())
                    center_loss_sum += center_loss_l1_obj
                # For Shadow
                if gt_index >= len(gt_instance.gt_classes)/2:
                    sha_match_count+=1
                    instance_representations_sha = track_embedding_main[proposals["instances"].
                                                                           gt_inds == real_gd_index].float()
                    center_representation_sha = torch.mean(instance_representations_sha, 0)
                    center_representations_all_sha_.append(center_representation_sha)
                    center_representations_all_sha_index_.append(real_gd_index)
                    center_representation_repeat_sha = center_representation_sha.expand(len(instance_representations_sha),
                                                                                -1).float()
                    # Add Relation Info
                    '''
                    ######################
                    relative_gd_index = real_gd_index - int(len(gt_instance.gt_classes)/2)
                    if relative_gd_index in gt_inds_distinct:
                        sha_side_match_count+=1
                        instance_representations_sha_side = track_embedding_side[proposals["instances"].
                                                                                    gt_inds == relative_gd_index].float()
                        instance_representations_sha = torch.cat(
                            (instance_representations_sha, instance_representations_sha_side), 0)
                    '''
                    center_loss_l1_sha = \
                        loss_center_l1(instance_representations_sha, center_representation_repeat_sha)
                    center_loss_sum += center_loss_l1_sha

            accumulate_gt_index+=len(gt_instance.gt_classes)

        # For Object
        if obj_match_count>0:
            # equation 4 in "Learning to Track Instances without Video Annotations"
            center_representation_obj_all = torch.stack(center_representations_all_obj_).float()

            S_matrix_obj = torch.matmul(center_representation_obj_all,
                                         torch.transpose(center_representation_obj_all, 0, 1)).float()

            I_matrix_obj = [i for i in range(center_representation_obj_all.shape[0])]
            I_matrix_obj = torch.tensor(I_matrix_obj).to(self.device).long()
            center_loss_contra_obj = loss_center_contra(S_matrix_obj, I_matrix_obj)

            # equation 6 in "Learning to Track Instances without Video Annotations"
            S_matrix_obj_softmax = softmax(S_matrix_obj)
            maximum_entropy_loss_obj = torch.mean(
                (S_matrix_obj_softmax * torch.log(S_matrix_obj_softmax + 1e-20)).fill_diagonal_(0) * -1).float()

            maximum_entropy_loss_final_obj = maximum_entropy_loss_obj
        else:
            print("No Object Detection")
            center_loss_contra_obj = torch.tensor(0.0, requires_grad=True)
            maximum_entropy_loss_final_obj = torch.tensor(0.0, requires_grad=True)

        # For Shadow
        if sha_match_count>0:
            # equation 4 in "Learning to Track Instances without Video Annotations"
            center_representation_sha_all = torch.stack(center_representations_all_sha_).float()

            S_matrix_sha = torch.matmul(center_representation_sha_all,
                                        torch.transpose(center_representation_sha_all, 0, 1)).float()

            I_matrix_sha = [i for i in range(center_representation_sha_all.shape[0])]
            I_matrix_sha = torch.tensor(I_matrix_sha).to(self.device).long()
            center_loss_contra_sha = loss_center_contra(S_matrix_sha, I_matrix_sha)

            # equation 6 in "Learning to Track Instances without Video Annotations"
            S_matrix_sha_softmax = softmax(S_matrix_sha)
            maximum_entropy_loss_sha = torch.mean(
                (S_matrix_sha_softmax * torch.log(S_matrix_sha_softmax + 1e-20)).fill_diagonal_(0) * -1).float()

            maximum_entropy_loss_final_sha = maximum_entropy_loss_sha
        else:
            print("No Shadow Detection")
            center_loss_contra_sha = torch.tensor(0.0, requires_grad=True)
            maximum_entropy_loss_final_sha = torch.tensor(0.0, requires_grad=True)


        losses = {}
        center_loss_contra_final = (center_loss_contra_obj + center_loss_contra_sha)/2
        maximum_entropy_loss_final = (maximum_entropy_loss_final_obj+maximum_entropy_loss_final_sha)/2
        center_loss_l1_final = center_loss_sum/(obj_match_count+sha_match_count)

        losses.update({"center_l1_loss": center_loss_l1_final})
        losses.update({"contra_loss": center_loss_contra_final})
        losses.update({"MME_loss": maximum_entropy_loss_final})

        return losses

    def _forward_track_heads_generate(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        if self.training:
            pred_instances = proposals["instances"]
            if 0 <= self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                pred_instances = pred_instances[inds[:self.max_proposals]]

            pred_instances.mask_head_params3 = pred_instances.top_feats3
            pred_instances.mask_head_params4 = pred_instances.top_feats4

            track_embedding_main, track_embedding_side = self.track_head(
                mask_feats, self.mask_branch.out_stride,
                pred_instances, gt_instances
            )

        else:
            pred_instances = Instances.cat(proposals)
            pred_instances.mask_head_params3 = pred_instances.top_feat3
            pred_instances.mask_head_params4 = pred_instances.top_feat4

            track_embedding_main, track_embedding_side = self.track_head(
                mask_feats, self.mask_branch.out_stride,
                pred_instances, gt_instances
            )

        return track_embedding_main, track_embedding_side

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]
        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
            logger.info("clipping proposals from {} to {}".format(
                len(pred_instances), self.max_proposals
            ))
            pred_instances = pred_instances[inds[:self.max_proposals]]

        pred_instances.mask_head_params = pred_instances.top_feats
        pred_instances.mask_head_params2 = pred_instances.top_feats2

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat
        pred_instances.mask_head_params2 = pred_instances.top_feat2
        gt_instances = None
        track_embedding_main, track_embedding_side = self._forward_track_heads_generate(proposals, mask_feats, gt_instances)


        # Test Controller as tracking embedding
        # track_embedding_main = pred_instances.top_feat
        # track_embedding_side = pred_instances.top_feat2

        pred_instances.track_embedding_main = track_embedding_main
        pred_instances.track_embedding_side = track_embedding_side
        '''
        if len(pred_instances.pred_classes[pred_instances.pred_classes == 0]) > 0 and len(pred_instances.pred_classes[pred_instances.pred_classes == 1]) > 0:
            track_embedding = []
            for index, pred_class in enumerate(pred_instances.pred_classes):
                if pred_class == 0:
                    track_embedding.append(track_embedding_main[index])
                else:
                    track_embedding.append(track_embedding_side[index])
            track_embedding = torch.stack(track_embedding)
        else:
            track_embedding = track_embedding_main
        pred_instances.track_embedding = track_embedding  
        '''
        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        #print(pred_instances_w_masks.scores)
        #print(pred_instances_w_masks.ctrness_pred)
        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else:  # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """

        def mask_iou(mask1, mask2):

            n_1 = mask1.shape[0]
            n_2 = mask2.shape[0]
            iou = torch.zeros((n_1, n_2))
            for n_1, m_1 in enumerate(mask1):
                for n_2, m_2 in enumerate(mask2):
                    if n_1 == n_2:
                        continue
                    intersect = torch.logical_and(m_1, m_2).sum()
                    union = torch.logical_or(m_1, m_2).sum()
                    iou[n_1, n_2] = intersect.true_divide(union)
            return iou

        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size

        results = Instances((output_height, output_width), **results.get_fields())
        final_results, pred_associations = None, None
        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_asso_global_masks = aligned_bilinear(
                results.pred_asso_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_asso_global_masks = pred_asso_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )

            pred_asso_global_masks = F.interpolate(
                pred_asso_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            pred_asso_global_masks = pred_asso_global_masks[:, 0, :, :]
            pred_global_masks = (pred_global_masks > mask_threshold).float()
            asso_pred_masks = (pred_asso_global_masks > mask_threshold + 0.1).float()

            mask_ious = mask_iou(asso_pred_masks, pred_global_masks)
            pred_class = results.pred_classes.float().cpu()
            pred_asso_class = (1 - pred_class)

            class_map = torch.mm(pred_class[:, None], pred_asso_class[None]) + torch.mm(pred_asso_class[:, None],
                                                                                        pred_class[None])
            mask_ious *= class_map

            ious, inds = (mask_ious * (mask_ious > 0.5)).max(1)
            ious = ious != 0

            pairs = []
            record = {}
            for i, ind in enumerate(inds.numpy()):
                if ind not in record and i not in record and ious[i]:
                    record[ind] = i
                    record[i] = ind
                elif ind in record and ious[i]:
                    if results.scores[i] > results.scores[record[ind]]:
                        record.pop(record[ind])
                        record.pop(ind)
                        record[ind] = i
                        record[i] = ind
                elif i in record and ious[i]:
                    if results.scores[ind] > results.scores[record[i]]:
                        record.pop(record[i])
                        record.pop(i)
                        record[ind] = i
                        record[i] = ind

            for k, v in record.items():
                if k > v:
                    pairs.append((v, k))
                else:
                    pairs.append((k, v))
            pairs = list(set(pairs))

            pairs_inds = []
            pred_associations = []

            asso_mask = []
            asso_bbox = []
            asso_score = []
            asso_class = []
            asso_rela = []
            for i, pair in enumerate(pairs):
                pairs_inds += list(pair)
                pred_associations += [i + 1] * 2
                m, n = pair
                segm = (pred_global_masks[m] + pred_global_masks[n] > mask_threshold).float().cpu().numpy()
                asso_mask.append(segm)
                segm = segm[:, :, None]
                segm = encode(np.array(segm, order='F', dtype='uint8'))[0]
                asso_bbox.append(
                    BoxMode.convert(eval.maskUtils.toBbox(segm)[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)[0])
                asso_class.append(0)
                asso_score.append(((results.scores[m] + results.scores[n]) / 2.0).cpu().numpy())
                asso_rela.append(i + 1)

            # For Missing Object 20221001
            ################################
            un_paired_inds = []
            for i in range(pred_global_masks.shape[0]):
                if i not in pairs_inds:
                    un_paired_inds.append(i)
            un_paired_inds = torch.tensor(un_paired_inds).to(torch.int64)
            ###############################


            pairs_inds = torch.tensor(pairs_inds).to(torch.int64)
            pred_global_masks = pred_global_masks.cpu()
            final_results = Instances((output_height, output_width))
            final_results.pred_masks = pred_global_masks[pairs_inds]
            final_results.pred_classes = results.pred_classes[pairs_inds]
            final_results.ctrness = results.ctrness_pred[pairs_inds]

            final_results.pred_boxes = results.pred_boxes[pairs_inds]  # Boxes(torch.cat(pred_boxes))
            final_results.scores = results.scores[pairs_inds]
            final_results.pred_associations = pred_associations
            final_results.locations = results.locations[pairs_inds]
            final_results.fpn_levels = results.fpn_levels[pairs_inds]
            final_results.offset = results.offset_pred[pairs_inds]
            # SSIS_v1, track_embedding
            final_results.track_embedding = results.track_embedding_main[pairs_inds]
            # association
            pred_associations = Instances((output_height, output_width))

            pred_associations.pred_masks = np.array(asso_mask)
            pred_associations.pred_associations = asso_rela

            pred_associations.pred_boxes = Boxes(asso_bbox)
            pred_associations.pred_classes = np.array(asso_class)
            pred_associations.scores = np.array(asso_score)


            # For Missing Object 20221001
            pred_asso_global_masks = pred_asso_global_masks.cpu()
            final_results.rest_pred_masks = pred_global_masks[un_paired_inds]
            final_results.rest_pred_asso_masks = pred_asso_global_masks[un_paired_inds]
            final_results.rest_pred_boxes = results.pred_boxes[un_paired_inds]
            final_results.rest_scores = results.scores[un_paired_inds]
            final_results.rest_pred_classes = results.pred_classes[un_paired_inds]
            final_results.rest_track_embeddings = results.track_embedding_main[un_paired_inds]
            return final_results, pred_associations


