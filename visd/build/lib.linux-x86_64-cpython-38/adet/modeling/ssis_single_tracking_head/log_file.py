# SSIS_v1, track embedding buffer
def _dequeue_and_enqueue(self, keys, mode="all"):
    def concat_all_gather(tensor):
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    # gather keys before updating queue
    keys = concat_all_gather(keys)

    batch_size = keys.shape[0]

    if mode == "all":
        ptr = int(self.queue_ptr_all)
    elif mode == "object":
        ptr = int(self.queue_ptr_obj)
    elif mode == "shadow":
        ptr = int(self.queue_ptr_sha)
    # 判断字典的大小是否是batch_size的整数倍
    # print(ptr, batch_size, self.K)
    keys = keys[:self.K - ptr]
    batch_size = keys.shape[0]
    # assert self.K % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    self.queue_obj[:, ptr:ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % self.K  # move pointer

    self.queue_ptr_obj[0] = ptr


# SSIS_v1 Contrast ALL, version3
def _forward_track_heads_train(self, proposals, gt_instances):
    loss_center_l1 = nn.L1Loss(reduction='mean')
    loss_center_contra = nn.CrossEntropyLoss()

    track_embedding = self.track_embedding_fc_seq(proposals["instances"].top_feats3)
    gt_inds_distinct = list(set(proposals["instances"].gt_inds.tolist()))
    # print("gt_inds_distinct: ", gt_inds_distinct)

    # equation 2 in "Learning to Track Instances without Video Annotations"
    center_loss_l1_sum = 0
    center_representations_all_ = []
    for gt_ind_distinct in gt_inds_distinct:
        # print(proposals["instances"].gt_inds == gt_ind_distinct)
        # print(proposals["instances"].labels == 0)
        # print(torch.logical_and(proposals["instances"].gt_inds == gt_ind_distinct, proposals["instances"].labels == 0))
        instance_representations = track_embedding[proposals["instances"].gt_inds == gt_ind_distinct].float()
        center_resentation = torch.mean(instance_representations, 0)
        center_representations_all_.append(center_resentation)
        center_resentation_repeat = center_resentation.repeat(instance_representations.shape[0], 1).float()

        center_loss_l1 = loss_center_l1(instance_representations, center_resentation_repeat)
        center_loss_l1_sum += center_loss_l1

    # equation 4 in "Learning to Track Instances without Video Annotations"
    center_representations_all = torch.stack(center_representations_all_).float()

    S_matrix = torch.matmul(center_representations_all, torch.transpose(center_representations_all, 0, 1)).float()
    softmax = nn.Softmax(dim=1).float()
    S_matrix = softmax(S_matrix)
    # print(S_matrix.shape, center_representations_all.shape)

    I_matrix = [i for i in range(center_representations_all.shape[0])]
    I_matrix = torch.tensor(I_matrix).to(self.device).long()
    center_loss_contra = loss_center_contra(S_matrix, I_matrix)

    # equation 6 in "Learning to Track Instances without Video Annotations"
    maximum_entropy_loss = torch.sum((S_matrix * torch.log(S_matrix+1e-20)).fill_diagonal_(0) * -1).float()

    center_loss_l1_final = center_loss_l1_sum / len(gt_inds_distinct)
    center_loss_contra_final = center_loss_contra / len(center_representations_all)
    maximum_entropy_loss_final = maximum_entropy_loss / len(track_embedding)

    return center_loss_l1_final, center_loss_contra_final, maximum_entropy_loss_final

# SSIS_v1, Replay Offer Used!! Object Contrast Object, Shadow Contrast Shadow
def _forward_track_heads_train(self, proposals, gt_instances):
    loss_center_l1 = nn.L1Loss(reduction='mean')
    loss_center_contra = nn.CrossEntropyLoss()

    track_embedding = self.track_embedding_fc_seq(proposals["instances"].top_feats3)
    gt_inds_distinct = list(set(proposals["instances"].gt_inds.tolist()))
    # print("gt_inds_distinct: ", gt_inds_distinct)

    # equation 2 in "Learning to Track Instances without Video Annotations"
    center_loss_l1_sum = 0
    center_representations_all_obj_ = []
    center_representations_all_sha_ = []
    for gt_ind_distinct in gt_inds_distinct:
        # print(proposals["instances"].gt_inds == gt_ind_distinct)
        # print(proposals["instances"].labels == 0)
        # print(torch.logical_and(proposals["instances"].gt_inds == gt_ind_distinct, proposals["instances"].labels == 0))

        instance_representations = track_embedding[proposals["instances"].gt_inds == gt_ind_distinct].float()
        instance_representations_obj = track_embedding[
            torch.logical_and(proposals["instances"].gt_inds == gt_ind_distinct,
                              proposals["instances"].labels == 0)].float()
        instance_representations_sha = track_embedding[
            torch.logical_and(proposals["instances"].gt_inds == gt_ind_distinct,
                              proposals["instances"].labels == 1)].float()

        center_resentation = torch.mean(instance_representations, 0)

        if len(instance_representations_obj) != 0:
            center_resentation_obj = torch.mean(instance_representations_obj, 0)
            center_representations_all_obj_.append(center_resentation_obj)
        if len(instance_representations_sha) != 0:
            center_resentation_sha = torch.mean(instance_representations_sha, 0)
            center_representations_all_sha_.append(center_resentation_sha)

        center_resentation_repeat = center_resentation.repeat(instance_representations.shape[0], 1).float()
        center_loss_l1 = loss_center_l1(instance_representations, center_resentation_repeat)
        center_loss_l1_sum += center_loss_l1

    # equation 4 in "Learning to Track Instances without Video Annotations"
    center_representations_all_obj = torch.stack(center_representations_all_obj_)
    center_representations_all_sha = torch.stack(center_representations_all_sha_)
    # print(center_representations_all_obj.shape, center_representations_all_sha.shape)
    # print(self.queue_obj.clone().detach().shape, self.queue_sha.clone().detach().shape)

    len_center_representations_all_obj = len(center_representations_all_obj)
    len_center_representations_all_sha = len(center_representations_all_sha)

    center_representations_all_obj = torch.cat((center_representations_all_obj, torch.transpose(self.queue_obj.clone().detach(),0,1)[:self.K-len_center_representations_all_obj]), 0)
    center_representations_all_sha = torch.cat((center_representations_all_sha, torch.transpose(self.queue_sha.clone().detach(),0,1)[:self.K-len_center_representations_all_obj]), 0)
    # print(center_representations_all_obj.shape, center_representations_all_sha.shape)

    S_matrix_obj = torch.matmul(center_representations_all_obj,
                                torch.transpose(center_representations_all_obj, 0, 1)).float()
    S_matrix_sha = torch.matmul(center_representations_all_sha,
                                torch.transpose(center_representations_all_sha, 0, 1)).float()

    softmax = nn.Softmax(dim=1).float()

    S_matrix_obj = softmax(S_matrix_obj)
    S_matrix_sha = softmax(S_matrix_sha)

    I_matrix_obj = [i for i in range(center_representations_all_obj.shape[0])]
    I_matrix_sha = [i for i in range(center_representations_all_sha.shape[0])]

    I_matrix_obj = torch.tensor(I_matrix_obj).to(self.device).long()
    I_matrix_sha = torch.tensor(I_matrix_sha).to(self.device).long()

    center_loss_contra_obj = loss_center_contra(S_matrix_obj[:len_center_representations_all_obj], I_matrix_obj[:len_center_representations_all_obj])
    center_loss_contra_sha = loss_center_contra(S_matrix_sha[:len_center_representations_all_sha], I_matrix_sha[:len_center_representations_all_sha])

    # equation 6 in "Learning to Track Instances without Video Annotations"
    maximum_entropy_loss_obj = torch.sum((S_matrix_obj * torch.log(S_matrix_obj+1e-20)).fill_diagonal_(0)[:len_center_representations_all_obj] * -1).float()
    maximum_entropy_loss_sha = torch.sum((S_matrix_sha * torch.log(S_matrix_sha + 1e-20)).fill_diagonal_(0)[:len_center_representations_all_obj] * -1).float()

    center_loss_l1_final = center_loss_l1_sum / len(gt_inds_distinct)
    center_loss_contra_obj_final = center_loss_contra_obj/ len_center_representations_all_obj
    center_loss_contra_sha_final = center_loss_contra_sha / len_center_representations_all_sha
    maximum_entropy_loss_obj_final = 0.1 * maximum_entropy_loss_obj/ len_center_representations_all_obj
    maximum_entropy_loss_sha_final = 0.1 * maximum_entropy_loss_sha / len_center_representations_all_sha

    # dequeue and enqueue
    self._dequeue_and_enqueue(center_representations_all_obj, "object")
    self._dequeue_and_enqueue(center_representations_all_sha, "shadow")

    return center_loss_l1_final, center_loss_contra_obj_final, center_loss_contra_sha_final, \
           maximum_entropy_loss_obj_final, maximum_entropy_loss_sha_final


    # SSIS_v1 Contrast ASSO, version 5
    def _forward_track_heads_train(self, proposals, gt_instances):
        loss_center_l1 = nn.L1Loss(reduction='mean')
        loss_center_contra = nn.CrossEntropyLoss(reduction='mean')

        track_embedding = self.track_embedding_fc_seq(proposals["instances"].top_feats3)
        gt_inds_distinct = list(set(proposals["instances"].gt_inds.tolist()))
        total_number_gt = 0
        for gt_instance in gt_instances:
            total_number_gt += len(gt_instance.gt_classes)

        # equation 2 in "Learning to Track Instances without Video Annotations"
        center_loss_l1_sum = 0
        center_representations_all_ = []
        center_representations_all_index_ = []
        for gt_ind_distinct in gt_inds_distinct:
            instance_representations = track_embedding[proposals["instances"].gt_inds == gt_ind_distinct].float()
            center_resentation = torch.mean(instance_representations, 0)
            center_representations_all_.append(center_resentation)
            center_representations_all_index_.append(gt_ind_distinct)
            center_resentation_repeat = center_resentation.repeat(instance_representations.shape[0], 1).float()

            center_loss_l1 = loss_center_l1(instance_representations, center_resentation_repeat)
            center_loss_l1_sum += center_loss_l1

        centern_representation_asso_all_ = []
        for index in range(len(center_representations_all_)):
            # We assume first one is object embedding
            shadow_embedding_index = int(center_representations_all_index_[index] + total_number_gt / 2)
            if shadow_embedding_index in center_representations_all_index_:
                sha_index = center_representations_all_index_.index(shadow_embedding_index)
                centern_representation_asso_ = torch.cat(
                    (center_representations_all_[index], center_representations_all_[sha_index]), 0)
                centern_representation_asso_all_.append(centern_representation_asso_)

        # contrast association
        # equation 4 in "Learning to Track Instances without Video Annotations"
        centern_representation_asso_all = torch.stack(centern_representation_asso_all_).float()

        S_matrix_asso = torch.matmul(centern_representation_asso_all,
                                     torch.transpose(centern_representation_asso_all, 0, 1)).float()

        I_matrix_asso = [i for i in range(centern_representation_asso_all.shape[0])]
        I_matrix_asso = torch.tensor(I_matrix_asso).to(self.device).long()
        center_loss_contra_asso = loss_center_contra(S_matrix_asso, I_matrix_asso)

        # equation 6 in "Learning to Track Instances without Video Annotations"
        softmax = nn.Softmax(dim=1).float()
        S_matrix_asso_softmax = softmax(S_matrix_asso)
        maximum_entropy_loss_asso = torch.sum(
            (S_matrix_asso_softmax * torch.log(S_matrix_asso_softmax + 1e-20)).fill_diagonal_(0) * -1).float()

        center_loss_contra_final_asso = center_loss_contra_asso
        maximum_entropy_loss_final_asso = \
            maximum_entropy_loss_asso / (len(centern_representation_asso_all) * len(centern_representation_asso_all))

        center_loss_contra_final = center_loss_contra_final_asso
        maximum_entropy_loss_final = maximum_entropy_loss_final_asso
        center_loss_l1_final = center_loss_l1_sum / len(gt_inds_distinct)

        return center_loss_l1_final, center_loss_contra_final, maximum_entropy_loss_final


    # SSIS_v1 Contrast ASSO and Contrast Separate, version 6
    # def _forward_track_heads_train(self, proposals, gt_instances):
    #     loss_center_l1 = nn.L1Loss(reduction='mean')
    #     loss_center_contra = nn.CrossEntropyLoss(reduction='mean')
    #
    #     track_embedding = self.track_embedding_fc_seq(proposals["instances"].top_feats3)
    #     gt_inds_distinct = list(set(proposals["instances"].gt_inds.tolist()))
    #     total_number_gt = 0
    #     for gt_instance in gt_instances:
    #         total_number_gt += len(gt_instance.gt_classes)
    #
    #     # equation 2 in "Learning to Track Instances without Video Annotations"
    #     # contrast association
    #     center_loss_l1_sum = 0
    #     center_representations_all_ = []
    #     center_representations_all_obj_ = []
    #     center_representations_all_sha_ = []
    #     center_representations_all_index_ = []
    #
    #     for gt_ind_distinct in gt_inds_distinct:
    #         instance_representations = track_embedding[proposals["instances"].gt_inds == gt_ind_distinct].float()
    #         center_resentation = torch.mean(instance_representations, 0)
    #         center_representations_all_.append(center_resentation)
    #         center_representations_all_index_.append(gt_ind_distinct)
    #
    #         # Classify center representation: Object or Shadow
    #         if gt_ind_distinct < total_number_gt/2:
    #             center_resentation_obj = torch.mean(instance_representations, 0)
    #             center_representations_all_obj_.append(center_resentation_obj)
    #         else:
    #             center_resentation_sha = torch.mean(instance_representations, 0)
    #             center_representations_all_sha_.append(center_resentation_sha)
    #
    #         center_resentation_repeat = center_resentation.repeat(instance_representations.shape[0], 1).float()
    #         center_loss_l1 = loss_center_l1(instance_representations, center_resentation_repeat)
    #         center_loss_l1_sum += center_loss_l1
    #
    #     centern_representation_asso_all_ = []
    #     for index in range(len(center_representations_all_)):
    #         # We assume first one is object embedding
    #         shadow_embedding_index = int(center_representations_all_index_[index] + total_number_gt / 2)
    #         if shadow_embedding_index in center_representations_all_index_:
    #             sha_index = center_representations_all_index_.index(shadow_embedding_index)
    #             centern_representation_asso_ = torch.cat(
    #                 (center_representations_all_[index], center_representations_all_[sha_index]), 0)
    #             centern_representation_asso_all_.append(centern_representation_asso_)
    #
    #     # equation 4 in "Learning to Track Instances without Video Annotations"
    #     # contrast association
    #     centern_representation_asso_all = torch.stack(centern_representation_asso_all_).float()
    #
    #     S_matrix_asso = torch.matmul(centern_representation_asso_all,
    #                                  torch.transpose(centern_representation_asso_all, 0, 1)).float()
    #
    #     I_matrix_asso = [i for i in range(centern_representation_asso_all.shape[0])]
    #     I_matrix_asso = torch.tensor(I_matrix_asso).to(self.device).long()
    #     center_loss_contra_asso = loss_center_contra(S_matrix_asso, I_matrix_asso)
    #
    #     # contrast separate
    #     center_representations_all_obj = torch.stack(center_representations_all_obj_)
    #     center_representations_all_sha = torch.stack(center_representations_all_sha_)
    #
    #     S_matrix_obj = torch.matmul(center_representations_all_obj,
    #                                 torch.transpose(center_representations_all_obj, 0, 1)).float()
    #     S_matrix_sha = torch.matmul(center_representations_all_sha,
    #                                 torch.transpose(center_representations_all_sha, 0, 1)).float()
    #
    #     I_matrix_obj = [i for i in range(center_representations_all_obj.shape[0])]
    #     I_matrix_sha = [i for i in range(center_representations_all_sha.shape[0])]
    #
    #     I_matrix_obj = torch.tensor(I_matrix_obj).to(self.device).long()
    #     I_matrix_sha = torch.tensor(I_matrix_sha).to(self.device).long()
    #
    #     center_loss_contra_obj = loss_center_contra(S_matrix_obj, I_matrix_obj)
    #     center_loss_contra_sha = loss_center_contra(S_matrix_sha, I_matrix_sha)
    #
    #     # equation 6 in "Learning to Track Instances without Video Annotations"
    #     softmax = nn.Softmax(dim=1).float()
    #     S_matrix_asso_softmax = softmax(S_matrix_asso)
    #     maximum_entropy_loss_asso = torch.sum(
    #         (S_matrix_asso_softmax * torch.log(S_matrix_asso_softmax + 1e-20)).fill_diagonal_(0) * -1).float()
    #
    #     center_loss_contra_final_asso = center_loss_contra_asso
    #     maximum_entropy_loss_final_asso = \
    #         maximum_entropy_loss_asso / (len(centern_representation_asso_all)*len(centern_representation_asso_all))
    #
    #     center_loss_contra_final = center_loss_contra_final_asso + center_loss_contra_obj + center_loss_contra_sha
    #     maximum_entropy_loss_final = maximum_entropy_loss_final_asso
    #     center_loss_l1_final = center_loss_l1_sum / len(gt_inds_distinct)
    #
    #     return center_loss_l1_final, center_loss_contra_final, maximum_entropy_loss_final
