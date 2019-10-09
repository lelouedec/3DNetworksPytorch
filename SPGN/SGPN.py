import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time
import point
from net_utils import *
from SGPN_utils import *


class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        self.sa0 = PointNetSetAbstraction(4096, 0.1, 32, 6, [16, 16, 32], False)
        self.sa05 = PointNetSetAbstraction(2048, 0.1, 32, 32+3, [32, 32, 32], False)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 32+3, [32, 32, 64], False)# npoint, radius, nsample, in_channel, mlp, group_all
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])#in_channel, mlp
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])#in_channel, mlp
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])#in_channel, mlp
        self.fp1 = PointNetFeaturePropagation(160, [128, 128, 128])#in_channel, mlp
        self.fp05 = PointNetFeaturePropagation(160, [128, 128, 64])
        self.fp0 = PointNetFeaturePropagation(67, [128, 128, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)


        ## similarity
        self.conv2_1 = nn.Conv2d(64,128,kernel_size=(1,1),stride=(1,1))

        ## confidence map
        self.conv3_1 = nn.Conv2d(64,128,kernel_size=(1,1),stride=(1,1))
        self.conv3_2 = nn.Conv2d(128,1,kernel_size=(1,1),stride=(1,1))

        self.criterion_semseg = nn.BCELoss().cuda()
        self.criterion2  = nn.MSELoss(reduction='mean')


    def forward(self, xyz,color,target,training,epoch,just_seg):
        xyz = xyz.permute(0, 2, 1)
        color = color.permute(0, 2, 1)
        l0_xyz, l0_points = self.sa0(xyz, color)
        l05_xyz, l05_points = self.sa05(l0_xyz, l0_points)
        l1_xyz, l1_points = self.sa1(l05_xyz, l05_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l05_points = self.fp1(l05_xyz, l1_xyz, l05_points, l1_points)
        l0_points = self.fp05(l0_xyz, l05_xyz, l0_points, l05_points)
        l0_points = self.fp0(xyz, l0_xyz, color, l0_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        semseg = self.conv2(x)
        semseg_logit = torch.sigmoid(semseg)


        ##similarity
        Fsim = self.conv2_1(l0_points.unsqueeze(2)).squeeze(2)
        r = torch.sum(Fsim*Fsim,dim=1)
        r = r.view((l0_points.shape[0],-1,1)).permute(0,2,1)
        trans = torch.transpose(Fsim ,2, 1)
        mul = 2 * torch.matmul(trans, Fsim)
        sub = r - mul
        D = sub + torch.transpose(r, 2, 1)
        D[D<=0.0] = 0.0

        ##Confidence
        conf_logit = self.conv3_2(self.conv3_1(l0_points.unsqueeze(2))).squeeze(2)
        conf = torch.sigmoid(conf_logit)

        # print(semseg.shape,semseg_logit.shape,D.shape,conf.shape,conf_logit.shape)
        ## COmputing loss
        if(just_seg):
            return  [0.0,self.criterion_semseg(semseg_logit,target['semseg']),0.0,0.0],semseg_logit
        if(training):
            return self.compute_loss(semseg ,semseg_logit ,D ,conf ,conf_logit,target,epoch),semseg_logit
        else:
            pts_semseg_label,pts_semseg = self.convert_seg_to_one_hot(target['semseg'])
            pts_group_label, group_mask = self.convert_groupandcate_to_one_hot(target['ptsgroup'])
            group_mat_label = torch.matmul(pts_group_label,torch.transpose(pts_group_label,2,1))
            pts_corr_val = D[0].squeeze()
            pred_confidence_val = conf[0].squeeze()
            ptsclassification_val = semseg_logit[0].squeeze()
            NUM_CATEGORY = 2
            ths = np.zeros(NUM_CATEGORY)
            ths_ = np.zeros(NUM_CATEGORY)
            cnt = np.zeros(NUM_CATEGORY)
            # ths,ths_,cnt = Get_Ths(pts_corr_val, target['semseg'].cpu().numpy()[0], target['ptsgroup'].cpu().numpy()[0], ths, ths_, cnt)
            # ths = [ths[i]/cnt[i] if cnt[i] != 0 else 0.2 for i in range(len(cnt))]
            groupids_block = Create_groups(pts_corr_val, pred_confidence_val, ptsclassification_val,xyz[0].permute(1,0))
            # groupids_block, refineseg, group_seg = GroupMerging(pts_corr_val, pred_confidence_val, ptsclassification_val,[0.1,0.1])
            # groupids_block[groupids_block==-1] = 0.0
            return 0.0,groupids_block,ptsclassification_val


    def compute_loss(self,semseg ,semseg_logit ,D ,conf ,conf_logit, gt_truth,epoch):
        #gt_truth = {"ptsgroup":mask_target,"semseg":target,"bounding_boxes":bb_target}

        pts_semseg_label,pts_semseg = self.convert_seg_to_one_hot(gt_truth['semseg'])
        pts_group_label, group_mask = self.convert_groupandcate_to_one_hot(gt_truth['ptsgroup'])

        alpha=2.0
        margin=[1.,2.]
        if(epoch%5==0 and epoch!=0):
            alpha = alpha/2


        ## Similarity loss
        B = pts_group_label.shape[0]
        N = pts_group_label.shape[1]

        group_mat_label = torch.matmul(pts_group_label,torch.transpose(pts_group_label,1,2))
        diag_idx = torch.arange(0,group_mat_label.shape[1], out=torch.LongTensor())
        group_mat_label[:,diag_idx,diag_idx] = 1.0

        sem_mat_label = torch.matmul(pts_semseg_label,torch.transpose(pts_semseg_label,1,2))
        sem_mat_label[:,diag_idx,diag_idx] = 1.0

        samesem_mat_label = sem_mat_label
        diffsem_mat_label = 1.0 - sem_mat_label

        samegroup_mat_label = group_mat_label
        diffgroup_mat_label = 1.0 - group_mat_label
        diffgroup_samesem_mat_label = diffgroup_mat_label *  samesem_mat_label
        diffgroup_diffsem_mat_label = diffgroup_mat_label *  diffsem_mat_label

        num_samegroup = torch.sum(samegroup_mat_label)
        num_diffgroup_samesem = torch.sum(diffgroup_samesem_mat_label)
        num_diffgroup_diffsem = torch.sum(diffgroup_diffsem_mat_label)

        pos = samegroup_mat_label * D
        sub = margin[0] - D
        sub[sub<=0.0] = 0.0

        sub2 = margin[1] - D
        sub2[sub2<=0.0] = 0.0

        neg_samesem = alpha * (diffgroup_samesem_mat_label * sub)
        neg_diffsem = diffgroup_diffsem_mat_label * sub2

        simmat_loss = neg_samesem + neg_diffsem + pos

        group_mask_weight = torch.matmul(group_mask.unsqueeze(2), torch.transpose(group_mask.unsqueeze(2), 2, 1))
        simmat_loss = simmat_loss * group_mask_weight
        simmat_loss = torch.mean(simmat_loss)

        ## Confidence map loss
        Pr_obj = torch.sum(pts_semseg_label,dim=2).float().cuda()
        ng_label = group_mat_label
        ng_label = torch.gt(ng_label,0.5)
        ng  = torch.lt(D,margin[0])
        epsilon = torch.ones(ng_label.shape[:2]).float().cuda() * 1e-6

        up = torch.sum((ng & ng_label).float())
        down = torch.sum((ng | ng_label).float()) + epsilon
        pts_iou = torch.div(up,down)
        confidence_label = pts_iou * Pr_obj

        confidence_loss = self.criterion2(confidence_label.unsqueeze(1),conf_logit.squeeze(2))##MSE


        ##semseg loss
        sem_seg_loss = self.criterion_semseg(semseg_logit,gt_truth['semseg'])



        loss = simmat_loss + sem_seg_loss + confidence_loss

        grouperr = torch.abs(ng.float()-ng_label.float())

        return (loss,simmat_loss,sem_seg_loss,confidence_loss)#, grouperr.mean(), torch.sum(grouperr+diffgroup_samesem_mat_label),num_diffgroup_samesem \
               #torch.sum(grouperr * diffgroup_diffsem_mat_label), num_diffgroup_diffsem, \
               #torch.sum(grouperr * samegroup_mat_label), num_samegroup



    def convert_seg_to_one_hot(self,labels):
        # labels:BxN
        labels = labels.permute(0,2,1)
        NUM_CATEGORY = 2
        label_one_hot = torch.zeros((labels.shape[0], labels.shape[1], NUM_CATEGORY)).cuda()
        pts_label_mask = torch.zeros((labels.shape[0], labels.shape[1])).cuda()

        un, cnt = torch.unique(labels, return_counts=True)
        label_count_dictionary = {}
        for v,u in enumerate(un):
            label_count_dictionary[int(u.item())] = cnt[v].item()

        totalnum = 0
        for k_un, v_cnt in label_count_dictionary.items():
            if k_un != -1:
                totalnum += v_cnt

        for idx in range(labels.shape[0]):
            for jdx in range(labels.shape[1]):
                if labels[idx, jdx] != -1:
                    label_one_hot[idx, jdx, int(labels[idx, jdx])] = 1
                    pts_label_mask[idx, jdx] = float(totalnum) / float(label_count_dictionary[int(labels[idx, jdx])]) # 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum
        return label_one_hot, pts_label_mask

    def convert_groupandcate_to_one_hot(self,grouplabels):
        # grouplabels: BxN
        NUM_GROUPS = 50
        group_one_hot = torch.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS)).cuda()
        pts_group_mask = torch.zeros((grouplabels.shape[0], grouplabels.shape[1])).cuda()

        un, cnt = torch.unique(grouplabels, return_counts=True)
        group_count_dictionary = {}
        for v,u in enumerate(un):
            group_count_dictionary[int(u.item())] = cnt[v].item()
        totalnum = 0
        for k_un, v_cnt in group_count_dictionary.items():
            if k_un != -1:
                totalnum += v_cnt

        for idx in range(grouplabels.shape[0]):
            for jdx in range(grouplabels.shape[1]):
                if grouplabels[idx, jdx] != -1:
                    group_one_hot[idx, jdx, int(grouplabels[idx, jdx])] = 1
                    pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[int(grouplabels[idx, jdx])]) # 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

        return group_one_hot.float(), grouplabels

    def freeze_pn(self):
        for params in self.sa0.parameters():
            params.requires_grad = False
        for params in self.sa1.parameters():
            params.requires_grad = False
        for params in self.sa2.parameters():
            params.requires_grad = False
        for params in self.sa3.parameters():
            params.requires_grad = False
        for params in self.sa4.parameters():
            params.requires_grad = False

        for params in self.fp4.parameters():
            params.requires_grad = False
        for params in self.fp4.parameters():
            params.requires_grad = False
        for params in self.fp2.parameters():
            params.requires_grad = False
        for params in self.fp1.parameters():
            params.requires_grad = False
        for params in self.fp05.parameters():
            params.requires_grad = False
        for params in self.fp0.parameters():
            params.requires_grad = False

        self.conv1.requires_grad  = False
        self.conv2.requires_grad  = False

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

if __name__ == '__main__':
    for i in range(10):
        xyz = torch.rand(1, 30000,3).cuda()
        colors = torch.rand(1, 30000,3).cuda()
        net = PointNet2SemSeg(2)
        net.cuda()
        x = net(xyz,colors)
        print(x)
