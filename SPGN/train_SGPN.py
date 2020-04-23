import open3d as opend
import torch
import numpy as np
import SGPN
import SGPN2
# import pptk
# import load_data2
import torch.optim as optim
import torch.nn as nn
import tqdm
import torch.nn as nn
import torch.nn.init as init
import load_data
from torch.utils.tensorboard import SummaryWriter
import time




def get_boxe_from_p(boxes):
    vertices = []
    lines = np.array( [[0,0]])
    i = 0
    for b in boxes:
        first_p = b[1]
        second_p = b[0]
        width = first_p[0] - second_p[0]
        height = first_p[1] - second_p[1]
        depth = first_p[2] - second_p[2]

        vertices.append(first_p) # top front right
        vertices.append(first_p-[width,0,0]) # top front left
        vertices.append(first_p-[width,height,0]) # bottom front left
        vertices.append(first_p-[0,height,0]) # botton front right

        vertices.append(second_p) # bottom back left
        vertices.append(first_p-[width,0,depth]) # top back left
        vertices.append(first_p-[0,height,depth]) # bottom back right
        vertices.append(first_p-[0,0,depth]) # top back right

        edges = [[0+(i*8),1+(i*8)],[1+(i*8),2+(i*8)],[2+(i*8),3+(i*8)],[3+(i*8),0+(i*8)]
                ,[4+(i*8),5+(i*8)],[4+(i*8),6+(i*8)],[6+(i*8),7+(i*8)],[7+(i*8),5+(i*8)]
                ,[0+(i*8),7+(i*8)],[1+(i*8),5+(i*8)],[4+(i*8),2+(i*8)],[3+(i*8),6+(i*8)]]
        lines = np.concatenate([lines,edges],axis = 0)
        i = i+1


    line_set = opend.geometry.LineSet()
    line_set.points = opend.utility.Vector3dVector(vertices)
    line_set.lines = opend.utility.Vector2iVector(lines[1:])
    line_set.colors = opend.utility.Vector3dVector([[1, 0, 0] for i in range(lines[1:].shape[0])])
        # i = i + 1

    return line_set


seed = list(map(ord, 'toto'))
seed = map(str, seed)
seed = ''.join(seed)
seed = int(seed)
torch.manual_seed(seed)
np.random.seed(1)

torch.set_num_threads(1)
OMP_NUM_THREADS=1
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

print("loading data...")
points,colors,annotations,centers,bb_list,mask_list,label_list = load_data.load_data("../test")
model = SGPN.PointNet2SemSeg(2).cuda()
# model = torch.load("./models/test_model_SGPN_only_pn.ckpt").cuda()
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.00005}])
epochs = 100


writer = SummaryWriter()
print("training...")
for p in range(0,epochs):
    lost = []
    lost_seg = []
    lost_simmat_loss = []
    lost_confidence_loss = []
    for i in tqdm.tqdm(range(0,len(points))):
        input_tensor = torch.from_numpy(points[i]).float().unsqueeze(0).cuda()
        # input_tensor3 = torch.from_numpy(centers[i]).float().unsqueeze(0).cuda()
        if(input_tensor.shape[1]>1000):# and input_tensor3.shape[1]>0):
            input_tensor2 = torch.from_numpy(colors[i]).float().unsqueeze(0).cuda()
            target = torch.from_numpy(annotations[i]).unsqueeze(0).unsqueeze(1).cuda().float()
            bb_target = torch.from_numpy(bb_list[i]).unsqueeze(0).cuda().float()
            mask_target = torch.from_numpy(mask_list[i]).unsqueeze(0).cuda().float()
            optimizer.zero_grad
            gt = {"ptsgroup":mask_target,"semseg":target,"bounding_boxes":bb_target}
            loss,mask = model(input_tensor,input_tensor2,gt,True,p,False)
            lost.append(loss[0].data.item())
            lost_seg.append(loss[1].data.item())
            lost_simmat_loss.append(loss[2].data.item())
            lost_confidence_loss.append(loss[3].data.item())
            loss = loss[0]
            loss.backward()
            optimizer.step()

            if(p%100 ==0 and p!=0 ):
                torch.save(model.cpu(), "./models/test_model"+str(p)+".ckpt")
                model.cuda()
    writer.add_scalar('Loss', np.array(lost).mean(), p)
    writer.add_scalar('Loss seg', np.array(lost_seg).mean(), p)
    writer.add_scalar('Loss simmat_loss ', np.array(lost_simmat_loss).mean(), p)
    writer.add_scalar('Loss confidence_loss', np.array(lost_confidence_loss).mean(), p)
torch.save(model.cpu(), "./models/test_model_SGPN_0_008.ckpt")
