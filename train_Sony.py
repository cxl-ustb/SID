import glob
import os,time,scipy.io
import numpy as np
import rawpy
import glob
import torch
import torch.nn as nn
import scipy.misc as misc
import logging
train_parameter={
    'start_epoch':0,
    'num_epoches':4001,
    'patch_size':512,
    'save_freq':200,
    'learning_rate':1e-4,
    'DEBUG':0,
    'data_prefix':'/home/cxl/study/dataset/',
    'output_prefix':'./result/',
    'checkpoint_load_dir':'./checkpoint/' 
}
input_dir=os.path.join(train_parameter['data_prefix'],'Sony/Sony/short')
gt_dir=os.path.join(train_parameter['data_prefix'],'Sony/Sony/long/')
checkpoint_dir=os.path.join(train_parameter['output_prefix'],'result_Sony')
result_dir=os.path.join(train_parameter['output_prefix'],'result_Sony')

train_fns=glob.glob(os.path.join(gt_dir,'0*.ARW'))

train_ids=[int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps=train_parameter['patch_size']
save_freq=train_parameter['save_freq']

class DoubleConv(nn.Module):
    def __init__(self,input_channels,output_channels,filter_size):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(input_channels,output_channels,filter_size,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_channels,output_channels,filter_size,padding=1),
            nn.LeakyReLU(0.2))
    def forward(self,inputs):
        return self.conv(inputs)
    
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1=DoubleConv(4,32,3)
        self.pool1=nn.MaxPool2d(kernel_size=2)
        self.conv2=DoubleConv(32,64,3)
        self.pool2=nn.MaxPool2d(kernel_size=2)
        self.conv3=DoubleConv(64,128,3)
        self.pool3=nn.MaxPool2d(kernel_size=2)
        self.conv4=DoubleConv(128,256,3)
        self.pool4=nn.MaxPool2d(kernel_size=2)
        self.conv5=DoubleConv(256,512,3)
        
        self.up6=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.conv6=DoubleConv(512,256,3)
        self.up7=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.conv7=DoubleConv(256,128,3)
        self.up8=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv8=DoubleConv(128,64,3)
        self.up9=nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        self.conv9=DoubleConv(64,32,3)
        self.conv10=nn.Conv2d(32,12,kernel_size=2,stride=1,padding=1)
        self.pixel_shuffle=nn.PixelShuffle(2)
        
    def forward(self,inputs):
        x1_0=self.conv1(inputs)
        x1_1=self.pool1(x1_0)
        x2_0=self.conv2(x1_1)
        x2_1=self.pool2(x2_0)
        x3_0=self.conv3(x2_1)
        x3_1=self.pool3(x3_0)
        x4_0=self.conv4(x3_1)
        x4_1=self.pool4(x4_0)
        x5=self.conv5(x4_1)
        
        x6_0=self.up6(x5)
        x6_1=self.conv6(torch.cat((x6_0,x4_0),dim=1))
        x7_0=self.up7(x6_1)
        x7_1=self.conv7(torch.cat((x7_0,x3_0),dim=1))
        x8_0=self.up8(x7_1)
        x8_1=self.conv8(torch.cat((x8_0,x2_0),dim=1))
        x9_0=self.up9(x8_1)
        x9_1=self.conv9(torch.cat((x9_0,x1_0),dim=1))
        out=self.conv10(x9_1)[:,:,:-1,:-1]
        out=self.pixel_shuffle(out)
        return out

def pack_raw(raw):
    im=raw.raw_image_visible.astype(np.float32)
    im=np.maximum(im-512,0)/(16383-512)
    
    im=np.expand_dims(im,axis=2)
    img_shape=im.shape
    H=img_shape[0]
    W=img_shape[1]
    
    out=np.concatenate(
        (
            im[0:H:2,0:W:2,:],
            im[0:H:2,1:W:2,:],
            im[1:H:2,1:W:2,:],
            im[1:H:2,0:W:2,:]
        ),axis=2
    )
    return out
device='cuda:0'
net_model=Network().to(device)

G_loss=torch.nn.L1Loss(reduction='mean')
optimizer=torch.optim.Adam(lr=train_parameter['learning_rate'],params=net_model.parameters())

# Raw data takes long time to load.keep them in memory after loaded.
gt_images=[None]*6000


g_loss=np.zeros((5000,1))

for epoch in range(train_parameter['start_epoch'],train_parameter['num_epoches']):
    cnt=0
    
    for ind in np.random.permutation(len(train_ids)):

        train_id=train_ids[ind]
        in_files=glob.glob(os.path.join(input_dir,'%05d_00*.ARW'%train_id))
        in_path=in_files[np.random.randint(0,len(in_files))]
        in_fn=os.path.basename(in_path)
        
        gt_files=glob.glob(os.path.join(gt_dir,'%05d_00*.ARW'%train_id))
        gt_path=gt_files[0]
        gt_fn=os.path.basename(gt_path)
        in_exposure=float(in_fn[9:-5])
        gt_exposure=float(gt_fn[9:-5])
        ratio=min(gt_exposure/in_exposure,300)
        
        st=time.time()
        cnt+=1
        
        
        input_image=np.expand_dims(pack_raw(rawpy.imread(in_path)),axis=0)*ratio
        H=input_image.shape[1]
        W=input_image.shape[2]
        
        xx=np.random.randint(0,W-ps)
        yy=np.random.randint(0,H-ps)
        input_patch=input_image[:,yy:yy+ps,xx:xx+ps,:]
        
        gt_raw=rawpy.imread(gt_path)
        im=gt_raw.postprocess(use_camera_wb=True,half_size=False,no_auto_bright=True,output_bps=16)
        im=np.expand_dims(np.float32(im/65535),axis=0)
        gt_patch=im[:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]
        
        if np.random.randint(2,size=1)[0]==1:
            input_patch=np.flip(input_patch,axis=1)
            gt_patch=np.flip(gt_patch,axis=1)
            
        if np.random.randint(2,size=1)[0]==1:
            input_patch=np.flip(input_patch,axis=2)
            gt_patch=np.flip(gt_patch,axis=2)
        
        if np.random.randint(2,size=1)[0]==1:
            input_patch=np.transpose(input_patch,(0,2,1,3))
            gt_patch=np.transpose(gt_patch,(0,2,1,3))
            
        input_patch=np.minimum(input_patch,1.0)
        input_patch=torch.Tensor(np.transpose(input_patch,[0,3,1,2])).to(device)
        output=net_model(input_patch) 
        output=output.permute([0,2,3,1])
        output=torch.clamp(output,min=0.0,max=1.0)
        G_current=G_loss(output,torch.tensor(gt_patch.copy()).to(device))
        G_current.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        g_loss[ind]=G_current.detach().cpu().numpy()
        
        print("%d %d Loss=%.3f Time=%.3f Filename=%s"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st,gt_path))
        
        if epoch%save_freq==0:
            if not os.path.isdir(os.path.join(result_dir,'%04d'%epoch)):
                os.makedirs(os.path.join(result_dir,'%04d'%epoch))
            
            temp = np.concatenate((gt_patch[0, :, :, :], output.detach().cpu().numpy()[0, :, :, :]), axis=1)
        
            misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(result_dir, '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio)))
            torch.save(net_model.state_dict(), os.path.join(checkpoint_dir, 'model_%04d.pt' % epoch))
            
    
    
        
    
    