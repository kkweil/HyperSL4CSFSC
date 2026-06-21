import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from datautils.tools import split_train_test
from collections import OrderedDict
from sklearn.preprocessing import minmax_scale
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from engine.model import SpectralSharedEncoder
from engine.DomainAdaption import AdapterEncoder, print_trainable_parameters
import numpy as np
from skimage.segmentation import slic
import torch.nn.functional as F
import spectral
import matplotlib.pyplot as plt

def hyperspectral_superpixels(
    img_hwc,
    n_segments=50,
    compactness=10,
    max_iter=10,
    sigma=0,
    start_label=0,
):
    """
    基于 SLIC 的高光谱超像素分割，并返回：
      - 每个超像素的平均光谱向量 (N, C)
      - 每个超像素的重心 (N, 2)  (row, col)

    参数：
        img_hwc: np.ndarray, 形状 (H, W, C)
        n_segments: 期望的超像素数
        compactness: SLIC 的紧致度参数
        max_iter: SLIC 迭代次数
        sigma: 预平滑
        start_label: 超像素标签起始值（0 或 1）

    返回：
        mean_spectra: (N, C) 每个超像素的平均光谱
        centers:      (N, 2) 每个超像素重心 (row, col)
        labels:       (H, W) 每个像素对应的超像素ID
    """
    # img_hwc = img_hwc[200:225,150:175]

    H, W, C = img_hwc.shape

    # ---- 1. 超像素分割 ----
    
    labels = slic(
        img_hwc,
        n_segments=n_segments,
        compactness=compactness,
        max_num_iter=max_iter,
        sigma=sigma,
        start_label=start_label,
    )  # (H, W), 标签 0..N-1 或 1..N
    # import matplotlib.pyplot as plt
    # img_show = img_hwc[..., 20]
    # img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
    # img_rgb = np.stack([img_show]*3, axis=-1)
    # vis = mark_boundaries(img_rgb, labels, color=(1, 0, 0))
    
    # plt.imsave('slic.png', vis)
    
    # 若 start_label=0，则 N = labels.max() + 1
    # 若 start_label=1，则 N = labels.max()
    if start_label == 0:
        N = int(labels.max()) + 1
    else:
        N = int(labels.max())

    # print(f'Superpiexl:{n_segments}({N})')
    # ---- 2. 计算每个超像素的平均光谱 (N, C) ----
    labels_flat = labels.reshape(-1)        # (H*W,)
    img_flat = img_hwc.reshape(-1, C)       # (H*W, C)

    counts = np.bincount(labels_flat, minlength=N).astype(np.float64)  # (N,)

    mean_spectra = np.zeros((N, C), dtype=np.float32)

    # 对每个波段用带权 bincount 求均值
    for c in range(C):
        band = img_flat[:, c]  # (H*W,)
        sum_per_label = np.bincount(labels_flat, weights=band, minlength=N)  # (N,)
        mean_spectra[:, c] = (sum_per_label / np.maximum(counts, 1e-6)).astype(np.float32)

    # ---- 3. 计算每个超像素的重心 (N, 2) ----
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys_flat = ys.reshape(-1).astype(np.float64)
    xs_flat = xs.reshape(-1).astype(np.float64)

    sum_y = np.bincount(labels_flat, weights=ys_flat, minlength=N)  # (N,)
    sum_x = np.bincount(labels_flat, weights=xs_flat, minlength=N)  # (N,)

    center_y = sum_y / np.maximum(counts, 1e-6) / H  # (N,)
    center_x = sum_x / np.maximum(counts, 1e-6) / W  # (N,)

    centers = np.stack([center_y, center_x], axis=1).astype(np.float32)  # (N, 2), (row, col)
    if N>n_segments:
        n_segments = N
    mean_spectra_pad = np.zeros((n_segments, C), dtype=mean_spectra.dtype)
    centers_pad = np.zeros((n_segments, 2), dtype=centers.dtype)
    mean_spectra_pad[:N] = mean_spectra[:N]
    centers_pad[:N] = centers[:N]
    mask = np.zeros((n_segments,), dtype=bool)
    mask[:N] = True 
    return mean_spectra_pad, centers_pad, mask, labels




def read_Huston(data_path):
    data = loadmat(data_path)
    img = data['data']
    # W, H, C = img.shape
    # img = img.reshape(-1, C)
    # img = minmax_scale(img)
    # img = img.reshape(W, H,C)
    labels = data['gt']
    wavelength = data['wavelength'].squeeze()
    return img.astype('float32'), labels, wavelength.astype('float32')


class HyperDataset(Dataset):
    def __init__(self, data, label, wave, is_train, enable_superpixels=False):
        self.data = data
        self.label = label
        self.wave = wave
        self.is_train = is_train
        self.enable_superpixels = enable_superpixels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx].astype(np.float32)
        if self.is_train:
            # # 1. 随机翻转（保持光谱一致）
            # if np.random.rand() < 0.5:  
            #     data = np.flip(data, axis=0).copy()    # 上下翻转
            # if np.random.rand() < 0.5:
            #     data = np.flip(data, axis=1).copy()    # 左右翻转

            # # 2. 随机旋转 90/180/270（不会破坏光谱）
            # k = np.random.choice([0, 1, 2, 3])
            # data = np.rot90(data, k, axes=(0, 1)).copy()

            # 3. 光谱方向加噪（重要：保留光谱形状）
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.01, size=data.shape).astype(np.float32)
                data = data + noise

            # 4. 随机光谱缩放（模拟不同成像条件）
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                data = data * scale



        if self.enable_superpixels:
            data_, pos, mask, labels = hyperspectral_superpixels(data)
        else:
            data = torch.from_numpy(data)
            H,W,C = data.shape
            yy, xx = torch.meshgrid(
                            torch.arange(1,H+1) / H,
                            torch.arange(1,H+1) / W,
                            indexing='ij'
                        )
            pos = torch.stack([xx, yy], dim=-1)

            data_ = torch.flatten(data, start_dim=0, end_dim=1)
            
            pos = torch.flatten(pos, start_dim=0, end_dim=1)
            mask = np.ones(len(data_), dtype=bool)
            
        # img_show = data[..., 20]
        # img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
        # img_rgb = np.stack([img_show]*3, axis=-1)
        # vis = mark_boundaries(img_rgb, labels, color=(1, 0, 0))
        
        # plt.imsave('slic.png', vis)

        return data_, self.wave.astype(np.float32), pos, mask, self.label[idx].astype(np.longlong)
    





if __name__ == '__main__':


    batch_size = 16
    factor = 1.
    patch_size = 9
    np.random.seed(5)
    enable_superpixels = True
    

    source_data_path = './data/Huston/huston2013.mat'

    source_data, source_labels, source_wavelength = read_Huston(source_data_path)
    
    # from util.color import palette
    # rgb_s = palette[source_labels.astype('int')]
    # plt.imsave('Huston2013_gt.png', rgb_s)
    



    target_data_path = './data/Huston/huston2018.mat'

    target_data, target_labels, target_wavelength = read_Huston(target_data_path)
    
    
    Ws, Hs, Cs = source_data.shape
    Wt, Ht, Ct = target_data.shape
    source_data = source_data.reshape(-1, Cs)
    target_data = target_data.reshape(-1, Ct)
    data = np.concat((source_data,target_data))
    data = minmax_scale(data)
    source_data = data[:Hs*Ws]
    target_data = data[Hs*Ws:]
    source_data = source_data.reshape(Ws, Hs, Cs)
    target_data = target_data.reshape(Wt, Ht, Ct)
    
    
    source_x_train, source_x_test, source_y_train, source_y_test, source_coordinate_train, source_coordinate_test = split_train_test(
    source_data, source_labels, patch_size=patch_size, ratio=5, augment_times=4)

    # 训练集
    source_train_Dataset = HyperDataset(source_x_train, source_y_train, source_wavelength, is_train=True, enable_superpixels=enable_superpixels)
    source_train_DataLoader = DataLoader(source_train_Dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True,num_workers=4,persistent_workers=True)
    
    # 测试集
    source_test_Dataset = HyperDataset(source_x_test, source_y_test, source_wavelength, is_train=False, enable_superpixels=enable_superpixels)
    source_test_DataLoader = DataLoader(source_test_Dataset, batch_size=150, shuffle=False, drop_last=False, pin_memory=True,num_workers=4,persistent_workers=True)
    

    
    target_x_train, target_x_test, target_y_train, target_y_test, target_coordinate_train, target_coordinate_test\
    = split_train_test(target_data, target_labels, patch_size=patch_size, ratio=0)
    
    
    
    target_test_Dataset = HyperDataset(target_x_test, target_y_test, target_wavelength, is_train=False)
    target_test_DataLoader = DataLoader(target_test_Dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True, num_workers=4, persistent_workers=True)






    epochs = 200
    foundationmodel = SpectralSharedEncoder(
                    embedding_dim=256,
                    encoder_depth=8,
                    decoder_depth=4,
                    num_heads=8,)

    ckpt = torch.load('10_base_mask95_checkpoint.pt')
    weights = OrderedDict()
    for k, v in ckpt['model'].items():
        name = k[7:]
        weights[name] = v
    foundationmodel.load_state_dict(weights)
    print('✅ Weights loaded.')

    model = AdapterEncoder(backbone=foundationmodel,
                           class_num=7,
                           frez_patch_embeding=True,
                           unfrez_block_num=0,
                           frez_normal=False,
                           frez_global_token=True,
                           )
    
    print_trainable_parameters(model)

    model.cuda()

    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05,
    #                         betas=(0.9, 0.95), eps=1e-6)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=1e-5,
                            weight_decay=5e-4,
                            betas=(0.9, 0.95),
                            eps=1e-6)
    
    optimizer.zero_grad()
    loss_ce = nn.CrossEntropyLoss()


    steps = 0
    
    best_OA = 0
    best_AA = 0
    best_Kappa = 0
    best_aa = []
    
    for epoch in range(epochs):
        # len_dataloader = min(len(source_train_DataLoader), len(target_train_DataLoader))
        len_dataloader = len(source_train_DataLoader)
        data_source_iter = iter(source_train_DataLoader)
        # print(f"Epoch{epoch}: Total samples:{len_dataloader}***************************************************************")
        iters = 0
        while iters < len_dataloader:
            model.train()
            model.zero_grad()
            
            source_data, source_wave, source_pos, source_mask, class_labels  = data_source_iter.__next__()
            
            source_data = source_data.cuda()
            source_wave = source_wave.cuda()
            source_pos = source_pos.cuda()
            source_mask = source_mask.cuda()
            class_labels = class_labels.cuda()
            domain_label = torch.zeros((batch_size,),dtype=torch.long).cuda()

            source_cls_logits, source_feats = model(source_data,
                                                    source_wave,
                                                    source_pos,
                                                    source_mask)
            
            # prob_s = F.softmax(source_cls_logits, dim=1)
            cls_loss  = loss_ce(source_cls_logits, class_labels) 
            
            loss  = cls_loss
            loss.backward()
            optimizer.step()
            iters += 1  
            steps += 1
            if steps % 100 == 0:
                # print(f"[Step {steps}] Loss: {loss.item():.4f} | cls Loss: {cls_loss.item():.4f} | jmmd loss: {jmmd_loss.item():.4f}")  
                print(f"[Step {steps}] Loss: {loss.item():.4f} ")  

        if (epoch+1) % 50 == 0 or epoch==epochs-1:
            
            model.eval()
            target_preds = np.empty(shape=(0,))
            target_labels = np.empty(shape=(0,))

            # 源域测试
            with torch.no_grad():
                for (target_data, target_wave, target_pos, target_mask, class_labels) in source_test_DataLoader:
                    target_data = target_data.cuda()
                    target_wave = target_wave.cuda()
                    target_pos = target_pos.cuda()
                    target_mask = target_mask.cuda()
                    class_labels = class_labels.cuda()
                    
                    
                    target_cls_logits, target_domain_logits = model(target_data,
                                                                target_wave,
                                                                target_pos,
                                                                target_mask)
                    
                    pred = target_cls_logits.argmax(-1).cpu().numpy()
                
                    target_preds = np.concatenate((target_preds, pred), axis=0)
                    target_labels = np.concatenate((target_labels, class_labels.cpu().numpy()), axis=0)
                
            target_labels = target_labels.astype('int')
            cls_nums = {}
            for i in range(target_labels.max()+1):
                cls_nums[i] = sum(target_labels == i)
        
            from sklearn.metrics import confusion_matrix
            
            CM = confusion_matrix(target_labels, target_preds)
            aa = []
            for i in range(target_labels.max()+1):
                aa.append(CM[i, i] / cls_nums[i])
            AA = sum(aa) / len(aa)
            
            l = target_preds - target_labels
            OA = sum(l == 0) / len(l)
            
            k = 0
            for i in range(target_labels.max()+1):
                c = sum(CM[i, :])
                r = sum(CM[:, i])
                k = k + r * c
            k = k / (len(target_preds) ** 2)
            Kappa = (OA - k) / (1 - k)
            print(f'[Step {steps}]:  OA:{OA}  AA:{AA}  Kappa:{Kappa}')
            if best_OA < OA:
                torch.save(model.state_dict(), "Huston_7_5.pth")
                best_OA = OA
                best_AA = AA
                best_aa = aa
                best_Kappa = Kappa
    print('Few Shot Reslut:')            
    print(f'[Best]:  OA:{best_OA}  AA:{best_AA}  Kappa:{best_Kappa}')
    for i, acc in enumerate(best_aa):
        print(f'class {i}: {acc}')
    
   
    all_labels = np.concat((source_y_train, target_preds))+1
    all_coordinate_train=np.concat((source_coordinate_train, source_coordinate_test))
    ys = all_coordinate_train[:, 0]
    xs = all_coordinate_train[:, 1]
    pred_map = np.pad(np.zeros_like(source_labels), patch_size//2, mode='constant') 
    pred_map[ys, xs]=all_labels
    pred_map = pred_map[patch_size//2:-(patch_size//2), patch_size//2:-(patch_size//2)]
    from util.color import palette
    rgb_s = palette[pred_map.astype('int')]
    plt.imsave('Huston_7.png', rgb_s)
    
    
    # ###### cross sence inference 
    
    # model = AdapterEncoder(backbone=foundationmodel,
    #                        class_num=7,
    #                        frez_patch_embeding=True,
    #                        unfrez_block_num=0,
    #                        frez_normal=False,
    #                        frez_global_token=True,
    #                        )
    
    # state_dict = torch.load("Huston_7_5.pth", map_location="cpu")
    # model.load_state_dict(state_dict)
    # model.cuda()
    
    


    
    
    
    # model.eval()
    # target_preds = np.empty(shape=(0,))
    # target_labels = np.empty(shape=(0,))

    # with torch.no_grad():
    #     for (target_data, target_wave, target_pos, target_mask, class_labels) in target_test_DataLoader:
    #         target_data = target_data.cuda()
    #         target_wave = target_wave.cuda()
    #         target_pos = target_pos.cuda()
    #         target_mask = target_mask.cuda()
    #         class_labels = class_labels.cuda()
            
            
    #         target_cls_logits, target_domain_logits = model(target_data,
    #                                                     target_wave,
    #                                                     target_pos,
    #                                                     target_mask)
            
    #         pred = target_cls_logits.argmax(-1).cpu().numpy()
        
    #         target_preds = np.concatenate((target_preds, pred), axis=0)
    #         target_labels = np.concatenate((target_labels, class_labels.cpu().numpy()), axis=0)
        
    # target_labels = target_labels.astype('int')
    # cls_nums = {}
    # for i in range(target_labels.max()+1):
    #     cls_nums[i] = sum(target_labels == i)

    # from sklearn.metrics import confusion_matrix
    
    # CM = confusion_matrix(target_labels, target_preds)
    # aa = []
    # for i in range(target_labels.max()+1):
    #     aa.append(CM[i, i] / cls_nums[i])
    # AA = sum(aa) / len(aa)
    
    # l = target_preds - target_labels
    # OA = sum(l == 0) / len(l)
    
    # k = 0
    # for i in range(target_labels.max()+1):
    #     c = sum(CM[i, :])
    #     r = sum(CM[:, i])
    #     k = k + r * c
    # k = k / (len(target_preds) ** 2)
    # Kappa = (OA - k) / (1 - k)
    # print(f'[Cross Sence Inference]:  OA:{OA}  AA:{AA}  Kappa:{Kappa}')

    # for i, acc in enumerate(aa):
    #     print(f'class {i}: {acc}')
    # # a = 0
