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
from datautils.superpixel import fast_spectral_kmeans_aggregate_numpy

def hyperspectral_superpixels(
    img_hwc,
    n_segments=30,
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

    # labels, _, _ = fast_spectral_kmeans_aggregate_numpy(
    #     img_hwc, K_max=n_segments, A_min=4, iters=max_iter, feat_dim=compactness, init="kmeans++", return_std=True
    # )

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


def read_paviaU(data_path, label_path, wave_range):
    data = loadmat(data_path)
    labels = loadmat(label_path)
    data = data['paviaU'].astype('float32')
    labels = labels['paviaU_gt']
    W, H, C = data.shape
    data = data.reshape(-1, C)
    data = minmax_scale(data)
    data = data.reshape(W, H, C)
    wavelength = np.linspace(wave_range[0], wave_range[1], C)

    return data, labels, wavelength


def read_pavia(data_path, label_path, wave_range):
    data = loadmat(data_path)
    labels = loadmat(label_path)
    data = data['pavia'].astype('float32')/5000.
    labels = labels['pavia_gt_7']
    W, H, C = data.shape
    # data = data.reshape(-1, C)
    # data = minmax_scale(data)
    # data = data.reshape(W, H, C)*2-1
    wavelength = np.linspace(wave_range[0], wave_range[1], C)

    return data, labels, wavelength






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
            # 1. 随机翻转（保持光谱一致）
            if np.random.rand() < 0.5:  
                data = np.flip(data, axis=0).copy()    # 上下翻转
            if np.random.rand() < 0.5:
                data = np.flip(data, axis=1).copy()    # 左右翻转

            # 2. 随机旋转 90/180/270（不会破坏光谱）
            k = np.random.choice([0, 1, 2, 3])
            data = np.rot90(data, k, axes=(0, 1)).copy()

        # # 3. 光谱方向加噪（重要：保留光谱形状）
        # if np.random.rand() < 0.7:
        #     noise = np.random.normal(0, 0.01, size=data.shape).astype(np.float32)
        #     data = data + noise

        # # 4. 随机光谱缩放（模拟不同成像条件）
        # if np.random.rand() < 0.5:
        #     scale = np.random.uniform(0.9, 1.1)
        #     data = data * scale



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
    patch_size = 15
    # np.random.seed(5) #5
    pertrain = True
    PEFT = True
    enable_superpixels = True
    
   

    source_data_path = './data/Pavia/PaviaU.mat'
    source_label_path = './data/Pavia/paviaU_gt.mat'

    source_data, source_labels, source_wavelength = read_paviaU(source_data_path, source_label_path,
                                                                wave_range=[430, 860])
    
    source_x_train, source_x_test, source_y_train, source_y_test, source_coordinate_train, source_coordinate_test = split_train_test(
        source_data, source_labels, patch_size=patch_size, ratio=5, augment_times=4)

    # 训练集
    source_train_Dataset = HyperDataset(source_x_train, source_y_train, source_wavelength, is_train=True, enable_superpixels=enable_superpixels)
    source_train_DataLoader = DataLoader(source_train_Dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True,num_workers=4,persistent_workers=True)
    
    # 测试集
    source_test_Dataset = HyperDataset(source_x_test, source_y_test, source_wavelength, is_train=False, enable_superpixels=enable_superpixels)
    source_test_DataLoader = DataLoader(source_test_Dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=True,num_workers=4,persistent_workers=True)


    target_data_path = './data/Pavia/pavia.mat'
    target_label_path = './data/Pavia/pavia_gt_7.mat'

    target_data, target_labels, target_wavelength = read_pavia(target_data_path, target_label_path,
                                                                wave_range=[430, 860])
    
    target_x_train, target_x_test, target_y_train, target_y_test, target_coordinate_train, target_coordinate_test\
    = split_train_test(target_data, target_labels, patch_size=patch_size, ratio=0)
    
    
    
    target_test_Dataset = HyperDataset(target_x_test, target_y_test, target_wavelength, is_train=False)
    target_test_DataLoader = DataLoader(target_test_Dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=True, num_workers=4, persistent_workers=True)


    # from util.color import palette
    # rgb_s = palette[source_labels.astype('int')]
    # plt.imsave('Pavia_gt.png', rgb_s)
    
    # target_data_path = './data/Pavia/pavia.mat'
    # target_label_path = './data/Pavia/pavia_gt_7.mat'

    # target_data, target_labels, target_wavelength = read_pavia(target_data_path, target_label_path,
    #                                                             wave_range=[430, 860])
   
    # data_, pos, mask, labels = hyperspectral_superpixels(source_data[160:240,100:180] ,n_segments=50)

    # img_show = source_data[..., [67,30,10]].astype('float32')
    
    # for i in range(3):
    #     img_show[..., i] = (img_show[..., i] - img_show[..., i].min()) / (img_show[..., i].max() - img_show[..., i].min())
    # # img_rgb = np.stack([img_show]*3, axis=-1)
    # # vis = mark_boundaries(img_show, labels, color=(1, 1, 0.5))
    
    # plt.imsave('PaviaU.png', img_show)
    
    # from util.color import cmap_
    # plt.imsave('PaviaU_gt.png', source_labels, cmap=cmap_, vmin=0, vmax=16)
    # plt.imsave('Huston18_gt.png', label_t, cmap=cmap_)
    
   
    

    epochs = 800
    foundationmodel = SpectralSharedEncoder(
                    embedding_dim=256,
                    encoder_depth=8,
                    decoder_depth=4,
                    num_heads=8,)
    if pertrain:
        ckpt = torch.load('10_base_mask95_checkpoint.pt')
        weights = OrderedDict()
        for k, v in ckpt['model'].items():
            name = k[7:]
            weights[name] = v
        foundationmodel.load_state_dict(weights)
        print('✅ Weights loaded.')

    if PEFT:
        model = AdapterEncoder(backbone=foundationmodel,
                           class_num=9,
                           frez_patch_embeding=True,
                           unfrez_block_num=0,
                           frez_normal=False,
                           frez_global_token=True,
                           )
    else:
        model = AdapterEncoder(backbone=foundationmodel,
                            class_num=9,
                            frez_patch_embeding=False,
                            unfrez_block_num=8,
                            frez_normal=False,
                            frez_global_token=False,
                            )
    
    print_trainable_parameters(model)
    
    from util.count_flops import evaluate_model_compute_thop
    
    data_source_iter = iter(source_train_DataLoader)
    source_data, source_wave, source_pos, source_mask, class_labels  = data_source_iter.__next__()
    
    flops = evaluate_model_compute_thop(model, (source_data, source_wave, source_pos, source_mask))
    print(flops)
    
    
    

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
        len_dataloader = len(source_train_DataLoader)
        data_source_iter = iter(source_train_DataLoader)
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
            with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                source_cls_logits, source_feats = model(source_data,
                                                        source_wave,
                                                        source_pos,
                                                        source_mask)
                
                cls_loss  = loss_ce(source_cls_logits, class_labels) 
            
            loss  = cls_loss
            loss.backward()
            optimizer.step()
            iters += 1  
            steps += 1
            if steps % 100 == 0:
                print(f"[Step {steps}] Loss: {loss.item():.4f} ")  

        if (epoch+1) % 100 == 0 or epoch==epochs-1:
            
            model.eval()
            target_preds = np.empty(shape=(0,))
            target_labels = np.empty(shape=(0,))

            # 源域测试
            import time
            start = time.time()
            with torch.no_grad():
                for (target_data, target_wave, target_pos, target_mask, class_labels) in source_test_DataLoader:
                    target_data = target_data.cuda()
                    target_wave = target_wave.cuda()
                    target_pos = target_pos.cuda()
                    target_mask = target_mask.cuda()
                    class_labels = class_labels.cuda()
                    
                    with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        target_cls_logits, target_domain_logits = model(target_data,
                                                                    target_wave,
                                                                    target_pos,
                                                                    target_mask)
                    
                    pred = target_cls_logits.argmax(-1).cpu().numpy()
                
                    target_preds = np.concatenate((target_preds, pred), axis=0)
                    target_labels = np.concatenate((target_labels, class_labels.cpu().numpy()), axis=0)
            print(f"infer time {time.time() -start}s")    
            
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
                torch.save(model.state_dict(), "Pavia.pth")
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
    plt.imsave('Pavia_7_pred.png', rgb_s)
    
    
    ###### cross sence inference 
    
    # model = AdapterEncoder(backbone=foundationmodel,
    #                        class_num=7,
    #                        frez_patch_embeding=True,
    #                        unfrez_block_num=0,
    #                        frez_normal=False,
    #                        frez_global_token=True,
    #                        )
    
    # state_dict = torch.load("Pavia_7_da.pth", map_location="cpu")
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
    # a = 0
