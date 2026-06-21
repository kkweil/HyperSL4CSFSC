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

    data = data['paviaU']
    labels = labels['paviaU_gt_7']
    W, H, C = data.shape
    data = data.reshape(-1, C)
    data = minmax_scale(data)
    data = data.reshape(W, H, C)
    wavelength = np.linspace(wave_range[0], wave_range[1], C)

    return data, labels, wavelength


def read_pavia(data_path, label_path, wave_range):
    data = loadmat(data_path)
    labels = loadmat(label_path)
    data = data['pavia']
    labels = labels['pavia_gt_7']
    W, H, C = data.shape
    data = data.reshape(-1, C)
    data = minmax_scale(data)
    data = data.reshape(W, H, C)
    wavelength = np.linspace(wave_range[0], wave_range[1], C)

    return data, labels, wavelength



def read_indian_pines(data_path, label_path, wave_path):

    remove_bands = list(range(103, 108)) + list(range(149, 163)) + [119]
    hsi = loadmat(data_path)['input']
    gt = loadmat(label_path)['indian_pines_gt']
    W, H, C = hsi.shape
    hsi = hsi.reshape(-1, C)
    hsi = minmax_scale(hsi)
    hsi = hsi.reshape(W, H, C)
    waves = pd.read_csv(wave_path)
    waves = np.array(waves.iloc[:, 1])[:220]
    waves = np.delete(waves, remove_bands)

    return hsi, gt, waves




def read_Chikusei(data_path, label_path):
    img_data = spectral.open_image(data_path)
    lable_data = spectral.open_image(label_path)

    img = img_data.load()
    lable = lable_data.load().astype('int').squeeze()

    # img = np.array(img)
    img = img_data.open_memmap() # H W C
    wavelength = np.asarray([float(i) * 1000 for i in img_data.metadata['wavelength']][8:])

    img = img[:, :, 8:]  # 去除前8个波段

    W, H, C = img.shape
    img = img.reshape(-1, C)
    img = minmax_scale(img)
    img = img.reshape(W, H, C).astype(np.float32)
    lable = lable.astype(np.int64)
    return img, lable, wavelength.numpy()

class HyperDataset(Dataset):
    def __init__(self, data, label, wave):
        self.data = data
        self.label = label
        self.wave = wave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx].astype(np.float32)
        # # 1. 随机翻转（保持光谱一致）
        # if np.random.rand() < 0.5:  
        #     data = np.flip(data, axis=0).copy()    # 上下翻转
        # if np.random.rand() < 0.5:
        #     data = np.flip(data, axis=1).copy()    # 左右翻转

        # # 2. 随机旋转 90/180/270（不会破坏光谱）
        # k = np.random.choice([0, 1, 2, 3])
        # data = np.rot90(data, k, axes=(0, 1)).copy()

        # # 3. 光谱方向加噪（重要：保留光谱形状）
        # if np.random.rand() < 0.7:
        #     noise = np.random.normal(0, 0.01, size=data.shape).astype(np.float32)
        #     data = data + noise

        # # 4. 随机光谱缩放（模拟不同成像条件）
        # if np.random.rand() < 0.5:
        #     scale = np.random.uniform(0.9, 1.1)
        #     data = data * scale




        data_, pos, mask, labels = hyperspectral_superpixels(data)
        
        # img_show = data[..., 20]
        # img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
        # img_rgb = np.stack([img_show]*3, axis=-1)
        # vis = mark_boundaries(img_rgb, labels, color=(1, 0, 0))
        
        # plt.imsave('slic.png', vis)

        return data_, self.wave.astype(np.float32), pos, mask, self.label[idx].astype(np.longlong)
    





if __name__ == '__main__':


    batch_size = 16
    factor = 1.

    np.random.seed(0)
    

    # source_data_path = '/home/lab202/DA/V3_DA/data/Hyperspec_Chikusei_ENVI/Chikusei_ENVI/HyperspecVNIR_Chikusei_20140729.hdr'
    # source_label_path = '/home/lab202/DA/V3_DA/data/Hyperspec_Chikusei_ENVI/Chikusei_ENVI/HyperspecVNIR_Chikusei_20140729_Ground_Truth.hdr'


    # source_data, source_labels, source_wavelength = read_Chikusei(source_data_path, source_label_path)


    source_data_path =  './data/Pavia/pavia.mat'
    source_label_path = './data/Pavia/pavia_gt_7.mat'

    source_data, source_labels, source_wavelength = read_pavia(source_data_path, source_label_path,
                                                                wave_range=[430, 860])
    
    # img_show = source_data[..., [67,30,10]].astype('float32')
    
    # for i in range(3):
    #     img_show[..., i] = (img_show[..., i] - img_show[..., i].min()) / (img_show[..., i].max() - img_show[..., i].min())
    # # img_rgb = np.stack([img_show]*3, axis=-1)
    # # vis = mark_boundaries(img_show, labels, color=(1, 1, 0.5))
    
    # plt.imsave('PaviaC.png', img_show)
    
    # from util.color import cmap_
    # plt.imsave('PaviaC_gt.png', source_labels,  cmap=cmap_, vmin=0, vmax=16)
    
    from util.color import palette
    rgb_s = palette[source_labels.astype('int')]
    plt.imsave('PaviaC_gt.png', rgb_s)

    
    # source_data_path = '/home/lab202/DA/V3_DA/IN/IndianPine.mat'
    # source_label_path = '/home/lab202/DA/V3_DA/IN/Indian_pines_gt.mat'
    # source_wave_path = '/home/lab202/DA/V3_DA/IN/Calibration_Information_for_220_Channel_Data_Band_Set.csv'

    # source_data, source_labels, source_wavelength = read_indian_pines(source_data_path, source_label_path, source_wave_path)
    
    
    # mean_spectra, centers, labels = hyperspectral_superpixels(source_data)

    source_x_train, source_x_test, source_y_train, source_y_test, source_coordinate_train, source_coordinate_test = split_train_test(
        source_data, source_labels, patch_size=9, ratio=5)

    # target_data_path = './data/Pavia/pavia.mat'
    # target_label_path = './data/Pavia/pavia_gt_7.mat'

    # target_data, target_labels, target_wavelength = read_pavia(target_data_path, target_label_path,
    #                                                            wave_range=[430, 860])

    # target_x_train, target_x_test, target_y_train, target_y_test, target_coordinate_train, target_coordinate_test = split_train_test(
    #     target_data, target_labels, patch_size=9, ratio=10)

    # 源域训练集
    source_train_Dataset = HyperDataset(source_x_train, source_y_train, source_wavelength)
    source_train_DataLoader = DataLoader(source_train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,num_workers=4,persistent_workers=True)
    # 源域测试集
    source_test_Dataset = HyperDataset(source_x_test, source_y_test, source_wavelength)
    source_test_DataLoader = DataLoader(source_test_Dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True,num_workers=4,persistent_workers=True)
    
    
    
    # # 目标域训练集-标签不参与模型训练
    # target_train_Dataset = HyperDataset(target_x_train, target_y_train, target_wavelength)
    # target_train_DataLoader = DataLoader(target_train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # # 目标域测试集
    # target_test_Dataset = HyperDataset(target_x_train, target_y_train, target_wavelength)
    # target_test_DataLoader = DataLoader(target_test_Dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    epochs = 800

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

        if (epoch+1) % 50 == 0:
            
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
            for i in range(target_labels.max()):
                cls_nums[i] = sum(target_labels == i)
        
            from sklearn.metrics import confusion_matrix
            
            CM = confusion_matrix(target_labels, target_preds)
            aa = []
            for i in range(target_labels.max()):
                aa.append(CM[i, i] / cls_nums[i])
            AA = sum(aa) / len(aa)
            
            l = target_preds - target_labels
            OA = sum(l == 0) / len(l)
            
            k = 0
            for i in range(target_labels.max()):
                c = sum(CM[i, :])
                r = sum(CM[:, i])
                k = k + r * c
            k = k / (len(target_preds) ** 2)
            Kappa = (OA - k) / (1 - k)
            print(f'[Step {steps}]: target OA:{OA} target AA:{AA} target Kappa:{Kappa}')
            
            
            
    a = 0
