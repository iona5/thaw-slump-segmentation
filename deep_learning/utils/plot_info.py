import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

FLATUI = {'Turquoise': (0.10196078431372549, 0.7372549019607844, 0.611764705882353),
 'Emerald': (0.1803921568627451, 0.8, 0.44313725490196076),
 'Peter River': (0.20392156862745098, 0.596078431372549, 0.8588235294117647),
 'Amethyst': (0.6078431372549019, 0.34901960784313724, 0.7137254901960784),
 'Wet Asphalt': (0.20392156862745098, 0.28627450980392155, 0.3686274509803922),
 'Green Sea': (0.08627450980392157, 0.6274509803921569, 0.5215686274509804),
 'Nephritis': (0.15294117647058825, 0.6823529411764706, 0.3764705882352941),
 'Belize Hole': (0.1607843137254902, 0.5019607843137255, 0.7254901960784313),
 'Wisteria': (0.5568627450980392, 0.26666666666666666, 0.6784313725490196),
 'Midnight Blue': (0.17254901960784313, 0.24313725490196078, 0.3137254901960784),
 'Sun Flower': (0.9450980392156862, 0.7686274509803922, 0.058823529411764705),
 'Carrot': (0.9019607843137255, 0.49411764705882355, 0.13333333333333333),
 'Alizarin': (0.9058823529411765, 0.2980392156862745, 0.23529411764705882),
 'Clouds': (0.9254901960784314, 0.9411764705882353, 0.9450980392156862),
 'Concrete': (0.5843137254901961, 0.6470588235294118, 0.6509803921568628),
 'Orange': (0.9529411764705882, 0.611764705882353, 0.07058823529411765),
 'Pumpkin': (0.8274509803921568, 0.32941176470588235, 0.0),
 'Pomegranate': (0.7529411764705882, 0.2235294117647059, 0.16862745098039217),
 'Silver': (0.7411764705882353, 0.7647058823529411, 0.7803921568627451),
 'Asbestos': (0.4980392156862745, 0.5490196078431373, 0.5529411764705883)}

def flatui_cmap(*colors):
    ts = np.linspace(0, 1, len(colors))

    segmentdata = dict(
        red=[[t, FLATUI[col][0], FLATUI[col][0]] for col, t in zip(colors, ts)],
        green=[[t, FLATUI[col][1], FLATUI[col][1]] for col, t in zip(colors, ts)],
        blue=[[t, FLATUI[col][2], FLATUI[col][2]] for col, t in zip(colors, ts)],
    )

    return LinearSegmentedColormap('flatui', segmentdata=segmentdata, N=256)

def imageize(tensor):
    return np.clip(tensor.cpu().numpy().transpose(1, 2, 0), 0, 1)


def showexample(batch, preds, idx, filename, channels, writer=None):
    ## First plot
    ROWS = 5
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=0.12, wspace=m)
    N = 1 + int(np.ceil(len(preds) / ROWS))
    fig, ax = plt.subplots(ROWS, N, figsize=(3*N, 3*ROWS), gridspec_kw=gridspec_kw)
    ax = ax.T.reshape(-1)

    flatui = flatui_cmap('Emerald', 'Silver', 'Alizarin', 'Silver', 'Midnight Blue')
    heatmap_args = dict(cmap=flatui, vmin=0, vmax=2)
    heatmap_dem = dict(cmap='RdBu', vmin=0, vmax=0.9)

    batch_img, batch_target = batch
    batch_img = batch_img.to(torch.float)

    # Clear all axes
    for axis in ax:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    if 1 in channels and 2 in channels and 3 in channels:
        nir = channels.index(3)
        b = channels.index(2)
        g = channels.index(1)
        bgnir = imageize(batch_img[idx, [nir, b, g]])
        ax[0].imshow(bgnir)
        ax[0].set_title('B-G-NIR')
    if 4 in channels:
        c = channels.index(4)
        ndvi = imageize(batch_img[idx, [c, c, c]])
        ax[1].imshow(ndvi)
        ax[1].set_title('NDVI')
    if 5 in channels and 6 in channels and 7 in channels:
        r = channels.index(5)
        g = channels.index(6)
        b = channels.index(7)
        tcvis = imageize(batch_img[idx, [r, g, b]])
        ax[2].imshow(tcvis)
        ax[2].set_title('TCVis')
    if 8 in channels:
        c = channels.index(8)
        dem = batch_img[idx, [c, c, c]].cpu().numpy()
        ax[3].imshow(np.clip(dem[0], 0, 1), **heatmap_dem)
        ax[3].set_title('DEM')
    target = batch_target[idx, 0].cpu()
    ax[4].imshow(target, **heatmap_args)
    ax[4].set_title('Target')

    for i, pred in enumerate(preds):
        p = pred[idx].argmax(dim=0).cpu()
        ax[i+ROWS].imshow(p, **heatmap_args)
        ax[i+ROWS].set_title(f'Epoch {i} Prediction')

    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    if writer is not None:
        fig, ax = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw=gridspec_kw)
        if 1 in channels and 2 in channels and 3 in channels:
            nir = channels.index(3)
            b = channels.index(2)
            g = channels.index(1)
            bgnir = imageize(batch_img[idx, [nir, b, g]])
            ax[0].imshow(bgnir)
            ax[0].set_title('B-G-NIR')
        ax[1].imshow(batch_target[idx, 0].cpu(), **heatmap_args)
        ax[1].set_title('Ground Truth')

        pred = preds[-1][idx].argmax(dim=0)
        ax[2].imshow(pred, **heatmap_args)
        ax[2].set_title('Prediction')
        for axis in ax:
            axis.axis('off')
        writer.add_figure(filename.stem, fig, len(preds))


def read_metrics_file(file_path):
    with open(file_path) as src:
        lines = src.readlines()
    data = []
    for line in lines:
        if line.startswith('Phase'):
            continue
        epoch, remain = line.replace(' ', '').replace('\n', '').split('-')
        val_type = remain.split(',')[0].split(':')[0].strip()
        vals = remain.replace('Train:', '').replace('Val:', '').split(',')
        acc_vals = [float(v.split(':')[1]) for v in vals]

        data.append([int(epoch.replace('Epoch', '')), str(val_type), *acc_vals])

    df = pd.DataFrame(columns=['epoch', 'val_type', 'accuracy', 'precision', 'recall', 'f1', 'iou', 'loss'],
                      data=data)
    return df


def plot_metrics(train_metrics, val_metrics, outdir='.'):
    metrics = (set(train_metrics) | set(val_metrics)) - set(['step'])
    for metric in metrics:
        outfile = os.path.join(outdir, f'{metric}.png')
        ax = plt.subplot()
        fig = ax.get_figure()

        if metric in train_metrics:
            ax.plot(train_metrics['step'], train_metrics[metric], c='C0', label='Train')
        if metric in val_metrics:
            ax.plot(val_metrics['step'], val_metrics[metric], c='C1', label='Val')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid()
        fig.savefig(outfile, bbox_inches='tight')
        plt.close()


def plot_precision_recall(train_metrics, val_metrics, outdir='.'):
    fig = plt.figure(figsize=(10, 3))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.plot(train_metrics['step'], train_metrics['Precision'])
    ax1.plot(train_metrics['step'], train_metrics['Recall'])
    ax1.set_title('Train')
    ax2.plot(val_metrics['step'], val_metrics['Precision'])
    ax2.plot(val_metrics['step'], val_metrics['Recall'])
    ax2.set_title('Val')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Epoch')
        ax.legend(['Precision', 'Recall'])
        ax.grid()

    outfile = os.path.join(outdir, f'precision_recall.png')
    fig.savefig(outfile)
    fig.clear()
