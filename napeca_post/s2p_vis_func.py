import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def check_exist_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def reg_ref_imgs(output_ops, output_fig_dir):
    plt.subplot(1, 4, 1)

    plt.imshow(output_ops['refImg'], cmap='gray', )
    plt.title("Reference Image for Registration");
    refImg=output_ops['refImg'] 

    # maximum of recording over time
    plt.subplot(1, 4, 2)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection");

    plt.subplot(1, 4, 3)
    plt.imshow(output_ops['meanImg'], cmap='gray')
    plt.title("Mean registered image")

    plt.subplot(1, 4, 4)
    plt.imshow(output_ops['meanImgE'], cmap='gray')
    plt.title("High-pass filtered Mean registered image");
    
    
    #might have to change what output_fig_dir is in reference to output_ops['save_path']
    check_exist_dir(output_fig_dir)
    print(output_fig_dir)
    registration_summary_png = os.path.join(output_fig_dir, 'regsitration_summary.png')
    registration_summary_pdf = os.path.join(output_fig_dir, 'regsitration_summary.pdf')

    plt.savefig(registration_summary_png) 
    plt.savefig(registration_summary_pdf)

def reg_displacement_vis(output_ops, output_fig_dir):
    plt.figure(figsize=(18,8))

    plt.subplot(4,1,1)
    plt.plot(output_ops['yoff'][:1000])
    plt.ylabel('rigid y-offsets')

    plt.subplot(4,1,2)
    plt.plot(output_ops['xoff'][:1000])
    plt.ylabel('rigid x-offsets')

    plt.subplot(4,1,3)
    plt.plot(output_ops['yoff1'][:1000])
    plt.ylabel('nonrigid y-offsets')

    plt.subplot(4,1,4)
    plt.plot(output_ops['xoff1'][:1000])
    plt.ylabel('nonrigid x-offsets')
    plt.xlabel('frames')

    plt.show()
    
    check_exist_dir(output_fig_dir)
    registration_xy_displacement_png = os.path.join(output_fig_dir, 'regsitration_xy_displacement.png')
    registration_xy_displacement_pdf = os.path.join(output_fig_dir, 'regsitration_xy_displacement.pdf')

    plt.savefig(registration_xy_displacement_png) 
    plt.savefig(registration_xy_displacement_pdf)

def roi_mask_vis(output_ops, output_fig_dir, x):
    stats_file = os.path.join(output_ops['save_path'], 'stat.npy')
    iscell = np.load(os.path.join(output_ops['save_path'], 'iscell.npy'), allow_pickle=True)[:, 0].astype(int)
    stats = np.load(stats_file, allow_pickle=True)
    Ly=output_ops['Ly']
    Lx=output_ops['Lx']
    
    n_cells = len(stats)

    h = np.random.rand(n_cells)
    hsvs = np.zeros((2, Ly, Lx, 3), dtype=np.float32)

    for i, stat in enumerate(stats):
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
        hsvs[iscell[i], ypix, xpix, 0] = h[i]
        hsvs[iscell[i], ypix, xpix, 1] = 1
        hsvs[iscell[i], ypix, xpix, 2] = lam / lam.max()

    from colorsys import hsv_to_rgb
    rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)

    plt.figure(figsize=(18,18))
    plt.subplot(3, 1, 1)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(3, 1, 2)
    plt.imshow(rgbs[1])
    plt.title("All Cell ROIs")

    plt.subplot(3, 1, 3)
    plt.imshow(rgbs[0])
    plt.title("All non-Cell ROIs")

    plt.tight_layout()
    
    ROI_png = os.path.join(output_fig_dir, f'ROI_{x}.png'.format(x))
    ROI_pdf = os.path.join(output_fig_dir, f'ROI_{x}.pdf'.format(x))
    
    plt.savefig(ROI_png) 
    plt.savefig(ROI_pdf)

def roi_trace_vis(output_ops, output_fig_dir, x):
    f_cells = np.load(os.path.join(output_ops['save_path'], 'F.npy'))
    f_neuropils = np.load(os.path.join(output_ops['save_path'], 'Fneu.npy'))
    spks = np.load(os.path.join(output_ops['save_path'], 'spks.npy'))
    f_cells.shape, f_neuropils.shape, spks.shape

    plt.figure(figsize=[20,20])
    plt.suptitle("Fluorescence and Deconvolved Traces for Different ROIs", y=0.92);
    rois = np.arange(len(f_cells))[::200]
    for i, roi in enumerate(rois):
        plt.subplot(len(rois), 1, i+1, )
        f = f_cells[roi]
        f_neu = f_neuropils[roi]
        sp = spks[roi]
        # Adjust spks range to match range of fluroescence traces
        fmax = np.maximum(f.max(), f_neu.max())
        fmin = np.minimum(f.min(), f_neu.min())
        frange = fmax - fmin 
        sp /= sp.max()
        sp *= frange
        plt.plot(f, label="Cell Fluorescence")
        plt.plot(f_neu, label="Neuropil Fluorescence")
        plt.plot(sp + fmin, label="Deconvolved")
        plt.xticks(np.arange(0, f_cells.shape[1], f_cells.shape[1]/10))
        plt.ylabel(f"ROI {roi}", rotation=0)
        plt.xlabel("frame")
        if i == 0:
            plt.legend(bbox_to_anchor=(0.93, 2))
    
    traces_png = os.path.join(output_fig_dir, f'traces_{x}.png')
    traces_pdf = os.path.join(output_fig_dir, f'traces_{x}.pdf')

    plt.savefig(traces_png) 
    plt.savefig(traces_pdf)