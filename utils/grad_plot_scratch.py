import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from mpl_toolkits.axes_grid1 import make_axes_locatable


def renorm(x, pos, neg):
    return (np.maximum(x, 0) / pos) + (((np.minimum(x, 0) * -1) / neg) * -1)


def zscore(x, mu=None, sd=None):
    if not mu:
        mu = x.mean()
    if not sd:
        sd = x.std()
    return (x - mu) / sd


def prep_subplot(f, im, ax, ticks=[-4, 4]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])


def get_grad_plots(
        nne,
        nni,
        e,
        i,
        nims,
        ims,
        logits,
        nlogits,
        out_dir,
        ran=None,
        save_figs=False,
        dpi=300,
        sig=1):
    """Gradients from the hGRU on pathfinder."""
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 8}
    matplotlib.rc('font', **font)
    gmin = -8  # np.min([nze, nzi, ze, zi])
    gmax = 8  # np.max([nze, nzi, ze, zi])
    if ran is None:
        ran = range(len(nne))
    count = 0
    for im in ran:
        f = plt.figure()

        nze = zscore(nne[im], munne, sdnne)  # zscore(nni[im], munni, sdnni)
        nzi = zscore(nni[im], munni, sdnni)  # zscore(nne[im], munne, sdnne)
        ze = zscore(e[im], mue, sde)
        zi = zscore(i[im], mui, sdi)

        # -
        ax1 = plt.subplot(2, 4, 2)
        plt.title('$H^{(1)}$')
        im1 = ax1.imshow(gaussian(nze, sig, preserve_range=True), cmap='PuOr_r', vmin=-4, vmax=4)
        prep_subplot(f=f, im=im1, ax=ax1)

        ax2 = plt.subplot(2, 4, 3)
        plt.title('$H^{(2)}$')
        im2 = ax2.imshow(gaussian(nzi, sig, preserve_range=True), cmap='PuOr_r', vmin=-4, vmax=4)
        prep_subplot(f=f, im=im2, ax=ax2)

        ax3 = plt.subplot(2, 4, 4)
        ndif = gaussian(nze, sig, preserve_range=True) + gaussian(nzi, sig, preserve_range=True)
        plt.title('$H^{(2)} + H^{(1)}, max=%s$' % np.around(ndif.max(), 2))
        im3 = ax3.imshow((ndif), cmap='RdBu_r', vmin=gmin, vmax=gmax)
        prep_subplot(f=f, im=im3, ax=ax3, ticks=[-8, 8])
        plt.subplot(2, 4, 1)
        plt.title('Decision: %s' % nlogits[im])
        plt.imshow(nims[im], cmap='Greys_r')
        plt.axis('off')

        # +
        ax4 = plt.subplot(2, 4, 6)
        plt.title('$H^{(1)}$')
        im4 = ax4.imshow(gaussian(ze, sig, preserve_range=True), cmap='PuOr_r', vmin=-4, vmax=4)
        prep_subplot(f=f, im=im4, ax=ax4)
        ax5 = plt.subplot(2, 4, 7)
        plt.title('$H^{(2)}$')
        im5 = ax5.imshow(gaussian(zi, sig, preserve_range=True), cmap='PuOr_r', vmin=-4, vmax=4)
        prep_subplot(f=f, im=im5, ax=ax5)
        ax6 = plt.subplot(2, 4, 8)
        dif = gaussian(ze, sig, preserve_range=True) + gaussian(zi, sig, preserve_range=True)
        plt.title('$H^{(2)} + H^{(1)}, max=%s$' % np.around(dif.max(), 2))
        im6 = ax6.imshow((dif), cmap='RdBu_r', vmin=gmin, vmax=gmax)
        prep_subplot(f=f, im=im6, ax=ax6, ticks=[-8, 8])
        plt.subplot(2, 4, 5)
        plt.title('Decision: %s' % logits[im])
        plt.imshow(ims[im], cmap='Greys_r')
        plt.axis('off')
        if save_figs:
            out_path = os.path.join(
                out_dir,
                '%s.png' % count)
            count += 1
            plt.savefig(out_path, dpi=dpi)
        else:
            plt.show()
        plt.close(f)


####
neg_data = np.load('movies/hgru_2018_07_30_16_36_57_413272_val_gradients.npz')
pos_data = np.load('movies/hgru_2018_07_30_16_33_07_583801_val_gradients.npz')

nne = neg_data['e_grads'][0]
nni = neg_data['i_grads'][0]
nims = neg_data['og_ims'].squeeze()
nlogits = np.argmax(neg_data['val_logits'], axis=-1).ravel()

e = pos_data['e_grads'][0]
i = pos_data['i_grads'][0]
ims = pos_data['og_ims'].squeeze()
logits = np.argmax(pos_data['val_logits'], axis=-1).ravel()


pos_e = np.max(np.maximum(nne, e))
pos_i = np.max(np.maximum(nni, i))
neg_e = np.min(np.minimum(nne, e)) * -1
neg_i = np.min(np.minimum(nni, i)) * -1
mue = e.mean()
sde = e.std()
mui = i.mean()
sdi = i.std()
munne = nne.mean()
sdnne = nne.std()
munni = nni.mean()
sdnni = nni.std()
labels = ['Path' if x else 'No path' for x in logits]
nlabels = ['Path' if x else 'No path' for x in nlogits]
ran = [10, 12, 14, 19, 23, 2, 3, 26]

get_grad_plots(
        nne=nne,
        nni=nni,
        e=e,
        i=i,
        nims=nims,
        ims=ims,
        logits=labels,
        nlogits=nlabels,
        ran=ran,
        save_figs=True,
        out_dir='grad_ims',
        dpi=300)


