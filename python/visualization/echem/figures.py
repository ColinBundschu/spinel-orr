import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import ml.shared
import visualization.echem.utils as utils
import visualization.echem.plotters as plotters


def create_experimental_fig(base_folder: str, dpi: int, figsize: tuple[int, int] = (12, 8),
                            icon_fontsize: float = 18, fontsize: float = 13) -> None:
    def path(filename):
        return os.path.join(base_folder, filename)

    def create_image_subplot(filepath: str, ax: plt.Axes) -> None:
        img = plt.imread(filepath)
        ax.imshow(img)
        ax.axis('off')

    def scale(ax, ref_ax, scale):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        aspect = scale * (x_max - x_min) / (y_max - y_min)
        ax.set_aspect(aspect, 'box')
        box = ax.get_position()
        dy = ref_ax.get_position().y1 - box.y1
        box.y0 += dy
        box.y1 += dy
        ax.set_position(box)

    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi)

    create_image_subplot(path('cubic_stem_far.png'), axs[0, 0])
    plotters.create_histogram(path('cubic_nanoparticle_sizes.xlsx'), axs[0, 1], fontsize)
    plotters.plot_xrd_data(path('xrd.xlsx'), axs[0, 2], fontsize)
    create_image_subplot(path('cubic_stem_near.png'), axs[1, 0])
    create_image_subplot(path('cubic_diffraction.png'), axs[1, 1])

    df_dict = utils.dfs_from_spreadsheets(path('cv.xlsx'))
    plotters.plot_cv_data(df_dict, axs[1, 2], fontsize)

    for i, ax in enumerate(axs.flatten()):
        icon = chr(ord('a') + i)+')'
        fig.text(0.04, 0.96, icon, fontsize=icon_fontsize, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, backgroundcolor='white')

    fig.tight_layout(w_pad=2)
    scale(axs[0, 1], axs[0, 0], 0.95)
    scale(axs[0, 2], axs[0, 0], 0.95)
    scale(axs[1, 2], axs[1, 0], 0.92)

    plt.savefig(path('experimental_fig.png'), dpi=dpi, bbox_inches='tight')
    plt.show()


def create_S3_fig(
        base_folder: str,
        cv_data_filename: str,
        rde_data_filename: str,
        output_image_filename: str,
        *,
        dpi: int = 80,
        figsize: tuple[int, int] = (12, 8),
        fontsize: float = 13,
) -> None:
    cv_path = os.path.join(base_folder, cv_data_filename)
    rde_path = os.path.join(base_folder, rde_data_filename)
    image_path = os.path.join(base_folder, output_image_filename)

    df_dict = utils.dfs_from_spreadsheets(cv_path)
    fig, axs = plt.subplots(2, len(df_dict), figsize=figsize, dpi=dpi)
    for i, (sheet_name, df) in enumerate(df_dict.items()):
        plotters.plot_cv_data({sheet_name: df}, axs[0, i], fontsize, plot_corrected=True)
    plotters.plot_tafel_data(df_dict, axs[1, 0], fontsize)
    df_kl = pd.read_excel(rde_path, header=0)
    plotters.plot_kl_data(df_kl, axs[1, 1], fontsize)

    fig.tight_layout(w_pad=2)
    plt.savefig(image_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def create_temperature_fig(base_folder: str, dpi: int, figsize: tuple[int, int] = (12, 8),
                           fontsize: float = 13) -> None:
    def path(filename):
        return os.path.join(base_folder, filename)

    df = pd.read_excel(path('cv.xlsx'), header=[0, 1]).swaplevel(i=0, j=1, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    plotters.plot_pd_cv_data(axs[0], fontsize, df)
    plotters.plot_pd_cv_data(axs[1], fontsize, df, tafel=True)
    fig.tight_layout(w_pad=2)
    plt.savefig(path('temp.png'), dpi=dpi, bbox_inches='tight')
    plt.show()


def create_rde_fig(
        base_folder = '/mnt/d/Dropbox/School/PhD Research/2024_jonathan_core_shell/Kl_data',
        file_nums = (144, 145, 146, 150, 162),
        cv_data_filestem = 'JLR-6-X_levich',
        dpi: int = 120,
        fontsize: float = 13,
        ):
    
    nums_str = 'x'.join([str(fn) for fn in file_nums])
    image_path = os.path.join(base_folder, cv_data_filestem.replace('X', nums_str) + '.png')
    fig, axs = plt.subplots(len(file_nums), 2, figsize=(14, 3*len(file_nums)), dpi=dpi)
    for i, file_num in enumerate(file_nums):
        cv_path = os.path.join(base_folder, cv_data_filestem.replace('X', str(file_num)))
        full_df_dict = utils.dfs_from_spreadsheets(cv_path)
        df_dict = {key: df[df['Segment'] == 2] for key, df in full_df_dict.items()}
        plotters.plot_cv_data(df_dict, axs[i,0], fontsize)
        df_inv_j = utils.create_sampled_inv_j_df(df_dict, [0.825, 0.84, 0.85])
        n_e_raw, n_e_corrected = plotters.plot_kl_data(df_inv_j, axs[i,1], fontsize)
        print(f'{Path(cv_path).stem}  n_e_raw={n_e_raw:.2f}  n_e_corrected={n_e_corrected:.2f}')
        # print(df_inv_j)
    fig.tight_layout()
    plt.savefig(image_path, bbox_inches='tight')
    plt.show()
