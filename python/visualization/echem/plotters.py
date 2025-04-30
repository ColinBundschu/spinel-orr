import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator

import utils


def add_ticks(ax: plt.Axes, fontsize: float) -> None:
    # Set minor ticks to be exactly one between each major tick
    def minor_ticks(major_ticks):
        return [(major_ticks[i] + major_ticks[i + 1]) / 2 for i in range(len(major_ticks) - 1)]

    # Set the tick label font size
    ax.tick_params(axis='x', which='major', labelsize=fontsize*0.7)
    ax.tick_params(axis='y', which='major', labelsize=fontsize*0.7)

    # Increase the weight of plot box and ticks
    tick_linewidth = 1.1
    ax.tick_params(axis='both', which='both', width=tick_linewidth)
    for spine in ax.spines.values():
        spine.set_linewidth(tick_linewidth)

    x_major_ticks = ax.get_xticks()
    y_major_ticks = ax.get_yticks()
    x_minor_ticks = minor_ticks(x_major_ticks)
    y_minor_ticks = minor_ticks(y_major_ticks)
    ax.xaxis.set_minor_locator(FixedLocator(x_minor_ticks))
    ax.yaxis.set_minor_locator(FixedLocator(y_minor_ticks))

    # Set different tick sizes for major and minor ticks
    ax.tick_params(which='both', direction='inout')
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)


def plot_kl_data(df: pd.DataFrame,
                 ax: plt.Axes,
                 fontsize: float,
                 *,
                 KOH_1M_correction_factor: float = 0.925,
                 legend_loc: str = 'lower right',
                 linewidth: float = 1.5,
                 ) -> tuple[float, float]:

    def rpm_to_omega_neg_half(rpm: float) -> float:
        rad_per_second = 2 * np.pi * rpm / 60
        return rad_per_second**(-1/2)

    # In ORR reactant is O2. Reaction in 1 M KOH
    # Concentration of the reactant in the solution (mol/cm^3)
    # Calculated using eq 1 on page 290 10.1016/0013-4686(67)80007-0
    # C_O2_1M_KOH = 1.53e-6 / 1.08
    C_O2_0p1M_KOH = 1.2e-6
    C_O2_1M_KOH = C_O2_0p1M_KOH * KOH_1M_correction_factor
    # D_O2_1M_KOH = 1.42e-5
    D_O2_0p1M_KOH = 1.9e-5
    D_O2_1M_KOH = D_O2_0p1M_KOH * KOH_1M_correction_factor
    v_1M_KOH = 1e-2  # Kinematic viscosity of the electrolyte (cm^2/s)

    ax.set_xlabel(r'$\omega^{-1/2}$ (rad$^{-1/2}$s$^{1/2}$)', fontsize=fontsize)
    ax.set_ylabel(r'$j^{-1}\ ($mA$^{-1})$', fontsize=fontsize)

    if df.columns.get_loc('RPM') != 0:
        raise ValueError('RPM column is not the first column in the Excel file')
    
    # Infer the number of columns to handle arbitrary columns
    x_data = rpm_to_omega_neg_half(df.loc[:, 'RPM'])
    y_data_list = [df.iloc[:, i] for i in range(1, df.shape[1])]  # Handle all columns after 'RPM'

    markers = ['s', '^', 'o', 'd', 'x', '*']  # Extend marker types if needed
    for y_data, marker in list(zip(y_data_list, markers[:len(y_data_list)]))[::-1]:
        ax.plot(x_data, y_data, linewidth=linewidth, label=f'{y_data.name} V', marker=marker, markersize=6)
    
    add_ticks(ax, fontsize)

    # Compute the number of electrons transferred per reaction
    lowest_voltage_index = np.argmin([float(y_data_item.name) for y_data_item in y_data_list])
    y_lowest_voltage = y_data_list[lowest_voltage_index]
    
    n_electrons_raw = utils.calculate_n_electrons_transferred(
        x_data, y_lowest_voltage, C_O2_0p1M_KOH, D_O2_0p1M_KOH, v_1M_KOH)
    n_electrons = utils.calculate_n_electrons_transferred(x_data, y_lowest_voltage, C_O2_1M_KOH, D_O2_1M_KOH, v_1M_KOH)

    n_e_str = '$n^*_{e^-}$'
    ax.set_title(f'{n_e_str}(1 M KOH corrected)={n_electrons:.2f}     {n_e_str}(Raw)={n_electrons_raw:.2f}', fontsize=fontsize)
    ax.legend(fontsize=fontsize*0.8, loc=legend_loc)
    
    return n_electrons_raw, n_electrons


def plot_pd_cv_data(ax: plt.Axes, fontsize: float, df_all: pd.DataFrame, *,
                    linewidth: float = 1.5, tafel: bool = False) -> None:
    ax.set_xlim([0.9, 1.1])
    ax.set_ylim([-4, 0] if tafel else [-0.3, 0])
    ax.yaxis.get_major_locator().set_params(integer=True)

    temps = df_all.columns.levels[0]
    norm = mcolors.Normalize(vmin=1.3 * min(temps) - 0.3 * max(temps), vmax=max(temps))
    for temp in temps:
        color = plt.cm.hsv_r(norm(temp))
        [df_up, df_down] = utils.sign_change_partition_df(df_all[temp], 'E')
        up_V = df_up['E'].values
        # down_V = df_down['E'].values
        up_i = df_up['i/iL'].values
        # down_i = df_down['i/iL'].values
        if tafel:
            # ax.plot(down_V, np.log10(-down_i), linewidth=linewidth, label=r'$\log_{10}(-j^*/j_0)$', color=color)
            ax.plot(up_V, np.log10(-up_i), linewidth=linewidth, label=r'$\log_{10}(-j^*/j_0)$', color=color)
        else:
            # ax.plot(down_V, down_i, linewidth=linewidth, label='$j$', color=color)
            ax.plot(up_V, up_i, linewidth=linewidth, label='$j$', color=color)

    if tafel:
        ax.set_ylabel('ln($j$)', fontsize=fontsize)
        ax.set_xlabel('V', fontsize=fontsize)
    else:
        ax.set_xlabel('V (V vs. RHE)', fontsize=fontsize)
        ax.set_ylabel('Current Density (mA/$cm^2$)', fontsize=fontsize)
    add_ticks(ax, fontsize)
    # ax.legend(fontsize=fontsize*0.8, loc='center right')


def plot_tafel_data(
    df_dict: str,
    ax: plt.Axes,
    fontsize: float,
    *,
    V_eq: float = 1.23,
    segment: int = 2,
    x_lim: tuple[float, float] = (0.8, 0.95),
    y_lim: tuple[float, float] = (-3, 1),
    linewidth: float = 1.5,
) -> None:
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.yaxis.get_major_locator().set_params(integer=True)
    x_scale = x_lim[1] - x_lim[0]

    multi_data = len(df_dict) > 1
    for sheet_name, df in df_dict.items():
        potential_V = df.loc[df['Segment'] == segment, 'Potential (V)'].values
        j_mApcm2 = df.loc[df['Segment'] == segment, 'Current (mA/cm2)'].values
        V_half, p = utils.three_line_half_wave_potential(potential_V, j_mApcm2)
        corrected_j_mApcm2 = np.array([c - p['e']*(v - max(potential_V))
                                       for v, c in zip(potential_V, j_mApcm2 - max(j_mApcm2))])
        plot_labels = [sheet_name] if multi_data else []
        tafel_label = ' '.join(plot_labels + [r'$\log_{10}(-j^*/j_0)$'])
        log_V, log_j_c = zip(*[(V, np.log10(-j)) for V, j in zip(potential_V, corrected_j_mApcm2) if j < -1e-6])
        ax.plot(log_V, log_j_c, linewidth=linewidth, label=tafel_label)

        if not multi_data:
            ref_j_mApcm2 = np.max(corrected_j_mApcm2)*0.95 + np.min(corrected_j_mApcm2)*0.05
            V_onset = potential_V[np.abs(corrected_j_mApcm2 - ref_j_mApcm2).argmin()]
            E_half_text_y = ax.get_ylim()[0]*0.90+ax.get_ylim()[1]*0.1
            E_half_text_y = ax.get_ylim()[0]*0.95+ax.get_ylim()[1]*0.05

            ax.axvline(x=V_onset, color='#555555', linestyle='--', linewidth=1, zorder=0)
            ax.text(V_onset + x_scale*0.005,
                    E_half_text_y,
                    f'$\\eta_{{V_{{on}}}}={V_eq - V_onset:.2f}$',
                    color='k',
                    fontsize=fontsize,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0),
                    )

            ax.axvline(x=V_half, color='#555555', linestyle='--', linewidth=1, zorder=0)
            ax.text(V_half + x_scale*0.005,
                    E_half_text_y,
                    f'$\\eta_{{V_{{1/2}}}}={V_eq - V_half:.2f}$',
                    color='k',
                    fontsize=fontsize,
                    horizontalalignment='left',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0),
                    )

    ax.set_xlabel('ln($j$)', fontsize=fontsize)
    ax.set_ylabel(r'$\eta$ (V)', fontsize=fontsize)
    ax.legend(fontsize=fontsize*0.8)
    add_ticks(ax, fontsize)


def plot_cv_data(
    df_dict: str,
    ax: plt.Axes,
    fontsize: float,
    slope: float,
    *,
    # x_lim: tuple[float, float] = (0.5, 1.1),
    # y_lim: tuple[float, float] = (-4, 0.5),
    linewidth: float = 1.5,
    plot_corrected: bool = False,
    plot_both: bool = False,
) -> None:
    ax.yaxis.get_major_locator().set_params(integer=True)

    for sheet_name, df in df_dict.items():
        plot_labels = [sheet_name] if len(df_dict) > 1 else []
        plot_label = ' '.join(plot_labels + (['$j$ (Uncorrected)'] if plot_corrected else []))

        potential_V = df['Potential (V)'].values
        j_mApcm2 = df['Current (mA/cm2)'].values
        # ax.plot(potential_V, j_mApcm2, linewidth=linewidth, label=plot_label)
        V_half, p = utils.three_line_half_wave_potential(potential_V, j_mApcm2)
        corrected_j_mApcm2 = np.array([c - slope*(v - max(potential_V)) for v, c in zip(potential_V, j_mApcm2 - max(j_mApcm2))])
        corrected_label = ' '.join(plot_labels + ['$j^*$ (Corrected)'])
        for v, c in zip(potential_V, corrected_j_mApcm2):
            if c <= -0.249:
                print(f'Onset potential {c} for {sheet_name} is {v:.3f} V')
                break
        ax.plot(potential_V, corrected_j_mApcm2, linewidth=linewidth, label=sheet_name)

        # for segment in ([1, 2] if plot_both else [2]):
        #     potential_V = df.loc[df['Segment'] == segment, 'Potential (V)'].values
        #     j_mApcm2 = df.loc[df['Segment'] == segment, 'Current (mA/cm2)'].values
        #     ax.plot(potential_V, j_mApcm2, linewidth=linewidth, label=plot_label)

        #     # V_half, p = utils.three_line_half_wave_potential(potential_V, j_mApcm2)
        #     # corrected_j_mApcm2 = np.array([c - p['e']*(v - max(potential_V))
        #     #                                for v, c in zip(potential_V, j_mApcm2 - max(j_mApcm2))])
        #     if plot_corrected:
        #         corrected_label = ' '.join(plot_labels + ['$j^*$ (Corrected)'])
        #         ax.plot(potential_V, corrected_j_mApcm2, linewidth=linewidth, label=corrected_label)
            # if len(df_dict) == 1:
            #     ref_j_mApcm2 = np.max(corrected_j_mApcm2)*0.95 + np.min(corrected_j_mApcm2)*0.05
            #     V_onset = potential_V[np.abs(corrected_j_mApcm2 - ref_j_mApcm2).argmin()]
            #     E_half_text_y = ax.get_ylim()[0]*0.90+ax.get_ylim()[1]*0.1
            #     ax.axvline(x=V_onset, color='#555555', linestyle='--', linewidth=1, zorder=0)
            #     x_scale = ax.get_xlim()[1] - ax.get_xlim()[0]
            #     ax.text(V_onset + x_scale*0.005,
            #             E_half_text_y,
            #             f'$V_{{on}}={V_onset:.2f}$',
            #             color='k',
            #             fontsize=fontsize,
            #             horizontalalignment='left',
            #             verticalalignment='bottom',
            #             bbox=dict(facecolor='white', edgecolor='none', pad=0),
            #             )
            #     ax.axvline(x=V_half, color='#555555', linestyle='--', linewidth=1, zorder=0)
            #     ax.text(V_half + x_scale*0.005,
            #             E_half_text_y,
            #             f'$V_{{1/2}}={V_half:.2f}$',
            #             color='k',
            #             fontsize=fontsize,
            #             horizontalalignment='left',
            #             verticalalignment='top',
            #             bbox=dict(facecolor='white', edgecolor='none', pad=0),
            #             )

    # if len(df_dict) == 1:
    #     [sheet_name] = df_dict.keys()
    #     ax.set_title(sheet_name, fontsize=fontsize)
    ax.set_xlabel('V (V vs. RHE)', fontsize=fontsize)
    ax.set_ylabel('Current Density (mA/cm$^2$)', fontsize=fontsize)
    ax.legend(fontsize=fontsize*0.8, loc='center right')
    add_ticks(ax, fontsize)


def create_histogram(filepath: str, ax: plt.Axes, fontsize: float, line_width: float = 0.6,
                     x_buffer: int = 10, bin_width: int = 5) -> None:
    data = pd.read_excel(filepath, sheet_name=0)
    column_data = data.iloc[:, 0]

    # Calculate the average and standard deviation of the data
    average = np.mean(column_data)
    std_dev = np.std(column_data)

    # Create the histogram using matplotlib
    plt.rcParams['hatch.linewidth'] = line_width
    x_min = int(math.floor(min(column_data) / bin_width)) * bin_width
    x_max = int(math.ceil(max(column_data) / bin_width)) * bin_width
    n, _, _ = ax.hist(column_data, bins=range(x_min, x_max + bin_width, bin_width),
                      edgecolor='#1f77b4', color='white', hatch='\\\\\\\\\\\\\\\\', linewidth=line_width)
    ax.set_xlabel('Size (nm)', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_ylim(0, np.max(n) * 1.2)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    xticks = np.arange(x_min - x_buffer, x_max + x_buffer + bin_width, bin_width)
    ax.set_xticks(xticks)
    step = 2 * bin_width if x_max < 20 * bin_width else 4 * bin_width
    ax.set_xticklabels([str(int(x)) if int(x) % step == 0 else '' for x in ax.get_xticks()])

    # Add text with average and standard deviation in the upper right corner
    text = f'Size: {average:.0f} \u00B1 {std_dev:.0f} nm'
    ax.text(0.95, 0.95, text, fontsize=fontsize, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)


def plot_xrd_data(filepath: str, ax: plt.Axes, fontsize: float, x_min: float = 20,
                  x_max: float = 90, line_width: float = 0.8) -> None:
    # Read the data from the Excel workbook sheets
    xls = pd.read_excel(filepath, sheet_name=None)
    ref_label, cubic_label = list(xls.keys())[:2]
    assert (ref_label == 'PDF 01-076-1802' and cubic_label == 'Cubic')

    data_ref = xls[ref_label]
    data_cubic = xls[cubic_label]

    # Normalize each dataset to have the same height and start from zero
    for data in (data_ref, data_cubic):
        data.iloc[:, 1] -= data.iloc[:, 1].min()
        data.iloc[:, 1] /= data.iloc[:, 1].max()

    # Shift the datasets vertically to avoid overlap
    data_cubic.iloc[:, 1] += 1.2

    # Plot the data with specified colors and linewidth
    ax.plot(data_cubic.iloc[:, 0], data_cubic.iloc[:, 1], label='Measured', linewidth=line_width)
    ax.plot(data_ref.iloc[:, 0], data_ref.iloc[:, 1], label=ref_label, linewidth=line_width, color='black')

    ax.set_xlabel('$2\\Theta$', fontsize=fontsize)
    ax.set_ylabel('Intensity (arb. units)', fontsize=fontsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.3, 2.8)
    ax.set_yticks([])
    ax.legend(fontsize=fontsize*0.9)
