import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import linregress


def sigmoid(x: float, a: float, b: float, c: float, d: float, e: float) -> float:
    # Asymmetric sigmoid with a linear component
    return e*x + c + (d-c) / (1 + np.exp(-a*(x-b)))


def sigmoid_deriv(x: float, a: float, b: float, c: float, d: float, e: float) -> float:
    return e + ((d-c) * a * np.exp(-a*(x-b))) / (1 + np.exp(-a*(x-b)))**2


def three_line_half_wave_potential(potential_V: pd.Series, current_mApcm2: pd.Series, ax: plt.Axes = None
                                   ) -> tuple[float, dict[str, float]]:
    '''
    One way to fit a CV curve is to fit flat lines to the low and high potential
    regions, which ideally will be approximately parallel. Then a line can be 
    fit to the steepest part in the middle, which will intersect the other two lines.
    The average of the y-values from the two intersections is the half wave current, 
    and we can then find the corresponding potential.

    If a plot axis is passed in, this will also plot the resulting fit, lines, and points 
    '''

    # These parameter seeds roughly interpolated the optimized values for test cases
    params = lmfit.Parameters()
    params.add('a', value=40.0)
    params.add('b', value=0.8)
    params.add('c', value=-4.0)
    params.add('d', value=0.0)
    params.add('e', value=0.2)
    model = lmfit.Model(sigmoid)
    result = model.fit(current_mApcm2, params, x=potential_V)

    p: dict[str, float] = result.params.valuesdict()  # convert Parameters to dict
    v_mid = potential_V[np.argmax(sigmoid_deriv(potential_V, **p))]
    v_vals = [potential_V[0], v_mid, potential_V[-1]]

    # compute the slopes and y intercepts
    slopes = [sigmoid_deriv(v, **p) for v in v_vals]
    y_intercepts = [sigmoid(v, **p) - slope * v for v, slope in zip(v_vals, slopes)]

    # solve for intersection points
    x_intersect_01 = (y_intercepts[1] - y_intercepts[0]) / (slopes[0] - slopes[1])
    x_intersect_12 = (y_intercepts[2] - y_intercepts[1]) / (slopes[1] - slopes[2])

    # calculate y-values at the intersection points directly
    y_intersect_01 = slopes[0] * x_intersect_01 + y_intercepts[0]
    y_intersect_12 = slopes[1] * x_intersect_12 + y_intercepts[1]

    # Check that the intersections indeed happen at the calculated points
    if not (np.isclose(y_intersect_01, slopes[1] * x_intersect_01 + y_intercepts[1]) and
            np.isclose(y_intersect_12, slopes[2] * x_intersect_12 + y_intercepts[2])):
        raise ValueError('Intersections do not match calculated values')

    # compute the half wave potential
    half_wave_current = (y_intersect_01 + y_intersect_12) / 2
    [half_wave_potential] = fsolve(lambda x: sigmoid(x, **p) - half_wave_current, v_mid)

    if ax is not None:
        # Stash the current limits to avoid lines resizing the plot
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        # plot the best fit
        plt.plot(potential_V, result.best_fit, 'r-', label='fit')

        # plot the lines
        ax.plot(potential_V, slopes[0] * potential_V + y_intercepts[0], color='pink')
        ax.plot(potential_V, slopes[1] * potential_V + y_intercepts[1], color='green')
        ax.plot(potential_V, slopes[2] * potential_V + y_intercepts[2], color='blue')

        # Plot the intersect points
        ax.plot(x_intersect_01, slopes[0] * x_intersect_01 + y_intercepts[0], 'ko')
        ax.plot(x_intersect_12, slopes[1] * x_intersect_12 + y_intercepts[1], 'ko')

        # plot the half wave potential
        ax.axvline(x=half_wave_potential, color='lightblue', linestyle='--')
        ax.text(half_wave_potential, ax.get_ylim()[0], f'$E_{{1/2}}={half_wave_potential:.3f}$ V', color='blue')

        # Reset the limits to what they were before plotting
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    return half_wave_potential, p


# def modified_transfer_function(x, Von_V, leakage_mApcm2V, j_limit_mApcm2, n):
#     """Defines the modified high-pass transfer function with renamed parameters."""
#     return j_limit_mApcm2  / np.sqrt(1 + (x / Von_V) ** (2 * n)) + leakage_mApcm2V * x


def modified_transfer_function(x, Von_V, leakage_mApcm2V, j_limit_mApcm2, n):
    """Defines the modified high-pass transfer function with renamed parameters."""
    return j_limit_mApcm2 *( 1 -  1/ np.sqrt(1 + (Von_V/ x) ** (2 * n))) + leakage_mApcm2V * x


def fit_transfer_function(potential_V: pd.Series, current_mApcm2: pd.Series, ax: plt.Axes = None
                         ) -> tuple[float, dict[str, float]]:
    '''
    Fit the modified high-pass transfer function with linear component to RDE CV curve data.
    
    Parameters:
        potential_V: pd.Series - Potential values (V).
        current_mApcm2: pd.Series - Current density values (mA/cm^2).
        ax: plt.Axes (optional) - Matplotlib Axes to plot the fit and results.
    
    Returns:
        Von_V: float - Estimated onset potential.
        params: dict[str, float] - Dictionary of fitted parameters {Von_V, leakage_mApcm2V, j_limit_mApcm2}.
    '''
    # Set up initial parameter guesses for fitting
    params = lmfit.Parameters()
    params.add('Von_V', value=0.8)
    params.add('leakage_mApcm2V', value=0.2)
    params.add('j_limit_mApcm2', value=-4.0)
    params.add('n', value=25.0, min=1)

    model = lmfit.Model(modified_transfer_function)
    result = model.fit(current_mApcm2, params, x=potential_V)
    p: dict[str, float] = result.params.valuesdict()  # convert Parameters to dict
    Von_V = p['Von_V']

    if ax is not None:
        # Preserve original axis limits
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        # Plot the best fit curve
        ax.plot(potential_V, result.best_fit, 'r-', label='fit')

        # plot the onset potential
        ax.axvline(x=Von_V, color='lightblue', linestyle='--')
        ax.text(Von_V, ax.get_ylim()[0], '$V_{on}='f'{Von_V:.3f}$ V', color='blue')

        # Reset limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    p['leakage_mApcm2V'] = 0.0
    return Von_V, p


def calculate_n_electrons_transferred(omega_neg_half: float, inverse_current_cm2pmA: float, C_molpcm: float,
                                      D_cm2ps: float, v_cm2ps: float) -> float:
    '''
    From Rotating Disk Electrode Data:
        omega_neg_half: ω^(-1/2) ((s/radians)^(1/2))
        inverse_current_mA: 1/j (cm^2/mA)

    Koutecky-Levich equation
        1/i = 1/i_k + (1/B)ω^(-1/2)
        Linear equation: Slope of 1/i vs. ω^(-1/2) equals (1/B)

    B (Levich constant): 0.62*n*F*A*D^(2/3)*v^(-1/6)*C
        n: electrons transferred per reaction
        F: Faraday constant = 96485 (C/mol)
        A: Electrode area. For a 5mm glassy carbon electrode, A = 0.19635 (cm^2)
        D: Diffusion coefficient of reactant (cm^2 s^-1)
        v: Kinematic viscosity of the electrolyte (cm^2 s^-1)
        C: Bulk concentration reactant (mol cm^-1)
    '''
    # Perform a linear regression on the Koutecky-Levich plot (1/j vs. ω^(-1/2))
    slope, _, _, _, _ = linregress(omega_neg_half, inverse_current_cm2pmA*1000)
    F_Cpmol = 96485  # Faraday constant (C/mol)
    B_over_n = 0.62 * F_Cpmol * C_molpcm * D_cm2ps**(2/3) * v_cm2ps**(-1/6)
    n_electrons = 1/(B_over_n*slope)
    return n_electrons


def bin_data_sequential(df: pd.DataFrame, bin_size: float, *, x_col: str = 'Potential (V)',
                        y_col: str = 'Current (mA/cm2)') -> pd.DataFrame:
    binned_data = []
    bin_start_i = 0
    segment = 0  # Never use the 0 index
    diff_lt_0 = None
    for i in range(1, len(df)):
        diff = df.iloc[i][x_col] - df.iloc[bin_start_i][x_col]
        if abs(diff) > bin_size or i == len(df) - 1:
            if (diff < 0) != diff_lt_0:
                segment += 1
                diff_lt_0 = diff < 0
            bin_data = df.iloc[bin_start_i:i]
            binned_data.append({'Segment': segment, x_col: bin_data[x_col].mean(), y_col: bin_data[y_col].mean()})
            bin_start_i = i
    bin_data = df.iloc[bin_start_i:len(df)]
    binned_data.append({'Segment': segment, x_col: bin_data[x_col].mean(), y_col: bin_data[y_col].mean()})
    return pd.DataFrame(binned_data)


def dfs_from_spreadsheets(filepath: str) -> dict[str, pd.DataFrame]:
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    df_dict = {}
    for sheet_name, df in all_sheets.items():
        if 'Current (mA/cm^2)' in df:
            df['Current (mA/cm2)'] = df['Current (mA/cm^2)']
        elif 'Current (mA)' in df:
            df['Current (mA/cm2)'] = df['Current (mA)'] / 0.19635
        elif 'Current (µA)' in df:
            df['Current (mA/cm2)'] = df['Current (µA)'] / (1000 * 0.19635)
        else:
            df['Current (mA/cm2)'] = 1000 * df['Current (A)'] / 0.19635
        if 'Potential (mV)' in df:
            df['Potential (V)'] = 0.956 +  df['Potential (mV)'] / 1000
        if 'Segment' not in df:
            # df = bin_data_sequential(df, 0.0005)
            df = df[(df['Potential (V)'] >= 0.53) & (df['Potential (V)'] <= 1.08)]
        df_dict[sheet_name] = df

    return df_dict


def sign_change_partition_df(df: pd.DataFrame, index: str) -> list[pd.DataFrame]:
    sign_switch_indices = np.where(np.diff(np.sign(np.diff(df[index]))) != 0)[0] + 1
    dfs = []
    i_start = 0
    for i_end in sign_switch_indices:
        dfs.append(df.iloc[i_start:i_end])
        i_start = i_end
    dfs.append(df.iloc[i_start:])
    return dfs


def create_sampled_inv_j_df(df_dict: dict, sample_points: list, *,
                            x_col='Potential (V)', y_col='Current (mA/cm2)') -> pd.DataFrame:
    rows = []
    for key, df in df_dict.items():
        rpm = int(key.split('-')[-1].replace('rpm', ''))
        inverse_current_values = []
        sorted_df = df.sort_values(by=x_col)
        for point in sample_points:
            # Find the closest point below point and the closest above point in x_col
            below_point = sorted_df[sorted_df[x_col] <= point].iloc[-1]
            above_point = sorted_df[sorted_df[x_col] >= point].iloc[0]

            # Add the weighted average of the y_col values of these points to the sampled values list
            weight = (point - below_point[x_col]) / (above_point[x_col] - below_point[x_col])
            sampled_value = (1 - weight) * below_point[y_col] + weight * above_point[y_col]
            inverse_current_values.append(-1/sampled_value)
        rows.append([rpm] + inverse_current_values)

    final_df = pd.DataFrame(rows, columns=["RPM"] + sample_points)
    final_df = final_df.sort_values(by="RPM", ascending=False)
    return final_df
