import datetime
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tables
from tqdm import tqdm
#%matplotlib inline

import itertools


def radius(df): # you just need a dataframe with a reset index, you could aLSO  just preprocess the data of combined and sanity checked for an easier life!
    x = np.square(np.array(df['x_position'].tolist()))
    y = np.square(np.array(df['y_position'].tolist()))
    r = np.sqrt(np.add(x, y))
    df['radius'] = r
    return df
def find_distance_round_x_y(path_data,key,resample_to):

    df = pd.read_hdf(path_data, key)
    df = df[df['time']>120]
    df = radius(df)
    df = df[df['radius']<0.6]
    df.dropna(subset = ['x_position'], inplace=True)
    df.dropna(subset = ['y_position'], inplace=True)
    df = df.reset_index(drop = False)
    fishes = df['folder_name'].unique().tolist()
    stimuli = df['stimulus_name'].unique().tolist()
    df.set_index('time_absolute', inplace=True)
    # Resample the dataframe by a specific frequency (e.g., daily) for each level of the multi-index
    resampled_df = df.groupby(['folder_name', 'stimulus_name', 'trial'], as_index=True).resample(resample_to).mean(numeric_only = True)
    xes = resampled_df['x_position'].tolist()
    yes = resampled_df['y_position'].tolist()
    x_vals = np.array(xes)
    y_vals = np.array(yes)
    x_vals_1 = np.append([0], x_vals[:-1])
    y_vals_1 = np.append([0], y_vals[:-1])
    x = np.sqrt(np.add(np.square(x_vals_1-x_vals),np.square(y_vals_1-y_vals)))
    resampled_df['distance'] =x
    return resampled_df,fishes,stimuli



def calculate_angle_pointing(x_start, y_start, x_end, y_end):
    vectors = np.column_stack((x_end - x_start, y_end - y_start))
    angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles_deg = np.degrees(angles_rad)
    return angles_deg



#%%
import numpy as np

def calculate_angle_pointing(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angles_rad = np.arctan2(dy, dx)
    angles_deg = np.degrees(angles_rad)
    angles_deg[angles_deg < 0] += 360
    return angles_deg

#def calculate_angle_pointing(x1, y1, x2, y2):
#    vectors = np.column_stack((x2 - x1, y2 - y1))
#    angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
#    angles_deg = np.degrees(angles_rad)
#    return list(angles_deg)
#%%
# start n, start n+1 and start n+2
#
from math import sqrt
import numpy as np
import math
 # x  601 591 581

def calculate_angles_turning(x1, y1, x2, y2, x3, y3):
    # Calculate the length of each segment
    a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

    # Calculate the angle using the law of cosines
    angles =180- np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))
    return angles

def start_start(df):
    x1 = np.array(df['x_position'][0:len(df)-2])
    x2 = np.array(df['x_position'][1:len(df)-1])
    x3 = np.array(df['x_position'][2:len(df)])

    y1 = np.array(df['y_position'][0:len(df)-2])
    y2 = np.array(df['y_position'][1:len(df)-1])
    y3 = np.array(df['y_position'][2:len(df)])
    df['angle_pointing'] = list(calculate_angle_pointing(x1, y1, x2, y2 )) +[0,0]
    df['angle_turning'] = [0]+[0] + list(calculate_angles_turning(x1, y1, x2, y2, x3, y3))

    return df


def calculate_speed_lists(df):
    fish_es = df['folder_name'].unique().tolist()
    stimuli = df['stimulus_name'].unique().tolist()
    speed = []
    for stim in tqdm(range(0, len(stimuli))):
        df_fly_stim = df[df['stimulus_name'] == stimuli[stim]]
        speed_temp = []
        for fish in fish_es:
            df_fish = df_fly_stim[df_fly_stim['folder_name'] == fish]
            for tr in range(0, 12):
                df_trial = df_fish[df_fish['trial'] == tr]
                if len(df_trial) > 0:
                    df_round = df_trial.round({'time': 0})
                    adf_gr = df_round.groupby(by=["time"], dropna=False, as_index=True).mean(numeric_only = True)
                    x_vals = np.append(adf_gr['x_position'].tolist(), 0)
                    y_vals = np.append(adf_gr['y_position'].tolist(), 0)
                    x_vals_1 = np.append([0], x_vals[:-1])
                    y_vals_1 = np.append([0], y_vals[:-1])
                    disp = np.sqrt(np.add(np.square(x_vals_1 - x_vals), np.square(y_vals_1 - y_vals)))
                    dd = disp[1:len(disp) - 1]
                    speed_temp.append(list(dd))
        speed_temp = list(itertools.chain.from_iterable(speed_temp))
        speed.append(speed_temp)
    return speed
#%%
import numpy as np

def remove_nans(lst):
    arr = np.array(lst)
    clean_arr = arr[~np.isnan(arr)]
    clean_list = clean_arr.tolist()
    return clean_list

#%%
def put_together_dot_stimuli(speed):
    dot_speeds = speed[0]+speed[2]+speed[4] + speed[6]
    stimuli_speeds = speed[1]+speed[3]+speed[5] + speed[7]
    return remove_nans(dot_speeds), remove_nans(stimuli_speeds)
#%%
def put_together_stripe_stimuli(speed):
    gray_speeds = speed[0]
    stimuli_speeds = speed[1]+speed[2]+speed[3] + speed[4]
    return remove_nans(gray_speeds), remove_nans(stimuli_speeds)
#%%
def pdf_cdf(lis_val,from_val,to_val,bins):
    count, bins_count = np.histogram(lis_val, bins = np.arange(from_val,to_val,bins))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count,pdf,cdf
#%%
def plot_data(speed, title,experiment_speed,time_set,exp_condition):
    if exp_condition=="dots":
        control_speeds, stimuli_speeds =put_together_dot_stimuli(speed)
    elif exp_condition == "sine_waves":
        control_speeds, stimuli_speeds = put_together_stripe_stimuli(speed)
    plt.rcParams["figure.figsize"] = (5, 5)
    bins_count_gray,pdf_gray,cdf_gray =  pdf_cdf(lis_val = control_speeds)
    bins_count_stimuli,pdf_stimuli,cdf_stimuli =  pdf_cdf(lis_val = stimuli_speeds)

    plt.plot(bins_count_gray[1:],pdf_gray,  alpha = 0.5, color = 'blue', label ="gray")
    plt.plot(bins_count_stimuli[1:],pdf_stimuli,  alpha = 0.5, color = 'red', label ="stimuli")

    plt.title(title,size = 15)
    print(f"{title} control {len(control_speeds)}, and stimuli {len(stimuli_speeds)} speed ={experiment_speed}, t {time_set}")
    plt.legend()
    plt.show()

def put_relative_time(df, fishes):
    df_all = pd.DataFrame()
    for fish in fishes:
        df_fish = df.xs(fish, level='folder_name')
        df_fish.reset_index(drop=False, inplace=True)
        df_fish['relative_time'] = df_fish['time_absolute'] - df_fish['time_absolute'].min()
        df_fish['relative_time_minutes'] = df_fish['relative_time'].dt.total_seconds()
        df_fish = df_fish[df_fish['relative_time_minutes'] <= 1800]
        df_all = pd.concat([df_all, df_fish])
   # df_all.set_index(['folder_name', 'stimulus_name', 'trial'], inplace=True)

    return df_all


def circl_plot(angles):
    bins = 300
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    polar_ax = fig.add_subplot(1, 2, 2, projection="polar")
    count, bin = np.histogram(angles, bins = np.linspace(0,360,30), density=False)
    ax.plot(bin[:-1], count)# width=10)
    polar_ax.plot(bin[:-1], count)
    polar_ax.set_rlabel_position(0)
    fig.tight_layout()