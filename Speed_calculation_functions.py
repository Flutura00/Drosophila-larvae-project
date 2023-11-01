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


def df_subset(df,lower_bound,upper_bound):
    df = df[df['time'] >= lower_bound]
    df = df[df['time'] <= upper_bound]
    return df


def radius(df): # you just need a dataframe with a reset index, you could aLSO  just preprocess the data of combined and sanity checked for an easier life!
    x = np.square(np.array(df['x_position'].tolist()))
    y = np.square(np.array(df['y_position'].tolist()))
    r = np.sqrt(np.add(x, y))
    df['radius'] = r
  #  print('radius done')
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
                    df_trial = radius(df_trial)
                    df_round = df_trial.round({'time': 0})

                    adf_gr = df_round.groupby(by=["time"], dropna=True).mean(numeric_only = True)
                    x_vals = np.append(adf_gr['x_position'].tolist(), 0)
                    y_vals = np.append(adf_gr['y_position'].tolist(), 0)
                    x_vals_1 = np.append([0], x_vals[:-1])
                    y_vals_1 = np.append([0], y_vals[:-1])
                    disp = np.sqrt(np.add(np.square(x_vals_1 - x_vals), np.square(y_vals_1 - y_vals)))
                    dd = disp[1:len(disp) - 1]
                    speed_temp.append(list(dd))
                #    radius = np.append(adf_gr['radius'].tolist(), 0)
                #    speed_temp.append(list(radius))
        speed_temp = list(itertools.chain.from_iterable(speed_temp))
        speed.append(speed_temp)
    return speed
#%%

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
def pdf_cdf(lis_val):
    count, bins_count = np.histogram(lis_val) #, bins = np.arange(0,0.05,0.002))

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

    plt.plot(bins_count_gray[1:],pdf_gray,  alpha = 0.5, color = 'blue', label ="control")
    plt.plot(bins_count_stimuli[1:],pdf_stimuli,  alpha = 0.5, color = 'red', label ="stimuli")

    plt.title(title,size = 15)
    plt.ylabel("Probability", size = 15)
    plt.xlabel("speed", size = 15)
   # plt.ylim(0,0.3)
    print(f"{title} control {len(control_speeds)}, and stimuli {len(stimuli_speeds)} speed ={experiment_speed}, t <{time_set}")

    plt.legend()
    plt.show()
    return bins_count_stimuli, pdf_gray, pdf_stimuli