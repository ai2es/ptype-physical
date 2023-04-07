import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from scipy import interpolate


source = sys.argv[1]
if sys.argv[2].lower() == 'true':
    save_dir = source + '_hourafter'
else:
    save_dir = source
idx_s, idx_e = [int(a) for a in sys.argv[3:]]
print(source, save_dir, idx_s, idx_e)

with open(f"../notebooks/missing_{save_dir}_interpolated.pkl", "rb") as f:
    files = pickle.load(f)
if idx_e == len(files):
    files = files[idx_s:]
else:
    files = files[idx_s:idx_e]

path_source = f"/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/{save_dir}/"
path_save_interpolated = f"/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/{save_dir}_interpolated/"

pressure_levels = np.arange(100, 1025, 25)
height_levels = np.arange(0, 16750, 250)

# old column names
height_cols_p = [f"HGT_{x:d}_m" for x in pressure_levels]
dew_cols_p = [f"T_DEWPOINT_{x:d}_C" for x in pressure_levels]
tmp_cols_p = [f"TMP_{x:d}_C" for x in pressure_levels]
u_cols_p = [f"UGRD_{x:d}_m/s" for x in pressure_levels]
v_cols_p = [f"VGRD_{x:d}_m/s" for x in pressure_levels]

# new column names
dew_cols = [f"T_DEWPOINT_C_{x:d}_m" for x in height_levels]
tmp_cols = [f"TEMP_C_{x:d}_m" for x in height_levels]
u_cols = [f"UGRD_m/s_{x:d}_m" for x in height_levels]
v_cols = [f"VGRD_m/s_{x:d}_m" for x in height_levels]
pres_cols = [f"PRES_Pa_{x:d}_m" for x in height_levels]

cols_heights = dew_cols + tmp_cols + u_cols + v_cols + pres_cols
cols_p = dew_cols_p + tmp_cols_p + u_cols_p + v_cols_p + height_cols_p

precip_types = ['ra', 'sn', 'pl', 'fzra']

start = time.time()
start_str = time.ctime()
for file_i, file in enumerate(files):
    df = pd.read_parquet(os.path.join(path_source, file)).reset_index()

    # interpolate
    interp_df = pd.DataFrame(0.0, index=df.index, columns=cols_heights)
    for row in df.index:
        row_hgts = df.loc[row, height_cols_p] - df.loc[row, 'HGT_ON_SFC_m']
        row_dew = df.loc[row, dew_cols_p]
        row_tmp = df.loc[row, tmp_cols_p]
        row_u = df.loc[row, u_cols_p]
        row_v = df.loc[row, v_cols_p]
        # interpolate dewpoints
        f = interpolate.interp1d(row_hgts, row_dew, fill_value="extrapolate", bounds_error=False)
        interp_df.loc[row, dew_cols] = f(height_levels)
        # interpolate temperatures
        f = interpolate.interp1d(row_hgts, row_tmp, fill_value="extrapolate", bounds_error=False)
        interp_df.loc[row, tmp_cols] = f(height_levels)
        # interpolate U
        f = interpolate.interp1d(row_hgts, row_u, fill_value="extrapolate", bounds_error=False)
        interp_df.loc[row, u_cols] = f(height_levels)
        # interpolate V
        f = interpolate.interp1d(row_hgts, row_v, fill_value="extrapolate", bounds_error=False)
        interp_df.loc[row, v_cols] = f(height_levels)
        # interpolate pressure levels
        f = interpolate.interp1d(row_hgts, pressure_levels, fill_value="extrapolate", bounds_error=False)
        interp_df.loc[row, pres_cols] = f(height_levels)

    interp_df['T_DEWPOINT_C_0_m'] = df['DEWPOINT_2M_C']
    interp_df['TEMP_C_0_m'] = df['TEMPERATURE_2M_C']
    interp_df['UGRD_m/s_0_m'] = df['UGRD_10M_m/s']
    interp_df['VGRD_m/s_0_m'] = df['VGRD_10M_m/s']
    interp_df['PRES_Pa_0_m'] = df['PRES_ON_SURFACE_Pa']

    df.drop(cols_p, inplace=True, axis=1)
    df = pd.concat([interp_df, df], axis=1)
    df.to_parquet(os.path.join(path_save_interpolated, file))
    if file_i % 10 == 0:
        end = time.time()
        end_str = time.ctime()
        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end-start))
        print(f"Done with {file_i} -- start: {start_str}\tend: {end_str}\tduration: {elapsed_time}")
        start = time.time()
        start_str = time.ctime()