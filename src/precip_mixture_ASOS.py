import os
import pandas as pd


source = "ASOS_converted"
path_save = "/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/"

files = os.listdir(os.path.join(path_save, source))
files.sort()

df = pd.read_parquet(os.path.join(path_save, source, files[0]))
precip_types = ['ra', 'sn', 'pl', 'fzra']
cols_subset = ['obdate', 'obtime', 'lat', 'lon', 'precip', 'datetime', 'precip_count_byhr']
columns_rap = sorted(list(set(df.columns) - set(cols_subset)))
columns_new = ["datetime", "lat", "lon", "report_count", "ra_percent", "sn_percent", "pl_percent", "fzra_percent"] + columns_rap

for i, file in enumerate(files):
    df = pd.read_parquet(os.path.join(path_save, source, file))
    df_new = pd.DataFrame(columns=columns_new)
    i=0
    group = df.groupby(["datetime", "lat", "lon"])
    for name, chunk in group:
        df_new.loc[i, 'datetime'], df_new.loc[i, 'lat'] , df_new.loc[i, 'lon']  = name[0], name[1], name[2]
        df_new.loc[i, columns_rap] = chunk.loc[chunk.index[0], columns_rap]
        df_new.loc[i, 'report_count'] = chunk["precip_count_byhr"].sum()
        for precip_type in precip_types:
            if chunk[chunk['precip'] == precip_type].shape[0]:
                precip_count = float(chunk.loc[chunk['precip'] == precip_type, 'precip_count_byhr'].values)
                df_new.loc[i, f"{precip_type}_percent"] = precip_count / df_new.loc[i, 'report_count']
            else:
                df_new.loc[i, f"{precip_type}_percent"] = 0.0
        i += 1
    df_new.to_parquet(os.path.join(path_save, "ASOS_mixture", file))