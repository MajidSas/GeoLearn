import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/label_stats.json', 'r') as file:
    stats = json.loads(file.read())

with open('/rhome/msaee007/bigdata/pointnet_data/p1_real_data' + '/label_stats.json', 'r') as file:
    stats2 = json.loads(file.read())

# cols_to_normalize = ['k_value_0.025', 'k_value_0.05', 'k_value_0.1', 'e0', 'e2']

    # self.labels[c] = (self.labels[c] - stats[c]['min'])/(stats[c]['max']-stats[c]['min'])
        
ref_results = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/ground_truth.csv')
# for col in cols_to_normalize:
#     _col = col.replace('value','values').replace('e0','box_counts_e0').replace('e2', 'box_counts_e2')
#     ref_results[_col] = (ref_results[_col] - stats[col]['min'])/(stats[col]['max']-stats[col]['min'])

resnet_results = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/resnet_results_summary.csv')
pointnet_results = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/pointnet_results_summary.csv')

resnet_weather_results = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/resnet_weather_results_summary.csv')
pointnet_weather_results = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/pointnet_weather_results_summary.csv')


distributions =[ 'diagonal',  'gaussian', 'sierpinski', 'uniform',
       'bit', 'parcel', 'weather',
       'diagonal_large', 'sierpinski_large','gaussian_large', 'uniform_large']

labels = [
       'hotspots_16_min_val', 'hotspots_16_min_x', 'hotspots_16_min_y',
       'hotspots_16_max_val', 'hotspots_16_max_x', 'hotspots_16_max_y',
       'hotspots_32_min_val', 'hotspots_32_min_x', 'hotspots_32_min_y',
       'hotspots_32_max_val', 'hotspots_32_max_x', 'hotspots_32_max_y',
       'hotspots_64_min_val', 'hotspots_64_min_x', 'hotspots_64_min_y',
       'hotspots_64_max_val', 'hotspots_64_max_x', 'hotspots_64_max_y',
       'k_values_0.025', 'k_values_0.05', 'k_values_0.1', 'k_values_0.25', 'box_counts_e0',
       'box_counts_e2']

def get_stats(col_name, stats=stats):
    mean, std = 0, 1
    col_name2 = col_name
    if 'k_value' in col_name:
        col_name2 = 'k_values'
    elif 'hotspots' in col_name:
        col_name2 = 'hotspots' + col_name[col_name.find('_')+3:]
    elif 'e0' in col_name:
        col_name2 = 'e0'
    elif 'e2' in col_name:
        col_name2 = 'e2'
    j = stats[col_name2]
    return j['mean'], j['std'], j['min'], j['max']

def get_col_values(col_name, ref, pointnet, resnet,normalize=False,get_time=False, stats=stats):
    col_name2 = col_name
    param = None
    ref_time_col = None
    if 'hotspots' in col_name:
        col_name2 = 'hotspots' + col_name[col_name.find('_')+3:]
        param = int(col_name[col_name.find('_')+1:col_name.find('_')+3])
        ref_time_col = 'hotspots_%d_time' % param
        pointnet = pointnet[pointnet['hotspots_k'] == param]
        resnet = resnet[resnet['hotspots_k'] == param]
    if 'k_value' in col_name:
        col_name2 = 'k_values'
        param = float(col_name[col_name.rfind('_')+1:])
        ref_time_col = col_name + '_time'
        pointnet = pointnet[pointnet['kvalue_r'] == param]
        resnet = resnet[resnet['kvalue_r'] == param]
    if 'box_counts' in col_name:
        ref_time_col = 'box_counts_time'
    ref = ref[['dist','id', col_name, 'tree_build_time', 'histogram_time', ref_time_col]]
    pointnet = pointnet[['dist','id', 'hotspots_k', 'kvalue_r', col_name2, 'time']]
    pointnet.loc[:, 'id'] = pointnet['id'].astype(str)
    resnet = resnet[['dist','id', 'hotspots_k', 'kvalue_r', col_name2, 'time']]
    resnet.loc[:,'id'] = resnet['id'].astype(str)
    pointnet = pointnet.rename(columns={col_name2: 'pointnet_'+col_name, 'time': 'pointnet_time'})
    resnet = resnet.rename(columns={col_name2: 'resnet_'+col_name, 'time': 'resnet_time'})
    ref = ref.rename(columns={col_name: 'ref_'+col_name, ref_time_col: 'ref_time'})
    m1 = pd.merge(resnet, pointnet, how='left', on=['dist', 'id', 'hotspots_k', 'kvalue_r'])
    m2 = pd.merge(m1, ref, how='left', on=['dist', 'id'])
    if get_time:
        df = m2[['ref_time', 'resnet_time', 'pointnet_time', 'histogram_time', 'tree_build_time']]
        return (df['ref_time'] + df['tree_build_time']).values, df['pointnet_time'].values, (df['resnet_time'] + df['histogram_time']).values
    else:
        df = m2[['ref_' + col_name, 'resnet_'+ col_name, 'pointnet_'+col_name]]
        mean, std, _, _ = get_stats(col_name, stats)
        if normalize:
            df.loc[:, 'ref_' + col_name] = (df['ref_' + col_name] - mean) / std
        else:
            df.loc[:, 'pointnet_' + col_name] = (df['pointnet_' + col_name] * std) + mean
            df.loc[:, 'resnet_' + col_name] = (df['resnet_' + col_name] * std) + mean
        return df['ref_'+col_name].values, df['pointnet_'+col_name].values, df['resnet_'+col_name].values


# for col in labels:
#     resnet_results[col] = resnet_results[col].clip(lower=0,upper=1)
#     pointnet_results[col] = pointnet_results[col].clip(lower=0,upper=1)


def wmape_col_all(ref, pred, col):
    _ref = ref.sort_values(by=['dist','id'])
    _pred = pred.sort_values(by=['dist','id'])
    wmape = (abs(_ref[col] - _pred[col]).sum()) / abs(_ref[col]).sum()
    return wmape

def wmape_dist_all(ref, pred, dist):
    _ref = ref[ref['dist'] == dist].sort_values(by='id')
    _pred = pred[pred['dist'] == dist].sort_values(by='id')
    wmape = (abs(_ref[labels] - _pred[labels]).sum()).sum() / abs(_ref[labels]).sum().sum()
    return wmape

def compute_wmape(ref, pred):
    # _ref = ref[ref['dist'] == dist].sort_values(by='id')
    # _pred = pred[pred['dist'] == dist].sort_values(by='id')
    wmape = (abs(ref - pred).sum()) / abs(ref).sum()
    return wmape

def smape_dist_all(ref, pred, dist):
    y_true = []
    y_pred = []
    all_true = ref[ref['dist'] == dist].sort_values(by='id')
    all_pred = pred[pred['dist'] == dist].sort_values(by='id')

    for c in labels:
        y_true += list(all_true[c].values)
        y_pred += list(all_pred[c].values)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, 1e-10)
    numerator = np.abs(y_true - y_pred)
    smape = 100 * np.mean(numerator / denominator)
    return smape

def compute_smape(ref, pred, dist, col):
    # compute Symmetric Mean Absolute Percentage Error (sMAPE)
    y_true = ref[ref['dist'] == dist].sort_values(by='id')[col]
    y_pred = pred[pred['dist'] == dist].sort_values(by='id')[col]
    denominator = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, 1e-10)
    numerator = np.abs(y_true - y_pred)
    smape = 100 * np.mean(numerator / denominator)
    return smape

# wmape_dist_all(ref_results, pointnet_results, 'uniform')

# random_results = pointnet_results.copy()
# for c in labels:
#     _, _, minimum, maximum = get_stats(c)
#     random_results[c] = np.random.uniform(minimum, maximum, size=random_results.shape[0])


resnet_wmape = {'distributions': [], 'labels': [], 'wmape': []}
pointnet_wmape = {'distributions': [], 'labels': [], 'wmape': []}
resnet_weather_wmape = {'distributions': [], 'labels': [], 'wmape': []}
pointnet_weather_wmape = {'distributions': [], 'labels': [], 'wmape': []}
random_wmape = {'distributions': [], 'labels': [], 'wmape': []}

for dist in distributions:
    for col in labels:
        mean, std, minimum, maximum = get_stats(col)
        ref, pointnet, resnet = get_col_values(col,
                                ref_results[ref_results['dist'] == dist].copy(deep=True),
                                pointnet_results[pointnet_results['dist'] == dist].copy(deep=True),
                                resnet_results[resnet_results['dist'] == dist].copy(deep=True),
                                normalize=False)
        rand = np.random.uniform(minimum, maximum, size=pointnet.shape[0])
        resnet_wmape['distributions'].append(dist)
        resnet_wmape['labels'].append(col)
        resnet_wmape['wmape'].append(compute_wmape(ref, resnet))
        pointnet_wmape['distributions'].append(dist)
        pointnet_wmape['labels'].append(col)
        pointnet_wmape['wmape'].append(compute_wmape(ref, pointnet))
        random_wmape['distributions'].append(dist)
        random_wmape['labels'].append(col)
        random_wmape['wmape'].append(compute_wmape(ref, rand))
        if dist == 'weather':
            ref, pointnet, resnet = get_col_values(col,
                                ref_results[ref_results['dist'] == dist].copy(deep=True),
                                pointnet_weather_results[pointnet_weather_results['dist'] == dist].copy(deep=True),
                                resnet_weather_results[resnet_weather_results['dist'] == dist].copy(deep=True),
                                normalize=False,stats=stats2)
            resnet_weather_wmape['distributions'].append(dist)
            resnet_weather_wmape['labels'].append(col)
            resnet_weather_wmape['wmape'].append(compute_wmape(ref, resnet))
            pointnet_weather_wmape['distributions'].append(dist)
            pointnet_weather_wmape['labels'].append(col)
            pointnet_weather_wmape['wmape'].append(compute_wmape(ref, pointnet))
        else:
            resnet_weather_wmape['distributions'].append(dist)
            resnet_weather_wmape['labels'].append(col)
            resnet_weather_wmape['wmape'].append(None)
            pointnet_weather_wmape['distributions'].append(dist)
            pointnet_weather_wmape['labels'].append(col)
            pointnet_weather_wmape['wmape'].append(None)



df1 = pd.DataFrame(resnet_wmape).rename(columns={'wmape': 'resnet_wmape'})
df2 = pd.DataFrame(pointnet_wmape).rename(columns={'wmape': 'pointnet_wmape'})
df3 = pd.DataFrame(random_wmape).rename(columns={'wmape': 'random_wmape'})
df4 = pd.DataFrame(resnet_weather_wmape).rename(columns={'wmape': 'resnet_weather_wmape'})
df5 = pd.DataFrame(pointnet_weather_wmape).rename(columns={'wmape': 'pointnet_weather_wmape'})
merged_df = pd.merge(df1, df2, on=['distributions', 'labels'], how='inner')
merged_df = pd.merge(merged_df, df3, on=['distributions','labels'], how='inner')
merged_df = pd.merge(merged_df, df4, on=['distributions','labels'], how='inner')
merged_df = pd.merge(merged_df, df5, on=['distributions','labels'], how='inner')


combined_wmape = {'distribution': [], 'resnet_wmape': [], 'pointnet_wmape': [], 'random_wmape': []}
for dist in distributions:
    df = merged_df[merged_df['distributions'] == dist]
    df.to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/{dist}.csv', index=False)
    combined_wmape['distribution'].append(dist)
    combined_wmape['resnet_wmape'].append(df['resnet_wmape'].mean())
    combined_wmape['pointnet_wmape'].append(df['pointnet_wmape'].mean())
    combined_wmape['random_wmape'].append(df['random_wmape'].mean())
pd.DataFrame(combined_wmape).to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/all_dists_summary_wmape.csv', index=False)


## wmape all dists per column
resnet_wmape = {'distributions': [], 'labels': [], 'wmape': []}
pointnet_wmape = {'distributions': [], 'labels': [], 'wmape': []}
random_wmape = {'distributions': [], 'labels': [], 'wmape': []}
for col in labels:
    mean, std, minimum, maximum = get_stats(col)
    ref, pointnet, resnet = get_col_values(col,
                            ref_results.copy(deep=True),
                            pointnet_results.copy(deep=True),
                            resnet_results.copy(deep=True),
                            normalize=False)
    rand = np.random.uniform(minimum, maximum, size=pointnet.shape[0])
    resnet_wmape['distributions'].append('all_dist')
    resnet_wmape['labels'].append(col)
    resnet_wmape['wmape'].append(compute_wmape(ref, resnet))
    pointnet_wmape['distributions'].append('all_dist')
    pointnet_wmape['labels'].append(col)
    pointnet_wmape['wmape'].append(compute_wmape(ref, pointnet))
    random_wmape['distributions'].append('all_dist')
    random_wmape['labels'].append(col)
    random_wmape['wmape'].append(compute_wmape(ref, rand))

df1 = pd.DataFrame(resnet_wmape).rename(columns={'wmape': 'resnet_wmape'})
df2 = pd.DataFrame(pointnet_wmape).rename(columns={'wmape': 'pointnet_wmape'})
df3 = pd.DataFrame(random_wmape).rename(columns={'wmape': 'random_wmape'})
merged_df = pd.merge(df1, df2, on=['distributions', 'labels'], how='inner')
merged_df = pd.merge(merged_df, df3, on=['distributions','labels'], how='inner')
merged_df.to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/all_dists_wmape.csv', index=False)



## MSE by distribution
combined = {'distribution': [], 'resnet_mse': [], 'pointnet_mse': [], 'random_mse': []}
for dist in distributions:
    ref = []
    pointnet = []
    resnet = []
    rand = []
    for col in labels:
        mean, std, minimum, maximum = get_stats(col)
        _ref, _pointnet, _resnet = get_col_values(col,
                                ref_results[ref_results['dist'] == dist],
                                pointnet_results[pointnet_results['dist'] == dist],
                                resnet_results[resnet_results['dist'] == dist],normalize=True)
        # print(dist, col, _ref.shape, _pointnet.shape, _resnet.shape, np.isnan(_ref).sum(), np.isnan(_pointnet).sum())
        _rand = np.random.uniform(-1, 1, size=_pointnet.shape[0])
        ref.append(_ref)
        pointnet.append(_pointnet)
        resnet.append(_resnet)
        rand.append(_rand)
    ref = np.concatenate(ref)
    pointnet = np.concatenate(pointnet)
    resnet = np.concatenate(resnet)
    rand = np.concatenate(rand)
    combined['distribution'].append(dist)
    combined['resnet_mse'].append(np.mean((ref - resnet) ** 2))
    combined['pointnet_mse'].append(np.mean((ref - pointnet) ** 2))
    combined['random_mse'].append(np.mean((ref - rand) ** 2))

pd.DataFrame(combined).to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/all_dists_summary.csv', index=False)

## time by output type
resnet_time = {'distributions': [], 'labels': [], 'time': []}
pointnet_time = {'distributions': [], 'labels': [], 'time': []}
ref_time = {'distributions': [], 'labels': [], 'time': []}
for col in labels:
    mean, std, minimum, maximum = get_stats(col)
    ref, pointnet, resnet = get_col_values(col,
                            ref_results.copy(deep=True),
                            pointnet_results.copy(deep=True),
                            resnet_results.copy(deep=True),
                            normalize=False, get_time=True)
    resnet_time['distributions'].append('all_dist')
    resnet_time['labels'].append(col)
    resnet_time['time'].append(resnet.mean())
    pointnet_time['distributions'].append('all_dist')
    pointnet_time['labels'].append(col)
    pointnet_time['time'].append(pointnet.mean())
    ref_time['distributions'].append('all_dist')
    ref_time['labels'].append(col)
    ref_time['time'].append(ref.mean())

df1 = pd.DataFrame(resnet_time).rename(columns={'time': 'resnet_time'})
df2 = pd.DataFrame(pointnet_time).rename(columns={'time': 'pointnet_time'})
df3 = pd.DataFrame(ref_time).rename(columns={'time': 'ref_time'})
merged_df = pd.merge(df1, df2, on=['distributions', 'labels'], how='inner')
merged_df = pd.merge(merged_df, df3, on=['distributions','labels'], how='inner')
merged_df.to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/all_dists_time.csv', index=False)

# time by distribution
resnet_time = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/resnet_time_summary.csv')[['dist','time']].rename(columns={'time': 'resnet_time'})
histogram_time = ref_results.groupby('dist')['histogram_time'].sum().reset_index()
pointnet_time = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/synthetic_data/pointnet_time_summary.csv')[['dist', 'time']].rename(columns={'time': 'pointnet_time'})
ref_results['ground_truth_total_time'] = ref_results[['tree_build_time', 'hotspots_16_time', 'hotspots_32_time', 'hotspots_64_time', 'k_values_0.025_time', 'k_values_0.05_time', 'k_values_0.1_time', 'k_values_0.25_time', 'box_counts_time']].sum(axis=1)
ref_time = ref_results.groupby('dist')['ground_truth_total_time'].sum().reset_index()
merged = pd.merge(resnet_time, pointnet_time, on='dist', how='inner')
merged = pd.merge(merged, histogram_time, on='dist', how='inner')
merged['resnet+hist_time'] = merged['resnet_time'] + merged['histogram_time']
merged = pd.merge(merged, ref_time, on='dist', how='inner')
merged.to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/time_by_distribution.csv', index=False)



# validation loss by epoch
import re
pointnet_losses = ""
with open('/rhome/msaee007/PointNet/01_data_synth/pointnet_synth_train_log.out', 'r') as file:
    pointnet_losses = [float(re.search(r"[-+]?\d*\.\d+|\d+", l).group()) for l in file.readlines() if 'val loss:' in l]

resnet_losses = ""
with open('/rhome/msaee007/PointNet/01_data_synth/resnet_synth_train_log.out', 'r') as file:
    resnet_losses = [float(re.search(r"[-+]?\d*\.\d+|\d+", l).group()) for l in file.readlines() if 'val loss:' in l]

pd.DataFrame({'resnet': resnet_losses, 'pointnet': pointnet_losses}).to_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/val_loss_by_epoch_synth.csv', index=False)

#### PLOT ALL DISTS SUMMARY FIGURE


# Convert to DataFrame
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['legend.title_fontsize'] = 10
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.color'] = 'gray'
mpl.rcParams['grid.alpha'] = 0.7

df = pd.read_csv(f'/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/all_dists_summary.csv')
df = pd.read_csv(f'./all_dists_summary.csv')

# Define seen and unseen distributions
seen_distributions = ['diagonal', 'gaussian', 'sierpinski', 'uniform']

df['type'] = df['distribution'].apply(lambda x: 'Seen' if x in seen_distributions else 'Unseen')

df = df.iloc[::-1]

# Define colors for grayscale
bar_colors = {'pointnet_mse': 'black', 'resnet_mse': 'dimgray', 'random_mse': 'lightgray'}
label_colors = {'Seen': 'black', 'Unseen': 'gray'}

# Plot setup
fig, ax = plt.subplots(figsize=(7, 4.5))  # Adjust for a wider layout

# Define bar positions
y = np.arange(len(df))  # the label locations
bar_height = 0.3  # Keep the bars thicker
offsets = [bar_height, 0, -bar_height]  # Reverse the order of the bars

# Create horizontal bars for each model
for i, (model, color) in enumerate(bar_colors.items()):
    bars = ax.barh(y + offsets[i], df[model], height=bar_height, label=model.split('_')[0].capitalize(), color=color)
    # Add values to each bar
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}',
                va='center', fontsize=8)

# Format the y-axis labels
ax.set_yticks(y)
ax.set_yticklabels(df['distribution'], fontsize=9)

# Reduce the white space between the distribution labels
ax.tick_params(axis='y', pad=2)

for label, tick in zip(df['type'], ax.get_yticklabels()):
    tick.set_color(label_colors[label])  # Color based on seen/unseen

# Add labels, legend, and title
ax.set_xlabel('MSE', fontsize=10)
ax.set_ylabel('Distributions', fontsize=10)
ax.legend(title='Model', fontsize=9, title_fontsize=10, frameon=True, loc='best')

# legend = ax.legend(title='Model', fontsize=9, loc='upper center', title_fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.1), ncol=3)
# plt.setp(legend.get_title(), loc='left')

# Use a research-style grid and layout
ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.tight_layout()

plt.savefig('/rhome/msaee007/PointNet/01_data_synth/figures/p1_all_dists.pdf')
# plt.show()
#####
#### PLOT FIGURE FOR SEIPINSKI

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Table data

# Convert to DataFrame
df = pd.read_csv('/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/sierpinski.csv')
df = pd.read_csv('./sierpinski.csv')
df = df[['labels', 'pointnet_wmape', 'resnet_wmape', 'random_wmape']]
###############################################################################
# 1. Clip all WMAPE values to a maximum of 10
###############################################################################
# for model_col in ['pointnet_wmape', 'resnet_wmape', 'random_wmape']:
#     df[model_col + '_clipped'] = df[model_col].apply(lambda x: min(x, 5))

###############################################################################
# 2. Create a horizontal bar chart with clipped values
###############################################################################
# Define grayscale colors for bars
bar_colors = {
    'pointnet_wmape': 'black',
    'resnet_wmape': 'dimgray',
    'random_wmape_clipped': 'lightgray'
}

fig, ax = plt.subplots(figsize=(3.5, 7))  # Single-column layout

# Define bar positions
y = np.arange(len(df))         # Label locations
bar_height = 0.2               # Height of the bars
offsets = [-bar_height, 0, bar_height]  # Offsets for grouped bars

models = list(bar_colors.keys())  # e.g. ["resnet_wmape_clipped", "pointnet_wmape_clipped", "random_wmape_clipped"]

# Plot horizontal bars for each model
for i, model_col in enumerate(models):
    ax.barh(
        y + offsets[i],
        df[model_col],
        height=bar_height,
        label=model_col.split('_')[0].capitalize(),  # "resnet" -> "Resnet"
        color=bar_colors[model_col]
    )

# Format y-axis labels
ax.set_yticks(y)
ax.set_yticklabels(df['labels'], fontsize=8)

ax.set_xlabel('WMAPE', fontsize=10)
ax.set_ylabel('Output Value', fontsize=10)
# ax.set_title('Clipped WMAPE Performance Across Labels', fontsize=12)
ax.legend(title='Model', fontsize=9)

# Use a research-style grid and layout
ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

###############################################################################
# 3. Label the bars that exceed 10 with the original value
###############################################################################
for row_idx in range(len(df)):
    for i, original_model_col in enumerate(['resnet_wmape', 'pointnet_wmape', 'random_wmape']):
        original_val = df[original_model_col][row_idx]
        clipped_val = df[original_model_col + '_clipped'][row_idx]
        # If original exceeds 10, label the bar at the clipped position
        if original_val > 5:
            ax.text(
                x=4,             # x-position at the clipped max (10)
                y=row_idx + offsets[i],    # y-position matching the bar
                s=f"{original_val:.1f}", # Show original
                ha='left', va='center',
                fontsize=8
            )

plt.tight_layout()
plt.savefig('./p1_sierpinski_all_outputs.pdf')


##############



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# df = pd.read_csv('dbscan_global_0.920392.csv').astype(int)
df = pd.read_csv('unet_global_0.795797.csv').astype(int)
# df = pd.read_csv('road_poi_0.681000.csv')[['actual', 'predicted']].astype(int)
# df = pd.read_csv('walkability_resnet_0.599000.csv')[['actual', 'predicted']].astype(int)

# df = df[df['batches'] == 1]
# df = df[df['batches'] == 327]
# Compute the confusion matrix
cm = confusion_matrix(df['actual'], df['predicted'])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

ticks = list(range(8))
# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=ticks, yticklabels=ticks)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()