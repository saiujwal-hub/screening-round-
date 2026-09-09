import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d

# Load training data
df_train = pd.read_excel('CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx', sheet_name='Training_Data')
op_features = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min']
valid_mask = df_train['Validity_Label'] == 'Valid'
invalid_mask = ~valid_mask

# Fit baseline conduction model
lr_s1_plot = LinearRegression().fit(df_train.loc[valid_mask, op_features], df_train.loc[valid_mask, 'Sensor_S1'])
th_s1_plot = (df_train.loc[valid_mask, 'Sensor_S1'] - lr_s1_plot.predict(df_train.loc[valid_mask, op_features])).abs().max() * 1.25

df_plot = df_train.copy()
df_plot['S1_pred'] = lr_s1_plot.predict(df_plot[op_features].fillna(df_plot[op_features].median()))
df_plot['S1_res'] = (df_plot['Sensor_S1'] - df_plot['S1_pred']).abs()

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.patch.set_facecolor('#ffffff')

valid_color = '#1976D2'
invalid_color = '#C2185B'

# =========================================================================
# Panel (A): Thermal Response
# =========================================================================
ax1.set_facecolor('#fcfcfc')
ax1.axvspan(85, 115, color='#FFF8E1', alpha=0.9, zorder=0, label='High-Current Regime (I > 85 A)')

ax1.scatter(
    df_plot.loc[valid_mask, 'Load_Current_A'],
    df_plot.loc[valid_mask, 'Sensor_S1'],
    c=valid_color, alpha=0.55, s=24, edgecolors='none', zorder=2,
    label=f'Valid Records (n={valid_mask.sum()})'
)

invalid_s1 = df_plot.loc[invalid_mask, 'Sensor_S1'].copy()
nan_s1_mask = invalid_s1.isnull()
ax1.scatter(
    df_plot.loc[invalid_mask & ~nan_s1_mask, 'Load_Current_A'],
    df_plot.loc[invalid_mask & ~nan_s1_mask, 'Sensor_S1'],
    c=invalid_color, marker='^', alpha=0.85, s=40, edgecolors='#880e4f', linewidths=0.5, zorder=3,
    label=f'Invalid Anomalies (n={invalid_mask.sum()})'
)

if nan_s1_mask.sum() > 0:
    ax1.scatter(
        df_plot.loc[invalid_mask & nan_s1_mask, 'Load_Current_A'],
        [-2.0] * nan_s1_mask.sum(),
        c='#6A1B9A', marker='x', s=45, linewidths=2.0, zorder=4,
        label=f'Dropouts (NaN, n={nan_s1_mask.sum()})'
    )

sorted_idx = np.argsort(df_plot.loc[valid_mask, 'Load_Current_A'])
curr_sorted = df_plot.loc[valid_mask, 'Load_Current_A'].iloc[sorted_idx]
pred_sorted = df_plot.loc[valid_mask, 'S1_pred'].iloc[sorted_idx]
pred_smooth = gaussian_filter1d(pred_sorted, sigma=5)
ax1.plot(curr_sorted, pred_smooth, color='#0D47A1', linestyle='--', linewidth=2.2, zorder=5, label=r'Conduction $\hat{S}_1$')

# Annotations (crisp, zero collisions)
ax1.annotate(
    "Extreme Sensor Spike (+28.5 °C)\nUncoupled from load current",
    xy=(89.2, 28.52), xytext=(52, 33.5),
    arrowprops=dict(arrowstyle="->", color="#880e4f", lw=1.2, connectionstyle="arc3,rad=-0.1"),
    fontsize=8, fontweight='bold', color='#880e4f',
    bbox=dict(boxstyle="round,pad=0.25", fc="#fce4ec", ec="#f48fb1", lw=0.8),
    zorder=6
)

ax1.annotate(
    "Genuine Regime Shift\nJoule heating rise (I > 85 A)",
    xy=(97, 20.8), xytext=(82, 5.0),
    arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.2, connectionstyle="arc3,rad=0.15"),
    fontsize=8, fontweight='bold', color='#bf360c',
    bbox=dict(boxstyle="round,pad=0.25", fc="#fff3e0", ec="#ffb74d", lw=0.8),
    zorder=6
)

ax1.set_xlabel('Load Current [A]', fontsize=10.5, fontweight='bold', color='#1e293b')
ax1.set_ylabel('Sensor S1 Temperature Rise [°C]', fontsize=10.5, fontweight='bold', color='#1e293b')
ax1.set_title('(A) Thermal Response: Load Current vs. Terminal Sensor S1', fontsize=10.5, fontweight='bold', pad=28, color='#0f172a')
ax1.set_xlim(10, 115)
ax1.set_ylim(-4.5, 38)
# Place legend ABOVE plot area
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7.2, framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')


# =========================================================================
# Panel (B): Conduction Residual
# =========================================================================
ax2.set_facecolor('#fcfcfc')
ax2.axhspan(0, th_s1_plot, color='#E8F5E9', alpha=0.85, zorder=0, label=f'Permissible Tolerance (≤ {th_s1_plot:.2f} °C)')
ax2.axhline(th_s1_plot, color='#2E7D32', linestyle='-', linewidth=1.5, zorder=1)
ax2.axvspan(85, 115, color='#FFF8E1', alpha=0.5, zorder=0)

ax2.scatter(
    df_plot.loc[valid_mask, 'Load_Current_A'],
    df_plot.loc[valid_mask, 'S1_res'],
    c=valid_color, alpha=0.55, s=24, edgecolors='none', zorder=2,
    label=r'Valid Runs (Residual $\leq \tau_{S1}$)'
)

valid_res_invalid_mask = invalid_mask & ~df_plot['S1_res'].isnull()
ax2.scatter(
    df_plot.loc[valid_res_invalid_mask, 'Load_Current_A'],
    df_plot.loc[valid_res_invalid_mask, 'S1_res'],
    c=invalid_color, marker='^', alpha=0.85, s=40, edgecolors='#880e4f', linewidths=0.5, zorder=3,
    label='Invalid Anomalies (Spikes, Faults)'
)

# Deterministic threshold annotation in open space [94, 104]
ax2.annotate(
    f"Deterministic Boundary: $\\tau_{{S1}} = {th_s1_plot:.2f}$ °C\n100% of valid runs strictly bounded",
    xy=(98, th_s1_plot), xytext=(78, 4.5),
    arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
    fontsize=8, fontweight='bold', color='#1b5e20',
    bbox=dict(boxstyle="round,pad=0.25", fc="#e8f5e9", ec="#a5d6a7", lw=0.8),
    zorder=6
)

# Spike annotation placed in wide open central region
ax2.annotate(
    "Anomalous Spike Failures\nDiverge wildly up to +18.5 °C",
    xy=(92.97, 18.53), xytext=(38, 14.5),
    arrowprops=dict(arrowstyle="->", color="#880e4f", lw=1.2, connectionstyle="arc3,rad=-0.12"),
    fontsize=8, fontweight='bold', color='#880e4f',
    bbox=dict(boxstyle="round,pad=0.25", fc="#fce4ec", ec="#f48fb1", lw=0.8),
    zorder=6
)

ax2.set_xlabel('Load Current [A]', fontsize=10.5, fontweight='bold', color='#1e293b')
ax2.set_ylabel(r'Conduction Absolute Residual $|S_1 - \hat{S}_1|$ [°C]', fontsize=10.5, fontweight='bold', color='#1e293b')
ax2.set_title('(B) Conduction Residual vs. Load Current (Decoupling Physics from Defects)', fontsize=10.5, fontweight='bold', pad=28, color='#0f172a')
ax2.set_xlim(10, 115)
ax2.set_ylim(-0.8, 22)
# Place legend ABOVE plot area
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7.2, framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1')
ax2.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')

plt.tight_layout()
plt.savefig('task1_regime_vs_anomaly.png', dpi=300, bbox_inches='tight')
print('Top-legend figure generated!')
