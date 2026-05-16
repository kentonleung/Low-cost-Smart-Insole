"""
Smart Insole Activity Classification — Final Pipeline
======================================================
MEng Research Project — Dik Chun Leung, KCL Biomedical Engineering, 2025

HONEST RESULTS (Leave-One-Group-Out cross-validation)
------------------------------------------------------
Best model        : Random Forest
Balanced accuracy : 57.4% +/- 47.7%
Walking recall    : 97%
Standing recall   : 78%
Sitting recall    :  0%  — only 1 independent recording run
Jumping recall    :  0%  — only 1 independent recording run
Stair_Up recall   :  0%  — only 1 independent recording run
Stair_Down recall :  0%  — only 2 independent recording runs

WHY THESE NUMBERS ARE CORRECT
------------------------------
An earlier version reported 92.5% using stratified k-fold CV. That was
inflated by temporal data leakage: consecutive windows share 75% of the
same raw samples. Randomly shuffling windows across train/test folds meant
the model was memorising test data it had already seen in training.

Leave-One-Group-Out (LOGO) holds out one entire continuous activity run,
trains on all others, and predicts on the held-out run. No window in the
test fold shares any raw samples with any window in the training fold.
This is the honest estimate of how the model generalises to new data.

The four classes that drop to 0% reveal a real limitation: with only one
or two recording runs per class, there is nothing left in training when
that run is held out. More recording sessions are needed.

Requirements:  pip install pandas numpy scikit-learn matplotlib seaborn
Usage:         python insole_pipeline_final.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (LeaveOneGroupOut, cross_val_score,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_S   = 2.0    # window length in seconds
STEP_S     = 0.5    # step between windows — 75% overlap
MIN_PURITY = 0.80   # minimum fraction of window sharing one label

SENSOR_CHANNELS = ['Lateral', 'Heel', 'Medial', 'Total_FSR', 'X', 'Y', 'Z']

ACTIVITY_COLOURS = {
    'Standing':   '#42A5F5',
    'Walking':    '#66BB6A',
    'Stair_Up':   '#FFA726',
    'Stair_Down': '#AB47BC',
    'Jumping':    '#EF5350',
    'Sitting':    '#26C6DA',
}


# =============================================================================
# STAGE 1 — FEATURE EXTRACTION
# =============================================================================
#
# Each 2-second window of raw sensor data is compressed into 67 numbers.
#
# Features per channel (7 channels x 9 statistics = 63):
#   mean, std, min, max, range, median, IQR, energy, zero-crossing rate
#
# Plus 4 cross-channel features:
#   acceleration magnitude mean and std
#   medial/heel ratio, lateral/heel ratio (forefoot weight distribution)
#
# The same function is used for both data sources so the feature vectors
# are directly comparable and can be stacked.

def extract_features(window: pd.DataFrame) -> dict:
    features = {}
    for ch in SENSOR_CHANNELS:
        v = window[ch].values.astype(float)
        features[f'{ch}_mean']    = np.mean(v)
        features[f'{ch}_std']     = np.std(v)
        features[f'{ch}_min']     = np.min(v)
        features[f'{ch}_max']     = np.max(v)
        features[f'{ch}_range']   = np.ptp(v)
        features[f'{ch}_median']  = np.median(v)
        features[f'{ch}_iqr']     = np.percentile(v, 75) - np.percentile(v, 25)
        features[f'{ch}_energy']  = np.mean(v ** 2)
        features[f'{ch}_zcr']     = np.mean(
            np.diff(np.sign(v - np.mean(v))) != 0)
    mag = np.sqrt(window['X']**2 + window['Y']**2 + window['Z']**2)
    features['accel_mag_mean']     = np.mean(mag)
    features['accel_mag_std']      = np.std(mag)
    heel_safe = features['Heel_mean'] + 1e-6
    features['medial_heel_ratio']  = features['Medial_mean']  / heel_safe
    features['lateral_heel_ratio'] = features['Lateral_mean'] / heel_safe
    return features


def slide_windows(df: pd.DataFrame, fs: float, source_id: int = 0):
    """
    Slide a window across the labelled data and extract features.

    Returns X (features), y (labels), groups (run IDs for LOGO CV).
    Each uninterrupted bout of a single activity = one group. Groups from
    different data sources are offset by source_id * 1000 so they never
    collide when the two datasets are combined.
    """
    win_n  = int(WINDOW_S * fs)
    step_n = int(STEP_S   * fs)

    df = df.copy()
    df['run_id']  = (df['label'] != df['label'].shift()).cumsum() - 1
    df['run_id'] += source_id * 1000

    rows, labels, groups = [], [], []
    for i in range(0, len(df) - win_n + 1, step_n):
        w        = df.iloc[i: i + win_n]
        majority = w['label'].value_counts().idxmax()
        purity   = w['label'].value_counts().iloc[0] / win_n
        if purity >= MIN_PURITY:
            rows.append(extract_features(w))
            labels.append(majority)
            groups.append(int(w['run_id'].value_counts().idxmax()))

    return pd.DataFrame(rows), np.array(labels), np.array(groups)


# =============================================================================
# STAGE 2 — DATA SOURCE 1: DATALOG.TXT
# =============================================================================
#
# Free-living recording containing all six activity classes.
#
# TRIM: 18-84 s removes the shoe-going-on period at the front and the
# shoe-coming-off period at the back (identified from Z-axis settling and
# FSR collapse).
#
# LABELS: derived from a smartphone video of the session. The video clock
# runs 4 seconds ahead of the data clock (confirmed by matching anchor
# events). All labels were confirmed by the user watching the video.

DATALOG_PATH = 'DATALOG.TXT'
TRIM_START   = 18.0
TRIM_END     = 84.0
VIDEO_OFFSET = 4.0


def vt(video_seconds: float) -> float:
    """Convert video timestamp to trim-relative data time."""
    return video_seconds - VIDEO_OFFSET


DATALOG_SEGMENTS = [
    (vt(10),   vt(12),   'Standing'),
    (vt(12),   vt(20),   'Walking'),
    (vt(20),   vt(22),   'Standing'),
    (vt(22),   vt(24),   'Walking'),
    (vt(24),   vt(31),   'Stair_Up'),
    (vt(31),   vt(33),   'Standing'),
    (vt(33),   vt(41),   'Jumping'),
    (vt(42),   vt(46),   'Stair_Down'),
    (vt(46),   vt(55.6), 'Sitting'),
    (vt(55.6), vt(60),   'Stair_Down'),
    (vt(60),   vt(64),   'Standing'),
    (vt(64),   66.0,     'Walking'),
]


def load_datalog(path: str):
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['Millis', 'X', 'Y', 'Z',
                            'Lateral', 'Heel', 'Medial'])
    df['Time_s']    = (df['Millis'] - df['Millis'].iloc[0]) / 1000.0
    df['Total_FSR'] = df[['Lateral', 'Heel', 'Medial']].sum(axis=1)
    fs = 1000.0 / df['Millis'].diff().median()

    df = df[(df['Time_s'] >= TRIM_START) &
            (df['Time_s'] <= TRIM_END)].copy().reset_index(drop=True)
    df['t'] = df['Time_s'] - TRIM_START

    df['label'] = 'Unlabelled'
    for start, end, activity in DATALOG_SEGMENTS:
        df.loc[(df['t'] >= start) & (df['t'] < end), 'label'] = activity
    df = df[df['label'] != 'Unlabelled'].copy()

    print(f"[DATALOG]  {len(df)} samples | {fs:.1f} Hz")
    X, y, g = slide_windows(df, fs, source_id=0)
    print(f"           {len(X)} windows | {len(np.unique(g))} runs")
    for lbl, cnt in pd.Series(y).value_counts().items():
        n_runs = len(np.unique(g[y == lbl]))
        print(f"           {lbl:15s}: {cnt:3d} windows, {n_runs} run(s)")
    return X, y, g, df, fs


# =============================================================================
# STAGE 3 — DATA SOURCE 2: KentonData.csv
# =============================================================================
#
# Gait lab recording. Walking and Standing only.
#
# LABELLING STRATEGY — two steps:
#
#   Step 1: default the entire 27-104 s window to Walking.
#     The insole is worn on one foot. The FSR reads near zero during swing
#     phase (foot in the air). These zero gaps are part of the gait cycle,
#     not pauses. Defaulting to Walking means swing-phase samples are
#     correctly included inside walking windows rather than discarded.
#
#   Step 2: override six confirmed standing pauses.
#     These were identified from the signal (FSR mean high, std low) and
#     confirmed by the user comparing the signal plot against the force
#     plate data and their memory of the session.

CSV_PATH       = 'KentonData.csv'
CSV_TRIM_START = 27.0
CSV_TRIM_END   = 104.0

STANDING_PAUSES = [
    (28.5, 30.0),
    (35.5, 36.0),
    (42.0, 42.5),
    (49.0, 53.0),   # explicitly confirmed: "49-53s is definitely standing"
    (57.5, 59.5),
    (70.5, 71.0),
]


def load_csv(path: str):
    raw = pd.read_csv(path, header=0)
    df  = raw.iloc[:, 8:16].copy()
    df.columns = ['Millis', 'Time_ms', 'X', 'Y', 'Z',
                  'Lateral', 'Heel', 'Medial']
    df = (df.dropna(subset=['Millis'])
            .apply(pd.to_numeric, errors='coerce')
            .dropna()
            .reset_index(drop=True))
    df['Total_FSR'] = df[['Lateral', 'Heel', 'Medial']].sum(axis=1)
    df['t']         = df['Time_ms'] / 1000.0
    fs = 1000.0 / df['Time_ms'].diff().median()

    df = df[(df['t'] >= CSV_TRIM_START) &
            (df['t'] <= CSV_TRIM_END)].copy().reset_index(drop=True)

    df['label'] = 'Walking'
    for start, end in STANDING_PAUSES:
        df.loc[(df['t'] >= start) & (df['t'] < end), 'label'] = 'Standing'

    print(f"[CSV]      {len(df)} samples | {fs:.1f} Hz")
    X, y, g = slide_windows(df, fs, source_id=1)
    print(f"           {len(X)} windows | {len(np.unique(g))} runs")
    for lbl, cnt in pd.Series(y).value_counts().items():
        n_runs = len(np.unique(g[y == lbl]))
        print(f"           {lbl:15s}: {cnt:3d} windows, {n_runs} run(s)")
    return X, y, g


# =============================================================================
# STAGE 4 — TRAIN WITH LEAVE-ONE-GROUP-OUT CROSS-VALIDATION
# =============================================================================
#
# Three classifiers are compared. All use LOGO CV.
#
# Random Forest: an ensemble of decision trees. Each tree is trained on a
#   random subset of the data and votes on the label; the majority wins.
#   Does not require feature scaling. Naturally handles mixed-scale features.
#
# Gradient Boosting: trees built sequentially, each one correcting the
#   errors of the previous ensemble. Often more accurate but slower.
#
# SVM (RBF): finds the boundary that maximises the margin between classes
#   in a high-dimensional space. The RBF kernel allows curved boundaries.
#   Requires StandardScaler — fitted on training fold only, never test fold.
#
# class_weight='balanced' compensates for the unequal class sizes (Walking
#   has 162 windows; Jumping has only 14). Without it the model would ignore
#   minority classes.

def build_models() -> dict:
    return {
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', random_state=42))
        ]),
    }


def train_and_compare(X: pd.DataFrame,
                      y: np.ndarray,
                      g: np.ndarray):
    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(g))
    print(f"\n[Train]    Leave-One-Group-Out CV ({n_groups} folds)")
    print(f"           {'Model':22s}  balanced_acc   std")

    best_score, best_name, best_pipe = -1.0, None, None
    for name, pipe in build_models().items():
        scores = cross_val_score(pipe, X, y, cv=logo, groups=g,
                                 scoring='balanced_accuracy')
        m, s = scores.mean(), scores.std()
        tag = '  <- best' if m > best_score else ''
        print(f"           {name:22s}  {m:.3f}         {s:.3f}{tag}")
        if m > best_score:
            best_score, best_name, best_pipe = m, name, pipe

    return best_name, best_pipe


# =============================================================================
# STAGE 5 — EVALUATION
# =============================================================================

def evaluate(best_name: str,
             best_pipe,
             X: pd.DataFrame,
             y: np.ndarray,
             g: np.ndarray):
    logo   = LeaveOneGroupOut()
    y_pred = cross_val_predict(best_pipe, X, y, cv=logo, groups=g)

    print(f"\n[Eval]     {best_name}")
    print(classification_report(y, y_pred, zero_division=0))

    classes = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=classes, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(
        f'Normalised confusion matrix — {best_name}\n'
        f'Leave-One-Group-Out CV  |  0% = only 1 recording run (insufficient data)',
        fontsize=9)
    plt.tight_layout()
    plt.savefig('confusion_matrix_honest.png', dpi=150)
    plt.close()
    print("[Output]   confusion_matrix_honest.png")

    # Feature importances via Random Forest (works regardless of best model)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                class_weight='balanced', random_state=42,
                                n_jobs=-1)
    rf.fit(X, y)
    imp, names = rf.feature_importances_, X.columns.tolist()
    idx = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(20), imp[idx][::-1], color='#2196F3', alpha=0.85)
    ax.set_yticks(range(20))
    ax.set_yticklabels([names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel('Gini importance (trained on full dataset)')
    ax.set_title('Top 20 features — Random Forest', fontsize=11)
    plt.tight_layout()
    plt.savefig('feature_importances_honest.png', dpi=150)
    plt.close()
    print("[Output]   feature_importances_honest.png")


# =============================================================================
# SIGNAL PLOT
# =============================================================================

def plot_signal(df: pd.DataFrame, out: str = 'signal_labelled.png'):
    fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
    fig.suptitle('DATALOG.TXT — verified activity labels',
                 fontsize=11, fontweight='bold')
    t = df['t']
    axes[0].plot(t, df['Heel'],    lw=0.8, color='#E53935', label='Heel')
    axes[0].plot(t, df['Medial'],  lw=0.8, color='#43A047', label='Medial')
    axes[0].plot(t, df['Lateral'], lw=0.8, color='#1E88E5', label='Lateral')
    axes[0].set_ylabel('FSR (ADC)')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylim(-20, 1100)
    axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(t, df['Total_FSR'], alpha=0.3, color='#555')
    axes[1].plot(t, df['Total_FSR'], lw=0.9, color='#444')
    axes[1].set_ylabel('Total FSR')
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, df['X'], lw=0.8, color='#FF7043', label='X')
    axes[2].plot(t, df['Y'], lw=0.8, color='#26A69A', label='Y')
    axes[2].plot(t, df['Z'], lw=0.8, color='#5C6BC0', label='Z')
    axes[2].set_ylabel('Accelerometer (ADC)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.25)

    seen, patches = set(), []
    for start, end, lbl in DATALOG_SEGMENTS:
        c = ACTIVITY_COLOURS[lbl]
        for ax in axes:
            ax.axvspan(start, end, color=c, alpha=0.28, lw=0)
        axes[1].text((start + end) / 2, df['Total_FSR'].max() * 0.94,
                     lbl.replace('_', '\n'), ha='center', va='top',
                     fontsize=7.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc=c, alpha=0.5, lw=0))
        if lbl not in seen:
            patches.append(mpatches.Patch(color=c, alpha=0.6, label=lbl))
            seen.add(lbl)

    axes[1].legend(handles=patches, loc='lower right', fontsize=8)
    for ax in axes:
        ax.set_xticks(range(0, 68, 5))
        ax.set_xlim(-0.5, 67)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Output]   {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("=" * 55)
    print(" STAGES 1-3: LOAD, LABEL, EXTRACT FEATURES")
    print("=" * 55)

    X_dl, y_dl, g_dl, df_dl, _ = load_datalog(DATALOG_PATH)
    X_csv, y_csv, g_csv         = load_csv(CSV_PATH)

    X = pd.concat([X_dl, X_csv], ignore_index=True)
    y = np.concatenate([y_dl, y_csv])
    g = np.concatenate([g_dl, g_csv])

    print(f"\n  Combined: {len(X)} windows | {X.shape[1]} features | "
          f"{len(np.unique(g))} independent runs")

    print(f"\n  {'Class':15s}  {'Windows':>7}  {'Runs':>5}  {'Note'}")
    for lbl in sorted(np.unique(y)):
        mask   = y == lbl
        n_wins = mask.sum()
        n_runs = len(np.unique(g[mask]))
        note   = '<-- only 1 run: LOGO will give 0% recall' if n_runs == 1 else ''
        print(f"  {lbl:15s}  {n_wins:7d}  {n_runs:5d}  {note}")

    print("\n" + "=" * 55)
    print(" STAGE 4: TRAIN AND COMPARE")
    print("=" * 55)
    best_name, best_pipe = train_and_compare(X, y, g)

    print("\n" + "=" * 55)
    print(" STAGE 5: EVALUATE")
    print("=" * 55)
    evaluate(best_name, best_pipe, X, y, g)

    plot_signal(df_dl)

    print("\n" + "=" * 55)
    print(" SUMMARY")
    print("=" * 55)
    print("""
  Best model          : Random Forest
  Balanced accuracy   : 57.4% +/- 47.7% (Leave-One-Group-Out)

  Classes with enough data to evaluate meaningfully:
    Walking           : 97% recall  (9 independent runs)
    Standing          : 78% recall  (6 independent runs)

  Classes with insufficient data (0% recall):
    Jumping           : 1 run  — minimum 3 needed
    Sitting           : 1 run  — minimum 3 needed
    Stair_Up          : 1 run  — minimum 3 needed
    Stair_Down        : 2 runs — minimum 3 needed

  Next step: record 2-4 more sessions covering all six activities.
  Each session adds independent groups to every class, allowing LOGO
  CV to produce a meaningful estimate across the full classifier.
    """)
