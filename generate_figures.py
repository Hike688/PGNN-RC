import sys, io, os, argparse, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============ GLOBAL STYLE (Nature-style) ============
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
})
NATURE_COLORS = ["#2166AC", "#D6604D", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628", "#F781BF", "#999999"]

# ============ ARG PARSE ============
parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true", help="Load saved models instead of training")
parser.add_argument("--output", default="results", help="Output directory")
args = parser.parse_args()
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

# ============ LOAD DATA ============
df = pd.read_csv("data/metal_data.csv")
features = ["M", "n", "sigma_y", "Tg"]
y = df["E"].values.ravel()

# Engineered features
df["M_per_n"] = df["M"] / df["n"]
df["sigma_y_over_Tg"] = df["sigma_y"] / df["Tg"]
df["inv_Tg"] = 1.0 / df["Tg"]
df["sigma_y_sq"] = df["sigma_y"]**2
df["sigma_y_Tg"] = df["sigma_y"] * df["Tg"]
df["M_norm"] = df["M"] / df["M"].max()
df["log_M"] = np.log(df["M"])
engineered_features = ["M", "n", "sigma_y", "Tg", "M_per_n", "sigma_y_over_Tg",
                        "inv_Tg", "sigma_y_sq", "sigma_y_Tg", "M_norm", "log_M"]
X_eng = df[engineered_features].values

X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_basic = df[features].values
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_basic, y, test_size=0.2, random_state=42)
scaler_basic = StandardScaler()
Xb_train_s = scaler_basic.fit_transform(Xb_train)
Xb_test_s = scaler_basic.transform(Xb_test)

print("Data loaded. Total samples:", len(df))
print("Train:", len(X_train), "Test:", len(X_test))

sigma_y_train = X_train[:, 2]
Tg_train = X_train[:, 3]
sigma_y_test = X_test[:, 2]
Tg_test = X_test[:, 3]

# ============ TRAIN OR LOAD MODELS ============
models = {}

def _save_models(models, X_test, y_test, Xb_test, yb_test, X_train, y_train, Xb_train, yb_train, scaler, scaler_basic, rf):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/models.pkl", "wb") as f:
        pickle.dump({
            "models": models, "X_test": X_test, "y_test": y_test,
            "Xb_test": Xb_test, "yb_test": yb_test,
            "X_train": X_train, "y_train": y_train,
            "Xb_train": Xb_train, "yb_train": yb_train,
            "scaler": scaler, "scaler_basic": scaler_basic, "rf": rf,
        }, f)
    print(f"Models saved to {MODEL_DIR}/")

if args.load:
    print("\nLoading saved models...")
    with open(f"{MODEL_DIR}/models.pkl", "rb") as f:
        saved = pickle.load(f)
    models = saved["models"]
    X_test, y_test = saved["X_test"], saved["y_test"]
    Xb_test, yb_test = saved["Xb_test"], saved["yb_test"]
    X_train, y_train = saved["X_train"], saved["y_train"]
    Xb_train, yb_train = saved["Xb_train"], saved["yb_train"]
    scaler, scaler_basic = saved["scaler"], saved["scaler_basic"]
    rf = saved["rf"]
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    Xb_train_s = scaler_basic.transform(Xb_train)
    Xb_test_s = scaler_basic.transform(Xb_test)
    print(f"Loaded {len(models)} models from disk.")

def ref_nrm(sigma_y, Tg):
    return 25 + 41.4*sigma_y - 0.0046*Tg + 0.0015*sigma_y*Tg

if not args.load:

    # --- Reference NRM ---
    y_ref = ref_nrm(sigma_y_test, Tg_test)
    models["Reference NRM"] = {"pred": y_ref, "truth": y_test,
                               "r2": r2_score(y_test, y_ref),
                               "rmse": np.sqrt(mean_squared_error(y_test, y_ref))}

    # --- Linear Regression (basic features) ---
    lr_basic = LinearRegression()
    lr_basic.fit(Xb_train_s, yb_train)
    pred_lr_basic = lr_basic.predict(Xb_test_s)
    models["Linear Regression (basic)"] = {"pred": pred_lr_basic, "truth": yb_test,
                                           "r2": r2_score(yb_test, pred_lr_basic),
                                           "rmse": np.sqrt(mean_squared_error(yb_test, pred_lr_basic))}

    # --- ANN Baseline (matching ref) ---
    ann_base = MLPRegressor(hidden_layer_sizes=(10,), activation="relu",
                            solver="adam", max_iter=5000, random_state=42)
    ann_base.fit(Xb_train_s, yb_train)
    pred_ann_base = ann_base.predict(Xb_test_s)
    models["ANN Baseline"] = {"pred": pred_ann_base, "truth": yb_test,
                               "r2": r2_score(yb_test, pred_ann_base),
                               "rmse": np.sqrt(mean_squared_error(yb_test, pred_ann_base))}

    # --- ANN Deep ---
    ann_deep = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation="relu",
                            solver="adam", max_iter=5000, random_state=42)
    ann_deep.fit(X_train_s, y_train)
    pred_ann_deep = ann_deep.predict(X_test_s)
    models["ANN Deep"] = {"pred": pred_ann_deep, "truth": y_test,
                           "r2": r2_score(y_test, pred_ann_deep),
                           "rmse": np.sqrt(mean_squared_error(y_test, pred_ann_deep))}

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    models["Random Forest"] = {"pred": pred_rf, "truth": y_test,
                                "r2": r2_score(y_test, pred_rf),
                                "rmse": np.sqrt(mean_squared_error(y_test, pred_rf))}

    # --- XGBoost ---
    xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_m.fit(X_train_s, y_train)
    pred_xgb = xgb_m.predict(X_test_s)
    models["XGBoost"] = {"pred": pred_xgb, "truth": y_test,
                          "r2": r2_score(y_test, pred_xgb),
                          "rmse": np.sqrt(mean_squared_error(y_test, pred_xgb))}

    # ============ PGNN-RC MODEL (INNOVATION) ============
    class PGNNRC:
        """Physics-Guided Neural Network with Residual Correction"""
        def __init__(self, hidden_layers=(64, 32, 16), random_state=42):
            self.phys_nrm = lambda sigma_y, Tg: 25 + 41.4*sigma_y - 0.0046*Tg + 0.0015*sigma_y*Tg
            self.correction_nn = MLPRegressor(
                hidden_layer_sizes=hidden_layers, activation="relu",
                solver="adam", max_iter=5000, random_state=random_state,
                learning_rate_init=0.001
            )
            self.scaler = StandardScaler()
            
        def fit(self, X_full, y, sigma_y_idx=2, tg_idx=3):
            sigma_y = X_full[:, sigma_y_idx]
            Tg = X_full[:, tg_idx]
            y_phys = self.phys_nrm(sigma_y, Tg)
            residual = y - y_phys
            X_scaled = self.scaler.fit_transform(X_full)
            self.correction_nn.fit(X_scaled, residual)
            return self
            
        def predict(self, X_full, sigma_y_idx=2, tg_idx=3):
            sigma_y = X_full[:, sigma_y_idx]
            Tg = X_full[:, tg_idx]
            y_phys = self.phys_nrm(sigma_y, Tg)
            X_scaled = self.scaler.transform(X_full)
            residual_pred = self.correction_nn.predict(X_scaled)
            return y_phys + residual_pred
        
        def score(self, X_full, y, sigma_y_idx=2, tg_idx=3):
            y_pred = self.predict(X_full, sigma_y_idx, tg_idx)
            return r2_score(y, y_pred)

    # Train PGNN-RC
    pgnnrc = PGNNRC(hidden_layers=(64, 32, 16))
    pgnnrc.fit(X_train, y_train)
    pred_pgnnrc = pgnnrc.predict(X_test)
    models["PGNN-RC (Ours)"] = {"pred": pred_pgnnrc, "truth": y_test,
                                 "r2": r2_score(y_test, pred_pgnnrc),
                                 "rmse": np.sqrt(mean_squared_error(y_test, pred_pgnnrc))}

    # === Ablation: PGNN without attention (just physics + standard ANN) ===
    pgnn_plain = PGNNRC(hidden_layers=(10,))
    pgnn_plain.fit(X_train, y_train)
    pred_pgnn_plain = pgnn_plain.predict(X_test)
    models["PGNN (no attention)"] = {"pred": pred_pgnn_plain, "truth": y_test,
                                      "r2": r2_score(y_test, pred_pgnn_plain),
                                      "rmse": np.sqrt(mean_squared_error(y_test, pred_pgnn_plain))}

    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(f'{"Model":<30} {"R2":<10} {"RMSE":<10}')
    print("-"*50)
    for name, m in sorted(models.items(), key=lambda x: x[1]["r2"], reverse=True):
        print(f"{name:<30} {m['r2']:<10.4f} {m['rmse']:<10.2f}")

    _save_models(models, X_test, y_test, Xb_test, yb_test, X_train, y_train, Xb_train, yb_train, scaler, scaler_basic, rf)

# ============ GENERATE FIGURES ============

# ---- FIGURE 1: Data distribution (E vs features) ----
fig1, axes = plt.subplots(2, 2, figsize=(7.08, 5.5))  # Nature single-col width
ax_flat = axes.flatten()
feature_names = ["M (g/mol)", "n (components)", "$\\sigma_y$ (GPa)", "$T_g$ (K)"]
for i, (feat, fname) in enumerate(zip(features, feature_names)):
    ax = ax_flat[i]
    ax.scatter(df[feat], df["E"], c=NATURE_COLORS[0], s=12, alpha=0.6, edgecolors="none")
    ax.set_xlabel(fname)
    ax.set_ylabel("E (GPa)")
    ax.text(0.95, 0.95, f"({chr(97+i)})", transform=ax.transAxes, va="top", ha="right", fontsize=8, fontweight="bold")
fig1.tight_layout(pad=1.5)
fig1.savefig(f"{output_dir}/fig1_data_distribution.pdf", bbox_inches="tight")
fig1.savefig(f"{output_dir}/fig1_data_distribution.svg", bbox_inches="tight")
print("Fig1 saved.")

# ---- FIGURE 2: Correlation heatmap ----
corr_cols = ["M", "n", "sigma_y", "Tg", "E"]
corr_df = df[corr_cols].copy()
corr_df.columns = ["M", "n", "$\\sigma_y$", "$T_g$", "E"]
fig2, ax2 = plt.subplots(figsize=(4.5, 4))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool), k=1)
sns.heatmap(corr_df.corr(), mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax2)
ax2.set_title("Pearson Correlation Matrix")
fig2.tight_layout()
fig2.savefig(f"{output_dir}/fig2_correlation.pdf", bbox_inches="tight")
fig2.savefig(f"{output_dir}/fig2_correlation.svg", bbox_inches="tight")
print("Fig2 saved.")

# ---- FIGURE 3: Model comparison bar chart (International Journal of Fatigue style) ----
model_names_display = ["Ref.\nNRM", "Linear\nReg.", "ANN\n(10)", "ANN\nDeep",
               "Random\nForest", "XGBoost", "PGNN\n(shallow)", "PGNN-RC\n(Ours)"]
model_keys = ["Reference NRM", "Linear Regression (basic)", "ANN Baseline",
              "ANN Deep", "Random Forest", "XGBoost",
              "PGNN (no attention)", "PGNN-RC (Ours)"]
r2_vals = [models[m]["r2"] for m in model_keys]
r2_ref = models["Reference NRM"]["r2"]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(7.08, 3.5))
bar_colors = ["#999999"]*6 + ["#D6604D", "#2166AC"]
x_pos = range(len(model_names_display))

# R2 bar with annotations
bars1 = ax3a.bar(x_pos, r2_vals, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)
ax3a.set_xticks(x_pos)
ax3a.set_xticklabels(model_names_display, fontsize=6)
ax3a.set_ylabel("$R^2$ Score")
ax3a.set_ylim(0.92, 1.0)
ax3a.axhline(y=r2_ref, color="red", linestyle="--", linewidth=0.8, label=f"Ref. NRM = {r2_ref:.4f}")
ax3a.legend(fontsize=6, loc="lower right")
for i, (bar, val) in enumerate(zip(bars1, r2_vals)):
    ax3a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
              f"{val:.4f}", ha="center", va="bottom", fontsize=5.5, rotation=45)
ax3a.text(0.02, 0.98, "(a)", transform=ax3a.transAxes, va="top", fontsize=8, fontweight="bold")

# RMSE bar with annotations
rmse_vals = [models[m]["rmse"] for m in model_keys]
bars2 = ax3b.bar(x_pos, rmse_vals, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)
ax3b.set_xticks(x_pos)
ax3b.set_xticklabels(model_names_display, fontsize=6)
ax3b.set_ylabel("RMSE (GPa)")
ax3b.set_ylim(0, 16)
for i, (bar, val) in enumerate(zip(bars2, rmse_vals)):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
              f"{val:.1f}", ha="center", va="bottom", fontsize=5.5)
ax3b.text(0.02, 0.98, "(b)", transform=ax3b.transAxes, va="top", fontsize=8, fontweight="bold")

fig3.tight_layout(pad=1.5)
fig3.savefig(f"{output_dir}/fig3_model_comparison.pdf", bbox_inches="tight")
fig3.savefig(f"{output_dir}/fig3_model_comparison.svg", bbox_inches="tight")
print("Fig3 saved.")

# ---- FIGURE 4: Predicted vs Actual with error bands (International Journal of Fatigue style) ----
fig4, axes4 = plt.subplots(2, 2, figsize=(7.08, 6.5))
model_subset = [("Reference NRM", "Reference\nNRM"), ("ANN Deep", "ANN\nDeep"),
                ("XGBoost", "XGBoost"), ("PGNN-RC (Ours)", "PGNN-RC\n(Ours)")]
for idx, (mname, label) in enumerate(model_subset):
    ax = axes4.flatten()[idx]
    truth = models[mname]["truth"]
    pred = models[mname]["pred"]
    ax.scatter(truth, pred, c=NATURE_COLORS[0], s=15, alpha=0.6, edgecolors="black", linewidth=0.3)
    lims = [min(truth.min(), pred.min()) - 5, max(truth.max(), pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    # Add 10% error bands like reference paper
    ax.fill_between(lims, [l*0.9 for l in lims], [l*1.1 for l in lims], alpha=0.08, color="gray", label=r"$\pm$10%")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Experimental E (GPa)")
    ax.set_ylabel("Predicted E (GPa)")
    r2v = models[mname]["r2"]
    rmsv = models[mname]["rmse"]
    maev = mean_absolute_error(truth, pred)
    ax.text(0.05, 0.9, f"$R^2$ = {r2v:.4f}\nRMSE = {rmsv:.1f} GPa\nMAE = {maev:.1f} GPa",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    ax.text(0.95, 0.05, f"({chr(97+idx)})", transform=ax.transAxes, va="bottom", ha="right", fontsize=8, fontweight="bold")
fig4.tight_layout(pad=1.5)
fig4.savefig(f"{output_dir}/fig4_predicted_vs_actual.pdf", bbox_inches="tight")
fig4.savefig(f"{output_dir}/fig4_predicted_vs_actual.svg", bbox_inches="tight")
print("Fig4 saved.")

# ---- FIGURE 5: Feature Importance (reference paper style) ----
fi = rf.feature_importances_
top_n = 10
top_idx = np.argsort(fi)[-top_n:]
top_names = [engineered_features[i] for i in top_idx]
top_vals = [fi[i] for i in top_idx]
# Add percentage labels
top_pcts = [f"{v*100:.1f}%" for v in top_vals]

fig5, ax5 = plt.subplots(figsize=(5.5, 3.5))
bars = ax5.barh(range(len(top_names)), top_vals, color=[NATURE_COLORS[0]]*3 + ["#999999"]*7, 
                height=0.6, edgecolor="black", linewidth=0.5)
# Highlight top 3
for i in range(3):
    bars[i].set_color(NATURE_COLORS[i])
ax5.set_yticks(range(len(top_names)))
ax5.set_yticklabels(top_names, fontsize=7)
ax5.set_xlabel("Feature Importance (Random Forest)")
ax5.invert_yaxis()
# Add percentage annotations
for i, (bar, pct) in enumerate(zip(bars, top_pcts)):
    ax5.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             pct, ha="left", va="center", fontsize=6, fontweight="bold" if i < 3 else "normal")
fig5.tight_layout(pad=1.0)
fig5.savefig(f"{output_dir}/fig5_feature_importance.pdf", bbox_inches="tight")
fig5.savefig(f"{output_dir}/fig5_feature_importance.svg", bbox_inches="tight")
print("Fig5 saved.")

# ---- FIGURE 6a: NRM 3D surface (standalone) ----
fig6a = plt.figure(figsize=(5.5, 4.5))
ax = fig6a.add_subplot(111, projection="3d")
sigma_y_grid = np.linspace(df["sigma_y"].min(), df["sigma_y"].max(), 30)
Tg_grid = np.linspace(df["Tg"].min(), df["Tg"].max(), 30)
Sy, Tgg = np.meshgrid(sigma_y_grid, Tg_grid)
E_surf = 25 + 41.4*Sy - 0.0046*Tgg + 0.0015*Sy*Tgg
ax.plot_surface(Sy, Tgg, E_surf, alpha=0.4, cmap="viridis", edgecolor="none")
ax.scatter(df["sigma_y"], df["Tg"], df["E"], c=NATURE_COLORS[1], s=8, alpha=0.7)
ax.set_xlabel("$\\sigma_y$ (GPa)")
ax.set_ylabel("$T_g$ (K)")
ax.set_zlabel("E (GPa)")
ax.view_init(elev=25, azim=-45)
fig6a.tight_layout(pad=1.5)
fig6a.savefig(f"{output_dir}/fig_nrm_surface.pdf", bbox_inches="tight")
fig6a.savefig(f"{output_dir}/fig_nrm_surface.svg", bbox_inches="tight")
print("Fig6 (NRM surface) saved.")

# ---- FIGURE 6b: Residual analysis panels (b)–(e) ----
fig6 = plt.figure(figsize=(7.08, 6.5))
gs = GridSpec(2, 2, figure=fig6, hspace=0.35, wspace=0.3)

# 6b: PGNN-RC residuals histogram
ax6b = fig6.add_subplot(gs[0, 0])
residual_vals = models["PGNN-RC (Ours)"]["truth"] - models["PGNN-RC (Ours)"]["pred"]
ax6b.hist(residual_vals, bins=20, color=NATURE_COLORS[0], edgecolor="black", linewidth=0.5, alpha=0.8)
ax6b.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
ax6b.set_xlabel("Residual (GPa)",fontsize=9)
ax6b.set_ylabel("Count",fontsize=9)
ax6b.text(0.02, 0.98, "(a) PGNN-RC residuals", transform=ax6b.transAxes, va="top", fontsize=7, fontweight="bold")

# 6c: PGNN-RC predicted vs actual
ax6c = fig6.add_subplot(gs[0, 1])
tr = models["PGNN-RC (Ours)"]["truth"]
pr = models["PGNN-RC (Ours)"]["pred"]
ax6c.scatter(tr, pr, c=NATURE_COLORS[0], s=12, alpha=0.6, edgecolors="none")
lims = [min(tr.min(), pr.min())-5, max(tr.max(), pr.max())+5]
ax6c.plot(lims, lims, "k--", linewidth=0.8)
ax6c.set_xlim(lims); ax6c.set_ylim(lims)
ax6c.set_xlabel("Experimental E (GPa)",fontsize=9)
ax6c.set_ylabel("Predicted E (GPa)",fontsize=9)
ax6c.text(0.05, 0.85, f"$R^2$ = {models['PGNN-RC (Ours)']['r2']:.4f}", transform=ax6c.transAxes, fontsize=7)
ax6c.text(0.02, 0.98, "(b) PGNN-RC prediction", transform=ax6c.transAxes, va="top", fontsize=7, fontweight="bold")

# 6d: Comparison of E/sigma_y ratio
ax6d = fig6.add_subplot(gs[1, 0])
E_over_sy = df["E"] / df["sigma_y"]
ax6d.hist(E_over_sy, bins=25, color=NATURE_COLORS[2], edgecolor="black", linewidth=0.5, alpha=0.8)
ax6d.axvline(x=49.8, color="red", linestyle="--", linewidth=0.8, label="$E/\\sigma_y = 49.8$")
ax6d.axvline(x=np.median(E_over_sy), color="blue", linestyle=":", linewidth=0.8, label=f"Median={np.median(E_over_sy):.1f}")
ax6d.set_xlabel("$E/\\sigma_y$",fontsize=9)
ax6d.set_ylabel("Count",fontsize=9)
ax6d.legend(fontsize=6)
ax6d.text(0.02, 0.98, "(c) $E/\\sigma_y$ ratio", transform=ax6d.transAxes,va="top", fontsize=7, fontweight="bold")

# 6e: By alloy system
ax6e = fig6.add_subplot(gs[1, 1])
alloy_bases = df["materials_name"].str.extract(r"^([A-Z][a-z]?)")[0].values
alloy_summary = df.groupby(alloy_bases)["E"].agg(["mean", "std", "count"]).sort_values("count", ascending=False)
top_systems = alloy_summary.head(8)
ax6e.bar(range(len(top_systems)), top_systems["mean"], yerr=top_systems["std"],
         color=NATURE_COLORS[3], edgecolor="black", linewidth=0.5, capsize=3)
ax6e.set_xticks(range(len(top_systems)))
ax6e.set_xticklabels(top_systems.index, fontsize=6)
ax6e.set_ylabel("Mean E (GPa)",fontsize=9)
ax6e.text(0.02, 0.98, "(d) By alloy system", transform=ax6e.transAxes, va="top", fontsize=7, fontweight="bold")

fig6.tight_layout(pad=1.5)
fig6.savefig(f"{output_dir}/fig6_residual_analysis.pdf", bbox_inches="tight")
fig6.savefig(f"{output_dir}/fig6_residual_analysis.svg", bbox_inches="tight")
print("Fig6 (residual analysis) saved.")

# ============ SAVE RESULTS TABLE ============
results_df = pd.DataFrame([{
    "Model": k, "R2": v["r2"], "RMSE": v["rmse"],
    "MAE": mean_absolute_error(v["truth"], v["pred"]),
    "MAPE": np.mean(np.abs((v["truth"] - v["pred"]) / (v["truth"] + 1e-10))) * 100
} for k, v in models.items()])
try:
    results_df.to_csv(f"{output_dir}/model_results.csv", index=False)
except PermissionError:
    results_df.to_csv(f"{output_dir}/model_results_v2.csv", index=False)
print("\nResults saved.")
print("\nBest model:", results_df.loc[results_df["R2"].idxmax(), "Model"])
print("Best R2:", results_df["R2"].max())

# Print final table
print("\n=== FINAL RESULTS TABLE ===")
print(results_df.to_string(index=False))
