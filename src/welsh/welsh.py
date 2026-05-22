import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy.optimize import minimize
from scipy import stats

#config
SHP_PATH = r'localauthorities.shp'
SHP_NAME_FIELD = 'name_en'
OUTPUT_DIR = r"output directory"

# M0 excluded: sigma is uninterpretable for Welsh
# with 2001 census methodology change (question rewrite + age expansion)
MODELS_TO_FIT = ['M0ns', 'MR']

CENSUS_YEARS = np.array([1991, 2001, 2011, 2021])

# fixed order so arrays line up with census dicts
SHAPEFILE_ORDER = [
    "Swansea", "Blaenau Gwent", "Vale of Glamorgan", "Newport", "Conwy",
    "Pembrokeshire", "Flintshire", "Cardiff", "Neath Port Talbot", "Gwynedd",
    "Merthyr Tydfil", "Torfaen", "Wrexham", "Caerphilly", "Bridgend",
    "Ceredigion", "Denbighshire", "Isle of Anglesey", "Powys",
    "Rhondda Cynon Taf", "Monmouthshire", "Carmarthenshire"
]

ALIAS_TO_FIELD = {name: name for name in SHAPEFILE_ORDER}

DISTRICT_REGION = {
    "Isle of Anglesey": "North", "Gwynedd": "North", "Conwy": "North",
    "Denbighshire": "North", "Flintshire": "North", "Wrexham": "North",
    "Ceredigion": "Mid", "Powys": "Mid",
    "Carmarthenshire": "South", "Pembrokeshire": "South",
    "Swansea": "South", "Neath Port Talbot": "South", "Bridgend": "South",
    "Vale of Glamorgan": "South", "Cardiff": "South",
    "Rhondda Cynon Taf": "South", "Merthyr Tydfil": "South",
    "Caerphilly": "South", "Blaenau Gwent": "South",
    "Torfaen": "South", "Monmouthshire": "South", "Newport": "South"
}
REGION_COLOURS = {'North': '#2471A3', 'Mid': '#16A085', 'South': '#C0392B'}
REGION_LIST = ['North', 'Mid', 'South']

X_OBS_DICT = {
    "Swansea": [0.071743, 0.093782, 0.080613, 0.086036],
    "Blaenau Gwent": [0.012026, 0.065595, 0.055280, 0.047040],
    "Vale of Glamorgan": [0.046215, 0.088068, 0.082062, 0.093270],
    "Newport": [0.013462, 0.071833, 0.065478, 0.056648],
    "Conwy": [0.219969, 0.232307, 0.206428, 0.202182],
    "Pembrokeshire": [0.119911, 0.163475, 0.139080, 0.129916],
    "Flintshire": [0.089717, 0.109191, 0.092895, 0.085813],
    "Cardiff": [0.046095, 0.087533, 0.087073, 0.101293],
    "Neath Port Talbot": [0.102172, 0.128337, 0.108177, 0.101901],
    "Gwynedd": [0.595733, 0.606339, 0.559653, 0.553382],
    "Merthyr Tydfil": [0.043405, 0.073473, 0.065450, 0.071140],
    "Torfaen": [0.014856, 0.080818, 0.071149, 0.064702],
    "Wrexham": [0.088019, 0.108971, 0.091690, 0.091214],
    "Caerphilly":[0.040423, 0.085219, 0.084456, 0.084963],
    "Bridgend": [0.052525, 0.080936, 0.072526, 0.072028],
    "Ceredigion": [0.464977, 0.441071, 0.387680, 0.373389],
    "Denbighshire": [0.188602, 0.207326, 0.186044, 0.176862],
    "Isle of Anglesey": [0.480092, 0.505141, 0.456300, 0.451918],
    "Powys": [0.135539, 0.159676, 0.137307, 0.125141],
    "Rhondda Cynon Taf": [0.058696, 0.097859, 0.096535, 0.101361],
    "Monmouthshire": [0.012052, 0.068232, 0.071923, 0.067630],
    "Carmarthenshire": [0.374269, 0.389609, 0.328306, 0.309181]
}

POP_DICT = {
    "Swansea": [223173, 216226, 231155, 231892],
    "Blaenau Gwent": [ 72258, 67795, 67348, 64796],
    "Vale of Glamorgan": [116909, 115116, 122018, 128080],
    "Newport": [133338, 131820, 139849, 153880],
    "Conwy":[106306, 106316, 111724, 111805],
    "Pembrokeshire": [112100, 110182, 118392, 120208],
    "Flintshire": [141366, 143382, 146940, 150560],
    "Cardiff": [285386, 294208, 332273, 351240],
    "Neath Port Talbot": [138199, 130305, 135278, 138320],
    "Gwynedd": [113341, 112800, 117789, 114308],
    "Merthyr Tydfil": [ 59302, 54115, 56623, 56958],
    "Torfaen":[ 90535, 88062, 87844, 89379],
    "Wrexham":[123360, 124024, 129425, 131088],
    "Caerphilly": [169632, 163297, 171972, 170687],
    "Bridgend":[129253, 124284, 134545, 141293],
    "Ceredigion": [ 63087, 72884, 73847, 69911],
    "Denbighshire": [ 88228, 90085, 90527, 93061],
    "Isle of Anglesey": [ 69143, 64679, 67403, 67083],
    "Powys": [119294, 122473, 129083, 129901],
    "Rhondda Cynon Taf": [232572, 223924, 225555, 230316],
    "Monmouthshire": [ 79906, 82351, 88609, 90685],
    "Carmarthenshire": [168379, 167373, 177642, 182718]
}

ADJACENCY_EDGES_NAMED = [
    ("Swansea", "Neath Port Talbot"), ("Swansea", "Carmarthenshire"),
    ("Blaenau Gwent", "Caerphilly"), ("Blaenau Gwent", "Powys"),
    ("Blaenau Gwent", "Monmouthshire"), ("Blaenau Gwent", "Torfaen"),
    ("Vale of Glamorgan", "Cardiff"), ("Vale of Glamorgan", "Bridgend"),
    ("Vale of Glamorgan", "Rhondda Cynon Taf"),
    ("Newport", "Cardiff"), ("Newport", "Caerphilly"),
    ("Newport", "Monmouthshire"), ("Newport", "Torfaen"),
    ("Neath Port Talbot", "Bridgend"), ("Neath Port Talbot", "Powys"),
    ("Neath Port Talbot", "Rhondda Cynon Taf"), ("Neath Port Talbot", "Carmarthenshire"),
    ("Conwy", "Gwynedd"), ("Conwy", "Denbighshire"),
    ("Gwynedd", "Ceredigion"), ("Gwynedd", "Powys"), ("Gwynedd", "Denbighshire"),
    ("Gwynedd", "Isle of Anglesey"), # SYNTHETIC - Menai Bridge
    ("Merthyr Tydfil", "Caerphilly"), ("Merthyr Tydfil", "Powys"),
    ("Merthyr Tydfil", "Rhondda Cynon Taf"),
    ("Torfaen", "Caerphilly"), ("Torfaen", "Monmouthshire"),
    ("Cardiff", "Caerphilly"), ("Cardiff", "Rhondda Cynon Taf"),
    ("Caerphilly", "Powys"), ("Caerphilly", "Rhondda Cynon Taf"),
    ("Bridgend", "Rhondda Cynon Taf"),
    ("Ceredigion", "Powys"), ("Ceredigion", "Carmarthenshire"),
    ("Denbighshire", "Powys"),
    ("Flintshire", "Denbighshire"), ("Flintshire", "Wrexham"),
    ("Wrexham", "Powys"), ("Wrexham", "Denbighshire"),
    ("Powys", "Rhondda Cynon Taf"), ("Powys", "Monmouthshire"),
    ("Powys", "Carmarthenshire"),
    ("Pembrokeshire", "Ceredigion"), ("Pembrokeshire", "Carmarthenshire"),
]

N = len(SHAPEFILE_ORDER)
T_DATA = CENSUS_YEARS.astype(float)
X_DATA = np.array([X_OBS_DICT[n] for n in SHAPEFILE_ORDER])
POP_DATA = np.array([POP_DICT[n] for n in SHAPEFILE_ORDER], dtype=float)
X0_VEC = X_DATA[:, 0].copy()
X_LAST = X_DATA[:, -1].copy()
N_OBS = N * (len(CENSUS_YEARS) - 1)

def get_pop(t):
    return np.array([np.interp(t, T_DATA, POP_DATA[i]) for i in range(N)])

#shapefile check + adjacency graph

def validate_and_load():
    print("\nshapefile check")
    gdf = gpd.read_file(SHP_PATH)
    print("CRS:", gdf.crs)
    print("rows:", len(gdf))
    print("geom:", gdf.geom_type.value_counts().to_dict())
    print("name field:", SHP_NAME_FIELD)
    print("values:", sorted(gdf[SHP_NAME_FIELD].tolist()), "\n")

    field_to_geom = {}
    ok = True
    for name in SHAPEFILE_ORDER:
        field_val = ALIAS_TO_FIELD[name]
        rows = gdf[gdf[SHP_NAME_FIELD] == field_val]
        if len(rows) == 0:
            print(f"not found: '{name}' -> '{field_val}'")
            ok = False
        elif len(rows) > 1:
            print(f"duplicate: '{name}' -> '{field_val}' ({len(rows)} rows)")
            ok = False
        else:
            field_to_geom[name] = rows.iloc[0].geometry

    if not ok:
        raise ValueError("ALIAS_TO_FIELD has unresolved entries.")
    print(f"All {N} alias mappings resolved.\n")

    geom_edges = 0
    for i in range(N):
        for j in range(i + 1, N):
            inter = field_to_geom[SHAPEFILE_ORDER[i]].intersection(field_to_geom[SHAPEFILE_ORDER[j]])
            if not inter.is_empty and inter.length > 0:
                geom_edges += 1
    synthetic = 1 # Anglesey–Gwynedd
    print(f"Shapefile geographic edges : {geom_edges}")
    print(f"+ synthetic (Menai Bridge) : {synthetic}")
    print(f"= total edges used : {len(ADJACENCY_EDGES_NAMED)}\n")

    idx = {name: i for i, name in enumerate(SHAPEFILE_ORDER)}
    A_adj = np.zeros((N, N))
    for (a, b) in ADJACENCY_EDGES_NAMED:
        i, j = idx[a], idx[b]
        A_adj[i, j] = A_adj[j, i] = 1.0

    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if A_adj[i, j] == 1.0]
    print(f"Graph: {N} districts, {len(edges)} edges.\n")
    return A_adj, edges

A_adj, EDGES = validate_and_load()

# ODE pieces

def as_rhs(x, s_vec, a):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (1 - x) * s_vec * x**a - x * (1 - s_vec) * (1 - x)**a

def diffusion_term(x, t, D_global, r_vec):
    # diffusion across edges; resistance slows exchange
    n = get_pop(t)
    diff = np.zeros(N)
    for (i, j) in EDGES:
        Dij = D_global / (1.0 + r_vec[i] + r_vec[j])
        flow = Dij * (n[j] * x[j] - n[i] * x[i])
        diff[i] += flow / n[i]
        diff[j] -= flow / n[j]
    return diff

def ode_rhs(t, x, s_vec, a, D_global, r_vec):
    # No shock term: sigma=0 for Welsh (M0 excluded by design)
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return as_rhs(x, s_vec, a) + diffusion_term(x, t, D_global, r_vec)

#RK4 integrator

def rk4(x0, t_eval, s_vec, a, D_global, r_vec, steps_per_year=4):
    x = x0.copy()
    out = np.zeros((N, len(t_eval)))
    out[:, 0] = x
    tc = t_eval[0]
    for k in range(1, len(t_eval)):
        tn = t_eval[k]
        # step in small chunks 10 year census gaps are too large otherwise
        ns = max(1, int(round((tn - tc) * steps_per_year)))
        h = (tn - tc) / ns
        for _ in range(ns):
            k1 = ode_rhs(tc, x,s_vec, a, D_global, r_vec)
            k2 = ode_rhs(tc + h/2, x + h/2*k1, s_vec, a, D_global, r_vec)
            k3 = ode_rhs(tc + h/2, x + h/2*k2, s_vec, a, D_global, r_vec)
            k4 = ode_rhs(tc + h, x + h*k3, s_vec, a, D_global, r_vec)
            x = np.clip(x + h/6*(k1 + 2*k2 + 2*k3 + k4), 0, 1)
            tc += h
        out[:, k] = x
    return out

# parameter unpacking for each model

def unpack_M0ns(params):
    s = np.clip(params[:N], 0.01, 0.99)
    a = float(np.clip(params[N], 0.30, 3.00))
    D = float(np.clip(params[N+1], 0.0, 0.50))
    return s, a, D, np.zeros(N)

def unpack_MR(params):
    s = np.clip(params[:N], 0.01, 0.99)
    a = float(np.clip(params[N], 0.30, 3.00))
    D = float(np.clip(params[N+1], 0.0, 0.50))
    r = np.clip(params[N+2:2*N+2], 0.0, 10.0)
    return s, a, D, r

UNPACK_FN = {'M0ns': unpack_M0ns, 'MR': unpack_MR}
K_PARAMS = {'M0ns': N + 2, 'MR': 2 * N + 2}

BOUNDS = {'M0ns': [(0.01, 0.99)] * N + [(0.30, 3.0), (0.0, 0.50)], 'MR': [(0.01, 0.99)] * N + [(0.30, 3.0), (0.0, 0.50)] + [(0.0, 10.0)] * N}
X0_INIT = {'M0ns': list(np.full(N, 0.5)) + [1.0, 0.01], 'MR': list(np.full(N, 0.5)) + [1.0, 0.01] + [0.1] * N}

def make_sse(unpack_fn):
    def objective(params):
        s, a, D, r_vec = unpack_fn(params)
        pred = rk4(X0_VEC, T_DATA, s, a, D, r_vec)
        return float(np.sum((pred - X_DATA)**2))
    return objective

#fit models
OPT = {'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-9}
results = {}

for mname in MODELS_TO_FIT:
    print(f"Fitting {mname} (k={K_PARAMS[mname]})")
    res = minimize(make_sse(UNPACK_FN[mname]), X0_INIT[mname], method='L-BFGS-B', bounds=BOUNDS[mname], options=OPT)
    s, a, D, r_vec = UNPACK_FN[mname](res.x)
    pred = rk4(X0_VEC, T_DATA, s, a, D, r_vec)
    sse = float(np.sum((pred - X_DATA)**2))
    rmse = np.mean(np.sqrt(np.mean((pred - X_DATA)**2, axis=1))) * 100
    results[mname] = dict(s=s, a=a, D=D, r_vec=r_vec, pred=pred, sse=sse, rmse=rmse, res=res)
    print(f"a={a:.4f} D={D:.6f} SSE={sse:.6f} RMSE={rmse:.4f}pp\n")

# AIC, AICc, BIC

def aic(sse, k, n): return n * np.log(sse / n) + 2 * k
def aicc(sse, k, n): return aic(sse, k, n) + 2*k*(k+1) / (n - k - 1)
def bic(sse, k, n): return n * np.log(sse / n) + k * np.log(n)

IC = {}
for mname, r in results.items():
    k = K_PARAMS[mname]
    IC[mname] = {'AIC': aic(r['sse'], k, N_OBS), 'AICc': aicc(r['sse'], k, N_OBS), 'BIC': bic(r['sse'], k, N_OBS)}

best = {c: min(IC, key=lambda m: IC[m][c]) for c in ['AIC', 'AICc', 'BIC']}

print("\nAIC, BIC, AICc")
print(f"n={N_OBS} ({N} units x {len(CENSUS_YEARS)-1} transitions)\n")
print(f"{'Model':<5} {'k':>4} {'SSE':>10} {'RMSE(pp)':>9} {'AIC':>8} {'AICc':>8} {'BIC':>8} {'deltaAIC':>8} {'deltaBIC':>8}")
for mname in MODELS_TO_FIT:
    r = results[mname]
    ic = IC[mname]
    daic = ic['AIC'] - IC[best['AIC']]['AIC']
    dbic = ic['BIC'] - IC[best['BIC']]['BIC']
    print(f"{mname:<5} {K_PARAMS[mname]:>4} {r['sse']:>10.5f} {r['rmse']:>9.4f} "
          f"{ic['AIC']:>8.2f} {ic['AICc']:>8.2f} {ic['BIC']:>8.2f} "
          f"{daic:>+8.2f} {dbic:>+8.2f}")
print(f"\nAIC: {best['AIC']} | AICc: {best['AICc']} | BIC: {best['BIC']}\n")

# regressions fitted params

ref = results['M0ns']
s_slope, s_intercept, s_r, s_p, s_se = stats.linregress(X0_VEC, ref['s'])
s_r2 = s_r**2
s_rho, s_p_sp = stats.spearmanr(X0_VEC, ref['s'])

print("s_i vs x0 (M0ns)")
print(f"OLS: s_i = {s_intercept:+.4f} + {s_slope:+.4f}*x_0")
print(f"R2={s_r2:.4f} r={s_r:.4f} p={s_p:.6f}")
print(f"Spearman: rho={s_rho:.4f} p={s_p_sp:.6f}")
print(f"s_i range: [{ref['s'].min():.4f}, {ref['s'].max():.4f}]")

if 'MR' in results:
    mr = results['MR']
    slope_r, intercept_r, r_r, p_r, se_r = stats.linregress(X0_VEC, mr['r_vec'])
    r2_r = r_r**2
    rho_r, p_sp_r = stats.spearmanr(X0_VEC, mr['r_vec'])
    resid_r = mr['r_vec'] - (intercept_r + slope_r * X0_VEC)
    outlier_thr_r = 2 * np.std(resid_r)

    print("r_i vs x_0 REGRESSION (MR)")
    print(f"OLS: r_i = {intercept_r:+.4f} + {slope_r:+.4f}*x_0")
    print(f"R2={r2_r:.4f} r={r_r:.4f} p={p_r:.4f}")
    print(f"Spearman: rho={rho_r:.4f} p={p_sp_r:.4f}\n")

# stability analysis

def as_rhs_scalar(x, s, a):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return (1 - x)*s*x**a - x*(1 - s)*(1 - x)**a

def as_jacobian(x, s, a):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (-s*x**a + (1-x)*s*a*x**(a-1)- (1-s)*(1-x)**a + x*(1-s)*a*(1-x)**(a-1))

def interior_fixed_point(s, a):
    if abs(a - 1.0) < 1e-6:
        return float(s)
    r = ((1 - s) / s) ** (1 / (a - 1))
    return r / (1 + r)

a_M0ns = results['M0ns']['a']
s_M0ns = results['M0ns']['s']

print("\nstability")
print(f"a (M0ns)={a_M0ns:.4f}")
if a_M0ns > 1.0:
    print("a > 1: standard bistable regime.")
else:
    print(f"a < 1 (a={a_M0ns:.4f}): x* is stable attractor, no tipping point.")

print(f"\n{'Unit':<22} {'x_0':>6} {'s_i':>6} {'x*':>7} "
      f"{'lambda(x*)':>8} {'Basin':>10} {'Obs':>8} {'OK/FAIL':>4}")

stability_rows = []
for i, name in enumerate(SHAPEFILE_ORDER):
    s_i = float(s_M0ns[i])
    x0_i = float(X0_VEC[i])
    obs_dir = "rising" if X_LAST[i] > x0_i else "falling"
    x_int = interior_fixed_point(s_i, a_M0ns)
    lam_int = as_jacobian(x_int, s_i, a_M0ns)
    if a_M0ns > 1.0:
        basin = "recovery" if x0_i > x_int else "decline"
        pred_dir = "rising" if x0_i > x_int else "falling"
    else:
        basin = "toward x*"
        pred_dir = "rising" if x0_i < x_int else "falling"
    match = "OK" if pred_dir == obs_dir else "FAIL"
    stability_rows.append(dict(name=name, x0=x0_i, s=s_i, x_int=x_int, lam_int=lam_int, basin=basin, obs_dir=obs_dir, pred_dir=pred_dir, match=match, region=DISTRICT_REGION[name]))
    print(f"{name:<22} {x0_i:.3f} {s_i:.4f} {x_int:.4f} {lam_int:+.4f} {basin:>10} {obs_dir:>8} {match}")

n_correct = sum(1 for r in stability_rows if r['match'] == 'OK')
print(f"\nBasin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)\n")

#fig1
model_labels = {'M0ns': f'M0ns\nNo shock\n(k={K_PARAMS["M0ns"]})', 'MR': f'MR\nResistance\n(k={K_PARAMS["MR"]})'}
model_base_colours = {'M0ns': '#884EA0', 'MR': '#2471A3'}

fig1, axes1 = plt.subplots(1, 3, figsize=(10, 5))
fig1.patch.set_facecolor('#F5F4EF')
fig1.suptitle('AIC, AICc, BIC')
for ax, criterion in zip(axes1, ['AIC', 'AICc', 'BIC']):
    ax.set_facecolor('#FAFAF8')
    vals = [IC[m][criterion] for m in MODELS_TO_FIT]
    best_v = min(vals)
    colours = ['#27AE60' if IC[m][criterion] == best_v else model_base_colours[m] for m in MODELS_TO_FIT]
    labels = [model_labels[m] for m in MODELS_TO_FIT]
    bars = ax.bar(labels, vals, color=colours, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best_i = vals.index(best_v)
    bars[best_i].set_edgecolor('black')
    bars[best_i].set_linewidth(2.5)
    ax.set_title(criterion, fontsize=10, fontweight='bold')
    ax.set_ylabel('Information Criterion', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(min(vals) - 5, max(vals) + 10)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'welsh_fig1.png'), dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.show()
print("fig1 saved")

#fig2
if 'MR' in results:
    mr = results['MR']
    t_crit = stats.t.ppf(0.975, df=N-2)
    x_line = np.linspace(X0_VEC.min(), X0_VEC.max(), 100)
    y_line = intercept_r + slope_r * x_line
    se_band = se_r * np.sqrt(1/N + (x_line - X0_VEC.mean())**2 / np.sum((X0_VEC - X0_VEC.mean())**2))

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.patch.set_facecolor('#F5F4EF')
    ax2.set_facecolor('#FAFAF8')

    for i, name in enumerate(SHAPEFILE_ORDER):
        c = REGION_COLOURS[DISTRICT_REGION[name]]
        ax2.scatter(X0_VEC[i]*100, mr['r_vec'][i], color=c, s=80, zorder=4, edgecolors='white', linewidths=0.8)
        if abs(resid_r[i]) > outlier_thr_r or mr['r_vec'][i] > np.percentile(mr['r_vec'], 80):
            ax2.annotate(name, (X0_VEC[i]*100, mr['r_vec'][i]), xytext=(6, 4), textcoords='offset points', fontsize=7.5, color=c, fontweight='bold')

    ax2.plot(x_line*100, y_line, color='#2C3E50', lw=2, zorder=3)
    ax2.fill_between(x_line*100, y_line - t_crit*se_band, y_line + t_crit*se_band, alpha=0.12, color='#2C3E50')
    ax2.axhline(0, color='black', lw=0.8, ls='--', alpha=0.3)
    ax2.text(0.04, 0.96,
         f'OLS: r_i = {intercept_r:+.3f} + {slope_r:+.3f}*x_0\n'
         f'R2={r2_r:.3f}, r={r_r:.3f}, p={p_r:.4f}\n'
         f'Spearman: rho={rho_r:.3f}, p={p_sp_r:.4f}', transform=ax2.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax2.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in REGION_COLOURS.items()], fontsize=9)
    ax2.set_xlabel('Initial Welsh fraction x_0 (1991, %)', fontsize=11)
    ax2.set_ylabel('Fitted resistance r_i (MR)', fontsize=11)
    ax2.set_title('Resistance vs initial fraction', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.15)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'welsh_fig2.png'), dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.show()
    print("fig2 saved")

#fig3
    fig3, ax3 = plt.subplots(figsize=(16, 5))
    fig3.patch.set_facecolor('#F5F4EF')
    ax3.set_facecolor('#FAFAF8')
    bar_colours = [REGION_COLOURS[DISTRICT_REGION[n]] for n in SHAPEFILE_ORDER]
    bars3 = ax3.bar(np.arange(N), mr['r_vec'], color=bar_colours, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, rv in zip(bars3, mr['r_vec']):
        if rv > 0.001:
            ax3.text(bar.get_x() + bar.get_width()/2, rv + 0.01, f'{rv:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax3.axhline(0, color='black', lw=1.0)
    ax3.set_xticks(np.arange(N))
    ax3.set_xticklabels([n.replace(' ', '\n') for n in SHAPEFILE_ORDER], rotation=0, fontsize=6, ha='center')
    ax3.set_ylabel('r_i (resistance, >= 0)', fontsize=10)
    ax3.set_title('Resistance by district', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.25)
    ax3.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in REGION_COLOURS.items()], fontsize=9)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'welsh_fig3.png'), dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
    plt.show()
    print("fig3 saved")

#fig4 trajectories
T_PROJ = np.linspace(float(CENSUS_YEARS[0]), float(CENSUS_YEARS[-1]) + 50, 500)
m_simple = MODELS_TO_FIT[0] # M0ns
m_complex = MODELS_TO_FIT[-1] # MR

traj_simple = rk4(X0_VEC, T_PROJ, results[m_simple]['s'], results[m_simple]['a'], results[m_simple]['D'], results[m_simple]['r_vec'])
traj_complex = rk4(X0_VEC, T_PROJ, results[m_complex]['s'], results[m_complex]['a'], results[m_complex]['D'], results[m_complex]['r_vec'])

ncols = 5
nrows = int(np.ceil(N / ncols))
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(ncols*4.4, nrows*4.2))
fig4.patch.set_facecolor('#F5F4EF')
fig4.suptitle(
    f'{m_simple} vs {m_complex}\n'
    f'{m_simple} RMSE={results[m_simple]["rmse"]:.2f}pp  |  '
    f'{m_complex} RMSE={results[m_complex]["rmse"]:.2f}pp', fontsize=11, fontweight='bold', y=0.99)

for idx, name in enumerate(SHAPEFILE_ORDER):
    row, col = divmod(idx, ncols)
    ax = axes4[row, col]
    ax.set_facecolor('#FAFAF8')
    c = REGION_COLOURS[DISTRICT_REGION[name]]
    ax.plot(T_PROJ, traj_simple[idx]*100, color='#AAAAAA', lw=1.4, ls='--', zorder=2)
    ax.plot(T_PROJ, traj_complex[idx]*100, color=c, lw=2.0, zorder=3)
    ax.scatter(CENSUS_YEARS, X_DATA[idx]*100, color='black', s=22, zorder=5)
    r_s = np.sqrt(np.mean((results[m_simple]['pred'][idx] - X_DATA[idx])**2)) * 100
    r_c = np.sqrt(np.mean((results[m_complex]['pred'][idx] - X_DATA[idx])**2)) * 100
    badge_c = '#27AE60' if r_c < 3.0 else ('#F39C12' if r_c < 6.0 else '#E74C3C')
    ax.text(0.98, 0.04, f'{m_simple} {r_s:.1f}pp\n{m_complex} {r_c:.1f}pp\ndelta {r_c-r_s:+.1f}pp', transform=ax.transAxes, fontsize=5.5, ha='right', va='bottom', color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_c, alpha=0.88))
    ax.set_title(f"{name}\n({DISTRICT_REGION[name]})", fontsize=7.5, fontweight='bold', color=c, pad=2)
    ax.axvline(2001, color='orange', lw=0.8, ls=':', alpha=0.7)
    ax.axvline(CENSUS_YEARS[-1], color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.axhline(50, color='black', lw=0.5, ls='--', alpha=0.2)
    ax.set_xlim(CENSUS_YEARS[0]-2, T_PROJ[-1]+2)
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15)
    if col == 0: ax.set_ylabel('%', fontsize=7)

for idx in range(N, nrows*ncols):
    row, col = divmod(idx, ncols)
    axes4[row, col].set_visible(False)

fig4.legend(handles=[ Line2D([0],[0], color='#AAAAAA', lw=1.5, ls='--', label=m_simple), Line2D([0],[0], color='#2471A3', lw=2, label=m_complex), Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Observed'), Line2D([0],[0], color='orange', lw=1, ls=':', label='2001 discontinuity')], loc='lower center', ncol=4, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.025, 1, 0.985])
fig4.savefig(os.path.join(OUTPUT_DIR, 'welsh_fig4.png'), dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.show()
print("fig4 saved")

#fig5
t_crit_s = stats.t.ppf(0.975, df=N-2)
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 7))
fig5.patch.set_facecolor('#F5F4EF')
fig5.suptitle('Prestige vs stability', fontsize=12, fontweight='bold')

#left: prestige grounding
ax5a.set_facecolor('#FAFAF8')
s_x = np.linspace(X0_VEC.min(), X0_VEC.max(), 100)
s_y = s_intercept + s_slope * s_x
s_se_b = s_se * np.sqrt(1/N + (s_x - X0_VEC.mean())**2 / np.sum((X0_VEC - X0_VEC.mean())**2))
ax5a.fill_between(s_x*100, s_y - t_crit_s*s_se_b, s_y + t_crit_s*s_se_b, alpha=0.12, color='#884EA0')
ax5a.plot(s_x*100, s_y, color='#884EA0', lw=2, zorder=3)
for i, name in enumerate(SHAPEFILE_ORDER):
    c = REGION_COLOURS[DISTRICT_REGION[name]]
    ax5a.scatter(X0_VEC[i]*100, s_M0ns[i], color=c, s=70, zorder=4, edgecolors='white', linewidths=0.8)
ax5a.axhline(0.491, color='#F39C12', lw=1.2, ls=':', alpha=0.8)
ax5a.axhline(0.33, color='#7F8C8D', lw=1.2, ls=':', alpha=0.8)
ax5a.text(62, 0.496, 'Basque s~0.49', fontsize=7, color='#F39C12', va='center')
ax5a.text(62, 0.335, 'Scottish Gaelic s~0.33', fontsize=7, color='#7F8C8D', va='center')
ax5a.text(0.03, 0.97, f'R2={s_r2:.3f}, r={s_r:.3f}, p={s_p:.4f}\nSpearman: rho={s_rho:.3f}, p={s_p_sp:.4f}', transform=ax5a.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
ax5a.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in REGION_COLOURS.items()], fontsize=8)
ax5a.set_xlabel('Initial Welsh fraction x_0 (1991, %)', fontsize=11)
ax5a.set_ylabel('Fitted prestige s_i (M0ns)', fontsize=11)
ax5a.set_title('s_i vs x_0: Prestige Grounding', fontsize=10, fontweight='bold')
ax5a.set_xlim(-2, 70)
ax5a.grid(True, alpha=0.15)

#right: basin classification
ax5b.set_facecolor('#FAFAF8')
for row in stability_rows:
    c = REGION_COLOURS[row['region']]
    marker = '^' if row['pred_dir'] == 'rising' else 'v'
    edge = 'black' if row['match'] == 'OK' else 'red'
    ax5b.scatter(row['x0']*100, row['x_int']*100, color=c, marker=marker, s=100, zorder=4, edgecolors=edge, linewidths=1.5)
    ax5b.annotate(row['name'], (row['x0']*100, row['x_int']*100), xytext=(4, 3), textcoords='offset points', fontsize=5, color=c)
diag = np.linspace(0, 100, 100)
ax5b.plot(diag, diag, color='black', lw=1.2, ls='--', alpha=0.4)
ax5b.fill_between(diag, diag, 100, alpha=0.04, color='#27AE60')
ax5b.fill_between(diag, 0, diag, alpha=0.04, color='#E74C3C')
ax5b.text(0.98, 0.04, f'Basin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)\na = {a_M0ns:.3f}', transform=ax5b.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
ax5b.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in REGION_COLOURS.items()] + [Line2D([0],[0], color='black', lw=1.2, ls='--', label='x_0 = x*')], fontsize=8, loc='upper left')
ax5b.set_xlabel('Initial Welsh fraction x_0 (1991, %)', fontsize=11)
ax5b.set_ylabel('Interior fixed point x*_i (%)', fontsize=11)
ax5b.set_title('Basin Classification: x_0 vs x*\n(up) rising, (down) falling, red = mismatch', fontsize=10, fontweight='bold')
ax5b.set_xlim(-2, 70); ax5b.set_ylim(-2, 102)
ax5b.grid(True, alpha=0.15)

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'welsh_fig5.png'), dpi=150, bbox_inches='tight', facecolor=fig5.get_facecolor())
plt.show()
print("fig5 saved")

#summary
print("\nsummary")
print(f"N={N} units | N_obs={N_OBS} | Models: {MODELS_TO_FIT}")
for mname in MODELS_TO_FIT:
    r = results[mname]
    print(f"{mname}: a={r['a']:.4f} D={r['D']:.6f} RMSE={r['rmse']:.4f}pp")
print(f"\nAIC: {best['AIC']} AICc: {best['AICc']} BIC: {best['BIC']}")
print(f"Prestige grounding (M0ns): R2={s_r2:.3f} p={s_p:.4f}")
print(f"s_i range: [{s_M0ns.min():.4f}, {s_M0ns.max():.4f}]")
print(f"Basin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)")
print(f"\ndone!")