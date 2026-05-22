import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy.optimize import minimize
from scipy import stats

# config

GEO_PATH = r"Counties_National_Statutory_Boundaries.geojson"
GEO_NAME_FIELD = 'COUNTY'
OUTPUT_DIR = r"output directory"

# M0 included alongside M0ns and MR
MODELS_TO_FIT = ['M0ns', 'M0', 'MR']

CENSUS_YEARS = np.array([2006, 2011, 2016, 2022])
T0_SOURCE = 2006.0

SHAPEFILE_ORDER = [
    'Carlow', 'Dublin', 'Kildare', 'Kilkenny', 'Laois', 'Longford',
    'Louth', 'Meath', 'Offaly', 'Westmeath', 'Wexford', 'Wicklow',
    'Clare', 'Cork', 'Kerry', 'Limerick', 'Tipperary', 'Waterford',
    'Galway', 'Leitrim', 'Mayo', 'Roscommon', 'Sligo',
    'Cavan', 'Donegal', 'Monaghan'
]

ALIAS_TO_FIELD = {name: name.upper() for name in SHAPEFILE_ORDER}

COUNTY_PROVINCE = {
    'Carlow': 'Leinster', 'Dublin': 'Leinster', 'Kildare': 'Leinster',
    'Kilkenny': 'Leinster', 'Laois': 'Leinster', 'Longford': 'Leinster',
    'Louth': 'Leinster', 'Meath': 'Leinster', 'Offaly': 'Leinster',
    'Westmeath': 'Leinster', 'Wexford': 'Leinster', 'Wicklow': 'Leinster',
    'Clare': 'Munster', 'Cork': 'Munster', 'Kerry': 'Munster',
    'Limerick': 'Munster', 'Tipperary': 'Munster', 'Waterford': 'Munster',
    'Galway': 'Connacht', 'Leitrim': 'Connacht', 'Mayo': 'Connacht',
    'Roscommon': 'Connacht', 'Sligo': 'Connacht',
    'Cavan': 'Ulster(pt.)', 'Donegal': 'Ulster(pt.)', 'Monaghan': 'Ulster(pt.)'
}
PROVINCE_COLOURS = {'Leinster': '#2471A3', 'Munster': '#C0392B', 'Connacht': '#16A085', 'Ulster(pt.)': '#884EA0'}

X_OBS_DICT = {
    "Carlow": [0.014499, 0.009961, 0.008765, 0.007488],
    "Dublin": [0.014638, 0.012739, 0.012686, 0.011275],
    "Kildare": [0.026388, 0.013542, 0.012377, 0.010893],
    "Kilkenny": [0.014607, 0.010532, 0.009674, 0.008823],
    "Laois": [0.012258, 0.009546, 0.008123, 0.007379],
    "Longford": [0.012358, 0.008538, 0.006288, 0.006224],
    "Louth": [0.012394, 0.009414, 0.009132, 0.007502],
    "Meath": [0.017619, 0.013338, 0.012115, 0.011747],
    "Offaly": [0.010865, 0.007772, 0.007683, 0.007048],
    "Westmeath": [0.012578, 0.009157, 0.007998, 0.006568],
    "Wexford": [0.012600, 0.007989, 0.007955, 0.006851],
    "Wicklow": [0.014185, 0.011000, 0.011754, 0.010138],
    "Clare": [0.016827, 0.013132, 0.012566, 0.011060],
    "Cork": [0.018581, 0.014458, 0.013197, 0.011822],
    "Kerry": [0.033933, 0.028927, 0.024420, 0.024051],
    "Limerick": [0.010611, 0.010396, 0.009210, 0.008843],
    "Tipperary": [0.009682, 0.010028, 0.009088, 0.007493],
    "Waterford": [0.014950, 0.015247, 0.014340, 0.014133],
    "Galway": [0.062848, 0.058340, 0.053573, 0.048632],
    "Leitrim": [0.008843, 0.010189, 0.010080, 0.008779],
    "Mayo": [0.025137, 0.020714, 0.017693, 0.014851],
    "Roscommon": [0.008440, 0.009147, 0.007824, 0.007444],
    "Sligo": [0.009902, 0.010169, 0.009277, 0.008234],
    "Cavan": [0.010937, 0.006832, 0.006170, 0.005863],
    "Donegal": [0.060517, 0.056530, 0.049820, 0.046384],
    "Monaghan": [0.015215, 0.011921, 0.011273, 0.011089]
}

POP_DICT = {
    "Carlow": [50349, 54612, 56932, 61968],
    "Dublin": [1187176, 1273069, 1347359, 1458154],
    "Kildare": [186335, 210312, 222504, 247774],
    "Kilkenny": [87558, 95419, 99232, 104160],
    "Laois": [67059, 80559, 84697, 91877],
    "Longford": [34391, 39000, 40873, 46751],
    "Louth": [111267, 122897, 128884, 139703],
    "Meath": [162831, 184135, 195044, 220826],
    "Offaly": [70868, 76687, 77961, 83150],
    "Westmeath": [79346, 86164, 88770, 96221],
    "Wexford": [131749, 145320, 149722, 163919],
    "Wicklow": [126194, 136640, 142425, 155851],
    "Clare": [110950, 117196, 118817, 127938],
    "Cork": [481295, 519032, 542868, 584156],
    "Kerry": [139835, 145502, 147707, 156458],
    "Limerick": [184055, 191809, 194899, 209536],
    "Tipperary": [149244, 158754, 159553, 167895],
    "Waterford": [107961, 113795, 116176, 127363],
    "Galway": [231670, 250653, 258058, 277737],
    "Leitrim": [28950, 31798, 32044, 35199],
    "Mayo": [123839, 130638, 130507, 137970],
    "Roscommon": [58768, 64065, 64544, 70259],
    "Sligo": [60894, 65393, 65535, 70198],
    "Cavan": [64003, 73183, 76176, 81704],
    "Donegal": [147264, 161137, 159192, 167084],
    "Monaghan": [55997, 60483, 61386, 65288]
}


N = len(SHAPEFILE_ORDER)
T_DATA = CENSUS_YEARS.astype(float)
X_DATA = np.array([X_OBS_DICT[n] for n in SHAPEFILE_ORDER])
POP_DATA = np.array([POP_DICT[n] for n in SHAPEFILE_ORDER], dtype=float)
X0_VEC = X_DATA[:, 0].copy()
X_LAST = X_DATA[:, -1].copy()
N_OBS = N * (len(CENSUS_YEARS) - 1)

def get_pop(t):
    return np.array([np.interp(t, T_DATA, POP_DATA[i]) for i in range(N)])

#validate and load geojson

def validate_and_load():
    print("GeoJSON check")
    gdf = gpd.read_file(GEO_PATH)
    print(f"CRS : {gdf.crs}")
    print(f"Rows : {len(gdf)}")
    print(f"Geom types : {gdf.geom_type.value_counts().to_dict()}")
    print(f"Name field : '{GEO_NAME_FIELD}'")
    print(f"All values : {sorted(gdf[GEO_NAME_FIELD].tolist())}\n")

    field_to_geom = {}
    ok = True
    for name in SHAPEFILE_ORDER:
        field_val = ALIAS_TO_FIELD[name]
        rows = gdf[gdf[GEO_NAME_FIELD] == field_val]
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

    A_adj = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            inter = field_to_geom[SHAPEFILE_ORDER[i]].intersection(field_to_geom[SHAPEFILE_ORDER[j]])
            if not inter.is_empty and inter.length > 0:
                A_adj[i, j] = A_adj[j, i] = 1.0

    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if A_adj[i, j] == 1.0]
    print(f"Graph: {N} counties, {len(edges)} shared-border edges.\n")
    return A_adj, edges

A_adj, EDGES = validate_and_load()

#ode pieces
def as_rhs(x, s_vec, a):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (1 - x) * s_vec * x**a - x * (1 - s_vec) * (1 - x)**a

def shock_term(t, x, sigma, lam):
    if sigma == 0.0 or t < T0_SOURCE:
        return np.zeros(N)
    return sigma * np.exp(-lam * (t - T0_SOURCE)) * (1 - x)

def diffusion_term(x, t, D_global, r_vec):
    n = get_pop(t)
    diff = np.zeros(N)
    for (i, j) in EDGES:
        Dij = D_global / (1.0 + r_vec[i] + r_vec[j])
        flow = Dij * (n[j] * x[j] - n[i] * x[i])
        diff[i] += flow / n[i]
        diff[j] -= flow / n[j]
    return diff

def ode_rhs(t, x, s_vec, sigma, a, lam, D_global, r_vec):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (as_rhs(x, s_vec, a) + shock_term(t, x, sigma, lam) + diffusion_term(x, t, D_global, r_vec))

#rk4
def rk4(x0, t_eval, s_vec, sigma, a, lam, D_global, r_vec, steps_per_year=4):
    x = x0.copy()
    out = np.zeros((N, len(t_eval)))
    out[:, 0] = x
    tc = t_eval[0]
    for k in range(1, len(t_eval)):
        tn = t_eval[k]
        ns = max(1, int(round((tn - tc) * steps_per_year)))
        h = (tn - tc) / ns
        for _ in range(ns):
            k1 = ode_rhs(tc, x, s_vec, sigma, a, lam, D_global, r_vec)
            k2 = ode_rhs(tc + h/2, x + h/2*k1, s_vec, sigma, a, lam, D_global, r_vec)
            k3 = ode_rhs(tc + h/2, x + h/2*k2, s_vec, sigma, a, lam, D_global, r_vec)
            k4 = ode_rhs(tc + h, x + h*k3, s_vec, sigma, a, lam, D_global, r_vec)
            x = np.clip(x + h/6*(k1 + 2*k2 + 2*k3 + k4), 0, 1)
            tc += h
        out[:, k] = x
    return out

#parameter unpacking

def unpack_M0ns(params):
    s = np.clip(params[:N], 0.01, 0.99)
    a = float(np.clip(params[N], 0.30, 3.00))
    D = float(np.clip(params[N+1], 0.0, 0.50))
    return s, 0.0, a, 0.0, D, np.zeros(N)

def unpack_M0(params):
    s = np.clip(params[:N], 0.01, 0.99)
    sig = float(np.clip(params[N], 0.0, 0.30))
    a = float(np.clip(params[N+1], 0.30, 3.00))
    lam = float(np.clip(params[N+2], 0.0, 0.30))
    D = float(np.clip(params[N+3], 0.0, 0.50))
    return s, sig, a, lam, D, np.zeros(N)

def unpack_MR(params):
    s = np.clip(params[:N], 0.01, 0.99)
    sig = float(np.clip(params[N], 0.0, 0.30))
    a = float(np.clip(params[N+1], 0.30, 3.00))
    lam = float(np.clip(params[N+2], 0.0, 0.30))
    D = float(np.clip(params[N+3], 0.0, 0.50))
    r = np.clip(params[N+4:2*N+4], 0.0, 10.0)
    return s, sig, a, lam, D, r

UNPACK_FN = {'M0ns': unpack_M0ns, 'M0': unpack_M0, 'MR': unpack_MR}

# k counts with sigma/lam restored
K_PARAMS = {'M0ns': N+2, 'M0': N+4, 'MR': 2*N+4}

BOUNDS = {
    'M0ns': [(0.01, 0.99)]*N + [(0.30, 3.0), (0.0, 0.50)],
    'M0': [(0.01, 0.99)]*N + [(0.0, 0.30), (0.30, 3.0), (0.0, 0.30), (0.0, 0.50)],
    'MR': [(0.01, 0.99)]*N + [(0.0, 0.30), (0.30, 3.0), (0.0, 0.30), (0.0, 0.50)] + [(0.0, 10.0)]*N,
}
X0_INIT = {
    'M0ns': list(np.full(N, 0.5)) + [1.0, 0.01],
    'M0': list(np.full(N, 0.5)) + [0.05, 1.0, 0.05, 0.01],
    'MR': list(np.full(N, 0.5)) + [0.05, 1.0, 0.05, 0.01] + [0.1]*N,
}

def make_sse(unpack_fn):
    def objective(params):
        s, sig, a, lam, D, r_vec = unpack_fn(params)
        pred = rk4(X0_VEC, T_DATA, s, sig, a, lam, D, r_vec)
        return float(np.sum((pred - X_DATA)**2))
    return objective

#fit models

OPT = {'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-9}
results = {}

for mname in MODELS_TO_FIT:
    print(f"Fitting {mname} (k={K_PARAMS[mname]})")
    res = minimize(make_sse(UNPACK_FN[mname]), X0_INIT[mname], method='L-BFGS-B', bounds=BOUNDS[mname], options=OPT)
    s, sig, a, lam, D, r_vec = UNPACK_FN[mname](res.x)
    pred = rk4(X0_VEC, T_DATA, s, sig, a, lam, D, r_vec)
    sse = float(np.sum((pred - X_DATA)**2))
    rmse = np.mean(np.sqrt(np.mean((pred - X_DATA)**2, axis=1))) * 100
    results[mname] = dict(s=s, sig=sig, a=a, lam=lam, D=D, r_vec=r_vec, pred=pred, sse=sse, rmse=rmse, res=res)
    print(f"a={a:.4f} sigma={sig:.5f} lam={lam:.5f} D={D:.6f} SSE={sse:.6f} RMSE={rmse:.4f}pp\n")

delta_shock = results['M0']['rmse'] - results['M0ns']['rmse']
print("shock test")
print(f"M0ns: RMSE={results['M0ns']['rmse']:.4f}pp")
print(f"M0 : RMSE={results['M0']['rmse']:.4f}pp"
      f"delta={delta_shock:+.4f}pp"
      f"sigma={results['M0']['sig']:.5f}"
      f"lam={results['M0']['lam']:.5f}")
print()

#AIC, AICc, BIC
def aic(sse, k, n): return n * np.log(sse / n) + 2 * k
def aicc(sse, k, n): return aic(sse, k, n) + 2*k*(k+1) / (n - k - 1)
def bic(sse, k, n): return n * np.log(sse / n) + k * np.log(n)

IC = {}
for mname, r in results.items():
    k = K_PARAMS[mname]
    IC[mname] = {'AIC': aic (r['sse'], k, N_OBS), 'AICc': aicc(r['sse'], k, N_OBS), 'BIC': bic (r['sse'], k, N_OBS)}

best = {c: min(IC, key=lambda m: IC[m][c]) for c in ['AIC', 'AICc', 'BIC']}

print("AIC, BIC, AICc comparison")
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

#pure fit
#Three models:
    #AS_shared: one (s, a) for all counties (original A-S)
    #AS_indiv: per county s_i, shared a
    #AS_free: per county (s_i, a_i)

print("Pure A-S Fit")

def as_rhs_scalar(x, s, a):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return (1 - x)*s*x**a - x*(1 - s)*(1 - x)**a

def rk4_scalar(x0, t_eval, s, a, steps_per_year=4):
    """Single-unit RK4 for pure AS."""
    x = float(x0)
    tc = t_eval[0]
    out = [x]
    for k in range(1, len(t_eval)):
        tn = t_eval[k]
        ns = max(1, int(round((tn - tc) * steps_per_year)))
        h = (tn - tc) / ns
        for _ in range(ns):
            k1 = as_rhs_scalar(x, s, a)
            k2 = as_rhs_scalar(x + h/2*k1, s, a)
            k3 = as_rhs_scalar(x + h/2*k2, s, a)
            k4 = as_rhs_scalar(x + h*k3, s, a)
            x = np.clip(x + h/6*(k1 + 2*k2 + 2*k3 + k4), 0, 1)
            tc += h
        out.append(x)
    return np.array(out)

def sse_as_shared(params):
    s = np.clip(params[0], 0.01, 0.99)
    a = np.clip(params[1], 0.30, 3.00)
    total = 0.0
    for i in range(N):
        pred = rk4_scalar(X0_VEC[i], T_DATA, s, a)
        total += np.sum((pred[1:] - X_DATA[i, 1:])**2)
    return total

res_shared = minimize(sse_as_shared, [0.3, 1.0],
                      method='L-BFGS-B',
                      bounds=[(0.01, 0.99), (0.30, 3.0)],
                      options=OPT)
s_shared = float(np.clip(res_shared.x[0], 0.01, 0.99))
a_shared = float(np.clip(res_shared.x[1], 0.30, 3.00))
pred_shared = np.array([rk4_scalar(X0_VEC[i], T_DATA, s_shared, a_shared)
                        for i in range(N)])
sse_shared = float(np.sum((pred_shared[:, 1:] - X_DATA[:, 1:])**2))
rmse_shared = np.mean(np.sqrt(np.mean((pred_shared - X_DATA)**2, axis=1))) * 100

print(f"AS_shared (k=2): s={s_shared:.4f} a={a_shared:.4f} SSE={sse_shared:.6f} RMSE={rmse_shared:.4f}pp")

def sse_as_indiv(params):
    s_vec = np.clip(params[:N], 0.01, 0.99)
    a = float(np.clip(params[N], 0.30, 3.00))
    total = 0.0
    for i in range(N):
        pred = rk4_scalar(X0_VEC[i], T_DATA, s_vec[i], a)
        total += np.sum((pred[1:] - X_DATA[i, 1:])**2)
    return total

x0_indiv = list(np.full(N, 0.3)) + [1.0]
bds_indiv = [(0.01, 0.99)]*N + [(0.30, 3.0)]
res_indiv = minimize(sse_as_indiv, x0_indiv, method='L-BFGS-B', bounds=bds_indiv, options=OPT)
s_indiv = np.clip(res_indiv.x[:N], 0.01, 0.99)
a_indiv = float(np.clip(res_indiv.x[N], 0.30, 3.00))
pred_indiv = np.array([rk4_scalar(X0_VEC[i], T_DATA, s_indiv[i], a_indiv) for i in range(N)])
sse_indiv = float(np.sum((pred_indiv[:, 1:] - X_DATA[:, 1:])**2))
rmse_indiv = np.mean(np.sqrt(np.mean((pred_indiv - X_DATA)**2, axis=1))) * 100

# Prestige grounding for AS_indiv
slope_ai, intercept_ai, r_ai, p_ai, _ = stats.linregress(X0_VEC, s_indiv)
rho_ai, p_sp_ai = stats.spearmanr(X0_VEC, s_indiv)

print(f"AS_indiv (k=N+1={N+1}): shared a={a_indiv:.4f} SSE={sse_indiv:.6f}  RMSE={rmse_indiv:.4f}pp")
print(f"Prestige grounding: R2={r_ai**2:.4f} Spearman rho={rho_ai:.4f}  p={p_ai:.4f}")

def sse_as_free(params):
    s_vec = np.clip(params[:N], 0.01, 0.99)
    a_vec = np.clip(params[N:2*N], 0.30, 3.00)
    total = 0.0
    for i in range(N):
        pred = rk4_scalar(X0_VEC[i], T_DATA, s_vec[i], a_vec[i])
        total += np.sum((pred[1:] - X_DATA[i, 1:])**2)
    return total

x0_free = list(np.full(N, 0.3)) + list(np.full(N, 1.0))
bds_free = [(0.01, 0.99)]*N + [(0.30, 3.0)]*N
res_free = minimize(sse_as_free, x0_free, method='L-BFGS-B', bounds=bds_free, options=OPT)
s_free = np.clip(res_free.x[:N], 0.01, 0.99)
a_free = np.clip(res_free.x[N:2*N], 0.30, 3.00)
pred_free = np.array([rk4_scalar(X0_VEC[i], T_DATA, s_free[i], a_free[i]) for i in range(N)])
sse_free = float(np.sum((pred_free[:, 1:] - X_DATA[:, 1:])**2))
rmse_free = np.mean(np.sqrt(np.mean((pred_free - X_DATA)**2, axis=1))) * 100

print(f"AS_free (k=2N={2*N}): mean a={np.mean(a_free):.4f} "
      f"median a={np.median(a_free):.4f} std a={np.std(a_free):.4f} "
      f"SSE={sse_free:.6f} RMSE={rmse_free:.4f}pp")

# AIC for pure AS models (n = N x (T-1) = N_OBS)
n_as = N_OBS
ic_as = {}
for label, sse_v, k_v in [('AS_shared', sse_shared, 2), ('AS_indiv', sse_indiv, N+1),('AS_free', sse_free, 2*N)]:
    ic_as[label] = {'AIC': aic (sse_v, k_v, n_as),'AICc': aicc(sse_v, k_v, n_as),'BIC':  bic (sse_v, k_v, n_as)}

print(f"\n{'Model':<12} {'k':>4} {'AIC':>8} {'AICc':>8} {'BIC':>8}")
for label in ['AS_shared', 'AS_indiv', 'AS_free']:
    ic = ic_as[label]
    k_v = {'AS_shared': 2, 'AS_indiv': N+1, 'AS_free': 2*N}[label]
    print(f"{label:<12} {k_v:>4}  {ic['AIC']:>8.2f}  {ic['AICc']:>8.2f}  {ic['BIC']:>8.2f}")
best_as = min(ic_as, key=lambda m: ic_as[m]['AIC'])
print(f"\nBest by AIC: {best_as}")

print(f"AS_shared recovered: a = {a_shared:.4f}")
print(f"AS_indiv recovered: a = {a_indiv:.4f}")
print(f"AS_free mean: a = {np.mean(a_free):.4f} (range [{a_free.min():.3f}, {a_free.max():.3f}])")

if a_indiv < 1.0:
    print(f"Irish data does NOT recover a ~ 1.31.")
elif abs(a_indiv - 1.31) < 0.2:
    print(f"\na ~ {a_indiv:.2f} close to original AS value.")
else:
    print(f"\na = {a_indiv:.2f} recovered but differs from 1.31.")
print()

#regressions

ref = results['M0ns']
s_slope, s_intercept, s_r, s_p, s_se = stats.linregress(X0_VEC, ref['s'])
s_r2 = s_r**2
s_rho, s_p_sp = stats.spearmanr(X0_VEC, ref['s'])

print("s_i vs x_0: PRESTIGE GROUNDING (M0ns spatial)")
print(f"OLS: s_i = {s_intercept:+.4f} + {s_slope:+.4f}*x_0")
print(f"R2={s_r2:.4f} r={s_r:.4f} p={s_p:.6f}")
print(f"Spearman: rho={s_rho:.4f} p={s_p_sp:.6f}")
print(f"s_i range: [{ref['s'].min():.4f}, {ref['s'].max():.4f}]")
print(f"Basque benchmark: R2=0.89 | Welsh benchmark: R2=0.98\n")

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

# stability analysis (commented out: basin accuracy unstable across optimizer runs)

#def as_jacobian(x, s, a):
#    x = np.clip(x, 1e-9, 1 - 1e-9)
#    return (-s*x**a + (1-x)*s*a*x**(a-1)
#            - (1-s)*(1-x)**a + x*(1-s)*a*(1-x)**(a-1))

#def interior_fixed_point(s, a):
#    if abs(a - 1.0) < 1e-6:
#        return float(s)
#    exponent = (np.log(s) - np.log(1 - s)) / (a - 1)
#    return 1.0 / (1.0 + np.exp(exponent))

a_M0ns = results['M0ns']['a']
s_M0ns = results['M0ns']['s']

#print("Stability analysis")

#print(f"a (M0ns)={a_M0ns:.4f}")
#if a_M0ns > 1.0:
#    print("a > 1: standard bistable regime.")
#else:
#    print(f"a < 1 (a={a_M0ns:.4f}): x* is stable attractor.")

#print(f"\n{'Unit':<12} {'x_0':>6} {'s_i':>6} {'x*':>7} "
#      f"{'lambda(x*)':>8} {'Basin':>10} {'Obs':>8} {'OK/FAIL':>4}")

#stability_rows = []
#for i, name in enumerate(SHAPEFILE_ORDER):
#    s_i = float(s_M0ns[i])
#    x0_i= float(X0_VEC[i])
#    obs_dir = "rising"if X_LAST[i] > x0_i else "falling"
#    x_int = interior_fixed_point(s_i, a_M0ns)
#    lam_int = as_jacobian(x_int, s_i, a_M0ns)
#    dx0 = ode_rhs(T0_SOURCE, X0_VEC, s_M0ns, 0.0, a_M0ns, 0.0, results['M0ns']['D'], np.zeros(N))[i]
#    pred_dir = "rising" if dx0 > 0 else "falling"
#    basin = "ODE sign"
#    match = "OK"if pred_dir == obs_dir else "FAIL"
#    stability_rows.append(dict(name=name, x0=x0_i, s=s_i, x_int=x_int, lam_int=lam_int,basin=basin, obs_dir=obs_dir, pred_dir=pred_dir, match=match,province=COUNTY_PROVINCE[name]))
#    print(f"{name:<12} {x0_i:.3f} {s_i:.4f} {x_int:.4f} "
#          f"{lam_int:+.4f}   {basin:>10}   {obs_dir:>8}   {match}")
#
#n_correct = sum(1 for r in stability_rows if r['match'] == 'OK')
#print(f"\nBasin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)")

#fig1
model_labels = {
    'M0ns': f'M0ns\nNo shock\n(k={K_PARAMS["M0ns"]})',
    'M0': f'M0\nGlobal D + sigma\n(k={K_PARAMS["M0"]})',
    'MR': f'MR\nResistance + sigma\n(k={K_PARAMS["MR"]})',
}
model_base_colours = {'M0ns': '#884EA0', 'M0': '#7F8C8D', 'MR': '#2471A3'}

fig1, axes1 = plt.subplots(1, 3, figsize=(13, 5))
fig1.patch.set_facecolor('#F5F4EF')
fig1.suptitle('Model Comparison: AIC, AICc, BIC',fontsize=12, fontweight='bold')
for ax, criterion in zip(axes1, ['AIC', 'AICc', 'BIC']):
    ax.set_facecolor('#FAFAF8')
    vals = [IC[m][criterion] for m in MODELS_TO_FIT]
    best_v = min(vals)
    colours = ['#27AE60' if IC[m][criterion] == best_v else model_base_colours[m] for m in MODELS_TO_FIT]
    bars = ax.bar([model_labels[m] for m in MODELS_TO_FIT], vals, color=colours, alpha=0.85, edgecolor='white', linewidth=0.8)
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
fig1.savefig(os.path.join(OUTPUT_DIR, 'irish_fig1.png'), dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.show()
print("fig1 saved.")

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
        c = PROVINCE_COLOURS[COUNTY_PROVINCE[name]]
        ax2.scatter(X0_VEC[i]*100, mr['r_vec'][i], color=c, s=80, zorder=4, edgecolors='white', linewidths=0.8)
        if abs(resid_r[i]) > outlier_thr_r or mr['r_vec'][i] > np.percentile(mr['r_vec'], 80):
            ax2.annotate(name, (X0_VEC[i]*100, mr['r_vec'][i]), xytext=(6, 4), textcoords='offset points', fontsize=7.5, color=c, fontweight='bold')
    ax2.plot(x_line*100, y_line, color='#2C3E50', lw=2, zorder=3)
    ax2.fill_between(x_line*100, y_line - t_crit*se_band, y_line + t_crit*se_band, alpha=0.12, color='#2C3E50')
    ax2.axhline(0, color='black', lw=0.8, ls='--', alpha=0.3)
    ax2.text(0.03, 0.97,
             f'OLS: r_i = {intercept_r:+.3f} + {slope_r:+.3f}*x_0\n'
             f'R2={r2_r:.3f}, r={r_r:.3f}, p={p_r:.4f}\n'
             f'Spearman: rho={rho_r:.3f}, p={p_sp_r:.4f}', transform=ax2.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
    ax2.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=9)
    ax2.set_xlabel('Initial Irish fraction x_0 (2006, %)', fontsize=11)
    ax2.set_ylabel('Fitted resistance r_i (MR)', fontsize=11)
    ax2.set_title('Spatial Resistance vs Initial Irish Fraction', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.15)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'irish_fig2.png'), dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.show()
    print("fig2 saved.")

    fig3, ax3 = plt.subplots(figsize=(18, 5))
    fig3.patch.set_facecolor('#F5F4EF')
    ax3.set_facecolor('#FAFAF8')
    bar_colours = [PROVINCE_COLOURS[COUNTY_PROVINCE[n]] for n in SHAPEFILE_ORDER]
    bars3 = ax3.bar(np.arange(N), mr['r_vec'], color=bar_colours, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, rv in zip(bars3, mr['r_vec']):
        if rv > 0.001:
            ax3.text(bar.get_x() + bar.get_width()/2, rv + 0.01, f'{rv:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax3.axhline(0, color='black', lw=1.0)
    ax3.set_xticks(np.arange(N))
    ax3.set_xticklabels(SHAPEFILE_ORDER, rotation=45, fontsize=7, ha='right')
    ax3.set_ylabel('r_i (resistance, >= 0)', fontsize=10)
    ax3.set_title(f'Per-County Spatial Resistance r_i', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.25)
    ax3.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=9)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'irish_fig3.png'), dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
    plt.show()
    print("fig3 saved.")

#fig4 trajectories

T_PROJ = np.linspace(float(CENSUS_YEARS[0]), float(CENSUS_YEARS[-1]) + 30, 500)
m_simple = 'M0ns'
m_complex = 'MR'

traj_simple = rk4(X0_VEC, T_PROJ, results[m_simple]['s'], results[m_simple]['sig'], results[m_simple]['a'], results[m_simple]['lam'], results[m_simple]['D'], results[m_simple]['r_vec'])
traj_complex = rk4(X0_VEC, T_PROJ, results[m_complex]['s'], results[m_complex]['sig'], results[m_complex]['a'], results[m_complex]['lam'], results[m_complex]['D'], results[m_complex]['r_vec'])

ncols = 6
nrows = int(np.ceil(N / ncols))
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(ncols*4.0, nrows*3.8))
fig4.patch.set_facecolor('#F5F4EF')
fig4.suptitle(
    f'{m_simple} vs {m_complex}\n'
    f'x = daily Irish speakers outside education / total population', fontsize=10, fontweight='bold', y=0.99)

for idx, name in enumerate(SHAPEFILE_ORDER):
    row, col = divmod(idx, ncols)
    ax = axes4[row, col]
    ax.set_facecolor('#FAFAF8')
    c = PROVINCE_COLOURS[COUNTY_PROVINCE[name]]
    ax.plot(T_PROJ, traj_simple[idx]*100, color='#AAAAAA', lw=1.4, ls='--', zorder=2)
    ax.plot(T_PROJ, traj_complex[idx]*100, color=c, lw=2.0, zorder=3)
    ax.scatter(CENSUS_YEARS, X_DATA[idx]*100, color='black', s=22, zorder=5)
    r_s = np.sqrt(np.mean((results[m_simple]['pred'][idx] - X_DATA[idx])**2)) * 100
    r_c = np.sqrt(np.mean((results[m_complex]['pred'][idx] - X_DATA[idx])**2)) * 100
    badge_c = '#27AE60' if r_c < 1.0 else ('#F39C12' if r_c < 2.0 else '#E74C3C')
    ax.text(0.98, 0.04,
            f'{m_simple} {r_s:.2f}pp\n{m_complex} {r_c:.2f}pp\ndelta {r_c-r_s:+.2f}pp', transform=ax.transAxes, fontsize=5, ha='right', va='bottom', color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_c, alpha=0.88))
    ax.set_title(f"{name}\n({COUNTY_PROVINCE[name]})", fontsize=6.5, fontweight='bold', color=c, pad=2)
    ax.axvline(CENSUS_YEARS[-1], color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlim(CENSUS_YEARS[0]-1, T_PROJ[-1]+1)
    ax.set_ylim(-0.5, max(X_DATA.max()*100 * 1.3, 3))
    ax.tick_params(labelsize=5.5)
    ax.grid(True, alpha=0.15)
    if col == 0: ax.set_ylabel('%', fontsize=6)

for idx in range(N, nrows*ncols):
    row, col = divmod(idx, ncols)
    axes4[row, col].set_visible(False)

fig4.legend(handles=[
    Line2D([0],[0], color='#AAAAAA', lw=1.5, ls='--', label=m_simple),
    Line2D([0],[0], color='#2471A3', lw=2, label=m_complex),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Observed')
], loc='lower center', ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.02, 1, 0.985])
fig4.savefig(os.path.join(OUTPUT_DIR, 'irish_fig4.png'), dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.show()
print("fig4 saved.")

#fig5

t_crit_s = stats.t.ppf(0.975, df=N-2)
fig5, ax5a = plt.subplots(1, 1, figsize=(9, 7))
fig5.patch.set_facecolor('#F5F4EF')
fig5.suptitle('Prestige Grounding', fontsize=12, fontweight='bold')

ax5a.set_facecolor('#FAFAF8')
s_x = np.linspace(X0_VEC.min(), X0_VEC.max(), 100)
s_y = s_intercept + s_slope * s_x
s_se_b = s_se * np.sqrt(1/N + (s_x - X0_VEC.mean())**2 / np.sum((X0_VEC - X0_VEC.mean())**2))
ax5a.fill_between(s_x*100, s_y - t_crit_s*s_se_b, s_y + t_crit_s*s_se_b, alpha=0.12, color='#884EA0')
ax5a.plot(s_x*100, s_y, color='#884EA0', lw=2, zorder=3)
for i, name in enumerate(SHAPEFILE_ORDER):
    c = PROVINCE_COLOURS[COUNTY_PROVINCE[name]]
    ax5a.scatter(X0_VEC[i]*100, s_M0ns[i], color=c, s=70, zorder=4, edgecolors='white', linewidths=0.8)
    ax5a.annotate(name[:3], (X0_VEC[i]*100, s_M0ns[i]), xytext=(3, 2), textcoords='offset points', fontsize=5, color=c)
ax5a.axhline(0.491, color='#F39C12', lw=1.2, ls=':', alpha=0.8)
ax5a.axhline(0.40, color='#27AE60', lw=1.2, ls=':', alpha=0.8)
ax5a.text(0.65, 0.77, 'Basque s~0.49', fontsize=7, color='#F39C12', transform=ax5a.transAxes)
ax5a.text(0.65, 0.60, 'Welsh s~0.40', fontsize=7, color='#27AE60', transform=ax5a.transAxes)
ax5a.text(0.03, 0.55,
          f'M0ns spatial: R2={s_r2:.3f}, rho={s_rho:.3f}, p={s_p:.4f}\n'
          f'AS_indiv (pure AS): R2={r_ai**2:.3f}, rho={rho_ai:.3f}, p={p_ai:.4f}\n'
          f'Basque R2=0.89 | Welsh R2=0.98', transform=ax5a.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
ax5a.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=8)
ax5a.set_xlabel('Initial Irish fraction x_0 (2006, %)', fontsize=11)
ax5a.set_ylabel('Fitted prestige s_i', fontsize=11)
ax5a.set_title('s_i vs x_0: Prestige Grounding\n(x = daily use outside education)', fontsize=10, fontweight='bold')
ax5a.grid(True, alpha=0.15)

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'irish_fig5.png'), dpi=150, bbox_inches='tight', facecolor=fig5.get_facecolor())
plt.show()
print("fig5 saved")

#fig6

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(16, 6))
fig6.patch.set_facecolor('#F5F4EF')
fig6.suptitle(f'Pure AS Fit (no diffusion)\n'
              f'AS_shared: a={a_shared:.3f} | AS_indiv: a={a_indiv:.3f} | '
              f'AS_free mean a={np.mean(a_free):.3f} | AS reference: 1.31', fontsize=11, fontweight='bold')

# Left: per-county a_i (AS_free) bar chart
ax6a.set_facecolor('#FAFAF8')
bar_colours = [PROVINCE_COLOURS[COUNTY_PROVINCE[n]] for n in SHAPEFILE_ORDER]
bars6 = ax6a.bar(np.arange(N), a_free, color=bar_colours, alpha=0.85, edgecolor='white', linewidth=0.8)
ax6a.axhline(1.31, color='black', lw=1.5, ls='--', label='AS ref (1.31)', alpha=0.7)
ax6a.axhline(a_indiv, color='#884EA0', lw=1.5, ls=':', label=f'AS_indiv a={a_indiv:.3f}', alpha=0.9)
ax6a.axhline(1.0, color='#E74C3C', lw=1.0, ls=':', alpha=0.5, label='a=1 (boundary)')
ax6a.set_xticks(np.arange(N))
ax6a.set_xticklabels(SHAPEFILE_ORDER, rotation=45, fontsize=7, ha='right')
ax6a.set_ylabel('Fitted a_i', fontsize=10)
ax6a.set_title('Per-County Volatility a_i', fontsize=10, fontweight='bold')
ax6a.legend(fontsize=8)
ax6a.grid(axis='y', alpha=0.25)
ax6a.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()] + [Line2D([0],[0], color='black', lw=1.5, ls='--', label='AS ref 1.31'), Line2D([0],[0], color='#884EA0', lw=1.5, ls=':', label=f'AS_indiv a={a_indiv:.3f}')], fontsize=7)

# Right: s_i from AS_indiv vs x_0
ax6b.set_facecolor('#FAFAF8')
for i, name in enumerate(SHAPEFILE_ORDER):
    c = PROVINCE_COLOURS[COUNTY_PROVINCE[name]]
    ax6b.scatter(X0_VEC[i]*100, s_indiv[i], color=c, s=70, zorder=4, edgecolors='white', linewidths=0.8)
    ax6b.annotate(name[:3], (X0_VEC[i]*100, s_indiv[i]), xytext=(3, 2), textcoords='offset points', fontsize=5, color=c)
s_x2 = np.linspace(X0_VEC.min(), X0_VEC.max(), 100)
s_y2 = intercept_ai + slope_ai * s_x2
ax6b.plot(s_x2*100, s_y2, color='#884EA0', lw=2)
ax6b.text(0.03, 0.97,
          f'Pure AS_indiv\n'
          f'R2={r_ai**2:.3f}, rho={rho_ai:.3f}, p={p_ai:.4f}\n'
          f'Shared a={a_indiv:.4f}\n\n'
          f'Compare - M0ns spatial:\n'
          f'R2={s_r2:.3f}, rho={s_rho:.3f}, p={s_p:.4f}', transform=ax6b.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
ax6b.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=8)
ax6b.set_xlabel('Initial Irish fraction x_0 (2006, %)', fontsize=11)
ax6b.set_ylabel('Fitted prestige s_i (AS_indiv)', fontsize=11)
ax6b.set_title('Pure AS Prestige Grounding\n', fontsize=10, fontweight='bold')
ax6b.grid(True, alpha=0.15)

plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, 'irish_fig6.png'), dpi=150, bbox_inches='tight', facecolor=fig6.get_facecolor())
plt.show()
print("fig6 saved.")

#summary

print("summary")
print(f"N={N} counties | N_obs={N_OBS} | Models: {MODELS_TO_FIT}")
for mname in MODELS_TO_FIT:
    r = results[mname]
    print(f"{mname:<5}: a={r['a']:.4f} sigma={r['sig']:.5f} D={r['D']:.6f} RMSE={r['rmse']:.4f}pp")
print(f"\nAIC: {best['AIC']} AICc: {best['AICc']} BIC: {best['BIC']}")
print(f"Prestige grounding (M0ns spatial): R2={s_r2:.3f} p={s_p:.4f}")
#print(f"Basin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)")
print("\nPure AS results:")
print(f"AS_shared (k=2): a={a_shared:.4f} RMSE={rmse_shared:.4f}pp")
print(f"AS_indiv (k={N+1}): a={a_indiv:.4f} RMSE={rmse_indiv:.4f}pp R2(s~x0)={r_ai**2:.3f}")
print(f"AS_free (k={2*N}): mean a={np.mean(a_free):.4f} median a={np.median(a_free):.4f} RMSE={rmse_free:.4f}pp")
print("\ndone!")
