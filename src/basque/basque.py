# basque comarca model: main fitting script
# M0ns (no shock), M0 (with 1982 statute shock), MR (+ resistance)
# census data 1981-2021
# see basque_age.py for the age-cohort descriptives first

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy.optimize import minimize
from scipy import stats

#config

SHP_PATH = r'COMARCAS_5000_ETRS89.shp'
SHP_NAME_FIELD = 'COMARCA'
OUTPUT_DIR = r"output directory"

MODELS_TO_FIT = ['M0ns', 'M0', 'MR']

CENSUS_YEARS = np.array([1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021])
T0_SOURCE = 1982.0 # year for the shock term

#Internal names used as dict keys throughout
#fixed order so arrays line up with census dicts
SHAPEFILE_ORDER = [
    'Anana', 'Cantabrica_Alavesa', 'Gorbeialdea', 'Llanada_Alavesa',
    'Montana_Alavesa', 'Rioja_Alavesa',
    'Debagoiena', 'Bidasoa_Beherea', 'Debabarrena', 'Donostialdea',
    'Goierri', 'Tolosaldea', 'Urola_Kosta',
    'Arratia_Nerbioi', 'Durangaldea', 'Enkartazioak', 'Gernika_Bermeo',
    'Gran_Bilbao', 'Markina_Ondarroa', 'Plentzia_Mungia'
]

#alias to shapefile name field
ALIAS_TO_FIELD = {
    'Anana':'AÑANA',
    'Cantabrica_Alavesa':'CANTÁBRICA ALAVESA',
    'Gorbeialdea':'ESTRIBACIONES DEL GORBEA',
    'Llanada_Alavesa':'LLANADA ALAVESA',
    'Montana_Alavesa':'MONTAÑA ALAVESA',
    'Rioja_Alavesa':'RIOJA ALAVESA',
    'Debagoiena':'ALTO DEBA',
    'Bidasoa_Beherea':'BAJO BIDASOA',
    'Debabarrena':'BAJO DEBA',
    'Donostialdea':'DONOSTIALDEA',
    'Goierri':'GOIERRI',
    'Tolosaldea':'TOLOSALDEA',
    'Urola_Kosta':'UROLA KOSTA',
    'Arratia_Nerbioi':'ARRATIA-NERVIÓN',
    'Durangaldea':'DURANGUESADO',
    'Enkartazioak':'ENCARTACIONES',
    'Gernika_Bermeo':'GERNIKA-BERMEO',
    'Gran_Bilbao':'GRAN BILBAO',
    'Markina_Ondarroa':'MARKINA-ONDARROA',
    'Plentzia_Mungia':'PLENTZIA-MUNGIA'
}

COMARCA_PROVINCE = {
    'Anana': 'Alava', 'Cantabrica_Alavesa': 'Alava', 'Gorbeialdea': 'Alava',
    'Llanada_Alavesa': 'Alava', 'Montana_Alavesa': 'Alava', 'Rioja_Alavesa': 'Alava',
    'Debagoiena': 'Gipuzkoa', 'Bidasoa_Beherea': 'Gipuzkoa', 'Debabarrena': 'Gipuzkoa',
    'Donostialdea': 'Gipuzkoa', 'Goierri': 'Gipuzkoa', 'Tolosaldea': 'Gipuzkoa',
    'Urola_Kosta': 'Gipuzkoa',
    'Arratia_Nerbioi': 'Bizkaia', 'Durangaldea': 'Bizkaia', 'Enkartazioak': 'Bizkaia',
    'Gernika_Bermeo': 'Bizkaia', 'Gran_Bilbao': 'Bizkaia', 'Markina_Ondarroa': 'Bizkaia',
    'Plentzia_Mungia': 'Bizkaia'
}
PROVINCE_COLOURS = {'Alava': '#C0392B', 'Gipuzkoa': '#16A085', 'Bizkaia': '#2471A3'}
PROV_LIST = ['Alava', 'Gipuzkoa', 'Bizkaia']

X_OBS_DICT = {
    'Anana':[0.009,0.011,0.023,0.040,0.069,0.175,0.166,0.194,0.222],
    'Cantabrica_Alavesa':[0.042,0.086,0.124,0.201,0.216,0.260,0.280,0.331,0.358],
    'Gorbeialdea': [0.272,0.319,0.325,0.340,0.363,0.390,0.408,0.454,0.480],
    'Llanada_Alavesa':[0.033,0.058,0.074,0.134,0.148,0.241,0.222,0.269,0.298],
    'Montana_Alavesa':[0.005,0.022,0.054,0.074,0.116,0.177,0.206,0.245,0.277],
    'Rioja_Alavesa': [0.008,0.031,0.070,0.101,0.128,0.217,0.209,0.220,0.249],
    'Debagoiena': [0.519,0.555,0.596,0.629,0.641,0.621,0.614,0.645,0.640],
    'Bidasoa_Beherea': [0.247,0.280,0.306,0.348,0.373,0.402,0.399,0.447,0.456],
    'Debabarrena': [0.463,0.520,0.543,0.572,0.589,0.573,0.569,0.602,0.604],
    'Donostialdea': [0.254,0.298,0.318,0.365,0.382,0.422,0.430,0.480,0.491],
    'Goierri': [0.455,0.510,0.546,0.582,0.602,0.584,0.581,0.617,0.630],
    'Tolosaldea': [0.614,0.657,0.670,0.709,0.698,0.678,0.686,0.717,0.721],
    'Urola_Kosta': [0.715,0.759,0.768,0.782,0.772,0.745,0.735,0.758,0.762],
    'Arratia_Nerbioi': [0.468,0.500,0.523,0.556,0.548,0.535,0.530,0.568,0.588],
    'Durangaldea':[0.354,0.380,0.403,0.455,0.460,0.472,0.476,0.509,0.515],
    'Enkartazioak':[0.010,0.017,0.042,0.093,0.128,0.197,0.222,0.280,0.312],
    'Gernika_Bermeo':[0.696,0.718,0.744,0.758,0.733,0.708,0.702,0.716,0.712],
    'Gran_Bilbao':[0.063,0.089,0.102,0.152,0.164,0.239,0.235,0.295,0.313],
    'Markina_Ondarroa': [0.795,0.825,0.835,0.858,0.851,0.811,0.790,0.790,0.792],
    'Plentzia_Mungia': [0.453,0.487,0.482,0.485,0.458,0.476,0.479,0.518,0.532]
}
#population needed for diffusion weighting
POP_DICT = {
    'Anana':[4267, 4160, 3886, 4134, 4681, 5515, 6001, 5939, 6267],
    'Cantabrica_Alavesa': [34083, 34731, 34216, 33742, 32720, 33678, 34592, 34496, 34454],
    'Gorbeialdea': [5696, 5519, 5340, 6000, 6892, 8058, 8852, 8821, 9093],
    'Llanada_Alavesa': [200600,209997,216417,225147,228991,244174,256868,260157,265498],
    'Montana_Alavesa': [3645, 3533, 3197, 3140, 3092, 3146, 3231, 2990, 3180],
    'Rioja_Alavesa':[9559, 9788, 9391, 9658, 10011, 11251, 11710, 11486, 11697],
    'Debagoiena': [67136, 66356, 65109, 63562, 62010, 60954, 62095, 62795, 62642],
    'Bidasoa_Beherea': [64714, 67085, 66800, 69668, 71645, 76289, 76191, 76901, 77536],
    'Debabarrena': [66569, 63201, 59936, 56857, 54228, 53838, 54966, 55245, 55170],
    'Donostialdea': [317863,315759,309926,313093,311398,316254,323357,325524,328961],
    'Goierri':[70293, 69030, 66559, 64566, 63510, 64534, 67634, 67478, 68284],
    'Tolosaldea':[46696, 45965, 44996, 44289, 44344, 45467, 47517, 48669, 49543],
    'Urola_Kosta':[61410, 61826, 63162, 64173, 66428, 69329, 73450, 75507, 76751],
    'Arratia_Nerbioi': [22291, 21843, 21275, 21128, 21262, 22341, 23651, 23755, 24136],
    'Durangaldea': [90912, 91812, 90660, 90232, 90495, 94491, 97953, 98497, 99559],
    'Enkartazioak': [31317, 30702, 29680, 29791, 29663, 30500, 32013, 32081, 32110],
    'Gernika_Bermeo': [46234, 46077, 45240, 44402, 44110, 45010, 46172, 45797, 46172],
    'Gran_Bilbao': [936848,926949,905468,887977,865799,867777,872540,857748,857176],
    'Markina_Ondarroa': [28132, 28396, 27301, 26683, 26095, 26105, 26417, 26020, 26048],
    'Plentzia_Mungia': [33544, 33371, 35482, 39813, 45213, 50628, 54605, 56671, 58922]
}

N = len(SHAPEFILE_ORDER)
T_DATA = CENSUS_YEARS.astype(float)
X_DATA = np.array([X_OBS_DICT[n] for n in SHAPEFILE_ORDER])
POP_DATA = np.array([POP_DICT[n]  for n in SHAPEFILE_ORDER], dtype=float)
X0_VEC = X_DATA[:, 0].copy()
X_LAST = X_DATA[:, -1].copy()
N_OBS = N * (len(CENSUS_YEARS) - 1)

def get_pop(t):
    """linear interpolation of census population"""
    return np.array([np.interp(t, T_DATA, POP_DATA[i]) for i in range(N)])

#shapefile check + adjacency graph

def validate_and_load():
    """
    Load shapefile, validate alias mappings, build adjacency matrix.
    """
    print("\nshapefile check")
    gdf = gpd.read_file(SHP_PATH)
    print(f"CRS: {gdf.crs}")
    print(f"Rows: {len(gdf)}")
    print(f"Geometry types: {gdf.geom_type.value_counts().to_dict()}")
    print(f"Name field: '{SHP_NAME_FIELD}'")
    print(f"All values: {sorted(gdf[SHP_NAME_FIELD].tolist())}\n")

    field_to_geom = {}
    ok = True
    for name in SHAPEFILE_ORDER:
        field_val = ALIAS_TO_FIELD[name]
        rows = gdf[gdf[SHP_NAME_FIELD] == field_val]
        if len(rows) == 0:
            print(f"NOT FOUND: '{name}' -> '{field_val}'")
            ok = False
        elif len(rows) > 1:
            print(f"DUPLICATE: '{name}' -> '{field_val}' ({len(rows)} rows)")
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
    print(f"Graph: {N} comarcas, {len(edges)} shared-border edges.\n")
    return A_adj, edges

A_adj, EDGES = validate_and_load()

# ODE pieces

def as_rhs(x, s_vec, a):
    """Abrams-Strogatz RHS"""
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (1 - x) * s_vec * x**a - x * (1 - s_vec) * (1 - x)**a

def shock_term(t, x, sigma, lam):
    """
    Statute of autonomy decay effect
    No effect before T0 or if sigma is zero.
    """
    if sigma == 0.0 or t < T0_SOURCE:
        return np.zeros(N)
    return sigma * np.exp(-lam * (t - T0_SOURCE)) * (1 - x)

def diffusion_term(x, t, D_global, r_vec):
    """
    diffusion across borders, resistance slows it
    """
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


def rk4(x0, t_eval, s_vec, sigma, a, lam, D_global, r_vec, steps_per_year=4):
    """
    RK4 with small steps because census gaps are big
    """
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

# parameter unpacking same ODE for all models, they just zero out unused params

def unpack_M0ns(params):
    s = np.clip(params[:N], 0.01, 0.99)
    a = float(np.clip(params[N], 0.30, 3.00))
    D = float(np.clip(params[N+1], 0.0, 0.50))
    return s, 0.0, a, 0.0, D, np.zeros(N)

def unpack_M0(params):
    s = np.clip(params[:N], 0.01, 0.99)
    sig = float(np.clip(params[N], 0.0,0.30))
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
# brute SSE against observed trajectories
def make_sse(unpack_fn):
    def objective(params):
        s, sig, a, lam, D, r_vec = unpack_fn(params)
        pred = rk4(X0_VEC, T_DATA, s, sig, a, lam, D, r_vec)
        return float(np.sum((pred - X_DATA)**2))
    return objective

#FIT MODELS
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


if 'M0ns' in results and 'M0' in results:
    delta = results['M0']['rmse'] - results['M0ns']['rmse']
    
    print("shock test")
    
    print(f"M0ns: RMSE={results['M0ns']['rmse']:.4f}pp")
    print(f"M0 : RMSE={results['M0']['rmse']:.4f}pp"
          f"delta={delta:+.4f}pp"
          f"sigma={results['M0']['sig']:.5f}"
          f"lam={results['M0']['lam']:.5f}")
    print(f"delta RMSE: {delta:+.4f}pp")

#AIC, BIC, AICc
def aic(sse, k, n): return n * np.log(sse / n) + 2 * k
def aicc(sse, k, n): return aic(sse, k, n) + 2*k*(k+1) / (n - k - 1)
def bic(sse, k, n): return n * np.log(sse / n) + k * np.log(n)

IC = {}
for mname, r in results.items():
    k = K_PARAMS[mname]
    IC[mname] = {
        'AIC': aic (r['sse'], k, N_OBS),
        'AICc': aicc(r['sse'], k, N_OBS),
        'BIC': bic (r['sse'], k, N_OBS)
    }

best = {c: min(IC, key=lambda m: IC[m][c]) for c in ['AIC', 'AICc', 'BIC']}


print("\nAIC/BIC")
print(f"n={N_OBS}  ({N} units x {len(CENSUS_YEARS)-1} transitions)\n")
print(f"{'Model':<5} {'k':>4} {'SSE':>10} {'RMSE(pp)':>9} "
      f"{'AIC':>8} {'AICc':>8} {'BIC':>8} {'deltaAIC':>8} {'deltaBIC':>8}")
for mname in MODELS_TO_FIT:
    r = results[mname]
    ic = IC[mname]
    daic = ic['AIC'] - IC[best['AIC']]['AIC']
    dbic = ic['BIC'] - IC[best['BIC']]['BIC']
    print(f" {mname:<5} {K_PARAMS[mname]:>4} {r['sse']:>10.5f} {r['rmse']:>9.4f} "
          f"{ic['AIC']:>8.2f} {ic['AICc']:>8.2f} {ic['BIC']:>8.2f} "
          f"{daic:>+8.2f} {dbic:>+8.2f}")
print(f"\n AIC: {best['AIC']} | AICc: {best['AICc']} | BIC: {best['BIC']}\n")

#regressions
# check if s_i is basically x0
ref = results['M0ns']
s_slope, s_intercept, s_r, s_p, s_se = stats.linregress(X0_VEC, ref['s'])
s_r2 = s_r**2
s_rho, s_p_sp = stats.spearmanr(X0_VEC, ref['s'])


print("s_i vs x_0 (M0ns)")

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
    return (-s*x**a + (1-x)*s*a*x**(a-1) - (1-s)*(1-x)**a + x*(1-s)*a*(1-x)**(a-1))

def interior_fixed_point(s, a):
    if abs(a - 1.0) < 1e-6:
        return float(s)
    r = ((1 - s) / s) ** (1 / (a - 1))
    return r / (1 + r)

a_M0ns = results['M0ns']['a']
s_M0ns = results['M0ns']['s']


print("stability analysis")

print(f"a (M0ns)={a_M0ns:.4f}")
print(f"AS reference: a ~ 1.31\n")
if a_M0ns > 1.0:
    print("a > 1 standard bistable regime")
else:
    print(f"a={a_M0ns:.4f} x* is stable attractor")

print(f"\n {'Unit':<25} {'x_0':>6} {'s_i':>6} {'x*':>7} "
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
    stability_rows.append(dict(name=name, x0=x0_i, s=s_i, x_int=x_int, lam_int=lam_int, basin=basin, obs_dir=obs_dir, pred_dir=pred_dir, match=match, province=COMARCA_PROVINCE[name]))
    print(f" {name:<25} {x0_i:.3f} {s_i:.4f} {x_int:.4f} {lam_int:+.4f} {basin:>10} {obs_dir:>8} {match}")

n_correct = sum(1 for r in stability_rows if r['match'] == 'OK')
print(f"\nBasin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)\n")

#fig1
model_labels = {
    'M0ns': f'M0ns\nNo shock\n(k={K_PARAMS["M0ns"]})',
    'M0': f'M0\nGlobal D + sigma\n(k={K_PARAMS["M0"]})',
    'MR': f'MR\nResistance + sigma\n(k={K_PARAMS["MR"]})',
}

model_base_colours = {'M0ns': '#7B6FA0', 'M0': '#8A9BAB', 'MR': '#5B8DB8'}

fig1, axes1 = plt.subplots(1, 3, figsize=(13, 5))
fig1.patch.set_facecolor('#F5F4EF')
fig1.suptitle('Model Comparison: AIC, AICc, BIC', fontsize=12, fontweight='bold')

for ax, criterion in zip(axes1, ['AIC', 'AICc', 'BIC']):
    ax.set_facecolor('#FAFAF8')
    vals = [IC[m][criterion] for m in MODELS_TO_FIT]
    best_v = min(vals)
    colours = ['#27AE60' if IC[m][criterion] == best_v else model_base_colours[m] for m in MODELS_TO_FIT]
    labels = [model_labels[m] for m in MODELS_TO_FIT]
    bars = ax.bar(labels, vals, color=colours, alpha=0.85, edgecolor='white', linewidth=0.8)
    # Label each bar at its top (note: values are negative so top is near zero)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + (max(vals) - min(vals)) * 0.01, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best_i = vals.index(best_v)
    bars[best_i].set_edgecolor('black')
    bars[best_i].set_linewidth(2.5)
    ax.set_title(criterion, fontsize=10, fontweight='bold')
    ax.set_ylabel('Information Criterion', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    spread = max(vals) - min(vals)
    ax.set_ylim(min(vals) - spread * 0.08, max(vals) + spread * 0.15)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'basque_fig1.png'), dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
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
        c = PROVINCE_COLOURS[COMARCA_PROVINCE[name]]
        ax2.scatter(X0_VEC[i]*100, mr['r_vec'][i], color=c, s=80, zorder=4, edgecolors='white', linewidths=0.8)
        # Label only the outliers so the plot doesn't get too busy
        if abs(resid_r[i]) > outlier_thr_r or mr['r_vec'][i] > np.percentile(mr['r_vec'], 80):
            ax2.annotate(name.replace('_', ' '), (X0_VEC[i]*100, mr['r_vec'][i]), xytext=(6, 4), textcoords='offset points', fontsize=7.5, color=c, fontweight='bold')

    ax2.plot(x_line*100, y_line, color='#2C3E50', lw=2, zorder=3)
    ax2.fill_between(x_line*100, y_line - t_crit*se_band, y_line + t_crit*se_band, alpha=0.12, color='#2C3E50')
    ax2.axhline(0, color='black', lw=0.8, ls='--', alpha=0.3)
    ax2.text(0.02, 0.97,
             f'OLS: r_i = {intercept_r:+.3f} + {slope_r:+.3f}*x_0\n'
             f'R2={r2_r:.3f},  r={r_r:.3f}, p={p_r:.4f}\n'
             f'Spearman: rho={rho_r:.3f}, p={p_sp_r:.4f}', transform=ax2.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
    ax2.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=9)
    ax2.set_xlabel('Initial Basque fraction x_0 (1981, %)', fontsize=11)
    ax2.set_ylabel('Fitted resistance r_i (MR)', fontsize=11)
    ax2.set_title('Spatial Resistance vs Initial Basque Fraction', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.15)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'basque_fig2.png'), dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.show()
    print("Fig2 saved")

#fig3
    fig3, ax3 = plt.subplots(figsize=(15, 5))
    fig3.patch.set_facecolor('#F5F4EF')
    ax3.set_facecolor('#FAFAF8')
    bar_colours = [PROVINCE_COLOURS[COMARCA_PROVINCE[n]] for n in SHAPEFILE_ORDER]
    bars3 = ax3.bar(np.arange(N), mr['r_vec'], color=bar_colours, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, rv in zip(bars3, mr['r_vec']):
        if rv > 0.001:
            ax3.text(bar.get_x() + bar.get_width()/2, rv + 0.001, f'{rv:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax3.axhline(0, color='black', lw=1.0)
    ax3.set_xticks(np.arange(N))
    ax3.set_xticklabels([n.replace('_', '\n') for n in SHAPEFILE_ORDER], rotation=0, fontsize=6.5, ha='center')
    ax3.set_ylabel('r_i (resistance, >= 0)', fontsize=10)
    ax3.set_title(f'Per-Unit Spatial Resistance r_i (MR, k={K_PARAMS["MR"]})', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.25)
    ax3.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=9)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'basque_fig3.png'), dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
    plt.show()
    print("Fig3 saved")

#fig4 trajectories
T_PROJ = np.linspace(float(CENSUS_YEARS[0]), float(CENSUS_YEARS[-1]) + 50, 500)
m_simple = MODELS_TO_FIT[0]
m_complex = MODELS_TO_FIT[-1]

traj_simple = rk4(X0_VEC, T_PROJ, results[m_simple]['s'], results[m_simple]['sig'], results[m_simple]['a'], results[m_simple]['lam'], results[m_simple]['D'], results[m_simple]['r_vec'])
traj_complex = rk4(X0_VEC, T_PROJ, results[m_complex]['s'], results[m_complex]['sig'], results[m_complex]['a'], results[m_complex]['lam'], results[m_complex]['D'], results[m_complex]['r_vec'])

ncols = 5
nrows = int(np.ceil(N / ncols))
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(ncols*4.4, nrows*4.2))
fig4.patch.set_facecolor('#F5F4EF')
fig4.suptitle(f'{m_simple} vs {m_complex}\n', fontsize=11, fontweight='bold', y=0.99)

for idx, name in enumerate(SHAPEFILE_ORDER):
    row, col = divmod(idx, ncols)
    ax = axes4[row, col] if nrows > 1 else axes4[col]
    ax.set_facecolor('#FAFAF8')
    c = PROVINCE_COLOURS[COMARCA_PROVINCE[name]]
    ax.plot(T_PROJ, traj_simple[idx]*100, color='#AAAAAA', lw=1.4, ls='--', zorder=2)
    ax.plot(T_PROJ, traj_complex[idx]*100, color=c, lw=2.0, zorder=3)
    ax.scatter(CENSUS_YEARS, X_DATA[idx]*100, color='black', s=22, zorder=5)
    r_s = np.sqrt(np.mean((results[m_simple]['pred'][idx] - X_DATA[idx])**2)) * 100
    r_c = np.sqrt(np.mean((results[m_complex]['pred'][idx] - X_DATA[idx])**2)) * 100
    badge_c = '#27AE60' if r_c < 1.5 else ('#F39C12' if r_c < 3.0 else '#E74C3C')
    ax.text(0.98, 0.04,
            f'{m_simple} {r_s:.1f}pp\n{m_complex} {r_c:.1f}pp\ndelta {r_c-r_s:+.1f}pp', transform=ax.transAxes, fontsize=5.5, ha='right', va='bottom', color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_c, alpha=0.88))
    ax.set_title(f"{name.replace('_', ' ')}\n({COMARCA_PROVINCE[name]})", fontsize=7.5, fontweight='bold', color=c, pad=2)
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
    ax = axes4[row, col] if nrows > 1 else axes4[col]
    ax.set_visible(False)

fig4.legend(handles=[
    Line2D([0],[0], color='#AAAAAA', lw=1.5, ls='--', label=m_simple),
    Line2D([0],[0], color='#2471A3', lw=2, label=m_complex),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Observed')], loc='lower center', ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.025, 1, 0.985])
fig4.savefig(os.path.join(OUTPUT_DIR, 'basque_fig4.png'), dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.show()
print("Fig4 saved")

#prestige grounding and basin check

t_crit_s = stats.t.ppf(0.975, df=N-2)
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 7))
fig5.patch.set_facecolor('#F5F4EF')
fig5.suptitle('Prestige Grounding and Stability Analysis', fontsize=12, fontweight='bold')

#left: prestige grounding
ax5a.set_facecolor('#FAFAF8')
s_x = np.linspace(X0_VEC.min(), X0_VEC.max(), 100)
s_y = s_intercept + s_slope * s_x
s_se_b = s_se * np.sqrt(1/N + (s_x - X0_VEC.mean())**2 / np.sum((X0_VEC - X0_VEC.mean())**2))
ax5a.fill_between(s_x*100, s_y - t_crit_s*s_se_b, s_y + t_crit_s*s_se_b, alpha=0.15, color='#884EA0')
ax5a.plot(s_x*100, s_y, color='#884EA0', lw=2, zorder=3)
for i, name in enumerate(SHAPEFILE_ORDER):
    c = PROVINCE_COLOURS[COMARCA_PROVINCE[name]]
    ax5a.scatter(X0_VEC[i]*100, s_M0ns[i], color=c, s=70, zorder=4, edgecolors='white', linewidths=0.8)

ax5a.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()], fontsize=8, loc='lower right')
ax5a.text(0.03, 0.97, f'R2={s_r2:.3f}, r={s_r:.3f}, p={s_p:.4f}\n'
f'Spearman: rho={s_rho:.3f}, p={s_p_sp:.4f}', transform=ax5a.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
ax5a.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()] + [Line2D([0],[0], color='#27AE60', ls=':', lw=1.2, label='Welsh s~0.40'), Line2D([0],[0], color='#E67E22', ls=':', lw=1.2, label='Gaelic s~0.33')],fontsize=8, loc='lower right')
ax5a.set_xlabel('Initial Basque fraction x_0 (1981, %)', fontsize=11)
ax5a.set_ylabel('Fitted prestige s_i (M0ns)', fontsize=11)
ax5a.set_title('s_i vs x_0: Prestige Grounding', fontsize=10, fontweight='bold')

ax5a.set_ylim(0.28, s_M0ns.max() + 0.05)
ax5a.set_xlim(-2, X0_VEC.max()*100 + 5)
ax5a.grid(True, alpha=0.15)

# right: basin classification
ax5b.set_facecolor('#FAFAF8')
for row in stability_rows:
    c = PROVINCE_COLOURS[row['province']]
    marker = '^' if row['pred_dir'] == 'rising' else 'v'
    is_mis = row['match'] == 'FAIL'

    if is_mis:
        ax5b.scatter(row['x0']*100, row['x_int']*100, color='red', marker=marker, s=200, zorder=6, edgecolors='darkred', linewidths=2.5)
        ax5b.annotate( f"{row['name'].replace('_',' ')}\nx₀={row['x0']*100:.0f}%, x*={row['x_int']*100:.0f}%\nobs={row['obs_dir']} pred={row['pred_dir']}", (row['x0']*100, row['x_int']*100), xytext=(-15, -55), textcoords='offset points', fontsize=7.5, color='darkred', fontweight='bold', bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFE0E0', alpha=0.95), arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))
    else:
        ax5b.scatter(row['x0']*100, row['x_int']*100, color=c, marker=marker, s=100, zorder=4, edgecolors='black', linewidths=1.5)
        short = row['name'].replace('_', ' ')[:9]
        ax5b.annotate(short, (row['x0']*100, row['x_int']*100), xytext=(4, 3), textcoords='offset points', fontsize=5, color=c)

diag = np.linspace(0, 100, 100)
ax5b.plot(diag, diag, color='black', lw=1.2, ls='--', alpha=0.4)
ax5b.fill_between(diag, diag, 100, alpha=0.04, color='#27AE60')
ax5b.fill_between(diag, 0, diag, alpha=0.04, color='#E74C3C')

ax5b.set_title(f'Basin Classification: x_0 vs x*\n'
               f'(up) rising, (down) falling, red = mismatch\n', fontsize=9, fontweight='bold')
ax5b.text(0.98, 0.04,
          f'Basin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)\na = {a_M0ns:.3f}', transform=ax5b.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
ax5b.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()] + [Line2D([0],[0], color='black', lw=1.2, ls='--', label='x_0 = x*')], fontsize=8, loc='upper left')
ax5b.set_xlabel('Initial Basque fraction x_0 (%)', fontsize=11)
ax5b.set_ylabel('Interior fixed point x*_i (%)', fontsize=11)
ax5b.set_xlim(-2, 102)
ax5b.set_ylim(-2, 102)
ax5b.grid(True, alpha=0.15)

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'basque_fig5.png'), dpi=150, bbox_inches='tight', facecolor=fig5.get_facecolor())
plt.show()
print("fig5 saved")

#summary
print("summary")
print()
print(f"N={N} units | N_obs={N_OBS} | Models: {MODELS_TO_FIT}")
for mname in MODELS_TO_FIT:
    r = results[mname]
    print(f"{mname:<5}: a={r['a']:.4f} D={r['D']:.6f} "
          f"sigma={r['sig']:.5f} RMSE={r['rmse']:.4f}pp")
print(f"\nAIC: {best['AIC']} AICc: {best['AICc']} BIC: {best['BIC']}")
print(f"Prestige grounding (M0ns): R2={s_r2:.3f} p={s_p:.4f}")
print(f"Basin accuracy: {n_correct}/{N} ({100*n_correct/N:.0f}%)")
print(f"\ndone")