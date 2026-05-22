# synthetic validation: checks code recovers known parameters
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

# True params
TRUE_A = 1.31 # from A-S 2003
TRUE_D = 0.05 # some spatial diffusion
NOISE_SD = 0.015 # Gaussian noise
RANDOM_SEED = 78

#data
CENSUS_YEARS = np.array([1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021])
T0_SOURCE = 1982.0

SHAPEFILE_ORDER = [
    'Anana', 'Cantabrica_Alavesa', 'Gorbeialdea', 'Llanada_Alavesa',
    'Montana_Alavesa', 'Rioja_Alavesa',
    'Debagoiena', 'Bidasoa_Beherea', 'Debabarrena', 'Donostialdea',
    'Goierri', 'Tolosaldea', 'Urola_Kosta',
    'Arratia_Nerbioi', 'Durangaldea', 'Enkartazioak', 'Gernika_Bermeo',
    'Gran_Bilbao', 'Markina_Ondarroa', 'Plentzia_Mungia'
]

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
POP_DATA = np.array([POP_DICT[n] for n in SHAPEFILE_ORDER], dtype=float)
X0_VEC = X_DATA[:, 0].copy() # 1981 initial conditions

#adjacency
def build_adjacency():
    gdf = gpd.read_file(SHP_PATH)
    field_to_geom = {}
    for name in SHAPEFILE_ORDER:
        rows = gdf[gdf[SHP_NAME_FIELD] == ALIAS_TO_FIELD[name]]
        field_to_geom[name] = rows.iloc[0].geometry
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            inter = field_to_geom[SHAPEFILE_ORDER[i]].intersection(
                    field_to_geom[SHAPEFILE_ORDER[j]])
            if not inter.is_empty and inter.length > 0:
                A[i, j] = A[j, i] = 1.0
    edges = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 1.0]
    print(f"{len(edges)} shared-border edges.\n")
    return A, edges

A_adj, EDGES = build_adjacency()

#ode
def get_pop(t):
    return np.array([np.interp(t, T_DATA, POP_DATA[i]) for i in range(N)])

def as_rhs(x, s_vec, a):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (1 - x) * s_vec * x**a - x * (1 - s_vec) * (1 - x)**a

def diffusion_term(x, t, D_global, r_vec):
    n = get_pop(t)
    diff = np.zeros(N)
    for (i, j) in EDGES:
        Dij = D_global / (1.0 + r_vec[i] + r_vec[j])
        flow = Dij * (n[j] * x[j] - n[i] * x[i])
        diff[i] += flow / n[i]
        diff[j] -= flow / n[j]
    return diff

def ode_rhs(t, x, s_vec, a, D_global, r_vec):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return as_rhs(x, s_vec, a) + diffusion_term(x, t, D_global, r_vec)

def rk4(x0, t_eval, s_vec, a, D_global, r_vec, steps_per_year=4):
    x = x0.copy()
    out = np.zeros((N, len(t_eval)))
    out[:, 0] = x
    tc = t_eval[0]
    for k in range(1, len(t_eval)):
        tn = t_eval[k]
        ns = max(1, int(round((tn - tc) * steps_per_year)))
        h = (tn - tc) / ns
        for _ in range(ns):
            k1 = ode_rhs(tc, x,s_vec, a, D_global, r_vec)
            k2 = ode_rhs(tc + h/2, x + h/2*k1, s_vec, a, D_global, r_vec)
            k3 = ode_rhs(tc + h/2, x + h/2*k2, s_vec, a, D_global, r_vec)
            k4 = ode_rhs(tc + h,x + h*k3, s_vec, a, D_global, r_vec)
            x = np.clip(x + h/6*(k1 + 2*k2 + 2*k3 + k4), 0, 1)
            tc += h
        out[:, k] = x
    return out

#generate data

rng = np.random.default_rng(RANDOM_SEED)

TRUE_S = np.clip(0.40 + 0.30 * X0_VEC + rng.normal(0, 0.03, N), 0.01, 0.99)
TRUE_R = np.zeros(N)

print("synthetic data generation")
print(f"True a = {TRUE_A}")
print(f"True D = {TRUE_D}")
print(f"True s_i ~ 0.40 + 0.30*x_0 + N(0, 0.03)")
print(f"Noise SD = {NOISE_SD*100:.1f} pp")
print(f"Seed = {RANDOM_SEED}\n")

X_SYNTH_CLEAN = rk4(X0_VEC, T_DATA, TRUE_S, TRUE_A, TRUE_D, TRUE_R)

# Add noise and clip to [0,1]
noise = rng.normal(0, NOISE_SD, X_SYNTH_CLEAN.shape)
noise[:, 0] = 0.0 # no noise on initial condition
X_SYNTH = np.clip(X_SYNTH_CLEAN + noise, 0.01, 0.99)

print("Sample synthetic trajectories (first 5 comarcas):")
print(f"{'Comarca':<25} {'x_0(real)':>10} {'x_2021(clean)':>14} {'x_2021(noisy)':>14}")
for i in range(5):
    print(f"{SHAPEFILE_ORDER[i]:<25} {X0_VEC[i]:>10.3f} "
          f"{X_SYNTH_CLEAN[i,-1]:>14.3f} {X_SYNTH[i,-1]:>14.3f}")
print()

#fit to synthetic data
N_OBS_SYNTH = N * (len(CENSUS_YEARS) - 1)

def unpack_M0ns(params):
    s = np.clip(params[:N],0.01, 0.99)
    a = float(np.clip(params[N],0.30, 3.00))
    D = float(np.clip(params[N+1], 0.0, 0.50))
    return s, a, D, np.zeros(N)

BOUNDS_M0ns = [(0.01, 0.99)]*N + [(0.30, 3.0), (0.0, 0.50)]
X0_M0ns = list(np.full(N, 0.5)) + [1.0, 0.01]
OPT = {'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-9}

def make_sse_synth(unpack_fn, X_target):
    """SSE objective against X_target"""
    def objective(params):
        s, a, D, r = unpack_fn(params)
        pred = rk4(X0_VEC, T_DATA, s, a, D, r)
        return float(np.sum((pred - X_target)**2))
    return objective

print("fitting to synthetic data")

res_synth = minimize(make_sse_synth(unpack_M0ns, X_SYNTH), X0_M0ns, method='L-BFGS-B', bounds=BOUNDS_M0ns, options=OPT)
s_syn, a_syn, D_syn, r_syn = unpack_M0ns(res_synth.x)
pred_syn = rk4(X0_VEC, T_DATA, s_syn, a_syn, D_syn, r_syn)
rmse_syn = np.mean(np.sqrt(np.mean((pred_syn - X_SYNTH)**2, axis=1))) * 100

print(f"Fitted a = {a_syn:.4f} (true: {TRUE_A})")
print(f"Fitted D = {D_syn:.6f} (true: {TRUE_D})")
print(f"RMSE = {rmse_syn:.4f} pp")
print(f"Recovery error: deltaa = {a_syn - TRUE_A:+.4f}, deltaD = {D_syn - TRUE_D:+.6f}\n")

#fit to real data
print("fitting to real census")

res_real = minimize(make_sse_synth(unpack_M0ns, X_DATA), X0_M0ns, method='L-BFGS-B', bounds=BOUNDS_M0ns, options=OPT)
s_real, a_real, D_real, r_real = unpack_M0ns(res_real.x)
pred_real = rk4(X0_VEC, T_DATA, s_real, a_real, D_real, r_real)
rmse_real = np.mean(np.sqrt(np.mean((pred_real - X_DATA)**2, axis=1)))* 100

print(f"Fitted a = {a_real:.4f} (expected: ~0.91)")
print(f"Fitted D = {D_real:.6f} (expected: ~0.0)")
print(f"RMSE = {rmse_real:.4f} pp\n")


print("summary")
print(f"{'':30} {'a':>8} {'D':>10} {'RMSE (pp)':>10}")
print(f"{'Ground truth (synthetic)':30} {TRUE_A:>8.4f} {TRUE_D:>10.6f} {'-':>10}")
print(f"{'Recovered (synthetic data)':30} {a_syn:>8.4f} {D_syn:>10.6f} {rmse_syn:>10.4f}")
print(f"{'Fitted (real census data)':30} {a_real:>8.4f} {D_real:>10.6f} {rmse_real:>10.4f}")
print()

da = abs(a_syn - TRUE_A)
dD = abs(D_syn - TRUE_D)
a_ok = da < 0.10
D_ok = dD < 0.015

#s_i recovery check

rho_s, p_s = stats.spearmanr(TRUE_S, s_syn)
slope, intercept, r_val, p_ols, _ = stats.linregress(TRUE_S, s_syn)
r2_s = r_val**2

print("prestige recovery (s_i: true vs fitted)")
print(f"Spearman rho= {rho_s:.4f} p = {p_s:.6f}")
print(f"OLS R2 = {r2_s:.4f} slope = {slope:.4f}")
print(f"Mean |deltas_i| = {np.mean(np.abs(TRUE_S - s_syn)):.4f}\n")

#fig1

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor('#F5F4EF')
fig.suptitle(
    'Synthetic Validation: Pipeline Identification Test\n'
    f'True params: a={TRUE_A}, D={TRUE_D} | '
    f'Recovered: a={a_syn:.3f}, D={D_syn:.4f}',
    fontsize=12, fontweight='bold'
)

# True vs recovered a and D 
ax_a = axes[0, 0]
ax_a.set_facecolor('#FAFAF8')
labels_ab = ['True a', 'Recovered a\n(synthetic)', 'Fitted a\n(real census)']
values_ab = [TRUE_A, a_syn, a_real]
colours_ab = ['#2ECC71', '#3498DB', '#E74C3C']
bars = ax_a.bar(labels_ab, values_ab, color=colours_ab, alpha=0.85, edgecolor='white', linewidth=0.8)
for bar, v in zip(bars, values_ab):
    ax_a.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_a.axhline(1.0, color='black', lw=1.2, ls='--', alpha=0.5, label='a=1 (boundary)')
ax_a.axhline(TRUE_A, color='#2ECC71', lw=1.5, ls=':', alpha=0.7)
ax_a.set_ylim(0.5, 1.6)
ax_a.set_ylabel('Fitted a', fontsize=11)
ax_a.set_title('Volatility Exponent a', fontsize=10, fontweight='bold')
ax_a.legend(fontsize=8)
ax_a.grid(axis='y', alpha=0.2)

# True vs recovered D
ax_b = axes[0, 1]
ax_b.set_facecolor('#FAFAF8')
labels_d = ['True D', 'Recovered D\n(synthetic)', 'Fitted D\n(real census)']
values_d = [TRUE_D, D_syn, D_real]
colours_d = ['#2ECC71', '#3498DB', '#E74C3C']
bars2 = ax_b.bar(labels_d, values_d, color=colours_d, alpha=0.85, edgecolor='white', linewidth=0.8)
for bar, v in zip(bars2, values_d):
    ax_b.text(bar.get_x() + bar.get_width()/2, v + 0.0003, f'{v:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_b.axhline(TRUE_D, color='#2ECC71', lw=1.5, ls=':', alpha=0.7)
ax_b.set_ylim(0, TRUE_D * 1.6)
ax_b.set_ylabel('Fitted D', fontsize=11)
ax_b.set_title('Spatial Diffusion D', fontsize=10, fontweight='bold')
ax_b.grid(axis='y', alpha=0.2)

# True vs recovered s_i
ax_c = axes[1, 0]
ax_c.set_facecolor('#FAFAF8')
x_line = np.linspace(TRUE_S.min(), TRUE_S.max(), 100)
ax_c.plot(x_line, x_line, 'k--', alpha=0.4, lw=1.2, label='Perfect recovery')
ax_c.plot(x_line, intercept + slope * x_line, color='#884EA0', lw=1.8, label=f'OLS (R2={r2_s:.3f})')
for i, name in enumerate(SHAPEFILE_ORDER):
    c = PROVINCE_COLOURS[COMARCA_PROVINCE[name]]
    ax_c.scatter(TRUE_S[i], s_syn[i], color=c, s=60, zorder=4, edgecolors='white', linewidths=0.7)
ax_c.text(0.04, 0.96,
          f'Spearman rho= {rho_s:.3f}\np = {p_s:.4f}\nMean |deltas| = {np.mean(np.abs(TRUE_S-s_syn)):.4f}',transform=ax_c.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
ax_c.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROVINCE_COLOURS.items()] + [Line2D([0],[0], color='k', ls='--', label='Perfect'), Line2D([0],[0], color='#884EA0', lw=2, label='OLS fit')], fontsize=7, loc='lower right')
ax_c.set_xlabel('True s_i', fontsize=11)
ax_c.set_ylabel('Recovered s_i', fontsize=11)
ax_c.set_title('Prestige Recovery: True vs Fitted s_i', fontsize=10, fontweight='bold')
ax_c.grid(True, alpha=0.15)

# example trajectories
ax_d = axes[1, 1]
ax_d.set_facecolor('#FAFAF8')
showcase_names = ['Gran_Bilbao', 'Donostialdea', 'Markina_Ondarroa']
showcase_styles = ['-', '--', ':']
T_PROJ = np.linspace(float(CENSUS_YEARS[0]), float(CENSUS_YEARS[-1]) + 30, 400)

traj_syn_proj = rk4(X0_VEC, T_PROJ, s_syn, a_syn, D_syn, r_syn)
traj_real_proj = rk4(X0_VEC, T_PROJ, s_real, a_real, D_real, r_real)
traj_true_proj = rk4(X0_VEC, T_PROJ, TRUE_S, TRUE_A, TRUE_D, TRUE_R)

for name, ls in zip(showcase_names, showcase_styles):
    idx = SHAPEFILE_ORDER.index(name)
    c = PROVINCE_COLOURS[COMARCA_PROVINCE[name]]
    label_base = name.replace('_', ' ')
    ax_d.plot(T_PROJ, traj_true_proj[idx]*100, color=c, lw=2.5, ls=ls, label=f'{label_base} (true)')
    ax_d.plot(T_PROJ, traj_syn_proj[idx]*100, color=c, lw=1.2, ls=ls, alpha=0.5)
    ax_d.scatter(CENSUS_YEARS, X_SYNTH[idx]*100, color=c, s=18, zorder=5, marker='o')

ax_d.axvline(CENSUS_YEARS[-1], color='gray', lw=0.8, ls=':', alpha=0.6)
ax_d.text(0.02, 0.97, 'Solid/dark = true trajectory\nFaded = recovered fit', transform=ax_d.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_d.set_xlabel('Year', fontsize=11)
ax_d.set_ylabel('Basque fraction (%)', fontsize=11)
ax_d.set_title('Example Trajectories: True vs Recovered', fontsize=10, fontweight='bold')
ax_d.legend(fontsize=7, loc='lower right')
ax_d.set_xlim(CENSUS_YEARS[0]-2, T_PROJ[-1]+2)
ax_d.set_ylim(-2, 102)
ax_d.grid(True, alpha=0.15)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'synthetic_validation.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("\ndone")