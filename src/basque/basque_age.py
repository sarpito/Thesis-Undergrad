# basque age cohort analysis on EUSTAT ne03 sheet
# nine census waves 1981-2021, comarca breakdown
# separate from main fitting, just descriptives + cohort tracking
# parse() is specific to the file, will break on other sheets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# config

FILE = r"cohort_lang.xlsx"
OUTPUT_DIR = r"output directory"

YEARS = [1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021]

AGE_GROUPS = [
    '2 - 4', '5 - 9', '10 - 14', '15 - 19', '20 - 24',
    '25 - 29', '30 - 34', '35 - 39', '40 - 44', '45 - 49',
    '50 - 54', '55 - 59', '60 - 64', '65 - 69', '70 - 74', '>= 75'
]

AGE_ORDER = {a: i for i, a in enumerate(AGE_GROUPS)}

FLUENT_LEVEL = 'Basque national / speakers'

COMARCAS = [
    'Anana', 'Arabako Errioxa / Rioja Alavesa',
    'Arabako Kantaurialdea / Cantabrica Alavesa',
    'Arabako Lautada / Llanada Alavesa', 'Arabako Mendialdea / Montana Alavesa',
    'Arratia Nerbioi / Arratia-Nervion', 'Bidasoa Beherea / Bajo Bidasoa',
    'Bilbo Handia / Gran Bilbao', 'Debabarrena / Bajo Deba',
    'Debagoiena / Alto Deba', 'Donostialdea', 'Durangaldea / Duranguesado',
    'Enkartazioak / Encartaciones', 'Gernika-Bermeo', 'Goierri',
    'Gorbeialdea / Estribaciones del Gorbea', 'Markina-Ondarroa',
    'Plentzia-Mungia', 'Tolosaldea', 'Urola Kosta'
]

SHORT = {c: c.split('/')[0].strip() for c in COMARCAS}

COMARCA_PROVINCE = {
    'Anana': 'Alava',
    'Arabako Errioxa / Rioja Alavesa': 'Alava',
    'Arabako Kantaurialdea / Cantabrica Alavesa': 'Alava',
    'Arabako Lautada / Llanada Alavesa': 'Alava',
    'Arabako Mendialdea / Montana Alavesa': 'Alava',
    'Gorbeialdea / Estribaciones del Gorbea': 'Alava',
    'Bidasoa Beherea / Bajo Bidasoa': 'Gipuzkoa',
    'Debabarrena / Bajo Deba': 'Gipuzkoa',
    'Debagoiena / Alto Deba': 'Gipuzkoa',
    'Donostialdea': 'Gipuzkoa',
    'Goierri': 'Gipuzkoa',
    'Tolosaldea': 'Gipuzkoa',
    'Urola Kosta': 'Gipuzkoa',
    'Arratia Nerbioi / Arratia-Nervion': 'Bizkaia',
    'Bilbo Handia / Gran Bilbao': 'Bizkaia',
    'Durangaldea / Duranguesado': 'Bizkaia',
    'Enkartazioak / Encartaciones': 'Bizkaia',
    'Gernika-Bermeo': 'Bizkaia',
    'Markina-Ondarroa': 'Bizkaia',
    'Plentzia-Mungia': 'Bizkaia'
}

PROVINCE_COLOURS = {'Alava': '#C0392B', 'Gipuzkoa': '#16A085', 'Bizkaia': '#2471A3'}


def parse(filepath):
    
    raw = pd.read_excel(filepath, sheet_name='ne03', header=None, engine='openpyxl')

    # row 2 has the year values in cols 3-11
    year_cols = {col: int(raw.iloc[2, col]) for col in range(3, 12)}

    rows = []
    current_region = None
    current_level = None

    for _, row in raw.iterrows():
        col0 = row[0] if pd.notna(row[0]) else None
        col1 = row[1] if pd.notna(row[1]) else None
        col2 = row[2] if pd.notna(row[2]) else None

        if col0 is not None:
            current_region = col0.strip()

        # competence level rows always have col2 == 'Total'
        if col1 is not None and col2 == 'Total':
            current_level = col1.strip().lstrip('-').strip()

        if col2 in AGE_GROUPS and current_region and current_level:
            for col, year in year_cols.items():
                val = row[col]
                if pd.notna(val):
                    try:
                        count = int(val)
                    except (ValueError, TypeError):
                        count = 0 # '-' entries treated as zero
                    rows.append({'region': current_region,'level': current_level,'age': col2,'year': year,'count': count})

    return pd.DataFrame(rows)


df = parse(FILE)
print(f'{len(df):,} rows | {df["region"].nunique()} regions | {df["level"].nunique()} competence levels')

def fluency_rate(data, region=None):
    """
    Returns df with cols: age, year, fluency.
    """
    reg = region or 'Basque Country'
    sub = data[data['region'] == reg]

    total = sub[sub['level'] == 'Total'].set_index(['age', 'year'])['count']
    fluent = sub[sub['level'] == FLUENT_LEVEL].set_index(['age', 'year'])['count']

    rate = (fluent / total.replace(0, np.nan)).reset_index()
    rate.columns = ['age', 'year', 'fluency']
    return rate


bc_rate = fluency_rate(df)

def get_rate(age, year):
    """pulls a single fluency value from bc_rate, returns nan if that combo is missing"""
    row = bc_rate[(bc_rate['age'] == age) & (bc_rate['year'] == year)]
    return row['fluency'].values[0] if len(row) else np.nan

#stats
print('\nkey statistics ')
print(f'5-9 year olds: 1981={get_rate("5 - 9", 1981):.1%} 2021={get_rate("5 - 9", 2021):.1%}')
print(f'10-14 year olds: 1981={get_rate("10 - 14",1981):.1%} 2021={get_rate("10 - 14",2021):.1%}')
print(f'30-34 year olds: 1981={get_rate("30 - 34",1981):.1%} 2021={get_rate("30 - 34",2021):.1%}')
print(f'40-44 year olds: 1981={get_rate("40 - 44",1981):.1%} 2021={get_rate("40 - 44",2021):.1%}')

print('\nBC-level fluency by age group:')
print(f'{"age":<12} {"1981":>7} {"2021":>7} {"change":>8}')
for age in AGE_GROUPS[1:]:
    r81 = get_rate(age, 1981)
    r21 = get_rate(age, 2021)
    print(f'{age:<12} {r81*100:>6.1f}% {r21*100:>6.1f}% {(r21-r81)*100:>+7.1f}pp')

print('\npseudo-cohort entry fluency (age 5-9) over time:')
for year in YEARS:
    print(f'{year}: {get_rate("5 - 9", year):.1%}')

# figures

fig1, ax = plt.subplots(figsize=(9, 5))
fig1.patch.set_facecolor('#F5F4EF')
ax.set_facecolor('#FAFAF8')

for year, color, ls, marker in [(1981, '#2471A3', '-', 'o'), (2021, '#C0392B', '--', 's')]:
    sub = bc_rate[bc_rate['year'] == year].copy()
    sub['age_order'] = sub['age'].map(AGE_ORDER)
    sub = sub.sort_values('age_order')
    ax.plot(sub['age_order'], sub['fluency'] * 100, color=color, linestyle=ls, marker=marker, markersize=5, linewidth=2, label=str(year))

ax.set_xticks(range(len(AGE_GROUPS)))
ax.set_xticklabels(AGE_GROUPS, rotation=45, ha='right', fontsize=9)
ax.set_ylabel(r'Fluency rate (% of age group)', fontsize=11)
ax.set_xlabel('Age group', fontsize=11)
ax.set_title('Basque fluency rate by age group\n1981 vs 2021', fontsize=12, fontweight='bold')
ax.legend(title='Census year', fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fig1.savefig(str(Path(OUTPUT_DIR) / 'basque_age_fig1.png'), dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.show()
print('fig1 saved')

#fig2
fig2, ax = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#F5F4EF')
ax.set_facecolor('#FAFAF8')

for age, color, marker in [('5 - 9', '#27AE60', '^'), ('10 - 14', '#E67E22', 'D'), ('15 - 19', '#8E44AD', 'v'), ('20 - 24', '#5D6D7E', 'x')]:
    sub = bc_rate[bc_rate['age'] == age].sort_values('year')
    ax.plot(sub['year'], sub['fluency'] * 100, color=color, marker=marker, linewidth=2, markersize=6, label=age)

ax.axvline(1982, color='gray', linestyle=':', linewidth=1.5, label='Ley Normalizacion 1982')
ax.set_xlabel('Census year', fontsize=11)
ax.set_ylabel('Fluency rate (%)', fontsize=11)
ax.set_title('Basque fluency over time\nYoungest age groups', fontsize=12, fontweight='bold')
ax.legend(title='Age group', fontsize=10)
ax.set_xticks(YEARS)
ax.set_xticklabels(YEARS, rotation=45, ha='right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fig2.savefig(str(Path(OUTPUT_DIR) / 'basque_age_fig2.png'), dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.show()
print('fig2 saved')

#fig3

age_sequence = ['5 - 9', '10 - 14', '15 - 19', '20 - 24', '25 - 29', '30 - 34', '35 - 39', '40 - 44']

cohorts = {}
for entry_year in [1981, 1986, 1991, 1996, 2001]:
    track = []
    for step, age_lbl in enumerate(age_sequence):
        census_year = entry_year + step * 5
        if census_year > 2021:
            break
        row = bc_rate[(bc_rate['year'] == census_year) & (bc_rate['age'] == age_lbl)]
        if len(row):
            track.append((census_year, age_lbl, row['fluency'].values[0]))
    cohorts[entry_year] = track

palette = plt.cm.viridis(np.linspace(0.15, 0.9, 5))

fig3, ax = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor('#F5F4EF')
ax.set_facecolor('#FAFAF8')

for i, (entry_year, track) in enumerate(cohorts.items()):
    cx = [t[0] for t in track]
    fy = [t[2] * 100 for t in track]
    al = [t[1] for t in track]
    ax.plot(cx, fy, color=palette[i], marker='o', linewidth=2, markersize=6, label=f'born ~{entry_year-7}-{entry_year-3}')
    for x, y, label in zip(cx, fy, al):
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(0, 6), fontsize=6.5, ha='center', color=palette[i])

ax.set_xlabel('Census year', fontsize=11)
ax.set_ylabel('Fluency rate (%)', fontsize=11)
ax.set_title('Basque Country: pseudo-cohort tracking\neach line = birth cohort, aged 5-9 at entry year', fontsize=12, fontweight='bold')
ax.legend(title='Birth cohort', fontsize=9, loc='upper left')
ax.set_xticks(YEARS)
ax.set_xticklabels(YEARS, rotation=45, ha='right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fig3.savefig(str(Path(OUTPUT_DIR) / 'basque_age_fig3.png'), dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.show()
print('fig3 saved')

#fig4
young_ages = ['5 - 9', '10 - 14', '15 - 19']

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig4.patch.set_facecolor('#F5F4EF')

for ax, year in zip(axes4, [1981, 2021]):
    ax.set_facecolor('#FAFAF8')
    mat = []
    for comarca in COMARCAS:
        cr = fluency_rate(df, comarca)
        if len(cr) == 0:
            cr = fluency_rate(df)
        row_vals = []
        for age in young_ages:
            r = cr[(cr['age'] == age) & (cr['year'] == year)]
            row_vals.append(r['fluency'].values[0] * 100 if len(r) else np.nan)
        mat.append(row_vals)

    mat = np.array(mat)
    im = ax.imshow(mat, aspect='auto', vmin=0, vmax=100, cmap='YlGn')
    ax.set_xticks(range(len(young_ages)))
    ax.set_xticklabels(young_ages, fontsize=10)
    ax.set_yticks(range(len(COMARCAS)))
    ax.set_yticklabels([SHORT[c] for c in COMARCAS], fontsize=8)
    ax.set_title(str(year), fontsize=13, fontweight='bold')
    for i in range(len(COMARCAS)):
        for j in range(len(young_ages)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color='black' if val < 70 else 'white')

plt.colorbar(im, ax=axes4[1], label='Fluency rate (%)')
fig4.suptitle('Fluency rate in young age groups by comarca\n1981 vs 2021', fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout()
fig4.savefig(str(Path(OUTPUT_DIR) / 'basque_age_fig4.png'), dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.show()
print('fig4 saved')

#fig5
fig5,ax = plt.subplots(figsize=(8, 7))
fig5.patch.set_facecolor('#F5F4EF')
ax.set_facecolor('#FAFAF8')

for comarca in COMARCAS:
    cr = fluency_rate(df, comarca)
    r81 = cr[(cr['age'] == '5 - 9') & (cr['year'] == 1981)]
    r21 = cr[(cr['age'] == '5 - 9') & (cr['year'] == 2021)]
    if not len(r81) or not len(r21):
        continue
    x = r81['fluency'].values[0] * 100
    y = r21['fluency'].values[0] * 100
    prov = COMARCA_PROVINCE.get(comarca, 'Bizkaia')
    c = PROVINCE_COLOURS[prov]
    ax.scatter(x, y, color=c, s=80, zorder=3, edgecolors='white', linewidth=0.5)
    ax.annotate(SHORT[comarca], (x, y), textcoords='offset points', xytext=(4, 3), fontsize=7.5)

ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8, alpha=0.5, label='no change')
ax.set_xlabel('Fluency rate 1981, age 5-9 (%)', fontsize=11)
ax.set_ylabel('Fluency rate 2021, age 5-9 (%)', fontsize=11)
ax.set_title('Change in fluency for 5-9 year olds by comarca\n1981 vs 2021', fontsize=12, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color=v, label=k)for k, v in PROVINCE_COLOURS.items()] + [plt.Line2D([0],[0], color='black', lw=1, ls='--', label='no change')], fontsize=9)
ax.set_xlim(-2, 102)
ax.set_ylim(-2, 102)
ax.grid(alpha=0.25)

plt.tight_layout()
fig5.savefig(str(Path(OUTPUT_DIR) / 'basque_age_fig5.png'), dpi=150, bbox_inches='tight', facecolor=fig5.get_facecolor())
plt.show()
print('fig5 saved')

print("done")
