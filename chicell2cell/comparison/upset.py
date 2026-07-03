"""Petal Venn + UpSet plot — exact match to working notebook."""

import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
from matplotlib.patches import Circle

CENTER_COLOR = '#1B3A6B'
PETAL_COLORS = {
    'CellPhoneDB'      : '#C0392B',
    'ConnectomeDB2020' : '#D48B0A',
    'CellTalkDB'       : '#1A7A4A',
    'ICELLNET'         : '#7B2FA0',
    'iTALK'            : '#C75000',
    'CellCall'         : '#0E7490',
    'Cellinker'        : '#8B6914',
}


def _clean(df):
    df = df[['source', 'target', 'ligand', 'receptor']].dropna().copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df['lr_key'] = df['ligand'] + '|' + df['receptor']
    return df


def plot_venn_upset(
    chicelldb,
    cellphonedb_results,
    connectomedb2020_results,
    celltalkdb_results,
    icellnet_results,
    italk_results,
    cellcall_results,
    cellinker_results,
    save_path=None,
):
    """
    Panel A — Petal Venn diagram centred on Chicell2cell.
    Panel B — UpSet plot of LR pair overlaps.

    Parameters
    ----------
    All DataFrames must have columns: source, target, ligand, receptor
    save_path : str or None — PDF save path
    """
    try:
        from upsetplot import UpSet, from_memberships
    except ImportError:
        raise ImportError("Install upsetplot: pip install upsetplot")

    raw_dbs = {
        'Chicell2cell'     : chicelldb,
        'CellPhoneDB'      : cellphonedb_results,
        'ConnectomeDB2020' : connectomedb2020_results,
        'CellTalkDB'       : celltalkdb_results,
        'ICELLNET'         : icellnet_results,
        'iTALK'            : italk_results,
        'CellCall'         : cellcall_results,
        'Cellinker'        : cellinker_results,
    }

    cleaned  = {name: _clean(df) for name, df in raw_dbs.items()}
    lr_sets  = {name: set(df['lr_key']) for name, df in cleaned.items()}

    center_set      = lr_sets['Chicell2cell']
    others          = [n for n in lr_sets if n != 'Chicell2cell']
    chic_unique_all = len(center_set - set.union(*[lr_sets[n] for n in others]))

    petal_stats = {
        name: {
            'total' : len(lr_sets[name]),
            'shared': len(center_set & lr_sets[name]),
            'unique': len(lr_sets[name] - center_set),
        }
        for name in others
    }

    # UpSet
    all_lr      = sorted(set.union(*lr_sets.values()))
    memberships = [[name for name, s in lr_sets.items() if lr in s] for lr in all_lr]
    from upsetplot import from_memberships
    upset_series = from_memberships(memberships)
    upset_series = upset_series.sort_values(ascending=False).head(35)

    fig_upset = plt.figure(figsize=(18, 10), facecolor='white')
    from upsetplot import UpSet
    UpSet(upset_series, subset_size='count', show_counts=True,
          sort_by='cardinality', sort_categories_by=None).plot(fig=fig_upset)
    buf = io.BytesIO()
    fig_upset.savefig(buf, format='png', dpi=200,
                      bbox_inches='tight', facecolor='white')
    buf.seek(0)
    upset_img = mpimg.imread(buf)
    plt.close(fig_upset)

    # Combined figure
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(16, 28), facecolor='white',
        gridspec_kw={'height_ratios': [1.3, 0.9], 'hspace': 0.06})

    # Panel A — Petal Venn
    center_r = 2.60
    petal_r  = 1.35
    dist     = 3.40
    shared_d = (center_r + (dist - petal_r)) / 2
    unique_d = dist + petal_r * 0.48
    label_d  = dist + petal_r + 0.72
    angles   = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi,
                           len(others), endpoint=False)

    ax_a.set_aspect('equal')
    ax_a.axis('off')
    ax_a.set_xlim(-7.5, 7.5)
    ax_a.set_ylim(-7.5, 7.0)

    for name, angle in zip(others, angles):
        ax_a.add_patch(Circle(
            (dist * np.cos(angle), dist * np.sin(angle)), petal_r,
            facecolor=PETAL_COLORS[name], alpha=0.42,
            edgecolor=PETAL_COLORS[name], linewidth=2.0, zorder=2))

    ax_a.add_patch(Circle((0, 0), center_r, facecolor=CENTER_COLOR,
                           alpha=0.88, edgecolor='#0d2240', linewidth=2.5, zorder=4))
    ax_a.text(0,  0.85, 'Chicell2cell', ha='center', va='center',
              fontsize=14, fontweight='bold', color='white', zorder=10)
    ax_a.text(0, -0.05, f'{len(center_set):,}', ha='center', va='center',
              fontsize=24, fontweight='bold', color='white', zorder=10)
    ax_a.text(0, -1.00, 'LR pairs', ha='center', va='center',
              fontsize=11, color='#cce4ff', zorder=10)
    ax_a.text(0, -1.95,
              f'({chic_unique_all:,} unique\nto Chicell2cell only)',
              ha='center', va='center', fontsize=9, color='#99c4ff',
              style='italic', zorder=10, multialignment='center')

    for name, angle in zip(others, angles):
        d   = petal_stats[name]
        col = PETAL_COLORS[name]
        ax_a.text(shared_d * np.cos(angle), shared_d * np.sin(angle),
                  f'{d["shared"]:,}', ha='center', va='center',
                  fontsize=12, fontweight='bold', color='white', zorder=11,
                  path_effects=[pe.withStroke(linewidth=2.5, foreground=col)])
        ax_a.text(unique_d * np.cos(angle), unique_d * np.sin(angle),
                  f'{d["unique"]:,}', ha='center', va='center',
                  fontsize=11, fontweight='bold', color='#111111', zorder=11,
                  path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
        short = name.replace('ConnectomeDB2020', 'ConnectomeDB\n2020')
        ax_a.text(label_d * np.cos(angle), label_d * np.sin(angle),
                  f'{short}\n({d["total"]:,} LR pairs)',
                  ha='center', va='center', fontsize=10, fontweight='bold',
                  color=col, zorder=12, multialignment='center', linespacing=1.5,
                  path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    legend_handles = [
        mpatches.Patch(facecolor=CENTER_COLOR, edgecolor='#0d2240', alpha=0.88,
                       label=f'Chicell2cell  ({len(center_set):,} LR pairs)')
    ]
    for name in others:
        legend_handles.append(
            mpatches.Patch(facecolor=PETAL_COLORS[name],
                           edgecolor=PETAL_COLORS[name], alpha=0.55,
                           label=f'{name}  ({petal_stats[name]["total"]:,} LR pairs)'))
    leg = ax_a.legend(handles=legend_handles, fontsize=9, frameon=True,
                      framealpha=0.95, edgecolor='#cccccc',
                      loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02),
                      borderpad=1.0, handlelength=1.5, labelspacing=0.55,
                      title='Database  (total unique LR pairs)', title_fontsize=9.5)
    leg.get_title().set_fontweight('bold')
    ax_a.text(-0.02, 1.01, 'A', transform=ax_a.transAxes,
              fontsize=22, fontweight='bold', va='top', ha='left')

    # Panel B — UpSet
    ax_b.imshow(upset_img, aspect='auto', interpolation='lanczos')
    ax_b.axis('off')
    ax_b.text(-0.01, 1.03, 'B', transform=ax_b.transAxes,
              fontsize=22, fontweight='bold', va='top', ha='left')

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight',
                    facecolor='white', dpi=300)
        print(f"Saved to {save_path}")

    plt.show()
    print('✓ Venn + UpSet plot complete')
