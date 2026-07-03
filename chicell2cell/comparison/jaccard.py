"""Jaccard similarity comparison of LR databases against Chicell2cell."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def _make_key_set(df, include_cells=False):
    """Build a set of unique keys from a database DataFrame."""
    df = df[['source', 'target', 'ligand', 'receptor']].dropna().copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    if include_cells:
        df['key'] = (df['source'] + '|' + df['target'] + '|'
                     + df['ligand'] + '|' + df['receptor'])
    else:
        df['key'] = df['ligand'] + '|' + df['receptor']
    return set(df['key'])


def analyze_ccc_jaccard_vs_chicell(
    chicelldb, cellphonedb, connectomedb2020,
    celltalkdb, icellnet, italk, cellcall, cellinker,
    include_cells=False,
    max_items_for_heatmap=30,
    save_path=None,
):
    """
    Compute pairwise Jaccard similarity between Chicell2cell and seven
    established CCC databases. Produces a six-panel comparison figure.

    Parameters
    ----------
    include_cells : bool
        If True, keys include source and target cell types.
        If False (default), only ligand-receptor pairs are compared.
    """
    print("=" * 80)
    print("JACCARD SIMILARITY — CHICELL2CELL vs OTHER DATABASES")
    print("=" * 80)

    databases = {
        'Chicell2cell'     : _make_key_set(chicelldb,        include_cells),
        'CellPhoneDB'      : _make_key_set(cellphonedb,      include_cells),
        'ConnectomeDB2020' : _make_key_set(connectomedb2020, include_cells),
        'CellTalkDB'       : _make_key_set(celltalkdb,       include_cells),
        'ICELLNET'         : _make_key_set(icellnet,         include_cells),
        'iTALK'            : _make_key_set(italk,            include_cells),
        'CellCall'         : _make_key_set(cellcall,         include_cells),
        'Cellinker'        : _make_key_set(cellinker,        include_cells),
    }
    db_names = list(databases.keys())
    n        = len(db_names)

    print("\nItems per method:")
    for db in db_names:
        print(f"  {db}: {len(databases[db])}")

    # Pairwise Jaccard matrix
    jac = np.zeros((n, n))
    for i, d1 in enumerate(db_names):
        for j, d2 in enumerate(db_names):
            inter  = len(databases[d1] & databases[d2])
            union  = len(databases[d1] | databases[d2])
            jac[i, j] = 1.0 if union == 0 else inter / union

    # Raw overlap matrix
    ov = np.zeros((n, n))
    for i, d1 in enumerate(db_names):
        for j, d2 in enumerate(db_names):
            ov[i, j] = (len(databases[d1]) if i == j
                        else len(databases[d1] & databases[d2]))

    ref_name    = 'Chicell2cell'
    ref_idx     = db_names.index(ref_name)
    other_names = [d for d in db_names if d != ref_name]
    jac_vs_ref  = {d: jac[ref_idx, db_names.index(d)] for d in other_names}

    palette = sns.color_palette("Set2", n_colors=n)

    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            height_ratios=[1.0, 1.4, 1.0, 1.6],
                            hspace=0.9)

    # A — raw overlap heatmap
    axA = fig.add_subplot(gs[0, 0])
    sns.heatmap(ov, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=db_names, yticklabels=db_names,
                ax=axA, annot_kws={'size': 10},
                cbar_kws={'label': 'Gene Overlap'})
    plt.setp(axA.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(axA.get_yticklabels(), rotation=0,  fontsize=10)
    axA.text(-0.12, 1.08, 'A', transform=axA.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # B — Jaccard vs Chicell2cell line plot
    axB = fig.add_subplot(gs[0, 1])
    x   = np.arange(len(other_names))
    axB.plot(x, [jac_vs_ref[d] for d in other_names],
             marker='o', color='teal', linewidth=1.5, markersize=7)
    axB.set_xticks(x)
    axB.set_xticklabels(other_names, rotation=45, ha='right', fontsize=10)
    axB.set_ylabel(f'Jaccard vs {ref_name}', fontsize=12)
    axB.grid(axis='y', linestyle='--', alpha=0.3)
    axB.spines[['top', 'right']].set_visible(False)
    axB.text(-0.07, 1.08, 'B', transform=axB.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # C — full Jaccard heatmap
    axC = fig.add_subplot(gs[1, 0])
    sns.heatmap(jac, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=db_names, yticklabels=db_names,
                vmin=0, vmax=1, ax=axC, annot_kws={'size': 10},
                cbar_kws={'label': 'Jaccard Index'})
    plt.setp(axC.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(axC.get_yticklabels(), rotation=0,  fontsize=10)
    axC.text(-0.12, 1.08, 'C', transform=axC.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # D — total LR pairs per database
    axD = fig.add_subplot(gs[1, 1])
    total = [len(databases[d]) for d in reversed(db_names)]
    bars  = axD.barh(list(reversed(db_names)), total,
                     color=palette[::-1], edgecolor='black', height=0.6)
    axD.set_xlabel('Number of LR pairs', fontsize=12)
    axD.spines[['top', 'right']].set_visible(False)
    x_max = max(total)
    for bar, val in zip(bars, total):
        axD.text(val + 0.02 * x_max,
                 bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9, fontweight='bold')
    axD.text(-0.07, 1.08, 'D', transform=axD.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # E — ranked bar chart of Jaccard vs Chicell2cell
    axE = fig.add_subplot(gs[2, :])
    ordered = sorted(other_names, key=lambda d: jac_vs_ref[d], reverse=True)
    bars_e  = axE.bar(ordered, [jac_vs_ref[d] for d in ordered],
                      color=palette[:len(ordered)], edgecolor='black', alpha=0.85)
    axE.set_xlabel('Method', fontsize=12)
    axE.set_ylabel('Jaccard Index vs Chicell2cell', fontsize=12)
    axE.tick_params(axis='x', rotation=45)
    axE.grid(axis='y', linestyle='--', alpha=0.3)
    axE.spines[['top', 'right']].set_visible(False)
    for bar in bars_e:
        axE.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    axE.text(-0.07, 1.08, 'E', transform=axE.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # F — presence/absence heatmap for top LR pairs
    axF = fig.add_subplot(gs[3, :])
    all_keys   = sorted(set().union(*databases.values()))
    key_counts = {k: sum(k in databases[d] for d in db_names) for k in all_keys}
    top_keys   = sorted(key_counts, key=key_counts.get,
                        reverse=True)[:max_items_for_heatmap]
    presence   = pd.DataFrame(
        {d: [1 if k in databases[d] else 0 for k in top_keys]
         for d in db_names},
        index=top_keys,
    )
    sns.heatmap(presence.T, cmap='mako', ax=axF,
                cbar_kws={'label': 'Pair presence (0/1)'})
    axF.set_xlabel('Ligand-Receptor Pairs', fontsize=12)
    axF.set_ylabel('Method', fontsize=12)
    axF.tick_params(axis='x', labelrotation=90, labelsize=7)
    axF.text(-0.07, 1.08, 'F', transform=axF.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.95, hspace=1.0)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()

    print("\nJaccard similarity to Chicell2cell:")
    for d in ordered:
        print(f"  {d}: {jac_vs_ref[d]:.3f}")

    return databases, jac, presence, jac_vs_ref
