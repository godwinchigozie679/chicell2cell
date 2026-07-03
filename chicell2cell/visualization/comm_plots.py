"""6-panel communication patterns figure."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

_LABEL_MAP = {
    'Microglia/Macrophages': 'Microglia/\nMacrophages',
    'Inhibition_neurons':    'Inhib.\nNeurons',
    'NPC-like-Tumor':        'NPC-like\nTumor',
    'OPC-like-tumor':        'OPC-like\nTumor',
    'MES-like':              'MES-like',
    'Endothelial':           'Endothelial',
    'Neurons':               'Neurons',
    'Astrocyte':             'Astrocyte',
    'AC-like':               'AC-like',
}


def _short(t):
    return _LABEL_MAP.get(t, t)


def plot_communication_patterns(
    filtered_lr,
    source_cell='Inhibition_neurons',
    targets=None,
    save_path=None,
):
    """
    6-panel figure: A-F showing LR communication from source_cell.
    filtered_lr must already be filtered to Secreted Signaling.
    """
    filtered_lr = filtered_lr.copy()
    filtered_lr['lr_pair'] = filtered_lr['ligand'] + '-' + filtered_lr['receptor']

    if targets is None:
        targets = (filtered_lr[filtered_lr['source'] == source_cell]
                   .groupby('target').size()
                   .sort_values(ascending=False).index.tolist())

    def _top10(tgt):
        sub = filtered_lr[(filtered_lr['source'] == source_cell)
                          & (filtered_lr['target'] == tgt)]
        g = sub.groupby(['ligand', 'receptor']).agg(
            communication_score=('communication_score', 'mean'),
            ligand_expression  =('ligand_expression',   'mean'),
            receptor_expression=('receptor_expression', 'mean'),
            pathway_name       =('pathway_name',        'first'),
        ).reset_index()
        return g.sort_values(['communication_score', 'ligand_expression',
                               'receptor_expression'],
                              ascending=False).head(10).copy()

    top10_dfs = {}
    for t in targets:
        df = _top10(t); df['target'] = t; top10_dfs[t] = df
    all_top10 = pd.concat(list(top10_dfs.values()), ignore_index=True)
    all_top10['lr_pair'] = all_top10['ligand'] + '-' + all_top10['receptor']

    qualified = set(zip(all_top10['ligand'], all_top10['receptor']))

    rank_df = all_top10.groupby(['ligand', 'receptor', 'pathway_name']).agg(
        communication_score=('communication_score', 'mean'),
        ligand_expression  =('ligand_expression',   'mean'),
        receptor_expression=('receptor_expression', 'mean'),
        n_targets          =('target',              'nunique'),
    ).reset_index()
    rank_df['primary'] = (rank_df['communication_score'].rank(ascending=True)
                          + rank_df['ligand_expression'].rank(ascending=True)
                          + rank_df['receptor_expression'].rank(ascending=True))
    rank_df = rank_df.sort_values('primary', ascending=False)
    top_row   = rank_df.iloc[0]
    TOP_LIG   = top_row['ligand']
    TOP_REC   = top_row['receptor']
    TOP_LABEL = f"{TOP_LIG}-{TOP_REC}"
    TOP_PATH  = top_row['pathway_name']

    tgt_colors = {t: plt.cm.tab10(i) for i, t in enumerate(targets)}
    inhib = filtered_lr[(filtered_lr['source'] == source_cell)
                        & (filtered_lr['target'].isin(targets))]

    fig = plt.figure(figsize=(20, 24))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.50)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax5, ax6 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    # A
    src_c = inhib.groupby('target').size().reset_index(name='count').sort_values('count')
    ax1.barh(src_c['target'], src_c['count'],
             color=[tgt_colors.get(t, 'grey') for t in src_c['target']],
             edgecolor='white', height=0.48)
    ax1.set_xlabel('Number of interactions')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_title('A', fontsize=16, fontweight='bold', loc='left')

    # B
    inc = inhib[inhib.apply(
        lambda r: (r['ligand'], r['receptor']) in qualified, axis=1)]
    inc_c = inc.groupby('target').size().reset_index(name='count').sort_values('count')
    ax2.barh(inc_c['target'], inc_c['count'],
             color=[tgt_colors.get(t, 'grey') for t in inc_c['target']],
             edgecolor='white', height=0.48)
    ax2.set_xlabel('Number of interactions')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_title('B', fontsize=16, fontweight='bold', loc='left')

    # C
    top_lr_sub = all_top10[(all_top10['ligand'] == TOP_LIG)
                           & (all_top10['receptor'] == TOP_REC)]
    ax3.bar(top_lr_sub['target'], top_lr_sub['communication_score'],
            color=[tgt_colors.get(t, 'grey') for t in top_lr_sub['target']],
            edgecolor='white', width=0.42)
    ax3.set_ylabel('Mean Communication Score')
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.set_title('C', fontsize=16, fontweight='bold', loc='left')
    ax3.text(0.02, 0.97, f'{TOP_LABEL}  |  {TOP_PATH}',
             transform=ax3.transAxes, fontsize=10, va='top', style='italic')

    # D
    pw = (inc.groupby(['pathway_name', 'target']).size()
          .reset_index(name='count'))
    pw_piv = pw.pivot_table(index='pathway_name', columns='target',
                             values='count', fill_value=0)
    pw_piv['_tot'] = pw_piv.sum(axis=1)
    pw_piv = pw_piv.sort_values('_tot').drop(columns='_tot')
    bottom = np.zeros(len(pw_piv))
    for t in pw_piv.columns:
        ax4.barh(pw_piv.index, pw_piv[t].values,
                 left=bottom, color=tgt_colors.get(t, 'grey'),
                 edgecolor='white', height=0.62, label=t)
        bottom += pw_piv[t].values
    ax4.set_xlabel('Number of interactions')
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.legend(title='Target', fontsize=9, loc='upper right')
    ax4.set_title('D', fontsize=16, fontweight='bold', loc='left')

    # E
    pair_order = (all_top10.groupby('lr_pair')['communication_score']
                  .mean().sort_values().index.tolist())
    tgt_order  = list(targets)
    all_top10['y_pos'] = all_top10['lr_pair'].map(
        {p: i for i, p in enumerate(pair_order)})
    all_top10['x_pos'] = all_top10['target'].map(
        {t: i for i, t in enumerate(tgt_order)})
    g_max  = max(all_top10['ligand_expression'].max(),
                 all_top10['receptor_expression'].max())
    lig_sz = (all_top10['ligand_expression']   / g_max) * 600 + 60
    rec_sz = (all_top10['receptor_expression'] / g_max) * 600 + 60

    sc = ax5.scatter(all_top10['x_pos'] - 0.18, all_top10['y_pos'],
                     s=lig_sz, c=all_top10['communication_score'],
                     cmap='YlOrRd', marker='o', alpha=0.85,
                     edgecolors='#4E79A7', linewidths=1.5)
    ax5.scatter(all_top10['x_pos'] + 0.18, all_top10['y_pos'],
                s=rec_sz, c=all_top10['communication_score'],
                cmap='YlOrRd', marker='D', alpha=0.85,
                edgecolors='#E63946', linewidths=1.5)
    if TOP_LABEL in pair_order:
        ty = pair_order.index(TOP_LABEL)
        ax5.axhspan(ty - 0.45, ty + 0.45, color='gold', alpha=0.2, zorder=0)
    ax5.set_xticks(range(len(tgt_order)))
    ax5.set_xticklabels(
        [_short(t) for t in tgt_order],
        rotation=30, ha='right', fontsize=9
    )
    ax5.set_yticks(range(len(pair_order)))
    ax5.set_yticklabels(pair_order, style='italic', fontsize=8)
    ax5.set_xlabel('Target cell type')
    ax5.set_ylabel('Ligand-receptor pair')
    ax5.spines[['top', 'right']].set_visible(False)
    ax5.set_title('E', fontsize=16, fontweight='bold', loc='left')
    plt.colorbar(sc, ax=ax5, orientation='horizontal',
                 shrink=0.4, pad=0.15, label='Mean Communication Score')

    # F
    expr_sub = all_top10[(all_top10['ligand'] == TOP_LIG)
                         & (all_top10['receptor'] == TOP_REC)].reset_index(drop=True)
    gap  = 2.8; bh = 0.65
    y_lig = np.arange(len(expr_sub)) * gap + 0.4
    y_rec = np.arange(len(expr_sub)) * gap - 0.4
    y_grp = np.arange(len(expr_sub)) * gap

    for i, t in enumerate(expr_sub['target']):
        ax6.axhspan(y_grp[i] - 1.0, y_grp[i] + 1.0,
                    color=tgt_colors.get(t, '#eeeeee'), alpha=0.12, zorder=0)

    ax6.barh(y_lig, expr_sub['ligand_expression'], height=bh,
             color='#4E79A7', label=f'{TOP_LIG} (ligand)', zorder=2)
    ax6.barh(y_rec, expr_sub['receptor_expression'], height=bh,
             color='#E63946', label=f'{TOP_REC} (receptor)', zorder=2)

    x_max = max(expr_sub['ligand_expression'].max(),
                expr_sub['receptor_expression'].max())
    for i, t in enumerate(expr_sub['target']):
        ax6.text(-x_max * 0.02, y_grp[i], t,
                 va='center', ha='right', fontsize=9,
                 color=tgt_colors.get(t, 'black'), fontweight='bold')

    ax6.set_yticks([])
    ax6.set_xlim(0, x_max * 1.15)
    ax6.set_xlabel('Mean Expression')
    ax6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
    ax6.spines[['top', 'right', 'left']].set_visible(False)
    ax6.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax6.set_title('F', fontsize=16, fontweight='bold', loc='left')
    ax6.text(0.02, 0.97, f'{TOP_LABEL}  |  {TOP_PATH}',
             transform=ax6.transAxes, fontsize=10, va='top', style='italic')

    legend_handles = [
        plt.scatter([], [], s=30,  marker='o', facecolor='#aaa',
                    edgecolors='#4E79A7', linewidths=1.5,
                    label='Ligand expression (circle)'),
        plt.scatter([], [], s=30,  marker='D', facecolor='#aaa',
                    edgecolors='#E63946', linewidths=1.5,
                    label='Receptor expression (diamond)'),
        plt.scatter([], [], s=30,  marker='o', facecolor='#aaa',
                    edgecolors='#555', linewidths=1.5,
                    label='Small = low expr'),
        plt.scatter([], [], s=400, marker='o', facecolor='#aaa',
                    edgecolors='#555', linewidths=1.5,
                    label='Large = high expr'),
        plt.scatter([], [], s=30,  marker='D', facecolor='#aaa',
                    edgecolors='#555', linewidths=1.5,
                    label='Small = low expr '),
        plt.scatter([], [], s=400, marker='D', facecolor='#aaa',
                    edgecolors='#555', linewidths=1.5,
                    label='Large = high expr '),
    ]
    fig.legend(
        handles=legend_handles,
        fontsize=9, frameon=True, framealpha=0.95,
        loc='lower center', bbox_to_anchor=(0.5, -0.03),
        ncol=3, borderpad=1.0, handletextpad=1.0,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()