"""Reproducible first-fold t-SNE of the actual input to ViRTA.classifier."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_tenfold import digest, save


def plot_features(source, output, seed=666, metadata=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    source, output = Path(source), Path(output)
    with np.load(source, allow_pickle=False) as data:
        features, labels, news_ids = data['features'], data['labels'], data['news_ids']
    if (features.ndim != 2 or len(features) < 4 or len(features) != len(labels)
            or len(labels) != len(news_ids) or set(labels.tolist()) != {0, 1}
            or not np.isfinite(features).all()):
        raise ValueError('Expected finite classifier features with aligned IDs and both binary classes')
    components = min(50, features.shape[1], len(features) - 1)
    pca = PCA(n_components=components, svd_solver='randomized', random_state=seed)
    reduced = pca.fit_transform(features)
    perplexity = min(30.0, (len(features) - 1) / 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto',
                max_iter=1500, random_state=seed, n_jobs=4)
    xy = tsne.fit_transform(reduced)
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite t-SNE coordinates')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.embedding.npz'), xy=xy, labels=labels, news_ids=news_ids)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = 'Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
    colors = np.asarray(['#FF8998', '#9868BD'])
    style = {'font.family': font, 'font.size': 7.5, 'svg.fonttype': 'none',
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.5, 2.45))
        ax = fig.add_axes([.025, .055, .95, .77])
        order = np.random.default_rng(seed).permutation(len(labels))
        ax.scatter(xy[order, 0], xy[order, 1], c=colors[labels[order]], s=3.2,
                   alpha=.72, edgecolors='none', linewidths=0)
        ax.margins(.035)
        ax.set_axis_off()
        handles = [Line2D([], [], marker='o', color='none', markerfacecolor=color,
                          markeredgecolor='none', markersize=3.5, label=label)
                   for color, label in zip(colors, ('False news (0)', 'Real news (1)'))]
        legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .97),
                            ncol=2, frameon=False, handlelength=.7, handletextpad=.6, columnspacing=1.5)
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        if box.x0 < 0 or box.x1 > 1 or box.y0 < ax.get_position().y1 or box.y1 > 1:
            raise ValueError('Legend is clipped or overlaps the data area')
        for suffix in ('.png', '.pdf', '.svg'):
            fig.savefig(output.with_suffix(suffix), dpi=600, facecolor='white')
        plt.close(fig)
    svg = output.with_suffix('.svg').read_text()
    if '<text' not in svg or 'False news (0)' not in svg or 'Real news (1)' not in svg:
        raise ValueError('Editable SVG legend did not survive export')
    save(output.with_suffix('.metadata.json'), {
        **(metadata or {}), 'feature_source': str(source.resolve()), 'feature_sha256': digest(source),
        'feature_layer': 'ViRTA.fgafm fused_features: exact input to the final Linear classifier, before Linear/ReLU',
        'split': 'first fold held-out test set; captured during the one test evaluation',
        'samples': len(labels), 'feature_dim': features.shape[1],
        'class_counts': {'false_0': int((labels == 0).sum()), 'real_1': int((labels == 1).sum())},
        'pca_components': components, 'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'tsne': {'seed': seed, 'perplexity': perplexity, 'init': 'pca', 'learning_rate': 'auto',
                 'max_iter': 1500, 'kl_divergence': float(tsne.kl_divergence_)},
        'style': {'font': font, 'colors': colors.tolist(), 'figsize_inches': [3.5, 2.45],
                  'dpi': 600, 'point_area_pt2': 3.2, 'alpha': .72, 'axes': False, 'title': False},
        'all_samples_plotted': True, 'class_labels_used_for_embedding': False,
        'qa': {'finite_features_and_coordinates': True, 'legend_inside_figure': True,
               'legend_separate_from_data': True, 'svg_text_editable': True,
               'visual_preview_inspected': False},
    })
    print('TSNE_SAVED', output, 'samples=', len(labels), 'feature_dim=', features.shape[1], flush=True)
    return xy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()
    plot_features(args.features, args.output, args.seed)
