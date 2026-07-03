"""HyperparameterTuner — exact match to your working notebook."""

from .config import FLAGS
from .gatvae import CELL2CELLGATVAE
from .train import train_model


class HyperparameterTuner:
    def __init__(self):
        self.results         = []
        self.best_result     = None
        self.best_score      = -1.0
        self.search_completed = False

    def _update_flags(self, params):
        for k, v in params.items():
            setattr(FLAGS, k, v)

    def _train_single_config(self, params, data_dict, config_id):
        self._update_flags(params)
        print(f"\n{'='*80}\nConfiguration {config_id + 1}\n{'='*80}")
        for k, v in params.items():
            print(f"  {k}: {v}")

        device = data_dict['cell_features'].device
        model  = CELL2CELLGATVAE(
            cell_features_dim=data_dict['cell_features'].shape[1],
            gene_features_dim=data_dict['gene_features'].shape[1],
            n_cells=data_dict['cell_features'].shape[0],
            n_genes=data_dict['gene_features'].shape[0],
            hidden_dim=FLAGS.hidden_dim,
            latent_dim=FLAGS.dimension,
        ).to(device)

        trained, history, best_epoch = train_model(
            model,
            data_dict['cell_features'], data_dict['cell_edge_index'],
            data_dict['gene_features'], data_dict['gene_edge_index'],
            data_dict['train_edges_cell'], data_dict['train_edges_neg_cell'],
            data_dict['train_edges_gene'], data_dict['train_edges_neg_gene'],
            data_dict['val_edges_cell'],   data_dict['val_edges_neg_cell'],
            data_dict['val_edges_gene'],   data_dict['val_edges_neg_gene'],
            data_dict['pos_weight_cell'],  data_dict['norm_cell'],
            data_dict['pos_weight_gene'],  data_dict['norm_gene'],
            data_dict['num_nodes_cell'],   data_dict['num_nodes_gene'],
            data_dict['exp'],
            epochs=FLAGS.epochs, patience=FLAGS.patience,
        )

        val_metrics = {
            'val_roc_cell': history['val_roc_cell'][-1],
            'val_ap_cell':  history['val_ap_cell'][-1],
            'val_roc_gene': history['val_roc_gene'][-1],
            'val_ap_gene':  history['val_ap_gene'][-1],
        }
        val_metrics['best_score'] = val_metrics['val_roc_cell']

        print(f"\n  Best Epoch: {best_epoch}")
        print(f"  Cell — Val ROC: {val_metrics['val_roc_cell']:.4f}  AP: {val_metrics['val_ap_cell']:.4f}")
        print(f"  Gene — Val ROC: {val_metrics['val_roc_gene']:.4f}  AP: {val_metrics['val_ap_gene']:.4f}")

        return {'config_id': config_id, 'params': params.copy(),
                'model': trained, 'history': history,
                'best_epoch': best_epoch, 'val_metrics': val_metrics}

    def search(self, param_grid, data_dict):
        print(f"\n{'='*80}\nHYPERPARAMETER SEARCH — {len(param_grid)} configurations\n{'='*80}")
        self.results     = []
        self.best_score  = -1.0
        self.best_result = None

        for idx, params in enumerate(param_grid):
            result = self._train_single_config(params, data_dict, idx)
            self.results.append(result)
            score = result['val_metrics']['best_score']
            if score > self.best_score:
                self.best_score  = score
                self.best_result = result
                print(f"\nNEW BEST! Cell ROC: {score:.4f}")

        self.search_completed = True
        self._update_flags(self.best_result['params'])
        print(f"\n{'='*80}\nSEARCH COMPLETE — best config {self.best_result['config_id']}\n{'='*80}")
        print(f"Best params: {self.best_result['params']}")

    def get_best_model(self):
        if not self.search_completed:
            raise RuntimeError("Call .search() first.")
        return {
            'model':       self.best_result['model'],
            'history':     self.best_result['history'],
            'best_epoch':  self.best_result['best_epoch'],
            'params':      self.best_result['params'],
            'val_metrics': self.best_result['val_metrics'],
            'config_id':   self.best_result['config_id'],
        }
