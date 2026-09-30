"""Construct representational dissimilarity matrices for the Part I analysis."""

from pathlib import Path

import rsatoolbox
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from config import NUM_CLASSES
from core import MODEL_CLASSES, SantoroDataset, extract_activations, load_yamnet_activations
from util import mean_and_ci

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL_DIR = Path(__file__).resolve().parent / "models"


def _build_rdm(measurements, santoro: SantoroDataset, method: str, source: str):
    # Basically builds an rdm based of provided measurements, dataset, dissimilarity measure and the layer name
    data = rsatoolbox.data.Dataset(
        measurements=measurements,
        descriptors={"source": source},
        obs_descriptors={
            "category": santoro.categories,
            "filename": santoro.filenames,
        },
    )

    # descriptor is used to explain rows and columns of the RDM
    return rsatoolbox.rdm.calc_rdm(data, method=method, descriptor="filename")


def build_stg_rdm(method: str = "euclidean", santoro: SantoroDataset | None = None):
    """Compute an STG RDM for each sound in SantoroDataset."""
    
    if santoro is None: # Create new Santoro dataset if none are here
        santoro = SantoroDataset()
    return _build_rdm(santoro.brain_responses.numpy(), santoro, method, "STG")


def build_model_rdms(trained: bool = False, method: str = "euclidean", batch_size: int = 32, santoro: SantoroDataset | None = None, checkpoint_dir: Path = MODEL_DIR,):#uses the model folder
    """Compute layer RDMs for untrained models or every available trained checkpoint."""
    if santoro is None:
        santoro = SantoroDataset()
    # Keep dataset order so extracted activations line up with the stimulus labels.
    dataloader = DataLoader(santoro, batch_size=batch_size, shuffle=False)
    model_rdms = {}

    for model_name, model_class in MODEL_CLASSES.items():

        # Use checkpoints if trained, not if untrained
        checkpoints = (
            sorted(checkpoint_dir.glob(f"{model_name}_run*_best.pt"))
            if trained
            else [None]
        )
        if trained and not checkpoints:
            raise FileNotFoundError(f"No trained checkpoints found for {model_name} in {checkpoint_dir}.")

        model_rdms[model_name] = {}
        for checkpoint in checkpoints:
            #name is only "runx"
            run_name = checkpoint.stem.removeprefix(f"{model_name}_").removesuffix("_best") if checkpoint else "untrained"
            model = model_class(num_classes=NUM_CLASSES)
            if checkpoint is not None:
                # load weights from checkpint
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

            activations = extract_activations(dataloader, model)
            #build rdm for each layer
            model_rdms[model_name][run_name] = {
                # rsa expects one feature vector per sound, so flatten spatial/time dims.
                layer_name: _build_rdm(
                    layer_activations.flatten(start_dim=1).numpy(),
                    santoro,
                    method,
                    f"{model_name}.{run_name}.{layer_name}",
                )
                for layer_name, layer_activations in activations.items()
            }

    return model_rdms


def build_yamnet_rdms(method: str = "euclidean",santoro: SantoroDataset | None = None,):
    """Compute an RDM for every pretrained YAMNet activation layer."""
    if santoro is None:
        santoro = SantoroDataset()
    #  one row per Santoro sound, in  dataset label order.
    return {
        layer_name: _build_rdm(activations,santoro,method,f"yamnet.{layer_name}",)
        for layer_name, activations in load_yamnet_activations().items()
    }

def model_brain_alignment(model_rdms, brain_rdm, method="corr"):
    """Calculate each model layer's alignment with the brain"""
    alignment_scores = {}

    # compare each layer's RDM with brain's RDM and provide score per layer
    for layer_name, model_rdm in model_rdms.items():
        alignment_score = rsatoolbox.rdm.compare(brain_rdm, model_rdm, method=method)
        alignment_scores[layer_name] = alignment_score.item()

    return alignment_scores

def plot_model_brain_alignment(model_name, untrained_statistics, trained_statistics):
    """Plot model-brain alignment across layers"""
    # collect plot values in the same order as layers
    layer_names = list(trained_statistics.keys())

    trained_means = [trained_statistics[layer]["mean"] for layer in layer_names]
    trained_cis = [trained_statistics[layer]["ci"] for layer in layer_names]
    untrained_means = [untrained_statistics[layer]["mean"] for layer in layer_names]
    untrained_cis = [untrained_statistics[layer]["ci"] for layer in layer_names]
    
    x = np.arange(len(layer_names))

    # plot trained and untrained means with CI error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, trained_means, yerr=trained_cis, marker="o", capsize=4, label="Trained")
    plt.errorbar(x, untrained_means, yerr=untrained_cis, marker="o", capsize=4, label="Untrained")
    plt.xticks(x, layer_names, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Brain Alignment (RDM Correlation)")
    plt.title(f"{model_name}: STG brain alignment")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=2, fontsize=8)
    plt.tight_layout()

    os.makedirs("assignment4/plots", exist_ok=True)
    plt.savefig(f"assignment4/plots/{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    santoro = SantoroDataset()
    stg_rdm = build_stg_rdm(santoro=santoro) #build stgs
    print(f"Constructed {stg_rdm.n_rdm} STG RDM with {stg_rdm.n_cond} sound conditions.")

    # load weights from .pt files and calculate layer RDMs
    trained_rdms = build_model_rdms(trained=True, santoro=santoro)

    # create 5 untrained models per model architecture and calculate layer RDMs for each run
    untrained_rdms = {}
    for run_number in range(5):
        model_rdms = build_model_rdms(trained=False, santoro=santoro)
        for model_name, runs in model_rdms.items():
            if model_name not in untrained_rdms:
                untrained_rdms[model_name] = {}
            untrained_rdms[model_name][f"run{run_number}"] = runs["untrained"]

    yamnet_rdms = build_yamnet_rdms(santoro=santoro) #finally build yamnet rdms
    print(f"Constructed {len(yamnet_rdms)} YAMNet layer RDMs.")

    for model_name, runs in trained_rdms.items():
        # multiple untrained model's alignment scores per layer
        untrained_scores = {}
        for run_name, layer_rdms in untrained_rdms[model_name].items():
            untrained_scores[run_name] = model_brain_alignment(layer_rdms, stg_rdm)

        # multiple trained model's alignment scores per layer
        trained_scores = {}
        for run_name, layer_rdms in runs.items():
            trained_scores[run_name] = model_brain_alignment(layer_rdms, stg_rdm)

        # stores trained mean and CI for each layer
        trained_statistics = {}
        untrained_statistics = {}

        layer_names = list(next(iter(trained_scores.values())).keys())

        # average trained scores for each layer, save summary, print the comparison with untrained scores, and proceed with plots
        for layer_name in layer_names:
            
            trained_values = np.array([scores[layer_name] for scores in trained_scores.values()])
            trained_mean, trained_ci = mean_and_ci(trained_values)
            trained_statistics[layer_name] = {"mean": float(trained_mean), "ci": float(trained_ci)}

            untrained_values = np.array([scores[layer_name] for scores in untrained_scores.values()])
            untrained_mean, untrained_ci = mean_and_ci(untrained_values)
            untrained_statistics[layer_name] = {"mean": float(untrained_mean), "ci": float(untrained_ci)}
            
            print(f"{model_name} {layer_name}")
            print(f"untrained={untrained_mean:.4f}, CI +/-{untrained_ci:.4f}")
            print(f"trained={trained_mean:.4f}, CI +/-{trained_ci:.4f}")

        plot_model_brain_alignment(model_name, untrained_statistics, trained_statistics)

    run_tsne(santoro) 

# part 1.3 
SEED = 0

def _flat(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.reshape(len(x), -1)

def _tsne_embed(X, seed=SEED):
    X = StandardScaler().fit_transform(X)
    X = PCA(min(50, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    perplexity = min(15, (len(X) - 1) / 3)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(X)


def _silhouette(X, y):
    return silhouette_score(StandardScaler().fit_transform(X), y, metric="cosine")


def _scatter(ax, Z, y, title, cmap, n_classes):
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, vmin=0, vmax=n_classes - 1,
               s=14, alpha=0.85, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _layer_features(model_name, dataloader, checkpoint=None, seed=None):
    """{layer: (n_sounds, n_features)} for a trained (checkpoint) or untrained (seed) model."""
    if seed is not None:
        torch.manual_seed(seed)
    model = MODEL_CLASSES[model_name](num_classes=NUM_CLASSES)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return {layer: _flat(a) for layer, a in extract_activations(dataloader, model).items()}


def run_tsne(santoro: SantoroDataset, checkpoint_dir: Path = MODEL_DIR, n_untrained: int = 5):
    """Part I.3: t-SNE per layer for brain, YAMNet, untrained and trained models + silhouette scores."""
    os.makedirs("assignment4/plots", exist_ok=True)
    dataloader = DataLoader(santoro, batch_size=32, shuffle=False)  

    le = LabelEncoder()
    y = le.fit_transform(np.asarray(santoro.categories))
    n_classes = len(le.classes_)
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")

    emb, sil, layer_order = {}, {}, {}

    def process(source, state, runs):
        """runs: iterable of {layer: features}. t-SNE on the first run, silhouette (mean+CI) over all runs."""
        scores = {}
        for i, feats in enumerate(runs):
            for layer, X in feats.items():
                scores.setdefault(layer, []).append(_silhouette(X, y))
                if i == 0:
                    emb[(source, state, layer)] = _tsne_embed(X)
        layer_order[(source, state)] = list(scores)
        for layer, vals in scores.items():
            if len(vals) > 1:
                m, ci = mean_and_ci(np.array(vals))
            else:
                m, ci = vals[0], float("nan")
            sil[(source, state, layer)] = (float(m), float(ci))
            print(f"{source:>11} {state:>9} {layer:<20} silhouette = {m:+.3f} +/- {ci:.3f}")

    process("brain", "-", [{"STG": _flat(santoro.brain_responses)}])
    process("yamnet", "trained", [{k: _flat(v) for k, v in load_yamnet_activations().items()}])

    for name in MODEL_CLASSES:
        process(name, "untrained",
                (_layer_features(name, dataloader, seed=s) for s in range(n_untrained)))
        ckpts = sorted(checkpoint_dir.glob(f"{name}_run*_best.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No trained checkpoints for {name} in {checkpoint_dir}.")
        process(name, "trained", (_layer_features(name, dataloader, checkpoint=c) for c in ckpts))

    yam = layer_order[("yamnet", "trained")]
    fig, axes = plt.subplots(1, 1 + len(yam), figsize=(2.6 * (1 + len(yam)), 2.9), squeeze=False)
    _scatter(axes[0, 0], emb[("brain", "-", "STG")], y,
             f"STG brain\nsil={sil[('brain', '-', 'STG')][0]:+.2f}", cmap, n_classes)
    for ax, layer in zip(axes[0, 1:], yam):
        _scatter(ax, emb[("yamnet", "trained", layer)], y,
                 f"YAMNet {layer}\nsil={sil[('yamnet', 'trained', layer)][0]:+.2f}", cmap, n_classes)
    fig.tight_layout()
    fig.savefig("assignment4/plots/tsne_brain_yamnet.png", dpi=200)
    plt.close(fig)

    # fig2
    for name in MODEL_CLASSES:
        layers = layer_order[(name, "trained")]
        fig, axes = plt.subplots(2, len(layers), figsize=(2.4 * len(layers), 5.3), squeeze=False)
        for r, state in enumerate(("untrained", "trained")):
            for c, layer in enumerate(layers):
                _scatter(axes[r, c], emb[(name, state, layer)], y,
                         f"{state}\n{layer}\nsil={sil[(name, state, layer)][0]:+.2f}", cmap, n_classes)
        fig.suptitle(f"t-SNE per layer - {name}")
        fig.tight_layout()
        fig.savefig(f"assignment4/plots/tsne_{name}.png", dpi=200)
        plt.close(fig)

    # fig 3 
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, name in enumerate(MODEL_CLASSES):
        for state, ls in (("untrained", "--"), ("trained", "-")):
            layers = layer_order[(name, state)]
            m = np.array([sil[(name, state, l)][0] for l in layers])
            ci = np.array([sil[(name, state, l)][1] for l in layers])
            depth = np.linspace(0, 1, len(layers))
            ax.plot(depth, m, ls, marker="o", color=f"C{i}", label=f"{name} ({state})")
            ax.fill_between(depth, m - ci, m + ci, color=f"C{i}", alpha=0.15)
    ax.axhline(sil[("brain", "-", "STG")][0], color="k", lw=1.5, label="STG brain")
    ax.set_xlabel("relative depth (0 = first layer, 1 = last layer)")
    ax.set_ylabel("silhouette score (sound categories)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("assignment4/plots/silhouette_vs_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
if __name__ == "__main__": 
    main()