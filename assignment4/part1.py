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

if __name__ == "__main__":
    main()