"""Construct representational dissimilarity matrices for the Part I analysis."""

from pathlib import Path

import rsatoolbox
import torch
from torch.utils.data import DataLoader

from config import NUM_CLASSES
from core import MODEL_CLASSES, SantoroDataset, extract_activations, load_yamnet_activations

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


def main():
    santoro = SantoroDataset()
    stg_rdm = build_stg_rdm(santoro=santoro) #build stgs
    print(f"Constructed {stg_rdm.n_rdm} STG RDM with {stg_rdm.n_cond} sound conditions.")

    for trained in (False, True): #build both untrained and trained models
        model_rdms = build_model_rdms(trained=trained, santoro=santoro)
        label = "trained" if trained else "untrained"
        for model_name, runs in model_rdms.items():
            for run_name, layer_rdms in runs.items():
                print(f"Constructed {len(layer_rdms)} layer RDMs for {label} {model_name} ({run_name}).")

    yamnet_rdms = build_yamnet_rdms(santoro=santoro) #finally build yamnet rdms
    print(f"Constructed {len(yamnet_rdms)} YAMNet layer RDMs.")


if __name__ == "__main__":
    main()