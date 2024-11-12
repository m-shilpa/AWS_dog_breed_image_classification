import os
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.logging_utils import setup_logger, task_wrapper, get_rich_progress

log = logging.getLogger(__name__)


def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever',
                    'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
 
@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    

    log_dir = Path(cfg.paths.log_dir)
    setup_logger(log_dir / "infer_log.log")

    # Get checkpoint path from command-line arguments
    if cfg.ckpt_path is None:
        raise ValueError("No checkpoint path provided")

    if not Path(cfg.ckpt_path).exists():
        raise FileNotFoundError(f"No checkpoint found at {cfg.ckpt_path}")

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))


    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(cfg.ckpt_path)

    # Set input folder
    if cfg.input_folder is None:
        download_and_extract_archive(
            url="https://github.com/m-shilpa/lightning-template-hydra/raw/main/dog_breed_10_test_images.zip",
            download_root=cfg.paths.data_dir,
            remove_finished=True
        )
        input_folder = Path(cfg.paths.data_dir) / "dog_breed_10_test_images"
    else:
        input_folder = Path(cfg.input_folder)

    output_folder = Path(cfg.paths.output_dir) / "output"
    output_folder.mkdir(exist_ok=True, parents=True)

    image_files = list(input_folder.glob('*'))
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))
        
        for image_file in image_files:
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device))
                
                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)
                
                progress.console.print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                progress.advance(task)

if __name__ == "__main__":
    main()
