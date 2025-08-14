# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.multiprocessing import set_sharing_strategy

from arguments import args
from utils import import_from_module, load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
printer = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*",
)
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

set_sharing_strategy("file_system")

# Load configuration and import modules
config = load_config(args.config)
TorchModel = import_from_module(
    "models." + config["model"]["module"], config["model"]["class"],
)
LitDataModule = import_from_module(
    "datamodules." + config["datamodule"]["module"], config["datamodule"]["class"],
)
LitModel = import_from_module(
    "models." + config["litmodule"]["module"], config["litmodule"]["class"],
)


def main(save_name: str) -> None:
    """Run evaluation on a trained PyTorch Lightning model using a stored checkpoint.

    This function:
    - Locates the appropriate model checkpoint based on the provided name.
    - Sets up logger, model, and data module based on the loaded configuration.
    - Loads model weights from the checkpoint, if available.
    - Runs the `test` routine via a PyTorch Lightning `Trainer`.

    Args:
        save_name (str): The identifier used to locate the saved model checkpoint.

    Raises:
        NameError: If no matching checkpoint file is found and dry_run is disabled.
        FileNotFoundError: If the checkpoint file cannot be loaded.

    """
    ds = config["dataset"]
    path = Path("saved_models") / ds / save_name

    # Check if checkpoint exists
    if path.with_suffix(".ckpt").exists():
        ckpt = path.with_suffix(".ckpt")
    elif path.with_name(path.name + "-v1").with_suffix(".ckpt").exists():
        ckpt = path.with_name(path.name + "-v1").with_suffix(".ckpt")
    elif not args.dry_run:
        msg = f"Could not find model with name: {save_name}"
        raise NameError(msg)

    # Determine the number of devices, and accelerator
    if torch.cuda.is_available() and args.use_cuda:
        devices, accelerator = -1, "auto"
    else:
        devices, accelerator = 1, "cpu"

    # Setup logger
    logger: bool | Logger
    if args.dry_run:
        logger = False
        args.small_ds = True
    elif not args.use_logger:
        logger = False
    else:
        logger = CSVLogger(save_dir=Path("lightning_logs") / ds, name=save_name)

    # Setup model
    net = TorchModel(config["model"])
    model = LitModel(net, config["training"])

    # Load checkpoint into model
    if ckpt and Path(ckpt).exists():
        msg = f"Loading checkpoint: {ckpt}"
        printer.info(msg)
        ckpt_dict = torch.load(ckpt, weights_only=True)
        model.load_state_dict(ckpt_dict["state_dict"], strict=False)
    elif not args.dry_run:
        msg = f"Could not find checkpoint: {ckpt}"
        raise FileNotFoundError(msg)

    # Setup datamodule
    if args.root:
        config["datamodule"]["root"] = args.root
    datamodule = LitDataModule(config["datamodule"], args)

    # Setup trainer
    trainer = Trainer(accelerator=accelerator, devices=devices, logger=logger)

    # Start testing
    trainer.test(model, datamodule=datamodule, verbose=True)


if __name__ == "__main__":
    seed_everything(args.seed, workers=True)

    mdl_name = config["model"]["class"]
    ds_name = config["dataset"]
    add_name = f"-{args.add_name}" if args.add_name else ""

    full_save_name = f"{mdl_name}{add_name}-{ds_name}"

    msg = f"\nGetting ready to test model: {full_save_name} \n"
    printer.info("----------------------------------------------------")
    printer.info(msg)
    printer.info("----------------------------------------------------")

    main(full_save_name)
