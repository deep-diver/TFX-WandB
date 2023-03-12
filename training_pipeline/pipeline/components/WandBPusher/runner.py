"""WandB Pusher runner module.
This module handles the workflow to publish machine 
learning model to Weights & Biases Model Registry.
"""
from typing import Text, List, Any, Dict, Optional

import os
import mimetypes
import tempfile
import ast
import tarfile
import tensorflow as tf
from absl import logging
from tfx.utils import io_utils
from pathlib import Path

import wandb

_MODEL_REPO_KEY = "MODEL_REPO_ID"
_MODEL_URL_KEY = "MODEL_REPO_URL"
_MODEL_VERSION_KEY = "MODEL_VERSION"

_DEFAULT_MODEL_REPO_PLACEHOLDER_KEY = "$MODEL_REPO_ID"
_DEFAULT_MODEL_URL_PLACEHOLDER_KEY = "$MODEL_REPO_URL"
_DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY = "$MODEL_VERSION"

def deploy_model_for_wandb_model_registry(
    access_token: str,
    project_name: str,
    run_name: str,
    model_name: str,
    aliases: List[str],
    model_path: str,
    model_version: str,
    space_config: [Dict[Text, Any]],
) -> Dict[str, str]:
    """Executes ML model deployment workflow to Weights & Biases Model
    Registry. Refer to the WandBPusher component in component.py for g
    eneric description of each parameter. This docstring only explains
    how the workflow works.
    step 1. push model to the Weights & Biases Model Registry
    step 1-1.
        login to the Weights & Biases w/ access token
    step 1-2.
        find the run path which is a unique ID of a certain Run belonin
        g to the project_name w/ run_name
    step 1-3.
        init wandb w/ project_name and the run path from 1-2
    step 1-4
        create an Weights & Biases Artifact and log the model file
    step 1-5.
        finish wandb
    step 2. push application to the Space Hub
    step 2-1.
        create a repository on the HuggingFace Hub. if there is an existing r
        epository with the given repo_name, that rpository will be overwritten.
    step 2-2.
        copies directory where the application related files are stored to a
        temporary directory. Since the files could be hosted in GCS bucket, t
        his process ensures every necessary files are located in the local fil
        e system.
    step 2-3.
        replace speical tokens in every files under the given directory.
    step 2-4.
        clone the created or existing remote repository to the local path.
    step 2-5.
        remove every files under the cloned repository(local), and copies the
        application related files to the cloned local repository path.
    step 2-6.
        push the updated repository to the remote Space Hub. note that the br
        anch is always set to "main", so that HuggingFace Space could build t
        he application automatically when pushed.
    """

    # 1-1
    wandb.login(key=access_token)

    # 1-2
    found_run = None
    for run in wandb.Api().runs(project_name):
        if run.name == run_name:
            found_run = run

    if found_run:
        print(f"found_run: {found_run.path}")

        # 1-3
        wandb.init(
            project=project_name,
            id=found_run.path[-1],
            resume=True
        )
        print(f"wandb initialized w/ project({project_name}), id({'/'.join(found_run.path)})")

        # 1-4
        tmp_dir = "model"
        os.mkdir(tmp_dir)
        print(f"created temporary dir({tmp_dir})")

        inside_model_path = tf.io.gfile.listdir(model_path)
        for content_name in inside_model_path:
            content = f"{model_path}/{content_name}"
            dst_content = f"{tmp_dir}/{content_name}"

            if tf.io.gfile.isdir(content):
                io_utils.copy_dir(content, dst_content)
            else:
                tf.io.gfile.copy(content, dst_content)

        print(f"copied SavedModel from {model_path} to the temporary dir({tmp_dir})")

        compressed_model_file = "model.tar.gz"
        
        tar = tarfile.open(compressed_model_file, "w:gz")
        tar.add(tmp_dir)
        tar.close()
        print(f"SavedModel compressed into {compressed_model_file}")
        
        art = wandb.Artifact(model_name, type="model")
        print(f"wandb Artifact({model_name}) is created")

        art.add_file(compressed_model_file)
        aliases = ast.literal_eval(aliases)
        aliases.append(model_version)
        wandb.log_artifact(art, aliases=aliases)
        print(f"added {compressed_model_file} to the Artifact")

        # step 1-5
        wandb.finish()
        print("finish up w/ wandb.finish()")

    return {
        "run_path": '/'.join(found_run.path) if found_run else "not found",
        "model_name": model_name,
        "model_version": model_version,
        "file": compressed_model_file,
    }
