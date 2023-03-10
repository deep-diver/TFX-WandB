"""WandB Pusher runner module.
This module handles the workflow to publish machine 
learning model to Weights & Biases Model Registry.
"""
from typing import Text, Any, Dict, Optional

import mimetypes
import tempfile
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
    aliases: str,
    model_path: str,
    model_version: str,
    space_config: Optional[Dict[Text, Any]] = None,
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
        init
    step 1-4
        write model card.
    step 1-5.
        push the updated repository to the given branch of remote Model Hub.
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

    wandb.login(key=access_token)

    return {}