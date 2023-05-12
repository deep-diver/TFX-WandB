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

from huggingface_hub import Repository
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

import wandb

_MODEL_PROJECT_KEY = "MODEL_REPO_ID"
_MODEL_NAME_KEY = "MODEL_REPO_URL"
_MODEL_VERSION_KEY = "MODEL_VERSION"
_MODEL_FILENAME_KEY = "MODEL_FILENAME"

_DEFAULT_MODEL_REPO_PLACEHOLDER_KEY = "$MODEL_PROJECT"
_DEFAULT_MODEL_URL_PLACEHOLDER_KEY = "$MODEL_NAME"
_DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY = "$MODEL_VERSION"
_DEFAULT_MODEL_FILENAME_KEY = "$MODEL_FILENAME"

def _is_text_file(path):
    mimetype = mimetypes.guess_type(path)
    if mimetype[0] != None:
        return 'text' in mimetype[0]
    
    return False

def _replace_files(src_paths, dst_path):
    """replace the contents(files/folders) of the repository with the
    latest contents"""

    not_to_delete = [".gitattributes", ".git"]

    inside_root_dst_path = tf.io.gfile.listdir(dst_path)

    for content_name in inside_root_dst_path:
        content = f"{dst_path}/{content_name}"

        if content_name not in not_to_delete:
            if tf.io.gfile.isdir(content):
                tf.io.gfile.rmtree(content)
            else:
                tf.io.gfile.remove(content)

    for src_path in src_paths:
        try:
            inside_root_src_path = tf.io.gfile.listdir(src_path)

            for content_name in inside_root_src_path:
                content = f"{src_path}/{content_name}"
                dst_content = f"{dst_path}/{content_name}"

                if tf.io.gfile.isdir(content):
                    io_utils.copy_dir(content, dst_content)
                else:
                    tf.io.gfile.copy(content, dst_content)

        except tf.errors.NotFoundError as e:
            logging.warning(f"Path not found: {src_path}")

def _replace_placeholders_in_files(
    root_dir: str, placeholder_to_replace: Dict[str, str]
):
    """Recursively open every files under the root_dir, and then
    replace special tokens with the given values in placeholder_
    to_replace"""
    files = tf.io.gfile.listdir(root_dir)
    for file in files:
        path = tf.io.gfile.join(root_dir, file)

        if tf.io.gfile.isdir(path):
            _replace_placeholders_in_files(path, placeholder_to_replace)
        else:
            _replace_placeholders_in_file(path, placeholder_to_replace)


def _replace_placeholders_in_file(
    filepath: str, placeholder_to_replace: Dict[str, str]
):
    """replace special tokens with the given values in placeholder_
    to_replace. This function gets called by _replace_placeholders
    _in_files function"""
    if _is_text_file(filepath):
        with tf.io.gfile.GFile(filepath, "r") as f:
            source_code = f.read()

        for placeholder in placeholder_to_replace:
            if placeholder_to_replace[placeholder] is not None:
                source_code = source_code.replace(
                    placeholder, placeholder_to_replace[placeholder]
                )

        with tf.io.gfile.GFile(filepath, "w") as f:
            f.write(source_code)


def _replace_placeholders(
    target_dir: str,
    placeholders: Dict[str, str],
    model_project: str,
    model_name: str,
    model_version: str,
    model_filename: str,
    additional_replacements: Optional[Dict[str, str]]
):
    # tfx-vit-pipeline/final_model:latest
    """set placeholder_to_replace before calling _replace_placeholde
    rs_in_files function"""

    if placeholders is None:
        placeholders = {
            _MODEL_PROJECT_KEY: _DEFAULT_MODEL_REPO_PLACEHOLDER_KEY,
            _MODEL_NAME_KEY: _DEFAULT_MODEL_URL_PLACEHOLDER_KEY,
            _MODEL_VERSION_KEY: _DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY,
            _MODEL_FILENAME_KEY: _DEFAULT_MODEL_FILENAME_KEY
        }

    placeholder_to_replace = {
        placeholders[_MODEL_PROJECT_KEY]: model_project,
        placeholders[_MODEL_NAME_KEY]: model_name,
        placeholders[_MODEL_VERSION_KEY]: model_version,
        placeholders[_MODEL_FILENAME_KEY]: model_filename
    }
    if additional_replacements is not None:
        placeholder_to_replace = {**placeholder_to_replace, **additional_replacements}
    _replace_placeholders_in_files(target_dir, placeholder_to_replace)

def _create_remote_repo(
    access_token: str, repo_id: str, repo_type: str = "space", space_sdk: str = None
):
    """create a remote repository on HuggingFace Hub platform. HTTPError
    exception is raised when the repository already exists"""

    logging.info(f"repo_id: {repo_id}")
    try:
        HfApi().create_repo(
            token=access_token,
            repo_id=repo_id,
            repo_type=repo_type,
            space_sdk=space_sdk,
        )
    except HTTPError:
        logging.warning(
            f"this warning is expected if {repo_id} repository already exists"
        )

def _clone_and_checkout(
    repo_url: str, local_path: str, access_token: str, version: Optional[str] = None
) -> Repository:
    """clone the remote repository to the given local_path"""

    repository = Repository(
        local_dir=local_path, clone_from=repo_url, use_auth_token=access_token
    )

    if version is not None:
        repository.git_checkout(revision=version, create_branch_ok=True)

    return repository


def _push_to_remote_repo(repo: Repository, commit_msg: str, branch: str = "main"):
    """push any changes to the remote repository"""

    repo.git_add(pattern=".", auto_lfs_track=True)
    repo.git_commit(commit_message=commit_msg)
    repo.git_push(upstream=f"origin {branch}")

def deploy_model_for_wandb_model_registry(
    access_token: str,
    project_name: str,
    run_name: str,
    model_name: str,
    aliases: List[str],
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
    outputs = {}

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
        list_aliases = ast.literal_eval(aliases)
        list_aliases.append(model_version)
        print(list_aliases)
        wandb.log_artifact(art, aliases=list_aliases)
        print(f"added {compressed_model_file} to the Artifact")

        # step 1-5
        wandb.finish()
        print("finish up w/ wandb.finish()")

        outputs["run_path"] = '/'.join(found_run.path) if found_run else "not found"
        outputs["model_name"] = model_name
        outputs["model_version"] = model_version
        outputs["file"] = compressed_model_file

    if space_config is not None:
        if "app_path" not in space_config:
            raise RuntimeError(
                "the app_path is not provided. "
                "app_path is required when space_config is set."
            )

        if "hf_username" not in space_config \
            or "hf_repo_name" not in space_config:
            raise RuntimeError(
                "the username or repo_name is not provided. "
            )

        if "hf_access_token" not in space_config:
            raise RuntimeError(
                "the access token to Hugging Face Hub is not provided. "
            )            

        repo_url_prefix = "https://huggingface.co"
        repo_id = f'{space_config["hf_username"]}/{space_config["hf_repo_name"]}'
        repo_url = f"{repo_url_prefix}/{repo_id}"

        app_path = space_config["app_path"]
        app_path = app_path.replace(".", "/")

        access_token = space_config["hf_access_token"]
        space_sdk = space_config.get("space_sdk", "gradio")

        # step 2-1
        _create_remote_repo(
            access_token=access_token,
            repo_id=repo_id,
            space_sdk=space_sdk
        )

        # step 2-2
        tmp_dir = tempfile.mkdtemp()
        io_utils.copy_dir(app_path, tmp_dir)

        # step 2-3
        _replace_placeholders(
            target_dir=tmp_dir,
            placeholders=space_config["placeholders"]
            if "placeholders" in space_config
            else None,
            model_project=project_name,
            model_name=model_name,
            model_version=model_version,
            model_filename=compressed_model_file,
            additional_replacements=space_config.get("additional_replacements", None),
        )

        # step 2-4
        local_path = "hf_space"
        repository = _clone_and_checkout(
            repo_url=repo_url,
            local_path=local_path,
            access_token=access_token,
        )

        # step 2-5
        _replace_files([tmp_dir], local_path)

        # step 2-6
        _push_to_remote_repo(
            repo=repository,
            commit_msg=f"upload {model_version} model",
        )

        outputs["space_url"] = repo_url

    return outputs