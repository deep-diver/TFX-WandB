import keras_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

import wandb

from .train_data import input_fn
from .ViT import MyHyperModel, MyTuner

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import get_hyperparameters


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    wandb_project = None
    if fn_args.custom_config and "wandb" in fn_args.custom_config:
        wandb_configs = fn_args.custom_config["wandb"]

        wandb.login(key=wandb_configs["API_KEY"])
        wandb_project = wandb_configs["PROJECT"]

    hyperparameters = fn_args.custom_config["hyperparameters"]
    tuner_configs = fn_args.custom_config["tuner"]

    tuner = MyTuner(
        wandb_project,
        MyHyperModel(),
        max_trials=tuner_configs["num_trials"],
        hyperparameters=get_hyperparameters(hyperparameters),
        allow_new_entries=False,
        objective=keras_tuner.Objective("val_accuracy", "max"),
        directory=fn_args.working_dir,
        project_name="ViT MLOps Advanced Part2",
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE,
    )

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE,
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": 1,
            "validation_steps": 1,
        },
    )
