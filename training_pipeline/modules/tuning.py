import keras_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

from .train_data import input_fn
from .ViT import MyHyperModel

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import get_hyperparameters


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    wandb.login(key="")

    hyperparameters = fn_args.custom_config["hyperparameters"]

    tuner = keras_tuner.RandomSearch(
        MyHyperModel(),
        max_trials=6,
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
            "epoch": 10,
            "steps_per_epoch": TRAIN_LENGTH // TRAIN_BATCH_SIZE,
            "validation_steps": EVAL_LENGTH // EVAL_BATCH_SIZE,
        },
    )
