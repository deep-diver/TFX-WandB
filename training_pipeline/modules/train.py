import keras_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

import wandb

from .train_data import input_fn
from .ViT import MyHyperModel
from .signatures import (
    model_exporter,
    transform_features_signature,
    tf_examples_serving_signature,
)

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import EPOCHS

from .utils import INFO


def run_fn(fn_args: FnArgs):
    wandb_project = None
    if fn_args.custom_config and "wandb" in fn_args.custom_config:
        wandb_configs = fn_args.custom_config["wandb"]

        wandb.login(key=wandb_configs["API_KEY"])
        wandb_project = wandb_configs["PROJECT"]

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

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

    if wandb_project:
        unique_id = wandb_configs["FINAL_RUN_ID"]

        wandb.init(
            project=wandb_project, 
            config=fn_args.hyperparameters,
            name=unique_id,
        )
    
    hp = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    INFO(f"HyperParameters for training: {hp.get_config()}")

    optimizer_type = hp.get("optimizer_type")
    learning_rate = hp.get("learning_rate")
    weight_decay = hp.get("weight_decay")
    epochs = hp.get("epochs")
    callbacks = []

    if wandb_project:
        wandb.log({"optimizer": optimizer_type})
        wandb.log({"learning_rate": learning_rate})
        wandb.log({"weight_decay": weight_decay})
        callbacks.append(wandb.keras.WandbMetricsLogger(log_freq='epoch'))

    model = MyHyperModel().build(hp)
    model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_LENGTH // TRAIN_BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=EVAL_LENGTH // TRAIN_BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
    )

    if wandb_project:
        wandb.finish()

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures={
            "serving_default": model_exporter(model),
            "transform_features": transform_features_signature(
                model, tf_transform_output
            ),
            "from_examples": tf_examples_serving_signature(model, tf_transform_output),
        },
    )
