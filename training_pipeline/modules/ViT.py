import tensorflow as tf
import keras_tuner
from transformers import TFViTForImageClassification

import wandb

from .common import LABELS
from .common import PRETRAIN_CHECKPOINT
from .utils import INFO

class MyHyperModel(keras_tuner.HyperModel):
  def build(self, hp):
    id2label={str(i): c for i, c in enumerate(LABELS)}
    label2id={c: str(i) for i, c in enumerate(LABELS)}

    model = TFViTForImageClassification.from_pretrained(
      PRETRAIN_CHECKPOINT,
      num_labels=len(LABELS),
      label2id=label2id,
      id2label=id2label,
    )

    model.layers[0].trainable=False

    with hp.conditional_scope("optimizer_type", ["AdamW"]):
      optimizer = tf.keras.optimizers.experimental.AdamW(
          learning_rate= hp.get("learning_rate"),
          weight_decay=hp.get("weight_decay"),
      )
    with hp.conditional_scope("optimizer_type", ["Adam"]):
      optimizer = tf.keras.optimizers.Adam(
          learning_rate=hp.get("learning_rate"),
          weight_decay=hp.get("weight_decay"),
      )      
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    INFO(model.summary())

    return model

class MyTuner(keras_tuner.RandomSearch):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run_trial(self, trial, *args, **kwargs):
    hp = trial.hyperparameters
    model = self.hypermodel.build(hp)

    optimizer_type = hp.get("optimizer_type")
    learning_rate = hp.get("learning_rate")
    weight_decay = hp.get("weight_decay")

    log_name = f"tuning@opt-{optimizer_type}@lr-{learning_rate}@wd-{weight_decay}"
    wandb.init(
      project="_PROJECT_NAME",
      config=hp.values,
      name=log_name,
    )

    wandb.log({"optimizer": optimizer_type})
    wandb.log({"optimizer": learning_rate})
    wandb.log({"optimizer": weight_decay})

    result = self.hypermodel.fit(
      hp,
      model,
      *args,
      callbacks=[
        wandb.keras.WandbMetricsLogger(log_freq='epoch')
      ],
      **kwargs
    )

    wandb.finish()
    return result