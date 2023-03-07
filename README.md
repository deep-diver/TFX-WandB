# TFX + W&B Integration

This repository is designed to contain the following implementations towards [TensorFlow Extended](https://www.tensorflow.org/tfx)(TFX) and [Weights and Biases](https://wandb.ai/site)(W&B) integrations

- [Experiment Tracking](https://wandb.ai/site/experiment-tracking): TFX and KerasTuner can be combined naturally, so different sets of hyper-parameter tunings from KerasTuner could be logged in W&B. 

- [Model Registry](https://model-registry.wandb.ai/): Trained & blessed (qualified) models from TFX pipeline could be pushed to the W&B Model Registry. Optionally push prototype application on [Hugging Face Space Hub](https://huggingface.co/docs/hub/spaces-overview) with the right model version.