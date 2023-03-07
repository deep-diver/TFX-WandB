# TFX + W&B Integration

This repository is designed to contain the following implementations towards [TensorFlow Extended](https://www.tensorflow.org/tfx)(TFX) and [Weights and Biases](https://wandb.ai/site)(W&B) integrations

- experiment tracking: TFX and KerasTuner can be combined naturally, so I can think of logging the results from different hyper-parameters on W&B. I thought about W&B sweep, and it would work fine within a single machine. However, it wouldn't be possible to run W&B seep on multiple machines hosted on GCP since the seep_id should be provided manually through API or CLI either way. 

- model registry: Trained & blessed (qualified) models from TFX pipeline could be pushed to the W&B Model Registry or Artifact store. Currently, I did a similar project to push models to the Hugging Face Dataset, and I added an optional feature to publish the Hugging Face Space
