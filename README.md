# TFX + W&B Integration

[![Pipeline Trigger(Local)](https://github.com/deep-diver/TFX-WandB/actions/workflows/local-trigger.yml/badge.svg)](https://github.com/deep-diver/TFX-WandB/actions/workflows/local-trigger.yml)

![](https://i.ibb.co/kgMByHM/overview.png)

This repository is designed to contain the following implementations towards [TensorFlow Extended](https://www.tensorflow.org/tfx)(TFX) and [Weights and Biases](https://wandb.ai/site)(W&B) integrations

- [Experiment Tracking](https://wandb.ai/site/experiment-tracking): TFX and KerasTuner can be combined naturally, so different sets of hyper-parameter tunings from KerasTuner could be logged in W&B. 

- [Model Registry](https://model-registry.wandb.ai/): Trained & blessed (qualified) models from TFX pipeline could be pushed to the W&B Model Registry. Optionally push prototype application on [Hugging Face Space Hub](https://huggingface.co/docs/hub/spaces-overview) with the right model version.

## Project structure

In order to keep the implementation simple, this repository uses only the following standard TFX components. However, note that you could easily integrate more of TFX components.

- `ImportExampleGen`: to bring prepared dataset into the TFX pipeline. 
- `Trainer`: to train the model with the best found hyper-parameters from `Tuner`
- `Tuner`: to search the best hyper-parameters. Experiment Tracking feature of W&B will be integrated into this component.

Additionally, this project will present a custom TFX component.

- `WandBPusher`: to push a trained model from `Trainer` to the Model Registry in W&B. This component could be integrated with `Evaluator` as well. 

### Model and dataset

Again, in order to simplify and focus on the purpose of this project, we are going to use a simple model and dataset as the following.

- [`ViT`](https://huggingface.co/docs/transformers/model_doc/vit) from HuggingFace's [`transformers`](https://huggingface.co/docs/transformers/index)
- [`beans`](https://huggingface.co/datasets/beans) dataset which will be prepared as TFRecord format from HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index) library.

## Todos

- [X] write a notebook to prepare TFRecord dataset
- [X] write a notebook to run the basic chain of TFX components
- [X] extend the notebook to add Experiment Tracking feature
- [X] extend the notebook to leverage KerasTuner's HyperModel and Tuner
- [X] port the notebook to standalone TFX pipeline w/o notebook
- [X] integrate GCP's [Vertex AI](https://cloud.google.com/vertex-ai) with [GitHub Action](https://github.com/features/actions)
- [ ] write a notebook for the new custom component (`WandBPusher`)
- [ ] write a notebook to integrate `WandBPusher` component into the existing TFX pipeline
- [X] port `WandBPusher` into the existing standadone TFX pipeline
- [X] modify `WandBPusher` to push prototype application to Hugging Face Space Hub
