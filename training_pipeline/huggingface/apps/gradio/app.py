import tarfile
import wandb

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import ViTFeatureExtractor

PRETRAIN_CHECKPOINT = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(PRETRAIN_CHECKPOINT)

MODEL = None 

RESOLTUION = 224

labels = []

with open(r"labels.txt", "r") as fp:
    for line in fp:
        labels.append(line[:-1])

def normalize_img(
    img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
):
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std

def preprocess_input(image):
    image = np.array(image)
    image = tf.convert_to_tensor(image)

    image = tf.image.resize(image, (RESOLTUION, RESOLTUION))
    image = normalize_img(image)

    image = tf.transpose(
        image, (2, 0, 1)
    )  # Since HF models are channel-first.

    return {
        "pixel_values": tf.expand_dims(image, 0)
    }

def get_predictions(wb_token, image):
    global MODEL
    
    if MODEL is None:
        wandb.login(key=wb_token)
        wandb.init(project="$MODEL_PROJECT", id="$MODEL_RUN", resume=True)
        path = wandb.use_artifact('tfx-vit-pipeline/$MODEL_NAME:$MODEL_VERSION', type='model').download()

        tar = tarfile.open(f"{path}/$MODEL_FILENAME")
        tar.extractall(path=".")

        MODEL = tf.keras.models.load_model("./model")
    
    preprocessed_image = preprocess_input(image)
    prediction = MODEL.predict(preprocessed_image)
    probs = tf.nn.softmax(prediction['logits'], axis=1)

    confidences = {labels[i]: float(probs[0][i]) for i in range(3)}
    return confidences

with gr.Blocks() as demo:
    gr.Markdown("## Simple demo for a Image Classification of the Beans Dataset with HF ViT model")

    wb_token_if = gr.Textbox(interactive=True, label="Your Weight & Biases API Key")

    with gr.Row():
        image_if = gr.Image()
        label_if = gr.Label(num_top_classes=3)

    classify_if = gr.Button()

    classify_if.click(
        get_predictions,
        [wb_token_if, image_if],
        label_if
    )

demo.launch(debug=True)