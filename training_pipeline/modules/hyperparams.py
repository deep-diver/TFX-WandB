import keras_tuner

EPOCHS = 10
BATCH_SIZE = 32

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

TRAIN_LENGTH = 1034
EVAL_LENGTH = 128

INPUT_IMG_SIZE = 224

def get_hyperparameters(hyperparameters) -> keras_tuner.HyperParameters:
    hp_set = keras_tuner.HyperParameters()

    for hp in hyperparameters:
        if hyperparameters[hp]["type"] == "choice":
            hp_set.Choice(
                hp, 
                hyperparameters[hp]["values"]
            )            
        elif hyperparameters[hp]["type"] == "float":
            hp_set.Float(
                hp, 
                hyperparameters[hp]["min_value"],
                hyperparameters[hp]["max_value"],
                sampling=hyperparameters[hp]["sampling"],
                step=hyperparameters[hp]["step"]
            )

    return hp_set
