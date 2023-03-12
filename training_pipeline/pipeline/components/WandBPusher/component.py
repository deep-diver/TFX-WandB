"""WandB(W&B) Pusher TFX Component.
The WandBPusher is used to push model to Model Registry of Weights and Biases 
and optionally prototype application to HuggingFace Space Hub.
"""
from typing import Text, List, Dict, Any, Optional

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from pipeline.components.WandBPusher import executor

MODEL_KEY = "model"
PUSHED_MODEL_KEY = "pushed_model"
MODEL_BLESSING_KEY = "model_blessing"


class WandBPusherSpec(types.ComponentSpec):
    """ComponentSpec for TFX HFPusher Component."""

    PARAMETERS = {
        "access_token": ExecutionParameter(type=str),
        "project_name": ExecutionParameter(type=str),
        "run_name":  ExecutionParameter(type=str),
        "model_name": ExecutionParameter(type=str),
        "aliases": ExecutionParameter(type=List[Text], optional=True),
        "space_config": ExecutionParameter(type=Dict[Text, Any], optional=True),
    }
    INPUTS = {
        MODEL_KEY: ChannelParameter(type=standard_artifacts.Model, optional=True),
        MODEL_BLESSING_KEY: ChannelParameter(
            type=standard_artifacts.ModelBlessing, optional=True
        ),
    }
    OUTPUTS = {
        PUSHED_MODEL_KEY: ChannelParameter(type=standard_artifacts.PushedModel),
    }


class WandBPusher(base_component.BaseComponent):
    """Component for pushing model and application to Weights & Biases
    Model Registry and HuggingFace Space Hub respectively.

    The `WandBPusher` is a [TFX Component](https://www.tensorflow.org/tfx
    /guide/understanding_tfx_pipelines#component), and its primary purpose 
    is to push a model from an upstream component such as [`Trainer`](http
    s://www.tensorflow.org/tfx/guide/trainer) to Weights & Biases Model Re
    gistry. It also provides a secondary feature that pushes an application 
    to HuggingFace Space Hub.
    """

    SPEC_CLASS = WandBPusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
        self,
        access_token: str,
        project_name: str,
        run_name: str,
        model_name: str,
        aliases: Optional[List[Text]] = None,
        space_config: Optional[Dict[Text, Any]] = None,
        model: Optional[types.Channel] = None,
        model_blessing: Optional[types.Channel] = None,        
    ):
        """The WandBPusher TFX component.

        WandBPusher pushes a trained or blessed model to Weights & Biases M
        odel Registry. This is designed to work as a downstream component of 
        Trainer and optionally Evaluator(optional) components. Trainer gives 
        trained model, and Evaluator gives information whether the trained m
        odel is blessed or not after evaluation of the model. HFPusher compo
        nent only publishes a model when it is blessed. If Evaluator is not 
        specified, the input model will always be pushed.

        Args:
        access_token: the access token obtained from Weights & Biases Refer 
            to [this document](https://wandb.ai/authorize) to know how to o
            btain one.
        run_name: a run name given to a particular run. This is used to ret
            rieve the underlying unique Run ID. 
        model_name: 
        aliases: 
        space_config: optional configurations set when to push an application
            to HuggingFace Space Hub. This is a dictionary, and the following
            information could be set.
            app_path: the path where the application related files are stored.
                this should follow the form either of app.gradio.segmentation
                or app/gradio/segmentation. This is a required parameter when
                space_config is set. This could be a local or GCS paths.
            space_sdk: Space Hub supports gradio, streamit, and static types
                of application. The default is set to gradio.
            placeholders: placeholders to replace in every files under the a
                pp_path. This is used to replace special string with the mod
                el related values. If this is not set, the default placehold
                ers will be used as follows.
                ```
                placeholders = {
                    "MODEL_REPO_ID" : "$MODEL_REPO_ID",
                    "MODEL_REPO_URL": "$MODEL_REPO_URL",
                    "MODEL_VERSION" : "$MODEL_VERSION",
                }
                ```
                In this case, "$MODEL_REPO_ID", "$MODEL_REPO_URL", "$MODEL_VE
                RSION" strings will be replaced with appropriate values at ru
                ntime. If placeholders are set, custom strings will be used.
            repo_name: the name of Space Hub repository where the application
                will be pushed. This should be unique name under the username
                within the Space Hub. repository is identified as {username}/
                {repo_name}. If this is not set, the same name to the Model H
                ub repository will be used.

        model: a TFX input channel containing a Model artifact. this is usually
            comes from the standard [`Trainer`]
            (https://www.tensorflow.org/tfx/guide/trainer) component.
        model_blessing: a TFX input channel containing a ModelBlessing artifact.
            this is usually comes from the standard [`Evaluator`]
            (https://www.tensorflow.org/tfx/guide/evaluator) component.
        Returns:
        a TFX output channel containing a PushedModel artifact. It contains
        information where the model is published at and whether the model is
        pushed or not.

        Raises:
            RuntimeError: if app_path is not set when space_config is provided.
        Example:

        Basic usage example:
        ```py
        trainer = Trainer(...)
        evaluator = Evaluator(...)
        hf_pusher = WandBPusher(
            access_token=<YOUR-HUGGINGFACE-ACCESS-TOKEN>,
            run_name="run_name",
            model_name="model_name",
            aliases="best",
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            space_config={
                "app_path": "apps.gradio.semantic_segmentation"
            }
        )
        ```
        """

        pushed_model = types.Channel(type=standard_artifacts.PushedModel)

        spec = WandBPusherSpec(
            access_token=access_token,
            project_name=project_name,
            run_name=run_name,
            aliases=aliases,
            space_config=space_config,
            model=model,
            model_blessing=model_blessing,
            pushed_model=pushed_model,
        )

        super().__init__(spec=spec)