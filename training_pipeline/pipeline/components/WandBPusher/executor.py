# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""HF Pusher TFX Component Executor. The HF Pusher Executor calls 
the workflow handler runner.deploy_model_for_hf_hub().
"""

import ast
import time
from typing import Any, Dict, List

from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils, standard_component_specs

from pipeline.components.WandBPusher import runner

_ACCESS_TOKEN_KEY = "access_token"
_PROJECT_NAME = "project_name"
_RUN_NAME = "run_name"
_MODEL_NAME = "model_name"
_ALIASES = "aliases"
_SPACE_CONFIG_KEY = "space_config"

class Executor(tfx_pusher_executor.Executor):
    """Pushes a model and an app to HuggingFace Model and Space Hubs respectively"""

    def Do(
        self,
        input_dict: Dict[str, List[types.Artifact]],
        output_dict: Dict[str, List[types.Artifact]],
        exec_properties: Dict[str, Any],
    ):
        """Overrides the tfx_pusher_executor to leverage some of utility methods
        Args:
          input_dict: Input dict from input key to a list of artifacts, including:
            - model_export: a TFX input channel containing a Model artifact.
            - model_blessing: a TFX input channel containing a ModelBlessing
              artifact.
          output_dict: Output dict from key to a list of artifacts, including:
            - pushed_model: a TFX output channel containing a PushedModel arti
              fact. It contains information where the model is published at an
              d whether the model is pushed or not. furthermore, pushed model
              carries the following information.
              - pushed : integer value to denote if the model is pushed or not.
                This is set to 0 when the input model is not blessed, and it is
                set to 1 when the model is successfully pushed.
          exec_properties: An optional dict of execution properties, including:
            ...
        """
        self._log_startup(input_dict, output_dict, exec_properties)

        model_push = artifact_utils.get_single_instance(
            output_dict[standard_component_specs.PUSHED_MODEL_KEY]
        )

        # if the model is not blessed
        if not self.CheckBlessing(input_dict):
            self._MarkNotPushed(model_push)
            return
        model_path = self.GetModelPath(input_dict)
        model_version_name = f"v{int(time.time())}"

        space_config = exec_properties.get(_SPACE_CONFIG_KEY, None)
        if space_config is not None:
            space_config = ast.literal_eval(space_config)

        pushed_properties = runner.deploy_model_for_wandb_model_registry(
            access_token=exec_properties.get(_ACCESS_TOKEN_KEY, None),
            project_name=exec_properties.get(_PROJECT_NAME, None),
            run_name=exec_properties.get(_RUN_NAME, None),
            model_name=exec_properties.get(_MODEL_NAME, None),
            model_version=model_version_name,
            aliases=exec_properties.get(_ALIASES, None),
            model_path=model_path,
            space_config=space_config,
        )

        self._MarkPushed(model_push, pushed_destination=pushed_properties["repo_url"])
        for key in pushed_properties:
            value = pushed_properties[key]

            if key != "repo_url":
                model_push.set_string_custom_property(key, value)