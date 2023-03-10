"""Tests for TFX WandB Pusher Custom Component."""

import tensorflow as tf
from tfx.types import standard_artifacts
from tfx.types import channel_utils

from pipeline.components.WandBPusher.component import WandBPusher


class HFPusherTest(tf.test.TestCase):
    def testConstruct(self):
        test_model = channel_utils.as_channel([standard_artifacts.Model()])
        wandb_pusher = WandBPusher(
            access_token="test_access_token",
            run_name="run_name",
            model_name="model_name",
            aliases="aliases",
            model=test_model,
        )

        self.assertEqual(
            standard_artifacts.PushedModel.TYPE_NAME,
            wandb_pusher.outputs["pushed_model"].type_name,
        )


if __name__ == "__main__":
    tf.test.main()