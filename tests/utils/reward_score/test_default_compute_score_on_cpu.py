# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.reward_score import default_compute_score


def test_math_500_alias_uses_math_reward():
    score = default_compute_score("math_500", "Reasoning... \\boxed{1}", "1")
    assert score == 1.0


def test_math_dapo_binary_reward_switches_incorrect_score_to_zero():
    result = default_compute_score(
        "math_dapo",
        "Reasoning...\nAnswer: 7",
        "5",
        math_dapo_binary_reward=True,
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["pred"] == "7"


def test_load_reward_manager_forwards_reward_kwargs_to_default_scorer():
    config = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {"path": None},
                "reward_manager": {"source": "register", "name": "naive"},
                "sandbox_fusion": {"url": None},
                "reward_kwargs": {"math_dapo_binary_reward": True},
            }
        }
    )

    manager = load_reward_manager(config, tokenizer=None)
    result = manager.compute_score(
        data_source="math_dapo",
        solution_str="Reasoning...\nAnswer: 7",
        ground_truth="5",
        extra_info={},
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
