# Copyright 2026 Limx Dynamics
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

from fluxvla.engines.utils.heterogeneous_runtime import \
    import_heterogeneous_runtime_symbols
try:  # flash_attn 在 Jetson(Thor) 上不可用；pi05/paligemma 不依赖 eagle
    from .eagle import EagleBackbone, EagleInferenceBackbone  # noqa: F401, F403
except ImportError as _eagle_import_error:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(
        f'eagle backbone disabled (missing optional dependency: '
        f'{_eagle_import_error}); safe to ignore for pi05 inference.')
    EagleBackbone = None
    EagleInferenceBackbone = None
from .outputs import VLMBackboneOutput  # noqa: F401
from .outputs import normalize_vlm_backbone_output  # noqa: F401

import_heterogeneous_runtime_symbols(
    __name__,
    globals(),
    {
        'cosmos25': ['Cosmos25Backbone'],
        'florence2': ['Florence2Backbone'],
        'groot_n17_qwen3_backbone': ['GrootN17Qwen3Backbone'],
        'paligemma': ['PaliGemma'],
        'qwen2_5_vl': ['QWen2_5VL'],
        'smolvlm': ['SmolVLMBackbone'],
        'qwen3_vl': ['Qwen3VL'],
        'wan_backbone': ['WanBaseBackbone'],
        'wan21_backbone': ['Wan21Backbone'],
        'wan22_backbone': ['Wan22Backbone'],
        'cosmos3': ['Cosmos3MoTBackbone'],
    },
    runtime_missing_names=['transformers.models.qwen3_vl'],
)
