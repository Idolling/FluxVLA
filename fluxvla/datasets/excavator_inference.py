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
"""Excavator inference dataset.

``PrivateInferenceDataset`` feeds ``qpos`` to the model but never exposes it as
``last_raw_state``.  ``FluxVLAROSPolicy`` reads ``dataset.last_raw_state`` to
put the current raw joint angles into the denormalization context, and
``DenormalizeDeltaAction`` refuses to run without it (delta actions cannot be
turned into absolute joint targets otherwise).

This subclass only adds that side effect; all preprocessing is inherited so the
serving pipeline stays byte-identical to training.

Note: the class name intentionally contains ``PrivateInferenceDataset`` because
``ros_server._prepare_dataset_config`` keys on that substring to inject
``model_path``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fluxvla.engines import DATASETS

from .parquet_dataset import PrivateInferenceDataset


@DATASETS.register_module()
class ExcavatorPrivateInferenceDataset(PrivateInferenceDataset):
    """PrivateInferenceDataset that also records the raw robot state."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        qpos = data.get('qpos')
        self.last_raw_state = (
            None if qpos is None else
            np.asarray(qpos, dtype=np.float32).reshape(-1).copy())
        return super().__call__(data)
