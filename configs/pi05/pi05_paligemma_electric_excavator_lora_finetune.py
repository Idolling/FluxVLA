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

model = dict(
    type='PI05FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    use_lora=True,
    lora_rank=256,
    lora_alpha=512,
    lora_dropout=0.0,
    lora_target_modules=[  # q/v/k/o_proj, gate/up/down_proj, projector, out_proj, fc1, fc2
        'q_proj','v_proj','k_proj','o_proj',
        'gate_proj','up_proj','down_proj',
        'projector.projector','out_proj','fc1','fc2'],
    modules_to_save=['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out'],
    n_action_steps=50,                       # ★ 必须 == action_window_size
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    # Match the OpenPI-aligned RoboCasa flow-matching objective.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/pi05_base/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=4,                        # ★ 7 -> 4
    # Supervise all padded model dimensions, as in OpenPI.
    loss_action_dim=32,                      # ★ 与 UR3 同（也对 padding 维监督）；想只监督 4 维就写 4
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        auto_compute_statistics=dict(        # ★ 替代 dataset_statistics/dataset_statistics_path
            profile='absolute',              # ★ PROFILES 里没有挖机，借 absolute 壳
            state_key='observation.state',
            action_key='actions',            # ★ 复数
            delta_mask=[True, True, True, True],   # ★ 与下面 RelativeActions.mask 必须一致
        ),
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'actions'],   # ★ 去掉 timestamp / eepose
        statistic_name='private',
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    '/home/zmc/work/electric_excavator_datasets/lerobot0930',  # noqa: E501
                ],
                action_key='actions',            # ★ 复数
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=['observation.images.head',
                                 'observation.images.wrist'],       # ★ 你的两个相机
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(type='RelativeActions', mask=[True] * 4),         # ★ delta
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=None,
                        state_dim=None,
                        state_key='proprio',
                        action_key='action',
                        norm_type='quantile',
                        output_dtype='float32'),
                    dict(type='PreparePromptWithState'),
                    dict[str, str | dict[str, str]](
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'checkpoints/pi05_base',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(type='PadStatesAndActions', model_action_dim=32),
                    dict(
                        type='ResizeImagesWithPad',
                        height=224,
                        width=224,
                        backend='pil'),
                    dict(type='SimpleNormalizeImages'),
                    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
                ],
                action_window_size=50,
                window_start_idx=0,
                supervise_terminal_padding=True)
        ]))

runner = dict(
    type='DDPTrainRunner',
    max_epochs=None,        # ★★ 必须显式写
    max_steps=20_000,
    # 8 samples/GPU x 4 GPUs x 2 accumulation steps = global batch 64.
    seed=42,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    # BF16 compute with FP32 sharded master parameters and reductions.
    collator=dict(
        type='DictCollator',
        keys=['states', 'timestamp', 'images', 'img_masks',
              'lang_tokens', 'lang_masks', 'actions', 'action_masks'],  # ★ 去掉 observation.eepose
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        decay_steps=30000,
        min_lr=2.5e-6),
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'checkpoints/pi05_base',
        # special_tokens={'pad_token': '<PAD>'}
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1),
    enable_gradient_checkpointing=True,        # ← 开（libero 也是 True）
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    save_iter_interval=4000,      # 默认 10000 → 20000 步只存 10 次
    max_keep_ckpts=1,             # 默认 2
    save_epoch_interval=99999,        # 默认 1
    )

inference = dict(
    type='URInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    seed=7,
    task_descriptions={
        '1': 'grasp the stopper of the dark-colored wide-mouth bottle',
        '2': 'place the bottle stopper upside down on the tabletop',
        '3': 'grasp the body of the dark-colored wide-mouth bottle',
        '4':
        'pour the liquid in the dark-colored wide-mouth bottle into the erlenmeyer flask',  # noqa: E501
        '5': 'put the dark-colored wide-mouth bottle back on the tabletop',
        '6': 'grasp the measuring cylinder',
        '7':
        'pour the liquid in the measuring cylinder into the erlenmeyer flask',
        '8': 'put the measuring cylinder back on the tabletop',
        '9': 'grasp the neck of the erlenmeyer flask',
        '10': 'shake the erlenmeyer flask',
        '11': 'place the erlenmeyer flask back on the tabletop',
    },
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict[str, str | dict[str, str]](
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'checkpoints/pi05_base',
                    # special_tokens={'pad_token': '<PAD>'}
                )),
            dict(type='PadStatesAndActions', model_action_dim=32),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        norm_type='quantile',
        action_dim=7,
        delta_action_mask=[True] * 6 + [False],
    ),
    action_chunk=50,
    operator=dict(
        type='UROperator',
        image_encoding='rgb8',
        command_mode='joint',
        img_left_topic='/wrist_camera/color/image_raw',
        img_front_topic='/front_camera/color/image_raw',
        joint_state_topic='/joint_states',
        gripper_state_topic='/gripper/position',
        cartesian_pose_topic='/arm/tcp_pose',
        joint_cmd_topic='/cmd/servoj',
        cartesian_cmd_topic='/cmd/servol',
        gripper_cmd_topic='/cmd/gripper',
        use_depth_image=False))
