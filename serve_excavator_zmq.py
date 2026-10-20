"""在 Thor 上起一个与 encode_predict_request 协议匹配的 ZMQ 推理服务。

为什么不用 scripts/zmq_inference_server.py：
  build_zmq_eval_server_from_config 注册的 _handle_predict(observation, ...)
  与 PolicyServer 的 msgpack 分发 `handler(**data)` 不匹配（请求里是 obs_data），
  会直接报 "got an unexpected keyword argument 'obs_data'"。
  zmq_server.create_server() 的 predict_action(obs_data, unnorm_key) 与
  encode_predict_request 是一对，且同样支持 dataset/denormalize/last_raw_state。

用法:
  python serve_excavator_zmq.py [--config ...] [--host 0.0.0.0] [--port 5555] [--device cuda:0]
"""
import argparse
import json
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file

from fluxvla.engines import (build_dataset_from_cfg, build_transform_from_cfg,
                             build_vla_from_cfg)
from fluxvla.engines.runners.serving.zmq_server import create_server
from fluxvla.engines.utils.torch_utils import \
    configure_inference_attention_defaults

DEFAULT_CFG = ('configs/pi05/'
               'pi05_paligemma_electric_excavator_lora_finetune.py')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=DEFAULT_CFG)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ckpt', default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_inference_attention_defaults()

    cfg = Config.fromfile(args.config)
    section = cfg.inference
    ckpt = args.ckpt or section['ckpt_path']
    print(f'[serve] ckpt = {ckpt}', flush=True)

    vla = build_vla_from_cfg(cfg.inference_model)
    state = load_file(ckpt, device='cpu')
    missing, unexpected = vla.load_state_dict(state, strict=False)
    print(f'[serve] weights loaded  missing={len(missing)} '
          f'unexpected={len(unexpected)}', flush=True)
    vla = vla.to(args.device).eval()

    # PrivateInferenceDataset 必填 norm_stats / model_path；
    # ROS 版由 ros_server._prepare_dataset_config 注入，这里手工补。
    ds_cfg = dict(section['dataset'])
    stats_path = Path(ckpt).parent.parent / 'dataset_statistics.json'
    ds_cfg.setdefault('norm_stats', str(stats_path))
    ds_cfg.setdefault('model_path', str(Path.cwd()))
    print(f'[serve] stats = {stats_path}  exists={stats_path.exists()}',
          flush=True)
    dataset = build_dataset_from_cfg(ds_cfg)
    with stats_path.open('r', encoding='utf-8') as stream:
        vla.norm_stats = json.load(stream)
    denorm_cfg = dict(section['denormalize_action'])
    denorm_cfg['norm_stats'] = str(stats_path)
    denorm = build_transform_from_cfg(denorm_cfg)
    task_suite = cfg.get('themis', {}).get('transport', {}).get(
        'unnorm_key', 'private')

    server = create_server(
        vla,
        dataset=dataset,
        denormalize_action=denorm,
        task_suite_name=task_suite,
        host=args.host,
        port=args.port,
        device=args.device,
        mixed_precision_dtype=torch.bfloat16,
    )
    print(f'[serve] ready on tcp://{args.host}:{args.port} '
          f'(task_suite={task_suite})', flush=True)
    server.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
