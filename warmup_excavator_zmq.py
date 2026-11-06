"""kernel 后端「丢弃式预热」：触发 CUDA Graph 录制，并复位 RTC 状态。

为什么需要：
    PI05FlowMatchingRTCInference 在**首次请求**时才 materialize 权重 + 录制
    CUDA Graph（A100 上约 13.5s，Jetson 更久），而录制期的输出是脏的；同时
    predict_action 会无条件把这次结果写进 rtc_state['prev_actions']。若不处理，
    客户端第一段真实请求会拿到垃圾动作，且带着假前缀继续推理。

做法：
    serve 起来后另开终端跑本脚本 —— 先发两次合成请求（第二次走 graph.replay，
    应为毫秒~百毫秒级），再调 reset 清掉 prev_actions。

用法：
    /home/zmc/.conda/envs/fluxvla/bin/python warmup_excavator_zmq.py \
        [--host 127.0.0.1] [--port 5555] [--unnorm-key private]
"""
import argparse
import io
import time

import msgpack
import numpy as np
import zmq

# 与训练一致的三路分辨率（dataset 侧还会 ResizeImagesWithPad 到 224，尺寸只影响耗时）
VIEWS = {
    'cam_head': (594, 960, 3),
    'cam_wrist': (480, 640, 3),
    'cam_side': (756, 1344, 3),
}
PROMPT = 'pick up the coffee and place it on the left table'


def encode_obs(obs: dict) -> bytes:
    encoded = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            encoded[key] = {'__ndarray__': True, 'data': stream.getvalue()}
        else:
            encoded[key] = value
    return msgpack.packb(encoded)


def call(sock, endpoint: str, data: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    sock.send(msgpack.packb({'endpoint': endpoint, 'data': data}))
    raw = sock.recv()
    elapsed = time.perf_counter() - t0
    response = msgpack.unpackb(raw, raw=False)
    if not isinstance(response, dict):
        raise RuntimeError(f'响应不是字典: {type(response)!r}')
    if 'error' in response:
        raise RuntimeError(f'服务端错误: {response["error"]}')
    return response, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--unnorm-key', default='private')
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()

    obs = {name: np.zeros(shape, dtype=np.uint8) for name, shape in VIEWS.items()}
    obs['qpos'] = np.zeros(4, dtype=np.float32)
    obs['prompt'] = PROMPT
    payload = encode_obs(obs)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 300000)
    sock.setsockopt(zmq.SNDTIMEO, 300000)
    sock.connect(f'tcp://{args.host}:{args.port}')
    print(f'[warmup] 连接 tcp://{args.host}:{args.port}', flush=True)

    try:
        for i in range(1, args.rounds + 1):
            response, elapsed = call(
                sock, 'predict_action',
                {'obs_data': payload, 'unnorm_key': args.unnorm_key})
            infer = float(response.get('infer_time', 0.0))
            note = '（含 CUDA Graph 录制，输出丢弃）' if i == 1 else ''
            print(f'[warmup] #{i}  infer={infer * 1000:.1f}ms  往返={elapsed:.2f}s {note}',
                  flush=True)

        call(sock, 'reset', {})
        print('[warmup] rtc prev_actions 已复位，可以开始真机', flush=True)
    finally:
        sock.close()
        ctx.term()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
