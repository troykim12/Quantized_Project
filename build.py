#!/usr/bin/env python3
# Jetson Nano (TRT 8.0.x) friendly ONNX -> TensorRT FP16 builder
# - Workspace 512MiB
# - builder_optimization_level = 2
# - Tactic sources: CUBLAS(+LT) only  (cuDNN 제외로 메모리 절약)
# - Static input: 1x3xH x W  (default H=480, W=640)

import os, sys, argparse
import tensorrt as trt

def set_workspace(config, bytes_):
    """TRT 8.0.x API: use max_workspace_size"""
    if hasattr(config, "set_memory_pool_limit"):
        # newer API (8.5+). Nano 8.0.1에는 없음
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)
        print("[i] workspace via config.set_memory_pool_limit")
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = bytes_
        print("[i] workspace via config.max_workspace_size")
    elif hasattr(config, "set_max_workspace_size"):
        config.set_max_workspace_size(bytes_)
        print("[i] workspace via config.set_max_workspace_size")
    else:
        print("[!] could not set workspace size (API not found)")

def set_tactics_cublas_only(config):
    # TRT 8.0.x 지원. 없으면 자동 스킵.
    if hasattr(config, "set_tactic_sources"):
        config.set_tactic_sources(trt.TacticSource.CUBLAS | trt.TacticSource.CUBLAS_LT)
        print("[i] tactic sources = CUBLAS | CUBLAS_LT (cuDNN 제외)")
    else:
        print("[i] set_tactic_sources not available on this TRT")

def build_engine(onnx_path, engine_path, input_name, H, W,
                 workspace_mb=512, fp16=True, opt_level=2):
    logger = trt.Logger(trt.Logger.INFO)
    EXPL = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    print(f"[i] TensorRT {trt.__version__}")
    print(f"[*] ONNX: {onnx_path}")

    with trt.Builder(logger) as builder, \
         builder.create_network(EXPL) as network, \
         trt.OnnxParser(network, logger) as parser:

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit("[x] ONNX parse failed")

        in_names = [network.get_input(i).name for i in range(network.num_inputs)]
        if not in_names:
            sys.exit("[x] No inputs found in ONNX")
        if input_name and input_name in in_names:
            in_tensor_name = input_name
        else:
            in_tensor_name = in_names[0]
        print("[i] Using input tensor:", in_tensor_name)

        config = builder.create_builder_config()
        set_workspace(config, workspace_mb * (1 << 20))

        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = int(opt_level)
            print(f"[i] builder_optimization_level = {opt_level}")

        set_tactics_cublas_only(config)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[i] FP16 enabled")

        # 입력 shape 고정 (동적이면 프로파일 생성)
        B, C = 1, 3
        try:
            network.get_input(0).shape = (B, C, H, W)
            print(f"[i] Set static input shape: {(B,C,H,W)}")
        except Exception as e:
            print("[i] Static shape set skipped:", e)
            prof = builder.create_optimization_profile()
            prof.set_shape(in_tensor_name, (B,C,H,W), (B,C,H,W), (B,C,H,W))
            config.add_optimization_profile(prof)
            print("[i] Added optimization profile (min=opt=max)")

        print("[*] Building engine... (this can take minutes on Nano)")
        engine = builder.build_engine(network, config)
        if engine is None:
            sys.exit("[x] Engine build failed")

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"[OK] Saved engine: {engine_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="yolov8n_sim.onnx")
    ap.add_argument("--engine", default="yolov8n_fp16.engine")
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--workspace", type=int, default=512)
    ap.add_argument("--opt-level", type=int, default=2)
    ap.add_argument("--fp32", action="store_true", help="force FP32")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.onnx):
        sys.exit(f"[x] ONNX not found: {args.onnx}")
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        input_name=args.input_name,
        H=args.height, W=args.width,
        workspace_mb=args.workspace,
        fp16=not args.fp32,
        opt_level=args.opt_level,
    )
