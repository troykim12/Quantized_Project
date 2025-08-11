#!/usr/bin/env python3
import os, sys
import tensorrt as trt

ONNX_PATH   = "int32_yolov8n.onnx"      # <-- 네 ONNX
ENGINE_PATH = "yolov8n_fp16.engine"
INPUT_NAME  = "images"
B, C, H, W  = 1, 3, 480, 640
WORKSPACE_MB = 1024

def set_workspace(config, builder, bytes_):
    """TRT 버전에 따라 workspace 설정 (8.0.1은 max_workspace_size 사용)"""
    # 최신(8.5+) 경로
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)
        print("[i] workspace via config.set_memory_pool_limit")
        return
    # 7.x~8.2 경로
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = bytes_
        print("[i] workspace via config.max_workspace_size")
        return
    # 아주 구버전(드물게)
    if hasattr(config, "set_max_workspace_size"):
        config.set_max_workspace_size(bytes_)
        print("[i] workspace via config.set_max_workspace_size")
        return
    print("[!] could not set workspace size (API not found)")

def main():
    if not os.path.exists(ONNX_PATH):
        sys.exit(f"[x] ONNX not found: {ONNX_PATH}")

    logger = trt.Logger(trt.Logger.INFO)
    EXPL = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    print(f"[i] TensorRT {trt.__version__}")

    with trt.Builder(logger) as builder, \
         builder.create_network(EXPL) as network, \
         trt.OnnxParser(network, logger) as parser:

        with open(ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit("[x] ONNX parse failed")

        in_names = [network.get_input(i).name for i in range(network.num_inputs)]
        input_name = INPUT_NAME if INPUT_NAME in in_names else in_names[0]
        print("[i] Using input tensor:", input_name)

        config = builder.create_builder_config()
        set_workspace(config, builder, WORKSPACE_MB * (1 << 20))

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[i] FP16 enabled")

        # 입력 shape 고정(동적이면 프로파일)
        try:
            network.get_input(0).shape = (B, C, H, W)
            print(f"[i] Set static input shape: {(B,C,H,W)}")
        except Exception as e:
            print("[i] Static shape set skipped:", e)
            prof = builder.create_optimization_profile()
            prof.set_shape(input_name, (B,C,H,W), (B,C,H,W), (B,C,H,W))
            config.add_optimization_profile(prof)

        print("[*] Building engine...")
        engine = builder.build_engine(network, config)
        if engine is None:
            sys.exit("[x] Engine build failed")

        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())
        print(f"[OK] Saved: {ENGINE_PATH}")

if __name__ == "__main__":
    main()
