#!/usr/bin/env python3
# ONNX -> TensorRT FP16 engine (explicit batch, 1x3x480x640)
import os, sys
import tensorrt as trt

ONNX_PATH   = "yolov8n_sim.onnx"        # 입력 ONNX
ENGINE_PATH = "yolov8n_fp16.engine"     # 저장할 엔진 경로
INPUT_NAME  = "images"                  # ONNX 입력 텐서명(다르면 아래에서 자동 감지)
B, C, H, W  = 1, 3, 480, 640            # 고정 입력 크기
WORKSPACE_MB = 1024                     # trtexec --workspace=1024

def main():
    if not os.path.exists(ONNX_PATH):
        sys.exit(f"[x] ONNX not found: {ONNX_PATH}")

    logger = trt.Logger(trt.Logger.INFO)
    EXPL = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    print(f"[i] TensorRT {trt.__version__}")
    print(f"[*] Loading ONNX: {ONNX_PATH}")

    with trt.Builder(logger) as builder, \
         builder.create_network(EXPL) as network, \
         trt.OnnxParser(network, logger) as parser:

        with open(ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit("[x] ONNX parse failed")

        # 입력 텐서 이름/개수 안내
        in_names = [network.get_input(i).name for i in range(network.num_inputs)]
        print("[i] ONNX inputs:", in_names)
        input_name = INPUT_NAME if INPUT_NAME in in_names else in_names[0]
        print(f"[i] Using input tensor: {input_name}")

        # Builder config
        config = builder.create_builder_config()
        # TRT 8+: memory pool API, 하위버전 대비 예외 처리
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1 << 20))
        except Exception:
            builder.max_workspace_size = WORKSPACE_MB * (1 << 20)

        # FP16 플래그
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[i] FP16 enabled")
        else:
            print("[!] Fast FP16 not available, building without FP16")

        # 네트워크 입력 shape 고정 세팅
        # 모델이 동적 입력이면 프로파일을 만들고, 이미 정적이면 shape 설정을 스킵
        try:
            tensor = network.get_input(0)
            tensor.shape = (B, C, H, W)
            print(f"[i] Set static input shape: {(B,C,H,W)}")
        except Exception as e:
            print("[i] Static shape set skipped:", e)
            print("[i] Creating optimization profile for dynamic input...")
            profile = builder.create_optimization_profile()
            mn = (B, C, H, W)
            op = (B, C, H, W)
            mx = (B, C, H, W)
            profile.set_shape(input_name, mn, op, mx)
            config.add_optimization_profile(profile)

        print("[*] Building engine... (this may take a while)")
        engine = builder.build_engine(network, config)
        if engine is None:
            sys.exit("[x] Engine build failed")

        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())
        print(f"[OK] Saved engine: {ENGINE_PATH}")

if __name__ == "__main__":
    main()
