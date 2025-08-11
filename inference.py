#!/usr/bin/env python3
import time, argparse, os
import numpy as np, cv2
import tensorrt as trt
import pycuda.autoinit  # keep
import pycuda.driver as cuda

# ---------- utils ----------
def letterbox_bgr(img, new_shape=(480,640), color=114):
    h,w = img.shape[:2]
    nh,nw = new_shape
    r = min(nw/w, nh/h)
    rw, rh = int(w*r), int(h*r)
    pad = np.full((nh,nw,3), color, np.uint8)
    pad[:rh,:rw] = cv2.resize(img, (rw,rh))
    x = pad[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.0  # CHW RGB
    return x, r, (rw, rh)

def nms_xyxy(boxes, scores, iou_th=0.45, conf_th=0.25):
    keep=[]
    idx = np.where(scores >= conf_th)[0]
    boxes, scores = boxes[idx], scores[idx]
    order = scores.argsort()[::-1]
    while order.size>0:
        i = order[0]; keep.append(idx[i])
        if order.size==1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2-xx1); h=np.maximum(0.0, yy2-yy1)
        inter = w*h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_o = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_th]
    return np.array(keep, dtype=np.int32)

# ---------- TensorRT runner (TRT 8.0 API) ----------
class TRTRunner:
    def __init__(self, engine_path, H=480, W=640):
        self.H, self.W = H, W
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.num_bindings = self.engine.num_bindings
        self.binding_addrs = [0]*self.num_bindings
        self.host = {}
        self.dev = {}
        self.input_index = None
        self.output_indices = []

        # 미리 입력 shape 지정(정적 네트워크 가정)
        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            if is_input:
                shape = (1,3,H,W)
                # 동적이면 여기서 set_binding_shape 필요
                try:
                    self.context.set_binding_shape(i, shape)
                except Exception:
                    pass
                self.input_index = i
            else:
                # 출력 shape 얻기 (정적일 가능성 높음)
                shape = tuple(self.engine.get_binding_shape(i))
                if -1 in shape:
                    # 입력 shape 지정 후 컨텍스트에서 다시 얻기
                    shape = tuple(self.context.get_binding_shape(i))
                self.output_indices.append(i)

            # 버퍼 할당
            size = int(np.prod(shape))
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.host[i] = host
            self.dev[i] = dev
            self.binding_addrs[i] = int(dev)

        assert self.input_index is not None, "No input binding found"

    def infer(self, bgr):
        x, _, _ = letterbox_bgr(bgr, (self.H, self.W))
        inp = self.host[self.input_index]
        inp[...] = x[None, ...]
        cuda.memcpy_htod(self.dev[self.input_index], inp)

        t0 = time.time()
        self.context.execute_v2(self.binding_addrs)
        ms = (time.time()-t0)*1000

        outs = {}
        for i in self.output_indices:
            cuda.memcpy_dtoh(self.host[i], self.dev[i])
            outs[self.engine.get_binding_name(i)] = self.host[i].astype(np.float32, copy=True)
        return outs, ms

# ---------- postprocess (nms=False ONNX/TRT) ----------
def decode_yolov8(outputs, orig_shape, input_hw=(480,640), conf_th=0.25, class_agnostic=False):
    """
    Handles raw YOLOv8 head or (rare) already-(N,6) output.
    Returns: (N,6) [x1,y1,x2,y2,conf,cls]
    """
    arr=None
    # 우선 (N,6) 또는 (1,N,6) 찾기 → NMS 포함 엔진일 때
    for v in outputs.values():
        if v.ndim==2 and v.shape[1] in (6,7):
            arr = v if v.shape[1]==6 else v[:,1:]
            break
        if v.ndim==3 and v.shape[0]==1 and v.shape[2] in (6,7):
            arr = v[0] if v.shape[2]==6 else v[0][:,1:]
            break
    if arr is not None:
        # 입력스케일 → 원본크기 보정
        H,W = orig_shape[:2]; inH,inW = input_hw
        boxes = arr[:,:4].copy()
        boxes[:,[0,2]] *= (W/inW); boxes[:,[1,3]] *= (H/inH)
        out = np.concatenate([boxes, arr[:,4:5], arr[:,5:6]], 1)
        return out.astype(np.float32)

    # raw head 처리
    raw=None
    for v in outputs.values():
        if v.ndim==3 and v.shape[0]==1 and (v.shape[1]>=6 or v.shape[2]>=6):
            raw=v; break
    if raw is None:
        return np.zeros((0,6), np.float32)

    # (N,C)로 통일
    pred = raw[0].transpose(1,0) if raw.shape[1] < raw.shape[2] else raw[0]
    xywh = pred[:,:4]; obj = pred[:,4:5]; cls = pred[:,5:]
    if cls.size==0: cls = np.ones_like(obj)
    if class_agnostic:
        scores = obj.squeeze(); cls_id = np.zeros_like(scores, np.int32)
    else:
        cls_id = cls.argmax(1)
        scores = (obj * cls[np.arange(cls.shape[0]), cls_id][:,None]).squeeze()
    x,y,w,h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    x1,y1 = x-w/2, y-h/2; x2,y2 = x+w/2, y+h/2
    boxes = np.stack([x1,y1,x2,y2],1)
    keep = nms_xyxy(boxes, scores, 0.45, conf_th)
    boxes = boxes[keep]; scores = scores[keep]; cls_id = cls_id[keep]
    H,W = orig_shape[:2]; inH,inW = input_hw
    boxes[:,[0,2]] *= (W/inW); boxes[:,[1,3]] *= (H/inH)
    return np.concatenate([boxes, scores[:,None], cls_id[:,None].astype(np.float32)],1)

def draw(img, dets, names=None, only_person=False):
    for x1,y1,x2,y2,conf,cls in dets:
        if only_person and int(cls)!=0:  # 0=person
            continue
        p1=(int(x1),int(y1)); p2=(int(x2),int(y2))
        cv2.rectangle(img,p1,p2,(0,255,0),2)
        lab = f"{int(cls)} {conf:.2f}" if not names else f"{names[int(cls)]} {conf:.2f}"
        cv2.putText(img, lab, (p1[0], max(0,p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
    return img

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help=".engine path")
    ap.add_argument("--image", help="image path")
    ap.add_argument("--cam", type=int, help="webcam index")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--names", help="classes.txt (optional)")
    ap.add_argument("--person-only", action="store_true")
    args = ap.parse_args()

    names=None
    if args.names and os.path.exists(args.names):
        with open(args.names) as f:
            names=[l.strip() for l in f if l.strip()]

    trt_runner = TRTRunner(args.engine, H=480, W=640)

    if args.image:
        img = cv2.imread(args.image); assert img is not None
        outs, ms = trt_runner.infer(img)
        dets = decode_yolov8(outs, img.shape, (480,640), conf_th=args.conf)
        print(f"Detections: {len(dets)} | infer {ms:.1f} ms")
        vis = draw(img, dets, names, only_person=args.person_only)
        cv2.imshow("result", vis); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    cam = cv2.VideoCapture(args.cam if args.cam is not None else 0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    t0=time.time(); f=0; ms_last=0
    while True:
        ok, frame = cam.read()
        if not ok: break
        outs, ms = trt_runner.infer(frame); ms_last=ms
        dets = decode_yolov8(outs, frame.shape, (480,640), conf_th=args.conf)
        frame = draw(frame, dets, names, only_person=args.person_only)
        f+=1
        if time.time()-t0>=1.0:
            fps=f/(time.time()-t0); f=0; t0=time.time()
            cv2.putText(frame, f"FPS ~ {fps:.1f} (infer {ms_last:.1f} ms)", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("YOLOv8 TRT (TRT8 raw+NMS)", frame)
        if cv2.waitKey(1)==27: break
    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
