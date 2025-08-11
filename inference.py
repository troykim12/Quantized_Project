#!/usr/bin/env python3
import time, argparse, os
import numpy as np, cv2
import tensorrt as trt
import pycuda.autoinit  # noqa
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

# ---------- TensorRT runner ----------
class TRT:
    def __init__(self, engine_path, in_name="images", H=480, W=640):
        self.H, self.W = H, W
        log = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(log) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = in_name
        if self.engine.get_tensor_mode(in_name) == trt.TensorIOMode.INPUT:
            self.ctx.set_input_shape(in_name, (1,3,H,W))
        self.bindings=[]; self.host={}; self.dev={}; self.names=[]
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_iotensor_name(i); self.names.append(name)
            is_in = self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT
            shape = (1,3,H,W) if is_in else tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            h = np.empty(shape, dtype=dtype); d = cuda.mem_alloc(h.nbytes)
            self.host[name]=h; self.dev[name]=d; self.bindings.append(int(d))

    def infer(self, bgr):
        x, _, _ = letterbox_bgr(bgr, (self.H,self.W))
        self.host[self.in_name][...] = x[None,...]
        cuda.memcpy_htod(self.dev[self.in_name], self.host[self.in_name])
        t0=time.time(); self.ctx.execute_v2(self.bindings); ms=(time.time()-t0)*1000
        outs={}
        for name in self.names:
            if name==self.in_name: continue
            cuda.memcpy_dtoh(self.host[name], self.dev[name])
            outs[name]=self.host[name].copy()
        return outs, ms

# ---------- postprocess for YOLOv8 raw head (nms=False) ----------
def decode_yolov8_raw(outputs, orig_shape, input_hw=(480,640), conf_th=0.25, class_agnostic=False):
    """
    Accepts any of:
      (1, N, 84/85)  or  (1, 84/85, N)
    Returns dets: [x1,y1,x2,y2,conf,cls_id]
    """
    arr=None
    for v in outputs.values():
        if v.ndim==3 and v.shape[0]==1 and (v.shape[2]>=6 or v.shape[1]>=6):
            arr=v; break
    if arr is None: return np.zeros((0,6), np.float32)

    # unify to (N, C)
    if arr.shape[1] < arr.shape[2]:
        pred = arr[0].transpose(1,0)  # (N, C)
    else:
        pred = arr[0]                 # (N, C)

    # YOLOv8 typical C = 4(xywh)+1(obj)+nc
    xywh = pred[:, :4]
    obj  = pred[:, 4:5]
    cls  = pred[:, 5:]
    if cls.size == 0:  # some exports: C=6 (single class)
        cls = np.ones_like(obj)

    if class_agnostic:
        scores = obj.squeeze()
        cls_id = np.zeros_like(scores, dtype=np.int32)
    else:
        cls_id = cls.argmax(1)
        scores = (obj * cls[np.arange(cls.shape[0]), cls_id][:,None]).squeeze()

    # xywh -> xyxy (in input scale)
    x,y,w,h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
    x1,y1 = x - w/2, y - h/2
    x2,y2 = x + w/2, y + h/2
    boxes = np.stack([x1,y1,x2,y2], 1)

    # NMS
    keep = nms_xyxy(boxes, scores, iou_th=0.45, conf_th=conf_th)
    boxes = boxes[keep]; scores = scores[keep]; cls_id = cls_id[keep]

    # scale to original frame
    H,W = orig_shape[:2]; inH,inW = input_hw
    boxes[:,[0,2]] *= (W/inW); boxes[:,[1,3]] *= (H/inH)

    dets = np.concatenate([boxes, scores[:,None], cls_id[:,None].astype(np.float32)], 1)
    return dets

def draw(img, dets, names=None, only_person=False):
    for x1,y1,x2,y2,conf,cls in dets:
        if only_person and int(cls)!=0:  # 0=person in COCO
            continue
        p1=(int(x1),int(y1)); p2=(int(x2),int(y2))
        cv2.rectangle(img,p1,p2,(0,255,0),2)
        lab = f"{int(cls)} {conf:.2f}" if not names else f"{names[int(cls)]} {conf:.2f}"
        cv2.putText(img, lab, (p1[0], max(0,p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
    return img

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT engine path (nms=False)")
    ap.add_argument("--image", help="image path")
    ap.add_argument("--cam", type=int, help="webcam index")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--names", help="classes.txt (optional)")
    ap.add_argument("--person-only", action="store_true")
    args = ap.parse_args()

    names = None
    if args.names and os.path.exists(args.names):
        with open(args.names, "r") as f:
            names=[l.strip() for l in f if l.strip()]

    trt_runner = TRT(args.engine, in_name=args.input_name, H=480, W=640)

    if args.image:
        img = cv2.imread(args.image); assert img is not None
        outs, ms = trt_runner.infer(img)
        dets = decode_yolov8_raw(outs, img.shape, (480,640), conf_th=args.conf)
        print(f"Detections: {len(dets)} | infer {ms:.1f} ms")
        vis = draw(img, dets, names, only_person=args.person_only)
        cv2.imshow("result", vis); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    cam = cv2.VideoCapture(args.cam if args.cam is not None else 0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    t0=time.time(); f=0; ms_last=0
    while True:
        ok, frame = cam.read()
        if not ok: break
        outs, ms = trt_runner.infer(frame); ms_last=ms
        dets = decode_yolov8_raw(outs, frame.shape, (480,640), conf_th=args.conf)
        frame = draw(frame, dets, names, only_person=args.person_only)
        f+=1
        if time.time()-t0>=1.0:
            fps = f/(time.time()-t0); f=0; t0=time.time()
            cv2.putText(frame, f"FPS ~ {fps:.1f} (infer {ms_last:.1f} ms)", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("YOLOv8 TRT (raw+NMS)", frame)
        if cv2.waitKey(1)==27: break
    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
