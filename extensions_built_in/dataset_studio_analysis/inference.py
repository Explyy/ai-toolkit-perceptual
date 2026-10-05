"""Actual local inference, imported only by an explicitly enabled Linux GPU job."""
from pathlib import Path
import os
import importlib.metadata
import numpy as np
from PIL import Image, ImageOps
from .recognition import recognition_normalization
from .artifacts import configuration, digest, VERSION

class Analyzer:
    def __init__(self, models):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import cv2
        import onnxruntime as ort
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, RTDetrForObjectDetection, VitPoseForPoseEstimation
        if not torch.cuda.is_available():
            raise RuntimeError('Dedicated NVIDIA GPU unavailable')
        self.cv2, self.torch = cv2, torch
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.manifest, self.config = configuration()
        self.normalization=recognition_normalization(models['face'] / 'glintr100.onnx')
        self.face = ort.InferenceSession(str(models['face'] / 'glintr100.onnx'), providers=['CPUExecutionProvider'])
        if self.face.get_inputs()[0].shape[-3:] != [3,112,112] or self.face.get_inputs()[0].type!='tensor(float)':
            raise ValueError('Unexpected pinned AuraFace input interface')
        self.detector = cv2.FaceDetectorYN.create(str(models['faceDetector'] / 'face_detection_yunet_2023mar.onnx'), '', (640,640), .7, .3, 5000)
        self.depth_processor = AutoImageProcessor.from_pretrained(models['depth'], local_files_only=True)
        self.depth = AutoModelForDepthEstimation.from_pretrained(models['depth'], local_files_only=True, use_safetensors=True).to('cuda').eval()
        self.person_processor = AutoImageProcessor.from_pretrained(models['person'], local_files_only=True)
        self.person = RTDetrForObjectDetection.from_pretrained(models['person'], local_files_only=True, use_safetensors=True).to('cuda').eval()
        self.pose_processor = AutoImageProcessor.from_pretrained(models['pose'], local_files_only=True)
        self.pose = VitPoseForPoseEstimation.from_pretrained(models['pose'], local_files_only=True, use_safetensors=True).to('cuda').eval()
        self.runtime = {'device':'cuda:0 + onnx-cpu', 'torch':torch.__version__, 'transformers':importlib.metadata.version('transformers'), 'onnxruntime':ort.__version__}

    def analyze(self, file: Path, output: Path, sha: str):
        torch, cv2 = self.torch, self.cv2
        if digest(file) != sha:
            raise ValueError('Original SHA changed before inference')
        with Image.open(file) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        width,height = image.size
        rgb = np.asarray(image)
        # YuNet runs on a bounded letterbox-free resize; map detections back to oriented pixels.
        scale = min(640/width,640/height)
        small = cv2.resize(rgb[:,:,::-1], (max(1,round(width*scale)),max(1,round(height*scale))))
        sx,sy=small.shape[1]/width,small.shape[0]/height
        self.detector.setInputSize((small.shape[1],small.shape[0]))
        _,detections = self.detector.detect(small)
        faces=[]
        reference=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=np.float64)
        if detections is not None:
            detections=sorted(detections,key=lambda f:(-float(f[2]*f[3]),float(f[0]),float(f[1])))[:32]
            for f in detections:
                # YuNet: right eye, left eye, nose, right mouth, left mouth in image coordinates.
                points=np.array(f[4:14],dtype=np.float64).reshape(5,2)/np.array([sx,sy])
                # Deterministic least-squares similarity alignment, no randomized RANSAC.
                rows=[]; targets=[]
                for (x,y),(u,v) in zip(points,reference):
                    rows += [[x,-y,1,0],[y,x,0,1]]; targets += [u,v]
                a,b,tx,ty=np.linalg.lstsq(np.asarray(rows),np.asarray(targets),rcond=None)[0]
                aligned=cv2.warpAffine(rgb,np.array([[a,-b,tx],[b,a,ty]],dtype=np.float32),(112,112))
                tensor=((aligned.astype(np.float32)-self.normalization['mean'])/self.normalization['std']).transpose(2,0,1)[None]
                embedding=self.face.run(None,{self.face.get_inputs()[0].name:tensor})[0].reshape(-1)
                norm=float(np.linalg.norm(embedding))
                if norm <= 0 or len(embedding)!=512:
                    raise ValueError('ArcFace embedding invalid')
                x,y,w,h=[float(n) for n in f[:4]]
                box=np.clip([x/small.shape[1],y/small.shape[0],(x+w)/small.shape[1],(y+h)/small.shape[0]],0,1).tolist()
                if box[2]>box[0] and box[3]>box[1]:
                    faces.append({'box':box,'confidence':float(np.clip(f[14],0,1)),'embedding':(embedding/norm).tolist(),'landmarks':np.clip(points/np.array([width,height]),0,1).tolist()})
        with torch.inference_mode():
            inputs=self.person_processor(images=image,return_tensors='pt').to('cuda')
            prediction=self.person(**inputs)
            detected=self.person_processor.post_process_object_detection(prediction,target_sizes=torch.tensor([[height,width]],device='cuda'),threshold=.5)[0]
            boxes=[]
            for box,label,score in zip(detected['boxes'],detected['labels'],detected['scores']):
                if self.person.config.id2label[int(label)]=='person':
                    b=np.clip(box.cpu().numpy(),[0,0,0,0],[width,height,width,height])
                    if b[2]>b[0] and b[3]>b[1]: boxes.append((b,float(score)))
            boxes.sort(key=lambda b:(-float((b[0][2]-b[0][0])*(b[0][3]-b[0][1])),float(b[0][0]),float(b[0][1])))
            boxes=boxes[:32]
            persons=[]
            if boxes:
                xywh=np.array([b for b,_ in boxes]); xywh[:,2:]-=xywh[:,:2]
                inputs=self.pose_processor(images=image,boxes=[xywh],return_tensors='pt').to('cuda')
                predicted=self.pose(**inputs)
                poses=self.pose_processor.post_process_pose_estimation(predicted,boxes=[xywh])[0]
                for (box,confidence),pose in zip(boxes,poses):
                    points=pose['keypoints'].cpu().numpy()/np.array([width,height])
                    scores=pose['scores'].cpu().numpy()
                    persons.append({'box':(box/np.array([width,height,width,height])).tolist(),'confidence':confidence,'keypoints':np.column_stack([np.clip(points,0,1),np.clip(scores,0,1)]).tolist()})
            inputs=self.depth_processor(images=image,return_tensors='pt').to('cuda')
            depth=self.depth(**inputs).predicted_depth
            depth=torch.nn.functional.interpolate(depth.unsqueeze(1),size=(128,128),mode='bicubic',align_corners=False)[0,0].cpu().numpy()
        if not np.isfinite(depth).all(): raise ValueError('Non-finite depth map')
        lo,hi=np.percentile(depth,[2,98])
        normalized=np.clip((depth-lo)/(hi-lo),0,1) if hi>lo else np.zeros_like(depth)
        Image.fromarray(np.round(normalized*65535).astype(np.uint16)).save(output/'depth.png')
        grid=cv2.resize(normalized,(8,8),interpolation=cv2.INTER_AREA).reshape(-1).tolist()
        return {'version':VERSION,'sha':sha,'config':self.config,'faces':faces,'persons':persons,'depth':{'grid':grid,'map':'depth.png','sha':digest(output/'depth.png'),'relative':True},'runtime':self.runtime}
