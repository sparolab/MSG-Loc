#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import hashlib
import os
from std_msgs.msg import Header
import json
from ultralytics import YOLO

def to_python_native(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def ensure_writable_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.access(path, os.W_OK)

class ImageProcessorNode:
    def __init__(self):
        rospy.init_node('yolo_image_processor_node')
        default_assets_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Detection_assets")
        )
        self.assets_dir = os.path.abspath(os.path.expanduser(
            str(rospy.get_param("~assets_dir", default_assets_dir)).strip()
        ))
        local_model_path = os.path.join(self.assets_dir, "yolov8x-seg.pt")
        if os.path.isfile(local_model_path):
            model_path = local_model_path
        else:
            rospy.logwarn(
                f"Detection asset not found: {local_model_path}; "
                "letting Ultralytics resolve yolov8x-seg.pt."
            )
            model_path = "yolov8x-seg.pt"
        self.model = YOLO(model_path)

        self.image_pub = rospy.Publisher("/output_image_topic", Image, queue_size=10)
        self.mask_image_pub = rospy.Publisher("/mask_image_topic", Image, queue_size=10)
        self.conf_thres = rospy.get_param("~conf_thres", 0.1)
        self.iou_thres = rospy.get_param("~iou_thres", 0.6)
        self.max_det = rospy.get_param("~max_det", 16)

        # Global (default) camera parameters
        self.fx = rospy.get_param("~fx", None)
        self.fy = rospy.get_param("~fy", None)
        self.cx = rospy.get_param("~cx", None)
        self.cy = rospy.get_param("~cy", None)
        self.depth_factor_default = rospy.get_param("~depth_factor", 1000.0)

        self.bridge = CvBridge()

        # ------ Input paths/camera parameters: based on ROS params ------
        self.base_path = str(rospy.get_param("~base_path", "")).strip()
        self.assoc_path = str(rospy.get_param("~assoc_path", "")).strip()
        self.cam_info_path = str(rospy.get_param("~cam_info", "")).strip()
        self.output_dir = str(rospy.get_param("~output_dir", "")).strip()
        default_save_name = rospy.get_param("~sequence", "slam_detections.json")
        self.save_name = str(rospy.get_param("~save_name", default_save_name)).strip()

        if not self.base_path:
            raise ValueError("Missing required parameter: ~base_path")
        if not self.assoc_path:
            raise ValueError("Missing required parameter: ~assoc_path")
        if not self.cam_info_path:
            raise ValueError("Missing required parameter: ~cam_info")
        if not self.output_dir:
            raise ValueError("Missing required parameter: ~output_dir")
        if not self.save_name:
            raise ValueError("Invalid parameter: ~save_name must not be empty")
        if not self.save_name.lower().endswith(".json"):
            self.save_name += ".json"

        self.base_path = os.path.abspath(self.base_path)
        self.assoc_path = os.path.abspath(self.assoc_path)
        self.cam_info_path = os.path.abspath(self.cam_info_path)
        self.output_dir = os.path.abspath(self.output_dir)

        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"base_path directory not found: {self.base_path}")
        if not os.path.isfile(self.assoc_path):
            raise FileNotFoundError(f"association file not found: {self.assoc_path}")
        if not os.path.isfile(self.cam_info_path):
            raise FileNotFoundError(f"cam_info file not found: {self.cam_info_path}")

        cam_params = self._load_camera_param_from_yaml(self.cam_info_path)
        self.sequences = [("sequence_0", self.base_path, self.assoc_path, cam_params)]

        ensure_writable_dir(self.output_dir)
        self.output_json_path = os.path.join(self.output_dir, self.save_name)

    def _load_camera_param_from_yaml(self, cam_info_path):
        fs = cv2.FileStorage(cam_info_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open cam_info file: {cam_info_path}")

        def read_real(key):
            node = fs.getNode(key)
            if node.empty():
                raise KeyError(f"Missing '{key}' entry in cam_info: {cam_info_path}")
            return float(node.real())

        try:
            return {
                "fx": read_real("Camera.fx"),
                "fy": read_real("Camera.fy"),
                "cx": read_real("Camera.cx"),
                "cy": read_real("Camera.cy"),
                "depth_factor": read_real("DepthMapFactor"),
            }
        finally:
            fs.release()

    def _resolve_cam_params(self, cam_params):
        fx = cam_params.get("fx", self.fx)
        fy = cam_params.get("fy", self.fy)
        cx = cam_params.get("cx", self.cx)
        cy = cam_params.get("cy", self.cy)
        depth_factor = cam_params.get("depth_factor", self.depth_factor_default)
        if "depth_scale" in cam_params and cam_params["depth_scale"]:
            depth_factor = 1.0 / float(cam_params["depth_scale"])
        elif rospy.has_param("~depth_scale"):
            depth_factor = 1.0 / float(rospy.get_param("~depth_scale"))
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_factor": depth_factor}

    def load_images_from_folder(self, association_filename, base_folder):
        rgb_filenames = []
        depth_filenames = []
        timestamps = []
        with open(association_filename, 'r') as f:
            for raw in f:
                line = raw.split('#', 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                timestamp_rgb = parts[0]
                rgb_filename = parts[1]
                timestamp_depth = parts[2]
                depth_filename = parts[3]
                timestamps.append(float(timestamp_rgb))
                full_rgb_path = os.path.join(base_folder, rgb_filename)
                full_depth_path = os.path.join(base_folder, depth_filename)
                rgb_filenames.append(full_rgb_path)
                depth_filenames.append(full_depth_path)
        return rgb_filenames, depth_filenames, timestamps

    def _median_depth_and_center3d(self, mask_bool, depth_img, cam, bbox_center_xy=None):
        if mask_bool is None or mask_bool.size == 0:
            return None, (None, None), None

        raw_vals = depth_img[mask_bool]
        if raw_vals.size == 0:
            return None, (None, None), None
        valid = raw_vals[raw_vals > 0]
        if valid.size == 0:
            return None, (None, None), None

        z = float(np.median(valid)) / float(cam["depth_factor"])

        ys, xs = np.nonzero(mask_bool)
        if xs.size > 0:
            u = float(xs.mean())
            v = float(ys.mean())
        else:
            if bbox_center_xy is not None:
                u, v = float(bbox_center_xy[0]), float(bbox_center_xy[1])
            else:
                return z, (None, None), None

        if None in (cam["fx"], cam["fy"], cam["cx"], cam["cy"]):
            return z, (u, v), None

        x = (u - float(cam["cx"])) * z / float(cam["fx"])
        y = (v - float(cam["cy"])) * z / float(cam["fy"])
        return z, (u, v), [float(x), float(y), float(z)]

    def publish_detection(self, detections, header, original_image, depth, cam_params, masks=None):

        detections_json_list = []
        del original_image
        depth_h, depth_w = depth.shape[:2]

        if detections is not None and getattr(detections, "xyxy", None) is not None and len(detections.xyxy) != 0:
            bounding_box, classes, confidence_score, masks, classes_topk, confidence_topk = \
                self.remove_all_overlapping_data(
                    detections.xyxy,
                    detections.class_id,
                    detections.confidence,
                    masks,
                    threshold=0.45,
                    classes_topk=getattr(detections, "class_id_topk", None),
                    confidences_topk=getattr(detections, "confidence_topk", None),
                )

            label_color_map = {}
            for idx, (bbox, cls, conf) in \
                    enumerate(zip(bounding_box, classes, confidence_score)):
                mask = masks[idx] if idx < len(masks) else None
                cls_k = classes_topk[idx] if classes_topk is not None and idx < len(classes_topk) else None
                conf_k = confidence_topk[idx] if confidence_topk is not None and idx < len(confidence_topk) else None

                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                bbox_f = [float(x_center), float(y_center), float(w), float(h)]

                label = str(cls)
                if label not in label_color_map:
                    label_color_map[label] = self.generate_color_for_label(label)
                color = np.array(label_color_map[label], dtype=np.uint8)
                color_array = [color]
                color_array_np = np.array(color_array, dtype=np.uint8)
                score = [float(conf)]
                score_topk = [float(s) for s in np.array(conf_k).reshape(-1)] if conf_k is not None else score
                concept_topk = [str(float(c)) for c in np.array(cls_k).reshape(-1)] if cls_k is not None else [str(float(cls))]

                if mask is None:
                    # If there is no segmentation mask, fallback to bbox-based bool mask
                    x1i = max(0, min(int(np.floor(x1)), depth_w - 1))
                    y1i = max(0, min(int(np.floor(y1)), depth_h - 1))
                    x2i = max(0, min(int(np.ceil(x2)), depth_w))
                    y2i = max(0, min(int(np.ceil(y2)), depth_h))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    mask_bool = np.zeros((depth_h, depth_w), dtype=bool)
                    mask_bool[y1i:y2i, x1i:x2i] = True
                else:
                    mask_bool = mask
                    if isinstance(mask_bool, np.ndarray):
                        if mask_bool.ndim == 3:
                            mask_bool = np.squeeze(mask_bool)
                        mask_bool = mask_bool.astype(bool)
                    else:
                        mask_bool = np.array(mask_bool).astype(bool)
                    if mask_bool.shape != (depth_h, depth_w):
                        rospy.logwarn(f"Skip mismatched mask shape: {mask_bool.shape} vs depth {(depth_h, depth_w)}")
                        continue

                median_depth, (_, _), center_3d = self._median_depth_and_center3d(
                    mask_bool=mask_bool,
                    depth_img=depth,
                    cam=cam_params,
                    bbox_center_xy=(bbox_f[0], bbox_f[1]),
                )
                if median_depth is None or not np.isfinite(median_depth) or median_depth <= 0.0:
                    continue
                if median_depth >= 15:
                    continue

                detections_json_list.append({
                    "bbox_xywh": bbox_f,
                    "color_rgb": color_array_np.tolist(),
                    "concept": concept_topk,
                    "score": score_topk,
                    "median_depth": median_depth,
                    "center_3d": center_3d,
                })

        frame_json_entry = {
            "stamp": float(header.stamp.to_sec()),
            "frame_id": header.frame_id,
            "detections": detections_json_list
        }
        return frame_json_entry

    def compute_iou(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        xi1 = max(x1_min, x2_min)
        yi1 = max(y1_min, y2_min)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    def remove_all_overlapping_data(self, boxes, classes, confidences, masks, threshold, classes_topk=None, confidences_topk=None):
        keep = []                 # final kept indices
        width, height = 640, 480
        compensation = 5

        for i, box in enumerate(boxes):
            # ───── 0. Basic filter ───────────────────────────
            area = (box[2] - box[0]) * (box[3] - box[1])
            if (area < 300 or area > 0.3 * width * height or
                    self.is_near_boundary(box, width, height, compensation)):
                continue

            max_score_i = float(confidences[i])
            cls_i = int(classes[i])

            replace_idx = None                    # replacement target (lower score)
            skip        = False                   # higher score exists -> skip

            # ───── 1. Overlap check against existing keep ──────────────
            for k in keep:
                iou = self.compute_iou(box, boxes[k])
                if iou < 0.3:               # pass if not overlapping enough
                    continue

                same_cls = (cls_i == int(classes[k]))
                if iou > 0.6 or (same_cls and iou > 0.3):
                    if max_score_i <= float(confidences[k]):
                        skip = True              # a higher score already exists
                        break
                    else:
                        replace_idx = k          # k will be replaced
                        break

            if skip:
                continue
            if replace_idx is not None:
                keep.remove(replace_idx)
            keep.append(i)

        # ───── 2. Return index-based results ──────────────────
        boxes      = [boxes[i]      for i in keep]
        classes    = [classes[i]    for i in keep]
        confidences= [confidences[i]for i in keep]
        filtered_masks = [None for _ in keep]
        if masks is not None:
            try:
                filtered_masks = [masks[i] for i in keep]
            except Exception:
                filtered_masks = [None for _ in keep]

        filtered_classes_topk = None
        if classes_topk is not None:
            try:
                filtered_classes_topk = [classes_topk[i] for i in keep]
            except Exception:
                filtered_classes_topk = None

        filtered_confidences_topk = None
        if confidences_topk is not None:
            try:
                filtered_confidences_topk = [confidences_topk[i] for i in keep]
            except Exception:
                filtered_confidences_topk = None

        return (boxes, classes, confidences, filtered_masks, filtered_classes_topk, filtered_confidences_topk)

    def generate_color_for_label(self, label):
        hash_object = hashlib.md5(label.encode())
        hash_digest = hash_object.hexdigest()
        r = int(hash_digest[:2], 16)
        g = int(hash_digest[2:4], 16)
        b = int(hash_digest[4:6], 16)
        return (r, g, b)


    def is_near_boundary(self, box, width, height, compensation=5):
        if (box[0] < compensation or box[1] < compensation or
            box[2] > (width - compensation) or box[3] > (height - compensation)):
            return True
        return False

    def process_sequence(self, seq_name, base_folder, association_file, cam_params):
        cam = self._resolve_cam_params(cam_params)
        rgb_filenames, depth_filenames, timestamps = self.load_images_from_folder(association_file, base_folder)

        frames_json = []
        start_time = time.time()
        for i, rgb_file in enumerate(rgb_filenames):
            if rospy.is_shutdown():
                break
            frame_start_time = time.time()

            depth_file = depth_filenames[i]
            timestamp = timestamps[i]
            depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            cv_image = cv2.imread(rgb_file)
            if cv_image is None or depth_image is None:
                rospy.logwarn(f"[{seq_name}] Skip invalid image pair: {rgb_file}, {depth_file}")
                continue
            cv_image1 = cv_image.copy()

            header = Header()
            header.frame_id = "base_link"
            header.stamp = rospy.Time.from_sec(timestamp)

            results = self.model.predict(
                source=cv_image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,   # already capped, but an extra safety slice is fine
                augment=False,
                agnostic_nms=False,
                verbose=True,
                topk_cls=5
            )
           
            # YOLO returns a list of Results. For a single image, use index 0.
            if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                res = results[0]

                # Convert to numpy
                xyxy = res.boxes.xyxy.cpu().numpy()            # shape: [N, 4]
                cls  = res.boxes.cls.cpu().numpy().astype(int) # shape: [N]
                conf = res.boxes.conf.cpu().numpy()            # shape: [N]
                cls_topk = None
                conf_topk = None
                if hasattr(res.boxes, "cls_topk"):
                    cls_topk = res.boxes.cls_topk.cpu().numpy().astype(int)
                if hasattr(res.boxes, "conf_topk"):
                    conf_topk = res.boxes.conf_topk.cpu().numpy()
                
                # Small wrapper to match GroundingDINO-compatible interface
                class _DetWrap:
                    pass
                detections = _DetWrap()
                detections.xyxy = xyxy
                detections.class_id = cls
                detections.confidence = conf
                detections.class_id_topk = cls_topk
                detections.confidence_topk = conf_topk

                # If this is a segmentation model, pass masks too (otherwise None)
                masks = None
                if hasattr(res, "masks") and res.masks is not None and hasattr(res.masks, "data"):
                    # res.masks.data: [N, H, W] (torch.Tensor)
                    masks = res.masks.data.cpu().numpy().astype(bool)

                frame_json_entry = self.publish_detection(
                    detections, header, cv_image1, depth_image, cam, masks=masks
                )
            else:
                rospy.logwarn(f"[{seq_name}] No detections found.")
                frame_json_entry = self.publish_detection(
                    None, header, cv_image, depth_image, cam, masks=None
                )

            frames_json.append(frame_json_entry)

            frame_runtime = time.time() - frame_start_time
            elapsed_time = time.time() - start_time
            detection_count = len(frame_json_entry.get("detections", []))
            rospy.loginfo(
                f"Image {i+1}/{len(rgb_filenames)} processed. "
                f"Detected objects: {detection_count}, "
                f"Frame runtime: {frame_runtime:.2f}s, "
                f"Cumulative runtime: {elapsed_time:.2f}s"
            )

        return frames_json

    def run(self):
        all_json = {"sequences": []}

        try:
            for (seq_name, base_folder, association_file, cam_params) in self.sequences:
                rospy.loginfo(f"Processing sequence: {seq_name}")
                frames_json = self.process_sequence(seq_name, base_folder, association_file, cam_params)
                seq_record = {"frames": frames_json}
                all_json["sequences"].append(seq_record)

            out_dir = os.path.dirname(self.output_json_path)
            ensure_writable_dir(out_dir)
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(to_python_native(all_json), f, indent=2, ensure_ascii=False)
            rospy.loginfo(f"[done] Saved all sequences to: {self.output_json_path}")

            rospy.signal_shutdown("All sequences processed and saved.")
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down by user request.")

if __name__ == '__main__':
    try:
        node = ImageProcessorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
