#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
import tf_conversions
import tf2_ros
import torch
from std_msgs.msg import Float32MultiArray
import time
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
import hashlib
from groundingdino.util.inference import Model
import torchvision
import os
from std_msgs.msg import Header
import json


def to_python_native(obj):
    import numpy as np
    import torch
    if isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return to_python_native(obj.detach().cpu().numpy())
    return obj

def ensure_writable_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.access(path, os.W_OK)


class ImageProcessorNode:
    def __init__(self):
        rospy.init_node('yolo_image_processor_node')

        self.grounding_dino_model = Model(
            model_config_path = "/root/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path = "/root/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            device="cuda"
        )
        self.classes = ["object"]
        self.model_type = "tap_vit_b"
        default_assets_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Detection_assets")
        )
        self.assets_dir = os.path.abspath(os.path.expanduser(
            str(rospy.get_param("~assets_dir", default_assets_dir)).strip()
        ))
        self.checkpoint = os.path.join(self.assets_dir, "tap_vit_b_v1_1.pkl")
        concept_weights = os.path.join(self.assets_dir, "coco_80.pkl")
        label_path = os.path.join(self.assets_dir, "coco.txt")

        for asset_path in (self.checkpoint, concept_weights, label_path):
            if not os.path.isfile(asset_path):
                raise FileNotFoundError(f"Detection asset not found: {asset_path}")

        self.tap_model = model_registry[self.model_type](checkpoint=self.checkpoint)
        self.tap_model.concept_projector.reset_weights(concept_weights)
        self.concept_topk = 5
        self.tap_model.text_decoder.reset_cache(max_batch_size=16)

        # ==== Preload label mapping ====
        (self.label_str_to_float_idx,
         self.float_idx_to_label_str,
         self.coco_label_list) = self._load_label_float_map(label_path)
        rospy.loginfo(f"[LabelMap] loaded {len(self.coco_label_list)} labels from {label_path}")


        # ------ ROS Pub/Sub ------
        self.image_pub = rospy.Publisher("/output_image_topic", Image, queue_size=10)
        self.mask_image_pub = rospy.Publisher("/mask_image_topic", Image, queue_size=10)

        # ------ Parameters ------
        self.box_thres = rospy.get_param("~box_thres", 0.14)
        self.text_thres = rospy.get_param("~text_thres", 0.14)
        self.max_det = rospy.get_param("~max_det", 16)
        # self.conf_thres = rospy.get_param("~conf_thres", 0.1)
        # self.iou_thres = rospy.get_param("~iou_thres", 0.6)
        # Global (default) camera parameters: used when per-sequence values are missing
        self.fx = rospy.get_param("~fx", None)
        self.fy = rospy.get_param("~fy", None)
        self.cx = rospy.get_param("~cx", None)
        self.cy = rospy.get_param("~cy", None)
        # meters = raw / depth_factor (e.g., RealSense mm -> 1000.0)
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

    # ----------------- Utilities -----------------
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

    def _load_label_float_map(self, path):
        names = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    names.append(s)
        except Exception as e:
            rospy.logwarn(f"[LabelMap] fail to read {path}: {e}")

        name_to_fstr = {}
        for i, n in enumerate(names):
            fstr = str(float(i))  # '0.0', '1.0', ...
            name_to_fstr[n] = fstr
            name_to_fstr[n.lower()] = fstr

        fstr_to_name = {v: k for k, v in name_to_fstr.items() if k == k.lower()}
        return name_to_fstr, fstr_to_name, names

    def to_float_idx_strings(self, labels, strict=False, sentinel="-1.0"):
        out = []
        for lb in labels:
            key = lb if isinstance(lb, str) else str(lb)
            fstr = (self.label_str_to_float_idx.get(key)
                    or self.label_str_to_float_idx.get(key.lower()))
            if fstr is None:
                fstr = sentinel if strict else key
            out.append(str(fstr))
        return out

    def _resolve_cam_params(self, cam_params):
        fx = cam_params.get("fx", self.fx)
        fy = cam_params.get("fy", self.fy)
        cx = cam_params.get("cx", self.cx)
        cy = cam_params.get("cy", self.cy)
        depth_factor = cam_params.get("depth_factor", self.depth_factor_default)
        # depth_scale compatibility (apply reciprocal if present)
        if "depth_scale" in cam_params and cam_params["depth_scale"]:
            depth_factor = 1.0 / float(cam_params["depth_scale"])
        elif rospy.has_param("~depth_scale"):
            depth_factor = 1.0 / float(rospy.get_param("~depth_scale"))
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_factor": depth_factor}

    def load_images_from_folder(self, association_filename, base_folder):
        rgb_filenames, depth_filenames, timestamps = [], [], []
        with open(association_filename, 'r') as f:
            for raw in f:
                line = raw.split('#', 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                timestamp_rgb = parts[0]
                rgb_filename   = parts[1]
                timestamp_depth= parts[2]
                depth_filename = parts[3]
                timestamps.append(float(timestamp_rgb))
                full_rgb_path   = os.path.join(base_folder, rgb_filename)
                full_depth_path = os.path.join(base_folder, depth_filename)
                rgb_filenames.append(full_rgb_path)
                depth_filenames.append(full_depth_path)
        return rgb_filenames, depth_filenames, timestamps

    def _median_depth_and_center3d(self, mask_bool, depth_img, cam, bbox_center_xy=None):
        """
        mask_bool: (H,W) bool mask
        depth_img: (H,W) uint16/uint32 depth (raw)
        cam: {"fx","fy","cx","cy","depth_factor"}
        Returns: (median_depth_m or None, (u,v), [X,Y,Z] or None)
        """
        if mask_bool is None or mask_bool.size == 0:
            return None, (None, None), None

        raw_vals = depth_img[mask_bool]
        if raw_vals.size == 0:
            return None, (None, None), None
        valid = raw_vals[raw_vals > 0]
        if valid.size == 0:
            return None, (None, None), None

        # meters = median(raw) / depth_factor
        Z = float(np.median(valid)) / float(cam["depth_factor"])

        ys, xs = np.nonzero(mask_bool)
        if xs.size > 0:
            u = float(xs.mean()); v = float(ys.mean())
        else:
            if bbox_center_xy is not None:
                u, v = float(bbox_center_xy[0]), float(bbox_center_xy[1])
            else:
                return Z, (None, None), None

        if None in (cam["fx"], cam["fy"], cam["cx"], cam["cy"]):
            return Z, (u, v), None

        X = (u - float(cam["cx"])) * Z / float(cam["fx"])
        Y = (v - float(cam["cy"])) * Z / float(cam["fy"])
        return Z, (u, v), [float(X), float(Y), float(Z)]

    # ----------------- Publish & Visualization -----------------
    def publish_detection(self, detections, header, original_image, depth,
                          masks=None, concepts=None, scores=None, cam=None):
        if cam is None:
            cam = self._resolve_cam_params({})
        detections_json_list = []
        del original_image
        depth_h, depth_w = depth.shape[:2]

        if detections is not None and getattr(detections, "xyxy", None) is not None and len(detections.xyxy) != 0:
            bounding_box, classes, confidence_score, masks, concepts, scores = \
                self.remove_all_overlapping_data(
                    detections.xyxy, detections.class_id, detections.confidence,
                    masks, concepts, scores, threshold=0.45
                )

            label_color_map = {}
            num_dets = len(bounding_box)
            for idx in range(num_dets):
                bbox = bounding_box[idx]
                cls = classes[idx]
                conf = confidence_score[idx]
                mask = masks[idx]
                concept = concepts[idx]
                score = scores[idx]

                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                bbox_f = [float(x_center), float(y_center), float(w), float(h)]

                # Fixed color per label
                color_array = []
                for j in range(len(concept)):
                    label = str(concept[j])
                    if label not in label_color_map:
                        label_color_map[label] = self.generate_color_for_label(label)
                    color = np.array(label_color_map[label], dtype=np.uint8)
                    color_array.append(color)

                # Mask processing: normalize to (H, W) bool
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
                color_array_np = np.array(color_array, dtype=np.uint8)  # [K,3]

                score = list(map(float, score))
                mapped = self.to_float_idx_strings(concept)

                # median depth & 3D center
                median_depth_m, (u, v), center3d = self._median_depth_and_center3d(
                    mask_bool=mask_bool,
                    depth_img=depth,
                    cam=cam,
                    bbox_center_xy=(bbox_f[0], bbox_f[1])
                )
                if median_depth_m is None or not np.isfinite(median_depth_m):
                    continue
                if median_depth_m >= 15.0:
                    continue

                # Write JSON record
                det_entry = {
                    "bbox_xywh": bbox_f,
                    "color_rgb": color_array_np.tolist(),
                    "concept": mapped,
                    "score": score,
                    "median_depth": median_depth_m,
                    "center_3d": center3d if center3d is not None else None
                }
                detections_json_list.append(det_entry)

        frame_json_entry = {
            "stamp": float(header.stamp.to_sec()),
            "frame_id": header.frame_id,
            "detections": detections_json_list
        }
        return frame_json_entry

    # ----------------- Misc utilities -----------------
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

    def remove_all_overlapping_data(self, boxes, classes, confidences, masks,
                                    concepts, scores, threshold):
        keep = []
        width, height = 640, 480
        compensation = 5
        for i, box in enumerate(boxes):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if (area < 300 or area > 0.3 * width * height or
                    self.is_near_boundary(box, width, height, compensation)):
                continue
            max_score_i = scores[i][0]
            cls_i = concepts[i][0]
            replace_idx = None
            skip = False
            for k in keep:
                iou = self.compute_iou(box, boxes[k])
                if iou < 0.3:
                    continue
                same_cls = (cls_i == concepts[k][0])
                if iou > 0.6 or (same_cls and iou > 0.3):
                    if max_score_i <= scores[k][0]:
                        skip = True
                        break
                    else:
                        replace_idx = k
                        break
            if skip:
                continue
            if replace_idx is not None:
                keep.remove(replace_idx)
            keep.append(i)
        boxes       = [boxes[i]       for i in keep]
        classes     = [classes[i]     for i in keep]
        confidences = [confidences[i] for i in keep]
        masks       = [masks[i]       for i in keep]
        concepts    = [concepts[i]    for i in keep]
        scores      = [scores[i]      for i in keep]
        
        return (boxes, classes, confidences, masks, concepts, scores)

    def generate_color_for_label(self, label):
        hash_object = hashlib.md5(label.encode())
        hash_digest = hash_object.hexdigest()
        r = int(hash_digest[:2], 16)
        g = int(hash_digest[2:4], 16)
        b = int(hash_digest[4:6], 16)
        return (r, g, b)

    def process_with_tap_model(self, detections, header, cv_image):
        del header
        with torch.no_grad():
            img_list, img_scales = im_rescale(cv_image, scales=[1024], max_size=1024)
            input_size, original_size = img_list[0].shape[:2], cv_image.shape[:2]
            img_batch = im_vstack(img_list, fill_value=self.tap_model.pixel_mean_value, size=(1024, 1024))
            inputs = self.tap_model.get_inputs({"img": img_batch})
            inputs.update(self.tap_model.get_features(inputs))

            batch_points = np.zeros((len(detections.xyxy), 2, 3), dtype=np.float32)
            for i in range(len(detections.xyxy)):
                bbox_xyxy = detections.xyxy[i]
                batch_points[i, 0, :2] = bbox_xyxy[:2]
                batch_points[i, 1, :2] = bbox_xyxy[2:]
                batch_points[i, 0, 2] = 2
                batch_points[i, 1, 2] = 3
            inputs["points"] = batch_points
            inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")

            outputs = self.tap_model.get_outputs(inputs)

            iou_pred = outputs["iou_pred"].cpu().numpy()
            point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
            rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
            mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
            mask_pred = outputs["mask_pred"]
            masks = mask_pred[mask_index]
            masks = self.tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
            masks = masks[..., :input_size[0], :input_size[1]]
            masks = self.tap_model.upscale_masks(masks, original_size).gt(0).cpu().numpy()

            sem_embeds = outputs["sem_embeds"]
            concepts, scores = self.tap_model.predict_concept(
                sem_embeds[mask_index], k=self.concept_topk
            )

            return masks, concepts, scores

    def is_near_boundary(self, box, width, height, compensation=5):
        if (box[0] < compensation or box[1] < compensation or
            box[2] > (width - compensation) or box[3] > (height - compensation)):
            return True
        return False

    # ----------------- Main processing -----------------
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

            detections = self.grounding_dino_model.predict_with_classes(
                image=cv_image,
                classes=self.classes,
                box_threshold=self.box_thres,
                text_threshold=self.text_thres
            )

            # YOLO returns a list of Results. For a single image, use index 0.
            if detections is not None and getattr(detections, "xyxy", None) is not None and len(detections.xyxy) > 0:
                top_indices = np.argsort(-detections.confidence)[:self.max_det]
                detections.xyxy = detections.xyxy[top_indices]
                detections.class_id = detections.class_id[top_indices]
                detections.confidence = detections.confidence[top_indices]
                
                masks, concepts, scores = \
                    self.process_with_tap_model(detections, header, cv_image)

                frame_json_entry = self.publish_detection(
                    detections, header, cv_image1, depth_image,
                    masks, concepts, scores, cam=cam
                )
            else:
                rospy.logwarn(f"[{seq_name}] No detections found.")
                frame_json_entry = self.publish_detection(
                    None, header, cv_image, depth_image, cam=cam
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

# ----------------- Entry point -----------------
if __name__ == '__main__':
    try:
        node = ImageProcessorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
