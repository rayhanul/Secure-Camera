"""Server-side existing-object prediction for GCL reconfiguration.

This module owns three responsibilities:

1. Store only real object observations in a bounded history per
   ``(camera_id, camera_location, track_id)``.
2. Convert that history into a normalized tensor for a GRU model.
3. Fill missing properties of omitted or partial existing objects.
4. Optionally predict the next frame continuously, even when complete object
   data is received.

A trained GRU checkpoint is optional. If no checkpoint is supplied, the
predictor uses a robust median-slope motion model. Predictions are never added
back to history, which prevents recursive prediction drift.
"""

from collections import Counter, deque
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import os

import numpy as np
import torch
from torch import nn


BASE_FEATURE_NAMES = (
    "cx",
    "cy",
    "width",
    "height",
    "distance_m",
    "bearing_deg",
    "confidence",
    "delta_time",
)
MODEL_INPUT_SIZE = len(BASE_FEATURE_NAMES) * 2  # values + validity masks
MODEL_OUTPUT_SIZE = 6  # delta cx, cy, width, height, distance, bearing


def parse_timestamp(value, fallback: float) -> float:
    """Parse numeric or ISO timestamps without using server-arrival time."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                iso_value = value.replace("Z", "+00:00")
                return datetime.fromisoformat(iso_value).timestamp()
            except ValueError:
                return fallback
    return fallback


def history_to_model_tensor(
    history: Sequence[Dict],
    history_size: int,
    frame_width: int,
    frame_height: int,
    fps: float,
    max_distance_m: float = 500.0,
    max_bearing_deg: float = 45.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert one track history to ``[batch, time, features]``.

    Missing numerical values are stored as zero and accompanied by a validity
    mask, allowing the model to distinguish missing data from a real zero.
    Histories shorter than ``history_size`` are left-padded with zeros.
    """
    history_size = max(int(history_size), 2)
    frame_width = max(int(frame_width), 1)
    frame_height = max(int(frame_height), 1)
    fps = max(float(fps), 0.001)
    max_distance_m = max(float(max_distance_m), 0.001)
    max_bearing_deg = max(float(max_bearing_deg), 0.001)

    rows: List[np.ndarray] = []
    previous_timestamp = None

    for observation in list(history)[-history_size:]:
        timestamp = _as_float(observation.get("timestamp"))
        if timestamp is None:
            timestamp = 0.0 if previous_timestamp is None else previous_timestamp

        if previous_timestamp is None:
            delta_time_in_frames = 0.0
        else:
            delta_time_in_frames = max(
                0.0,
                min(5.0, (timestamp - previous_timestamp) * fps),
            )

        raw_values = (
            _as_float(observation.get("cx")),
            _as_float(observation.get("cy")),
            _as_float(observation.get("width")),
            _as_float(observation.get("height")),
            _as_float(observation.get("distance_m")),
            _as_float(observation.get("bearing_deg")),
            _as_float(observation.get("confidence")),
            delta_time_in_frames,
        )
        scales = (
            frame_width,
            frame_height,
            frame_width,
            frame_height,
            max_distance_m,
            max_bearing_deg,
            1.0,
            1.0,
        )

        values = []
        masks = []
        for value, scale in zip(raw_values, scales):
            valid = value is not None and np.isfinite(value)
            masks.append(1.0 if valid else 0.0)
            values.append(float(value) / scale if valid else 0.0)

        rows.append(
            np.asarray(values + masks, dtype=np.float32)
        )
        previous_timestamp = timestamp

    pad_count = history_size - len(rows)
    if pad_count > 0:
        padding = [
            np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)
            for _ in range(pad_count)
        ]
        rows = padding + rows

    if not rows:
        rows = [
            np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)
            for _ in range(history_size)
        ]

    tensor = torch.tensor(np.stack(rows), dtype=torch.float32).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class ObjectMotionGRU(nn.Module):
    """GRU that predicts normalized changes in six object-state values."""

    def __init__(
        self,
        input_size: int = MODEL_INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, MODEL_OUTPUT_SIZE),
        )

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(model_input)
        return self.output_head(sequence_output[:, -1, :])


class ExistingObjectPredictor:
    """Store numerical histories and predict from lightweight descriptors.

    This class never needs the complete network packet, object crop, ReID
    embedding, encrypted content, or processed image. The server adapter sends
    only sanitized numerical observations when updating history and only a
    track ID (plus optional lightweight fields) when requesting inference.
    """

    def __init__(
        self,
        fps: float = 25.0,
        history_size: int = 10,
        max_missed_frames: int = 3,
        max_radial_speed_kmh: float = 180.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
        model_path: Optional[str] = None,
        model_device: str = "auto",
        model_hidden_size: int = 128,
        model_num_layers: int = 2,
        max_distance_m: float = 500.0,
        max_bearing_deg: float = 45.0,
        history_store_path: Optional[str] = None,
        continuous_prediction: bool = False,
    ):
        self.fps = max(float(fps), 0.001)
        self.history_size = max(int(history_size), 2)
        self.max_missed_frames = max(int(max_missed_frames), 1)
        self.max_radial_speed_kmh = max(float(max_radial_speed_kmh), 1.0)
        self.default_frame_width = max(int(frame_width), 1)
        self.default_frame_height = max(int(frame_height), 1)
        self.max_distance_m = max(float(max_distance_m), 0.001)
        self.max_bearing_deg = max(float(max_bearing_deg), 0.001)
        self.history_store_path = history_store_path
        self.continuous_prediction = bool(continuous_prediction)

        self.track_histories: Dict[Tuple[Tuple[str, str], str], deque] = {}
        self.latest_frame_by_camera: Dict[Tuple[str, str], int] = {}
        # Predictions are indexed by camera, target frame, and track ID so
        # they can be compared with the real object when that frame arrives.
        self.prediction_cache: Dict[
            Tuple[Tuple[str, str], int, str], Dict
        ] = {}
        self.last_comparisons: List[Dict] = []

        self.model = None
        self.model_device = self._resolve_device(model_device)
        if model_path:
            self.load_model(
                model_path=model_path,
                hidden_size=model_hidden_size,
                num_layers=model_num_layers,
            )

        if self.history_store_path:
            history_directory = os.path.dirname(self.history_store_path)
            if history_directory:
                os.makedirs(history_directory, exist_ok=True)

    @property
    def model_enabled(self) -> bool:
        return self.model is not None

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        requested = str(requested or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        return torch.device(requested)

    def load_model(
        self,
        model_path: str,
        hidden_size: int = 128,
        num_layers: int = 2,
    ) -> bool:
        """Load a trained checkpoint; retain robust fallback on failure."""
        try:
            checkpoint = torch.load(model_path, map_location=self.model_device)
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get(
                    "model_state_dict",
                    checkpoint.get("state_dict", checkpoint),
                )
                config = checkpoint.get("model_config", {})
                hidden_size = int(config.get("hidden_size", hidden_size))
                num_layers = int(config.get("num_layers", num_layers))

            model = ObjectMotionGRU(
                hidden_size=hidden_size,
                num_layers=num_layers,
            ).to(self.model_device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.model = model
            print(
                f"Object prediction GRU loaded from {model_path} "
                f"on {self.model_device}"
            )
            return True
        except Exception as error:
            self.model = None
            print(
                "Could not load object prediction GRU; using robust history "
                f"fallback: {error}"
            )
            return False

    @staticmethod
    def _first_value(mapping: Dict, *names, default=None):
        for name in names:
            value = mapping.get(name)
            if value is not None:
                return value
        return default

    @staticmethod
    def _track_key(track_id) -> str:
        return str(track_id)

    def _valid_bbox(self, obj: Dict) -> Optional[List[float]]:
        bbox = obj.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        coords = [_as_float(value) for value in bbox]
        if any(value is None for value in coords):
            return None
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            return None
        return coords

    def _missing_prediction_fields(self, obj: Dict) -> List[str]:
        missing = []
        if self._valid_bbox(obj) is None:
            missing.append("bbox")
        if _first_float(obj, "distance_m", "distance") is None:
            missing.append("distance_m")
        if _first_float(obj, "bearing_deg", "bearing_angle") is None:
            missing.append("bearing_deg")
        if not _first_text(obj, "direction", "motion_direction"):
            missing.append("direction")
        if _first_float(obj, "speed_kmh", "speed") is None:
            missing.append("speed_kmh")
        return missing

    def missing_prediction_fields(self, observation: Dict) -> List[str]:
        """Return the required motion fields absent from a sanitized record."""
        return self._missing_prediction_fields(observation)

    def has_history(
        self,
        camera_id,
        camera_location,
        track_id,
    ) -> bool:
        camera_key = (str(camera_id), str(camera_location))
        key = (camera_key, self._track_key(track_id))
        return bool(self.track_histories.get(key))

    def has_prior_history(
        self,
        camera_id,
        camera_location,
        track_id,
        current_frame_id: int,
    ) -> bool:
        """Return True only when the track was observed before this frame.

        A track first observed in ``current_frame_id`` is a new object.  It may
        be stored as an actual observation, but it must not be predicted until
        it appears again in a later frame.
        """
        camera_key = (str(camera_id), str(camera_location))
        key = (camera_key, self._track_key(track_id))
        frame_id = _as_int(current_frame_id, -1)
        return any(
            _as_int(observation.get("frame_id"), frame_id) < frame_id
            for observation in self.track_histories.get(key, ())
        )

    def history_length(
        self,
        camera_id,
        camera_location,
        track_id,
    ) -> int:
        """Return the number of real observations stored for one track."""
        camera_key = (str(camera_id), str(camera_location))
        key = (camera_key, self._track_key(track_id))
        return len(self.track_histories.get(key, ()))

    def known_track_ids(self, camera_id, camera_location) -> List[str]:
        """Return IDs currently known for one camera without exposing history."""
        camera_key = (str(camera_id), str(camera_location))
        return [
            track_key
            for (track_camera, track_key), history in self.track_histories.items()
            if track_camera == camera_key and history
        ]

    def _normalize_observation(
        self,
        obj: Dict,
        frame_id: int,
        timestamp: float,
    ) -> Optional[Dict]:
        track_id = self._first_value(
            obj, "track_id", "object_track_id", "object_id"
        )
        bbox = self._valid_bbox(obj)
        if track_id is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "track_id": track_id,
            "class_name": self._first_value(
                obj, "class_name", "class", default="unknown"
            ),
            "class_id": obj.get("class_id"),
            "confidence": _as_float(obj.get("confidence"), 0.0),
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "width": x2 - x1,
            "height": y2 - y1,
            "distance_m": _first_float(obj, "distance_m", "distance"),
            "bearing_deg": _first_float(obj, "bearing_deg", "bearing_angle"),
            "speed_kmh": _first_float(obj, "speed_kmh", "speed"),
            "direction": _first_text(
                obj, "direction", "motion_direction"
            ) or "unknown",
        }

    def _observe_actual_objects(
        self,
        camera_key: Tuple[str, str],
        frame_id: int,
        timestamp: float,
        objects: Iterable[Dict],
    ) -> None:
        persisted_observations = []
        for obj in objects:
            observation = self._normalize_observation(
                obj, frame_id, timestamp
            )
            if observation is None:
                continue
            key = (camera_key, self._track_key(observation["track_id"]))
            history = self.track_histories.setdefault(
                key, deque(maxlen=self.history_size)
            )
            if history and frame_id <= history[-1]["frame_id"]:
                continue
            history.append(observation)

            persistent_record = dict(observation)
            persistent_record.update(
                {
                    "camera_id": camera_key[0],
                    "camera_location": camera_key[1],
                    "is_observed": True,
                    "predicted": False,
                }
            )
            persisted_observations.append(persistent_record)

        self._persist_actual_observations(persisted_observations)

    def _persist_actual_observations(self, observations: Sequence[Dict]) -> None:
        """Append actual model-input rows to JSONL for later GRU training."""
        if not self.history_store_path or not observations:
            return
        try:
            with open(self.history_store_path, "a", encoding="utf-8") as output:
                for observation in observations:
                    output.write(
                        json.dumps(observation, default=str, separators=(",", ":"))
                    )
                    output.write("\n")
        except OSError as error:
            print(
                f"Could not store actual prediction-model input data at "
                f"{self.history_store_path}: {error}"
            )

    @staticmethod
    def _robust_extrapolate(
        history: Sequence[Dict],
        field: str,
        target_timestamp: float,
        maximum_absolute_rate: Optional[float] = None,
    ) -> Tuple[Optional[float], float, float]:
        points = []
        for observation in history:
            value = _as_float(observation.get(field))
            timestamp = _as_float(observation.get("timestamp"))
            if value is not None and timestamp is not None:
                points.append((timestamp, value))
        if not points:
            return None, 0.0, 0.0
        if len(points) == 1:
            return points[-1][1], 0.0, 0.0

        slopes = []
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                delta_time = points[second][0] - points[first][0]
                if delta_time > 0:
                    slopes.append(
                        (points[second][1] - points[first][1]) / delta_time
                    )
        if not slopes:
            return points[-1][1], 0.0, 0.0

        raw_rate = float(np.median(slopes))
        used_rate = raw_rate
        if maximum_absolute_rate is not None:
            used_rate = float(
                np.clip(raw_rate, -maximum_absolute_rate, maximum_absolute_rate)
            )
        horizon = max(0.0, target_timestamp - points[-1][0])
        return points[-1][1] + used_rate * horizon, used_rate, raw_rate

    def _model_state(
        self,
        history: Sequence[Dict],
        frame_width: int,
        frame_height: int,
        horizon_frames: int,
    ) -> Optional[Dict]:
        if self.model is None or not history:
            return None
        last = history[-1]
        required = (
            "cx", "cy", "width", "height", "distance_m", "bearing_deg"
        )
        if any(_as_float(last.get(field)) is None for field in required):
            return None

        model_input = history_to_model_tensor(
            history=history,
            history_size=self.history_size,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=self.fps,
            max_distance_m=self.max_distance_m,
            max_bearing_deg=self.max_bearing_deg,
            device=self.model_device,
        )
        with torch.no_grad():
            delta = self.model(model_input)[0].detach().cpu().numpy()

        horizon_frames = max(int(horizon_frames), 1)
        scales = np.asarray(
            [
                frame_width,
                frame_height,
                frame_width,
                frame_height,
                self.max_distance_m,
                self.max_bearing_deg,
            ],
            dtype=np.float32,
        )
        last_state = np.asarray(
            [last[field] for field in required], dtype=np.float32
        )
        predicted = last_state + delta * scales * horizon_frames
        return {
            "cx": float(predicted[0]),
            "cy": float(predicted[1]),
            "width": max(2.0, float(predicted[2])),
            "height": max(2.0, float(predicted[3])),
            "distance_m": max(0.0, float(predicted[4])),
            "bearing_deg": float(predicted[5]),
            "model_input_shape": list(model_input.shape),
        }

    def _predict_track(
        self,
        history: Sequence[Dict],
        target_frame_id: int,
        target_timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Dict]:
        last = history[-1]
        frames_since = target_frame_id - last["frame_id"]
        if frames_since <= 0 or frames_since > self.max_missed_frames:
            return None

        # Keep a small snapshot of the latest real observation used as the
        # prediction reference.  This contains numerical/object-identification
        # fields only; image crops, embeddings, and the full received packet
        # never enter the prediction result.
        last_bbox = [
            int(round(last["cx"] - last["width"] / 2.0)),
            int(round(last["cy"] - last["height"] / 2.0)),
            int(round(last["cx"] + last["width"] / 2.0)),
            int(round(last["cy"] + last["height"] / 2.0)),
        ]
        existing_object_used = {
            "frame_id": last.get("frame_id"),
            "timestamp": last.get("timestamp"),
            "track_id": last.get("track_id"),
            "class_name": last.get("class_name", "unknown"),
            "class_id": last.get("class_id"),
            "confidence": last.get("confidence"),
            "bbox": last_bbox,
            "distance_m": last.get("distance_m"),
            "bearing_deg": last.get("bearing_deg"),
            "direction": last.get("direction"),
            "speed_kmh": last.get("speed_kmh"),
        }

        cx, _, _ = self._robust_extrapolate(history, "cx", target_timestamp)
        cy, _, _ = self._robust_extrapolate(history, "cy", target_timestamp)
        width, _, _ = self._robust_extrapolate(
            history, "width", target_timestamp
        )
        height, _, _ = self._robust_extrapolate(
            history, "height", target_timestamp
        )
        distance, distance_rate, raw_distance_rate = self._robust_extrapolate(
            history,
            "distance_m",
            target_timestamp,
            self.max_radial_speed_kmh / 3.6,
        )
        bearing, _, _ = self._robust_extrapolate(
            history, "bearing_deg", target_timestamp, 120.0
        )
        if None in (cx, cy, width, height):
            return None

        source = "robust_history"
        model_input_shape = None
        model_state = self._model_state(
            history, frame_width, frame_height, frames_since
        )
        if model_state is not None:
            cx = model_state["cx"]
            cy = model_state["cy"]
            width = model_state["width"]
            height = model_state["height"]
            distance = model_state["distance_m"]
            bearing = model_state["bearing_deg"]
            model_input_shape = model_state["model_input_shape"]
            source = "gru_model"

            last_distance = _as_float(last.get("distance_m"))
            delta_time = max(target_timestamp - last["timestamp"], 1.0 / self.fps)
            if last_distance is not None:
                raw_distance_rate = (distance - last_distance) / delta_time
                distance_rate = float(
                    np.clip(
                        raw_distance_rate,
                        -self.max_radial_speed_kmh / 3.6,
                        self.max_radial_speed_kmh / 3.6,
                    )
                )
                distance = last_distance + distance_rate * delta_time

        width = max(2.0, width)
        height = max(2.0, height)
        bbox = [
            int(np.clip(round(cx - width / 2.0), 0, frame_width - 1)),
            int(np.clip(round(cy - height / 2.0), 0, frame_height - 1)),
            int(np.clip(round(cx + width / 2.0), 0, frame_width - 1)),
            int(np.clip(round(cy + height / 2.0), 0, frame_height - 1)),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        speed_kmh = abs(distance_rate) * 3.6 if distance is not None else 0.0
        raw_speed_kmh = (
            abs(raw_distance_rate) * 3.6 if distance is not None else 0.0
        )
        if speed_kmh < 2.0:
            direction = "stationary"
        elif distance_rate < 0:
            direction = "approaching"
        else:
            direction = "moving_away"

        class_name = Counter(
            str(item.get("class_name", "unknown")) for item in history
        ).most_common(1)[0][0]
        confidence = float(
            np.median(
                [
                    _as_float(item.get("confidence"), 0.0)
                    for item in list(history)[-3:]
                ]
            )
        )
        prediction_confidence = (
            confidence * min(1.0, len(history) / 3.0) * 0.85 ** frames_since
        )

        prediction = {
            "object_id": f"predicted_{last['track_id']}",
            "track_id": last["track_id"],
            "class_name": class_name,
            "class_id": last.get("class_id"),
            "confidence": round(confidence, 6),
            "bbox": bbox,
            "distance_m": (
                round(max(0.0, distance), 3) if distance is not None else None
            ),
            "bearing_deg": round(bearing, 2) if bearing is not None else None,
            "direction": direction,
            "speed_kmh": round(speed_kmh, 3),
            "raw_speed_kmh": round(raw_speed_kmh, 3),
            "speed_outlier": raw_speed_kmh > self.max_radial_speed_kmh,
            "prediction_confidence": round(prediction_confidence, 6),
            "prediction_history_length": len(history),
            "last_observed_frame_id": last["frame_id"],
            "existing_object_used_for_prediction": existing_object_used,
            "predicted_for_frame_id": target_frame_id,
            "timestamp": target_timestamp,
            "predicted": True,
            "prediction_source": source,
            "actual_payload_pending": True,
            "has_crop": False,
            "has_processed": False,
        }
        if model_input_shape is not None:
            prediction["model_input_shape"] = model_input_shape

        return prediction

    def _merge_partial_descriptor(
        self,
        prediction: Dict,
        partial_object: Dict,
        missing_fields: Sequence[str],
        history: Sequence[Dict],
        target_timestamp: float,
    ) -> Dict:
        prediction = dict(prediction)
        actual_bbox = self._valid_bbox(partial_object)
        if actual_bbox is not None:
            prediction["bbox"] = [int(round(value)) for value in actual_bbox]

        actual_distance = _first_float(
            partial_object, "distance_m", "distance"
        )
        actual_bearing = _first_float(
            partial_object, "bearing_deg", "bearing_angle"
        )
        actual_speed = _first_float(partial_object, "speed_kmh", "speed")
        actual_direction = _first_text(
            partial_object, "direction", "motion_direction"
        )
        if actual_distance is not None:
            prediction["distance_m"] = round(max(0.0, actual_distance), 3)
        if actual_bearing is not None:
            prediction["bearing_deg"] = round(actual_bearing, 2)
        if actual_speed is not None:
            prediction["speed_kmh"] = round(max(0.0, actual_speed), 3)
            prediction["raw_speed_kmh"] = round(max(0.0, actual_speed), 3)
            prediction["speed_outlier"] = (
                actual_speed > self.max_radial_speed_kmh
            )
        if actual_direction:
            prediction["direction"] = actual_direction

        if actual_distance is not None and (
            "speed_kmh" in missing_fields or "direction" in missing_fields
        ):
            points = list(history) + [
                {"timestamp": target_timestamp, "distance_m": actual_distance}
            ]
            _, rate, raw_rate = self._robust_extrapolate(
                points,
                "distance_m",
                target_timestamp,
                self.max_radial_speed_kmh / 3.6,
            )
            derived_speed = abs(rate) * 3.6
            if "speed_kmh" in missing_fields:
                prediction["speed_kmh"] = round(derived_speed, 3)
                prediction["raw_speed_kmh"] = round(abs(raw_rate) * 3.6, 3)
                prediction["speed_outlier"] = (
                    abs(raw_rate) * 3.6 > self.max_radial_speed_kmh
                )
            if "direction" in missing_fields:
                prediction["direction"] = (
                    "stationary"
                    if derived_speed < 2.0
                    else "approaching" if rate < 0 else "moving_away"
                )

        for field in (
            "track_id",
            "class_id",
        ):
            value = partial_object.get(field)
            if value is not None and value != "":
                prediction[field] = value
        class_name = _first_text(partial_object, "class_name", "class")
        if class_name:
            prediction["class_name"] = class_name
        confidence = _as_float(partial_object.get("confidence"))
        if confidence is not None:
            prediction["confidence"] = round(confidence, 6)

        prediction["partial_descriptor_received"] = True
        prediction["prediction_reason"] = "partial_existing_object"
        prediction["predicted_fields"] = list(missing_fields)
        prediction["actual_fields_received"] = [
            field
            for field in (
                "bbox",
                "distance_m",
                "bearing_deg",
                "direction",
                "speed_kmh",
            )
            if field not in missing_fields
        ]
        return prediction

    @staticmethod
    def _bbox_iou(first_bbox, second_bbox) -> Optional[float]:
        if first_bbox is None or second_bbox is None:
            return None
        first_x1, first_y1, first_x2, first_y2 = first_bbox
        second_x1, second_y1, second_x2, second_y2 = second_bbox
        intersection_width = max(
            0.0, min(first_x2, second_x2) - max(first_x1, second_x1)
        )
        intersection_height = max(
            0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
        )
        intersection = intersection_width * intersection_height
        first_area = max(0.0, first_x2 - first_x1) * max(
            0.0, first_y2 - first_y1
        )
        second_area = max(0.0, second_x2 - second_x1) * max(
            0.0, second_y2 - second_y1
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else None

    def _compare_received_objects(
        self,
        camera_key: Tuple[str, str],
        frame_id: int,
        objects: Sequence[Dict],
    ) -> List[Dict]:
        """Compare received values with predictions made for this frame."""
        comparisons = []
        for obj in objects:
            track_id = self._first_value(
                obj, "track_id", "object_track_id", "object_id"
            )
            if track_id is None:
                continue
            track_key = self._track_key(track_id)
            cache_key = (camera_key, frame_id, track_key)
            prediction = self.prediction_cache.get(cache_key)
            if prediction is None:
                continue

            predicted_bbox = self._valid_bbox(prediction)
            received_bbox = self._valid_bbox(obj)
            bbox_error = None
            if predicted_bbox is not None and received_bbox is not None:
                coordinate_errors = [
                    received - predicted
                    for received, predicted in zip(
                        received_bbox, predicted_bbox
                    )
                ]
                predicted_center = (
                    (predicted_bbox[0] + predicted_bbox[2]) / 2.0,
                    (predicted_bbox[1] + predicted_bbox[3]) / 2.0,
                )
                received_center = (
                    (received_bbox[0] + received_bbox[2]) / 2.0,
                    (received_bbox[1] + received_bbox[3]) / 2.0,
                )
                center_error = float(
                    np.hypot(
                        received_center[0] - predicted_center[0],
                        received_center[1] - predicted_center[1],
                    )
                )
                bbox_error = {
                    "coordinate_error_px": [
                        round(value, 3) for value in coordinate_errors
                    ],
                    "mean_absolute_error_px": round(
                        float(np.mean(np.abs(coordinate_errors))), 3
                    ),
                    "center_error_px": round(center_error, 3),
                    "iou": round(
                        self._bbox_iou(predicted_bbox, received_bbox), 6
                    ),
                }

            predicted_distance = _first_float(
                prediction, "distance_m", "distance"
            )
            received_distance = _first_float(obj, "distance_m", "distance")
            predicted_bearing = _first_float(
                prediction, "bearing_deg", "bearing_angle"
            )
            received_bearing = _first_float(
                obj, "bearing_deg", "bearing_angle"
            )
            predicted_speed = _first_float(
                prediction, "speed_kmh", "speed"
            )
            received_speed = _first_float(obj, "speed_kmh", "speed")
            predicted_direction = _first_text(
                prediction, "direction", "motion_direction"
            )
            received_direction = _first_text(
                obj, "direction", "motion_direction"
            )
            predicted_class = _first_text(
                prediction, "class_name", "class"
            )
            received_class = _first_text(obj, "class_name", "class")
            predicted_confidence = _as_float(prediction.get("confidence"))
            received_confidence = _as_float(obj.get("confidence"))

            def numerical_error(predicted, received):
                if predicted is None or received is None:
                    return None
                signed_error = received - predicted
                return {
                    "signed_error": round(signed_error, 6),
                    "absolute_error": round(abs(signed_error), 6),
                }

            comparison = {
                "camera_id": camera_key[0],
                "camera_location": camera_key[1],
                "frame_id": frame_id,
                "track_id": track_id,
                "class_name": _first_text(obj, "class_name", "class")
                or prediction.get("class_name", "unknown"),
                "prediction_source": prediction.get("prediction_source"),
                "prediction_reason": prediction.get("prediction_reason"),
                "simulated_missing_fields": prediction.get(
                    "simulated_missing_fields", []
                ),
                "prediction_confidence": prediction.get(
                    "prediction_confidence"
                ),
                "predicted": {
                    "class_name": predicted_class,
                    "confidence": predicted_confidence,
                    "bbox": prediction.get("bbox"),
                    "distance_m": predicted_distance,
                    "bearing_deg": predicted_bearing,
                    "direction": predicted_direction,
                    "speed_kmh": predicted_speed,
                },
                "received": {
                    "class_name": received_class,
                    "confidence": received_confidence,
                    "bbox": obj.get("bbox"),
                    "distance_m": received_distance,
                    "bearing_deg": received_bearing,
                    "direction": received_direction,
                    "speed_kmh": received_speed,
                },
                "error": {
                    "class_match": (
                        predicted_class.lower() == received_class.lower()
                        if predicted_class and received_class
                        else None
                    ),
                    "confidence": numerical_error(
                        predicted_confidence, received_confidence
                    ),
                    "bbox": bbox_error,
                    "distance_m": numerical_error(
                        predicted_distance, received_distance
                    ),
                    "bearing_deg": numerical_error(
                        predicted_bearing, received_bearing
                    ),
                    "direction_match": (
                        predicted_direction.lower()
                        == received_direction.lower()
                        if predicted_direction and received_direction
                        else None
                    ),
                    "speed_kmh": numerical_error(
                        predicted_speed, received_speed
                    ),
                },
            }
            comparisons.append(comparison)

            # Keep a prediction cached when only a partial descriptor arrived;
            # a delayed full payload can still be compared later.
            if not self._missing_prediction_fields(obj):
                del self.prediction_cache[cache_key]
        return comparisons

    def _cache_predictions(
        self,
        camera_key: Tuple[str, str],
        predictions: Sequence[Dict],
    ) -> None:
        for prediction in predictions:
            target_frame = _as_int(prediction.get("predicted_for_frame_id"))
            track_id = self._first_value(
                prediction, "track_id", "object_track_id", "object_id"
            )
            if target_frame is None or track_id is None:
                continue
            cache_key = (
                camera_key,
                target_frame,
                self._track_key(track_id),
            )
            self.prediction_cache[cache_key] = dict(prediction)

    @staticmethod
    def _lightweight_descriptor(descriptor) -> Optional[Dict]:
        """Keep only fields allowed at the prediction-module boundary."""
        if not isinstance(descriptor, dict):
            descriptor = {"track_id": descriptor}
        track_id = ExistingObjectPredictor._first_value(
            descriptor, "track_id", "object_track_id", "object_id"
        )
        if track_id is None:
            return None

        sanitized = {"track_id": track_id}
        aliases = {
            "class_name": ("class_name", "class"),
            "class_id": ("class_id",),
            "confidence": ("confidence", "conf"),
            "bbox": ("bbox",),
            "distance_m": ("distance_m", "distance"),
            "bearing_deg": ("bearing_deg", "bearing_angle"),
            "direction": ("direction", "motion_direction"),
            "speed_kmh": ("speed_kmh", "speed"),
        }
        for output_name, input_names in aliases.items():
            value = ExistingObjectPredictor._first_value(
                descriptor, *input_names
            )
            if value is not None:
                sanitized[output_name] = value
        return sanitized

    def compare_received_observations(
        self,
        camera_id,
        camera_location,
        frame_id: int,
        observations: Sequence[Dict],
    ) -> List[Dict]:
        """Compare sanitized actual observations with cached predictions."""
        camera_key = (str(camera_id), str(camera_location))
        frame_id = _as_int(
            frame_id,
            self.latest_frame_by_camera.get(camera_key, -1) + 1,
        )
        minimal_observations = [
            record
            for record in (
                self._lightweight_descriptor(item) for item in observations
            )
            if record is not None
        ]
        self.last_comparisons = self._compare_received_objects(
            camera_key, frame_id, minimal_observations
        )
        return list(self.last_comparisons)

    def update_history(
        self,
        camera_id,
        camera_location,
        frame_id: int,
        timestamp,
        observations: Sequence[Dict],
    ) -> None:
        """Store only sanitized numerical observations as model input.

        ``observations`` may contain the required motion fields, but large
        payload fields are discarded even if a caller accidentally includes
        them. Predictions are never written into history.
        """
        camera_key = (str(camera_id), str(camera_location))
        frame_id = _as_int(
            frame_id,
            self.latest_frame_by_camera.get(camera_key, -1) + 1,
        )
        timestamp = parse_timestamp(timestamp, float(frame_id) / self.fps)
        minimal_observations = [
            record
            for record in (
                self._lightweight_descriptor(item) for item in observations
            )
            if record is not None
        ]
        self._observe_actual_objects(
            camera_key,
            frame_id,
            timestamp,
            minimal_observations,
        )
        self.latest_frame_by_camera[camera_key] = max(
            frame_id, self.latest_frame_by_camera.get(camera_key, frame_id)
        )
        self._prune_stale_state(camera_key, frame_id)

    def predict_tracks(
        self,
        camera_id,
        camera_location,
        reference_frame_id: int,
        reference_timestamp,
        target_frame_id: int,
        track_descriptors: Optional[Sequence[Dict]] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        prediction_reason: str = "omitted_existing_object",
        merge_partial_values: bool = False,
        actual_payload_pending: bool = True,
    ) -> List[Dict]:
        """Predict known tracks using only lightweight inference requests.

        For a fully omitted object, each descriptor only needs ``track_id``.
        A partial descriptor may additionally contain any actually received
        motion fields; those values are preserved while absent fields are
        predicted from server history.
        """
        camera_key = (str(camera_id), str(camera_location))
        reference_frame_id = _as_int(
            reference_frame_id,
            self.latest_frame_by_camera.get(camera_key, -1) + 1,
        )
        target_frame_id = _as_int(target_frame_id, reference_frame_id)
        reference_timestamp = parse_timestamp(
            reference_timestamp,
            float(reference_frame_id) / self.fps,
        )
        target_timestamp = reference_timestamp + max(
            0, target_frame_id - reference_frame_id
        ) / self.fps
        frame_width = _as_int(frame_width, self.default_frame_width)
        frame_height = _as_int(frame_height, self.default_frame_height)

        if track_descriptors is None:
            track_descriptors = [
                {"track_id": track_id}
                for track_id in self.known_track_ids(
                    camera_id, camera_location
                )
            ]

        descriptors = []
        seen_track_keys = set()
        for descriptor in track_descriptors:
            minimal = self._lightweight_descriptor(descriptor)
            if minimal is None:
                continue
            track_key = self._track_key(minimal["track_id"])
            if track_key in seen_track_keys:
                continue
            seen_track_keys.add(track_key)
            descriptors.append((track_key, minimal))

        predictions = []
        all_motion_fields = [
            "bbox",
            "distance_m",
            "bearing_deg",
            "direction",
            "speed_kmh",
        ]
        for track_key, descriptor in descriptors:
            history = self.track_histories.get((camera_key, track_key))
            if not history:
                continue

            # A track must have appeared in a frame earlier than the current
            # reference frame. This prevents a newly received track from
            # becoming prediction-eligible merely because the current frame
            # was written to history immediately before this call.
            if not any(
                _as_int(item.get("frame_id"), reference_frame_id)
                < reference_frame_id
                for item in history
            ):
                continue

            prediction = self._predict_track(
                history,
                target_frame_id,
                target_timestamp,
                frame_width,
                frame_height,
            )
            if prediction is None:
                continue

            missing_fields = self._missing_prediction_fields(descriptor)
            has_received_motion_value = len(missing_fields) < len(
                all_motion_fields
            )
            if merge_partial_values and has_received_motion_value:
                prediction = self._merge_partial_descriptor(
                    prediction,
                    descriptor,
                    missing_fields,
                    history,
                    target_timestamp,
                )
                if prediction_reason != "omitted_existing_object":
                    prediction["prediction_reason"] = prediction_reason
            else:
                prediction["prediction_reason"] = prediction_reason
                prediction["predicted_fields"] = list(all_motion_fields)
                if descriptor.get("class_name"):
                    prediction["class_name"] = descriptor["class_name"]

            if prediction_reason == "simulated_missing_fields":
                prediction["simulated_missing_fields"] = list(
                    missing_fields
                )

            prediction["actual_payload_pending"] = bool(
                actual_payload_pending
            )
            prediction["track_status"] = "existing"
            prediction["track_id_source"] = "server_history"
            prediction["inference_input_fields"] = sorted(descriptor.keys())
            predictions.append(prediction)

        self._cache_predictions(camera_key, predictions)
        return predictions

    def _prune_stale_state(
        self,
        camera_key: Tuple[str, str],
        frame_id: int,
    ) -> None:
        stale_before = frame_id - max(self.max_missed_frames * 5, 25)
        for key, history in list(self.track_histories.items()):
            if (
                key[0] == camera_key
                and history
                and history[-1]["frame_id"] < stale_before
            ):
                del self.track_histories[key]

        for cache_key in list(self.prediction_cache):
            if cache_key[0] == camera_key and cache_key[1] < stale_before:
                del self.prediction_cache[cache_key]


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default=None):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _first_float(mapping: Dict, *names) -> Optional[float]:
    for name in names:
        value = _as_float(mapping.get(name))
        if value is not None:
            return value
    return None


def _first_text(mapping: Dict, *names) -> Optional[str]:
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "null", "nan", "unknown"}:
            return text
    return None


__all__ = [
    "ExistingObjectPredictor",
    "ObjectMotionGRU",
    "history_to_model_tensor",
]