from dataclasses import dataclass
from typing import List



@dataclass
class ReIDResult:
    object_id: str
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2,
    camera_id: str
    frame_id: int
    timestamp: str
    embedding_method: str
    reid_confidence: float
    person_id: str
    is_new_person: bool
    image: bytes  # raw image bytes
