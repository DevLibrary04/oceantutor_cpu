# app/services/image_matching_service.py (스케일 보정 매핑 최종 버전)

import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from llama_index.embeddings.clip import ClipEmbedding
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import logging

from app.rag import config
from app.ocr_service import get_ocr_reader

logger = logging.getLogger(__name__)

# --- 1. EnhancedImageRAG (기존과 동일) ---
class EnhancedImageRAG:
    def __init__(self, embed_model, reference_dir: str):
        self.embed_model = embed_model
        self.reference_dir = reference_dir
        self.ref_embeddings: Dict[str, List[float]] = {}

    def cache_reference_embeddings(self):
        logger.info("[Image RAG] 참조 이미지 임베딩 캐싱 시작...")
        if not os.path.isdir(self.reference_dir):
            logger.error(f"참조 이미지 디렉토리를 찾을 수 없습니다: {self.reference_dir}")
            return
        for filename in os.listdir(self.reference_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.reference_dir, filename)
                try:
                    embedding = self.embed_model.get_image_embedding(path)
                    self.ref_embeddings[path] = embedding
                    logger.info(f"  -> 임베딩 완료: {filename}")
                except Exception as e:
                    logger.error(f"참조 이미지 임베딩 실패 ({filename}): {e}")
        logger.info(f"[Image RAG] {len(self.ref_embeddings)}개의 참조 이미지 임베딩 캐싱 완료.")

    def find_best_match(self, query_image_path: str, similarity_threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        if not self.ref_embeddings:
            logger.warning("[Image RAG] 참조 이미지 임베딩 캐시가 비어있습니다.")
            return None
        try:
            query_embedding = self.embed_model.get_image_embedding(query_image_path)
        except Exception as e:
            logger.error(f"쿼리 이미지 임베딩 실패: {e}")
            return None
        similarities = []
        for ref_path, ref_embedding in self.ref_embeddings.items():
            sim = cosine_similarity([query_embedding], [ref_embedding])[0][0]
            similarities.append((ref_path, float(sim)))
        if not similarities:
            return None
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_match = similarities[0]
        if best_match[1] < similarity_threshold:
            logger.warning(f"가장 유사한 이미지의 유사도가 낮음: {best_match[1]:.3f} < {similarity_threshold}")
            return None
        logger.info(f"  -> 최적 매치: {os.path.basename(best_match[0])} (유사도: {best_match[1]:.3f})")
        return best_match


# --- 2. ObjectDetectorMapper (신형 터보 엔진으로 교체된 버전) ---
class ObjectDetectorMapper:
    def __init__(self, yolo_model, ocr_reader):
        self.yolo = yolo_model
        self.ocr_reader = ocr_reader

    # 포인터 탐지 로직 (기존과 동일)
    def find_pointer_box(self, image_np: np.ndarray) -> Optional[List[int]]:
        logger.info("  -> 1. YOLO로 포인터 탐색 시도...")
        results = self.yolo(image_np, conf=0.05, iou=0.45, verbose=False)
        all_pointers = []
        for result in results:
            boxes = result.boxes
            if boxes is None: continue
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                class_name = self.yolo.names.get(class_id, '').lower()
                if any(keyword in class_name for keyword in ['pointer', 'point', 'circle', 'marker']):
                    confidence = float(boxes.conf[i].cpu().numpy())
                    bbox = [int(c) for c in boxes.xyxy[i].cpu().numpy()]
                    all_pointers.append({'box': bbox, 'confidence': confidence, 'class_name': class_name})
                    logger.info(f"     -> YOLO 탐지: {class_name} (신뢰도: {confidence:.3f})")
        if all_pointers:
            best_pointers = self._non_max_suppression_custom(all_pointers)
            if best_pointers:
                best_pointer = best_pointers[0]
                logger.info(f"     [성공] YOLO 포인터 탐지: {best_pointer['box']} (클래스: {best_pointer['class_name']})")
                return best_pointer['box']
        logger.warning("     [YOLO 실패] OCR로 포인터 탐색 재시도...")
        return self._find_pointer_with_ocr(image_np)

    def _find_pointer_with_ocr(self, image_np: np.ndarray) -> Optional[List[int]]:
        try:
            pointer_keywords = [
                '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '①.', '②.', '③.', '④.', '⑤.', '1)', '2)', '3)', '4)', '5)', '(1)', '(2)', '(3)', '(4)', '(5)',
                '㉠', '㉡', '㉢', '㉣', '㉤', '㉥', '㉦', '㉧', '㉨', '㉩', '㉪', '㉫', '㉬', '㉭',
                '㉠.', '㉡.', '㉢.', '㉣.', '(ㄱ)', '(ㄴ)', '(ㄷ)', '(ㄹ)', 'ㄱ)', 'ㄴ)', 'ㄷ)', 'ㄹ)', 'ㄱ.', 'ㄴ.', 'ㄷ.', 'ㄹ.',
                '가', '나', '다', '라', '마', '가.', '나.', '다.', '라.', '(가)', '(나)', '(다)', '(라)', '가)', '나)', '다)', '라)',
            ]
            ocr_results = self.ocr_reader.readtext(image_np, detail=1)
            best_match = None
            best_confidence = 0
            for (bbox_coords, text, confidence) in ocr_results:
                clean_text = text.strip()
                if clean_text in pointer_keywords and confidence > best_confidence:
                    x_coords = [p[0] for p in bbox_coords]
                    y_coords = [p[1] for p in bbox_coords]
                    bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                    best_match = bbox
                    best_confidence = confidence
                    logger.info(f"     -> OCR 매치: '{clean_text}' (신뢰도: {confidence:.3f})")
            if best_match:
                logger.info(f"     [성공] OCR 포인터 탐지: {best_match}")
                return best_match
        except Exception as e:
            logger.error(f"     [오류] OCR 포인터 탐색 중 오류: {e}")
        logger.error("     [최종 실패] YOLO와 OCR 모두 포인터를 찾지 못했습니다.")
        return None

    def _non_max_suppression_custom(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        if not detections: return []
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        for det in detections:
            is_overlapping = False
            for kept_det in keep:
                if self._calculate_iou(det['box'], kept_det['box']) > iou_threshold:
                    is_overlapping = True
                    break
            if not is_overlapping:
                keep.append(det)
        return keep

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        x1_i, y1_i, x2_i, y2_i = box1
        x1_k, y1_k, x2_k, y2_k = box2
        inter_x1, inter_y1 = max(x1_i, x1_k), max(y1_i, y1_k)
        inter_x2, inter_y2 = min(x2_i, x2_k), min(y2_i, y2_k)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0: return 0.0
        box1_area = (x2_i - x1_i) * (y2_i - y1_i)
        box2_area = (x2_k - x1_k) * (y2_k - y1_k)
        union_area = box1_area + box2_area - inter_area
        return inter_area / float(union_area) if union_area > 0 else 0.0

    # ▼▼▼ 신형 터보 엔진 및 특수 공구 세트 이식 시작 ▼▼▼
    
    def map_pointer_to_reference(self, question_image_np: np.ndarray, reference_image_path: str,
                                 pointer_box: List[int], scale_factor: float = 1.0, debug_save: bool = True) -> str:
        """개선된 포인터 위치 매핑 (스케일 차이 고려)"""
        ref_image_color = cv2.imread(reference_image_path)
        if ref_image_color is None:
            return "매핑 실패 (참조 이미지 로드 불가)"

        # ⭐ 핵심 1: 이미지들을 매칭에 적합한 크기로 각각 리사이즈
        q_resized, q_scale = self._resize_question_image(question_image_np)
        ref_resized, ref_scale = self._resize_reference_image(ref_image_color, q_resized.shape[:2])

        q_gray = cv2.cvtColor(q_resized, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)

        # ⭐ 핵심 2: 포인터 좌표도 매칭용 이미지의 스케일에 맞게 조정
        # `pointer_box`는 `question_image_np` 기준 좌표이므로, `q_scale`을 곱해 `q_resized` 기준 좌표로 변환
        adjusted_pointer_box = [int(coord * q_scale) for coord in pointer_box]
        
        pointer_center = (int((adjusted_pointer_box[0] + adjusted_pointer_box[2]) / 2),
                          int((adjusted_pointer_box[1] + adjusted_pointer_box[3]) / 2))

        logger.info(f"  -> 매칭용 이미지 크기: 질문={q_resized.shape[:2]}, 참조={ref_resized.shape[:2]}")
        logger.info(f"  -> 조정된 포인터 중심: {pointer_center}")

        try:
            # 전략 1: 스케일 보정된 호모그래피 매칭
            homography_result = self._find_homography_with_scale_correction(q_gray, ref_gray)
            if homography_result:
                M, good_matches, kp1, kp2 = homography_result
                q_center_3d = np.float32([[pointer_center]]).reshape(-1, 1, 2)
                transformed_center_3d = cv2.perspectiveTransform(q_center_3d, M)
                transformed_center = tuple(map(int, transformed_center_3d[0][0]))

                # ⭐ 핵심 3: 변환된 좌표를 '원본' 참조 이미지 크기로 최종 스케일업
                original_transformed_center = (
                    int(transformed_center[0] / ref_scale),
                    int(transformed_center[1] / ref_scale)
                )
                logger.info(f"  -> 호모그래피 변환: {pointer_center} -> {transformed_center} (매칭용) -> {original_transformed_center} (원본용)")

                if debug_save:
                    self._save_debug_image_with_scales(
                        q_resized, ref_resized, kp1, kp2, good_matches,
                        pointer_center, transformed_center, reference_image_path, q_scale, ref_scale
                    )

                original_ref_gray = cv2.cvtColor(ref_image_color, cv2.COLOR_BGR2GRAY)
                keyword = self._find_nearest_text_with_fallback(original_ref_gray, original_transformed_center)
                if not keyword.startswith("매핑 실패"):
                    return keyword
                logger.warning("  -> 호모그래피 기반 텍스트 찾기 실패, fallback 시도")

            # 전략 2: 스케일 보정된 상대적 위치 매핑 (핵심 fallback)
            logger.info("  -> 상대적 위치 기반 매핑 시도...")
            keyword = self._map_by_relative_position(q_resized, ref_image_color, pointer_center)
            return keyword

        except Exception as e:
            logger.error(f"매핑 처리 중 오류: {e}", exc_info=True)
            return f"매핑 실패 (내부 오류: {e})"

    # --- 아래는 신형 엔진의 특수 공구들 (헬퍼 함수) ---

    def _resize_question_image(self, image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, float]:
        """질문 이미지를 매칭용 표준 크기로 리사이즈"""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= target_size:
            return image, 1.0
        scale = target_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _resize_reference_image(self, ref_image: np.ndarray, question_shape: Tuple[int, int],
                                scale_ratio: float = 0.8) -> Tuple[np.ndarray, float]:
        """참조 이미지를 질문 이미지와 유사한 크기로 리사이즈하여 스케일 차이를 줄임"""
        q_h, q_w = question_shape
        ref_h, ref_w = ref_image.shape[:2]
        target_w, target_h = int(q_w * scale_ratio), int(q_h * scale_ratio)
        scale = min(target_w / ref_w, target_h / ref_h)
        new_w, new_h = int(ref_w * scale), int(ref_h * scale)
        resized = cv2.resize(ref_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _find_homography_with_scale_correction(self, q_gray: np.ndarray, ref_gray: np.ndarray) -> Optional[Tuple]:
        """스케일에 강건한 매칭 방법을 사용하여 호모그래피 계산"""
        logger.info("  -> 스케일 보정 ORB 매칭 시도...")
        return self._try_orb_matching_enhanced(q_gray, ref_gray)

    def _try_orb_matching_enhanced(self, q_gray: np.ndarray, ref_gray: np.ndarray) -> Optional[Tuple]:
        """향상된 ORB 매칭 (전처리 및 강건한 파라미터 사용)"""
        q_processed = self._preprocess_for_matching(q_gray)
        ref_processed = self._preprocess_for_matching(ref_gray)
        orb = cv2.ORB_create(
            nfeatures=20000, scaleFactor=1.1, nlevels=16,
            edgeThreshold=10, patchSize=25, fastThreshold=15
        )
        kp1, des1 = orb.detectAndCompute(q_processed, None)
        kp2, des2 = orb.detectAndCompute(ref_processed, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10: return None
        logger.info(f"    -> 특징점: 질문={len(kp1)}, 참조={len(kp2)}")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        if matches and len(matches[0]) == 2:
            good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        
        logger.info(f"    -> 좋은 매칭: {len(good_matches)}개")
        if len(good_matches) < 10: return None
        
        return self._calculate_homography_robust(kp1, kp2, good_matches)

    def _preprocess_for_matching(self, gray_img: np.ndarray) -> np.ndarray:
        """특징점 매칭을 위한 이미지 전처리"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray_img)
        return cv2.GaussianBlur(processed, (3, 3), 0)

    def _calculate_homography_robust(self, kp1, kp2, matches) -> Optional[Tuple]:
        """더 관대한 파라미터로 호모그래피 계산"""
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC,
            ransacReprojThreshold=10.0, maxIters=5000, confidence=0.95
        )
        
        if M is None: return None
        
        inliers = np.sum(mask) if mask is not None else 0
        inlier_ratio = inliers / len(matches)
        logger.info(f"    -> 인라이어: {inliers}/{len(matches)} ({inlier_ratio:.2%})")
        
        if inlier_ratio < 0.1: return None # 최소 10%의 인라이어 비율 요구
        
        return M, matches, kp1, kp2

    def _map_by_relative_position(self, q_img: np.ndarray, ref_img_color: np.ndarray,
                                   pointer_center: Tuple[int, int]) -> str:
        """상대적 위치 기반 매핑 (Fallback)"""
        q_h, q_w = q_img.shape[:2]
        ref_h, ref_w = ref_img_color.shape[:2]
        rel_x, rel_y = pointer_center[0] / q_w, pointer_center[1] / q_h
        mapped_x, mapped_y = int(rel_x * ref_w), int(rel_y * ref_h)
        logger.info(f"    -> 상대 위치 매핑: ({rel_x:.3f}, {rel_y:.3f}) -> ({mapped_x}, {mapped_y})")
        ref_gray = cv2.cvtColor(ref_img_color, cv2.COLOR_BGR2GRAY)
        return self._find_nearest_text_with_fallback(ref_gray, (mapped_x, mapped_y))

    def _find_nearest_text_with_fallback(self, ref_gray: np.ndarray, center_point: Tuple[int, int]) -> str:
        """여러 반경으로 확장하며 가장 가까운 텍스트 검색"""
        search_radii = [50, 100, 150, 200, 300]
        try:
            ocr_results = self.ocr_reader.readtext(ref_gray, detail=1)
            if not ocr_results: return "매핑 실패 (참조 이미지 OCR 결과 없음)"
            
            for radius in search_radii:
                candidates = []
                for (bbox, text, conf) in ocr_results:
                    if conf < 0.4: continue
                    x_coords, y_coords = [p[0] for p in bbox], [p[1] for p in bbox]
                    box_center_x, box_center_y = sum(x_coords) / 4, sum(y_coords) / 4
                    distance = np.sqrt((center_point[0] - box_center_x)**2 + (center_point[1] - box_center_y)**2)
                    if distance <= radius:
                        candidates.append({'text': text.strip(), 'distance': distance, 'confidence': conf})
                if candidates:
                    candidates.sort(key=lambda x: x['distance']) # 거리가 가장 가까운 것을 최우선으로
                    best = candidates[0]
                    logger.info(f"    -> 반경 {radius}px에서 텍스트 발견: '{best['text']}' (거리: {best['distance']:.1f})")
                    return best['text']
        except Exception as e:
            logger.error(f"텍스트 탐색 중 오류: {e}")
            return f"매핑 실패 (OCR 오류: {e})"
        return "매핑 실패 (모든 반경에서 텍스트 없음)"

    def _save_debug_image_with_scales(self, q_img: np.ndarray, ref_img: np.ndarray,
                                      kp1, kp2, matches, pointer_center: Tuple[int, int],
                                      transformed_center: Tuple[int, int], ref_path: str,
                                      q_scale: float, ref_scale: float):
        """스케일 정보를 포함한 디버그 이미지 저장"""
        try:
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # 매칭이 너무 많으면 시각적으로 복잡하므로 상위 50개만 그립니다.
            matches_to_draw = sorted(matches, key=lambda x: x.distance)[:50]
            debug_img = cv2.drawMatches(q_img, kp1, ref_img, kp2, matches_to_draw, None, **draw_params)
            
            # 원 크기와 두께를 이미지 크기에 비례하도록 조정
            radius = int(max(q_img.shape[0], q_img.shape[1]) * 0.02)
            thickness = int(radius / 4)

            cv2.circle(debug_img, pointer_center, radius, (255, 0, 0), thickness) # 파란색
            transformed_x = transformed_center[0] + q_img.shape[1]
            cv2.circle(debug_img, (transformed_x, transformed_center[1]), radius, (0, 0, 255), thickness) # 빨간색
            
            info_text = f"Q_Scale: {q_scale:.3f} | Ref_Scale: {ref_scale:.3f}"
            cv2.putText(debug_img, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            
            debug_filename = f"debug_scale_mapping_{os.path.basename(ref_path)}"
            debug_save_path = os.path.join(config.TEMP_UPLOAD_DIR, debug_filename)
            cv2.imwrite(debug_save_path, debug_img)
            logger.info(f"★★★ 스케일 디버깅 이미지 저장: {debug_save_path} ★★★")
        except Exception as e:
            logger.warning(f"디버그 이미지 저장 실패: {e}")

    # ▲▲▲ 신형 터보 엔진 및 특수 공구 세트 이식 끝 ▲▲▲


# --- 3. 통합 서비스 (Singleton, 기존과 동일) ---
class ImageMatchingService:
    _instance = None
    _initialized = False
    

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ImageMatchingService, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        if self._initialized: return
        logger.info("--- ImageMatchingService 초기화 시작 ---")
        try:
            self.clip_embedding = ClipEmbedding(model_name="ViT-B/32")
            logger.info("  -> CLIP 임베딩 모델 로딩 완료")
            self.image_rag = EnhancedImageRAG(self.clip_embedding, config.REFERENCE_IMAGES_DIR)
            self.image_rag.cache_reference_embeddings()
            yolo_model = YOLO(config.YOLO_MODEL_PATH)
            logger.info(f"  -> YOLO 모델 로딩 완료: {config.YOLO_MODEL_PATH}")
            ocr_reader = get_ocr_reader()
            logger.info("  -> OCR 리더 초기화 완료")
            self.mapper = ObjectDetectorMapper(yolo_model, ocr_reader)
            self._initialized = True
            logger.info("--- ImageMatchingService 초기화 완료 ---")
        except Exception as e:
            logger.error(f"ImageMatchingService 초기화 실패: {e}", exc_info=True)
            raise

    def _resize_image_for_processing(self, image_np: np.ndarray, max_width: int = 1024, max_height: int = 768) -> Tuple[np.ndarray, float]:
        """이미지 크기를 일반 처리에 맞게 조정하고 스케일 팩터를 반환"""
        h, w = image_np.shape[:2]
        if w <= max_width and h <= max_height: return image_np, 1.0
        scale = min(max_width / w, max_height / h)
        new_width, new_height = int(w * scale), int(h * scale)
        resized = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"  -> 이미지 리사이즈 (처리용): {w}x{h} -> {new_width}x{new_height} (스케일: {scale:.3f})")
        return resized, scale

    def find_keyword_from_image(self, image_bytes: bytes) -> str:
        if not self._initialized:
            raise RuntimeError("ImageMatchingService가 초기화되지 않았습니다.")
        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=config.TEMP_UPLOAD_DIR) as tmp:
                tmp.write(image_bytes)
                temp_file_path = tmp.name
            
            logger.info(f"[Image Match] 처리 시작: {os.path.basename(temp_file_path)}")
            original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if original_image is None: return "키워드 찾기 실패 (이미지 로딩 불가)"
            
            # 일반 처리를 위한 1차 리사이즈
            image_np_proc, scale_factor = self._resize_image_for_processing(original_image)

            logger.info("[Image Match] 1/3: 가장 유사한 참조 이미지를 찾습니다...")
            match_result = self.image_rag.find_best_match(temp_file_path)
            if not match_result: return "키워드 찾기 실패 (유사 이미지 없음)"
            best_ref_path, _ = match_result

            logger.info("[Image Match] 2/3: 질문 이미지에서 포인터를 탐지합니다...")
            # 포인터는 처리용 이미지에서 탐지
            pointer_box = self.mapper.find_pointer_box(image_np_proc)
            if not pointer_box: return "키워드 찾기 실패 (포인터 탐지 불가)"
            logger.info(f"  -> 탐지된 포인터 BBox: {pointer_box}")

            logger.info("[Image Match] 3/3: 위치를 매핑하여 정답 키워드를 추출합니다...")
            # ⭐ 핵심: 매핑 함수에는 처리용 이미지와 포인터 박스를 그대로 전달
            # 내부적으로 매칭을 위해 한번 더 리사이즈하고 좌표를 보정함
            keyword = self.mapper.map_pointer_to_reference(
                image_np_proc, best_ref_path, pointer_box, scale_factor=scale_factor
            )
            logger.info(f"  -> 최종 추출된 키워드: '{keyword}'")

            if keyword.startswith("매핑 실패"):
                logger.warning(f"매핑 실패: {keyword}")
            else:
                logger.info(f"[성공] 키워드 추출 완료: '{keyword}'")
            return keyword
        except Exception as e:
            logger.error(f"find_keyword_from_image 처리 중 오류 발생: {e}", exc_info=True)
            return f"키워드 찾기 실패 (내부 오류: {str(e)})"
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as e: logger.warning(f"임시 파일 삭제 실패: {e}")

    def get_service_status(self) -> Dict:
        return {
            "initialized": self._initialized,
            "reference_images_count": len(self.image_rag.ref_embeddings) if self._initialized else 0,
            "reference_dir": config.REFERENCE_IMAGES_DIR if self._initialized else None,
        }

def get_image_matching_service() -> ImageMatchingService:
    service = ImageMatchingService()
    return service