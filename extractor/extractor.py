import cv2
import numpy as np
import easyocr
import json
from ultralytics import YOLO
import os  


MODEL_PATH = r"C:\Users\user\Downloads\parser-for-rag\parser-for-rag\yolo\runs\detect\train4\weights\best.pt"
IMAGE_PATH = r"C:\Users\user\Downloads\parser-for-rag\parser-for-rag\yolo\dataset\images\val\20.png"

OUTPUT_DIR = r"C:\Users\user\Downloads\detection_output" # 결과물이 저장될 기본 폴더
VISUALIZED_IMAGE_SAVE_PATH = os.path.join(OUTPUT_DIR, "detection_result_visualized-20.png") # 시각화된 전체 이미지
JSON_SAVE_PATH = os.path.join(OUTPUT_DIR, "detection_results-20.json") # JSON 결과 파일
CROPPED_OBJECTS_DIR = os.path.join(OUTPUT_DIR, "cropped_objects")   # 잘라낸 객체 이미지 저장 폴더

CONFIDENCE_THRESHOLD = 0.05     # 객체 탐지 최소 신뢰도
TEXT_CONFIDENCE_THRESHOLD = 0.1 # OCR 텍스트 최소 신뢰도
MIN_BOX_AREA = 25               # 노이즈로 간주할 최소 박스 면적

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPPED_OBJECTS_DIR, exist_ok=True)
print(f"결과 저장 폴더: '{OUTPUT_DIR}'")

print("EasyOCR 모델을 로딩 중입니다...")
try:
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    print("EasyOCR 로딩 완료 (GPU 사용).")
except Exception as e:
    print(f"GPU로 EasyOCR 로딩 중 오류 발생: {e}. CPU로 전환합니다.")
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    print("EasyOCR 로딩 완료 (CPU 사용).")

print(f"YOLOv8 모델을 '{MODEL_PATH}'에서 로딩 중입니다...")
try:
    model = YOLO(MODEL_PATH)
    print(f"모델에 정의된 클래스들: {model.names}")
    print("YOLOv8 모델 로딩 완료.")
except Exception as e:
    print(f"YOLOv8 모델 로딩 중 오류 발생: {e}")
    exit()

def get_box_center(box):
    """Bounding Box의 중심점을 계산합니다."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_box_area(box):
    """Bounding Box의 면적을 계산합니다."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def calculate_iou(box1, box2):
    """두 박스 간의 IoU를 계산합니다."""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1, x1_2)
    inter_y1 = max(y1, y1_2)
    inter_x2 = min(x2, x2_2)
    inter_y2 = min(y2, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def non_max_suppression_custom(detections, iou_threshold=0.5):
    """중복 탐지 제거를 위한 NMS 적용"""
    if not detections:
        return []

    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []

    for det1 in sorted_detections:
        should_keep = True
        for det2 in keep:
            if det1['original_class'] == det2['original_class']:
                if calculate_iou(det1['box'], det2['box']) > iou_threshold:
                    should_keep = False
                    break
        if should_keep:
            keep.append(det1)

    return keep

def detect_objects_yolo(image, model, conf_threshold=CONFIDENCE_THRESHOLD, min_area=MIN_BOX_AREA):
    """YOLO를 사용한 객체 탐지"""
    print(f"\n--- 객체 탐지 시작 (신뢰도: {conf_threshold}, 최소면적: {min_area}) ---")
    results = model(image, conf=conf_threshold, iou=0.45, verbose=False)

    all_detections = []
    print("\n🔍 YOLO 원시 탐지 결과:")

    for result in results:
        boxes = result.boxes
        if boxes is None:
            print("  - 탐지된 박스가 없습니다.")
            continue

        print(f"  - 총 {len(boxes)} 개의 원시 탐지 결과")

        for i in range(len(boxes)):
            box = [int(c) for c in boxes.xyxy[i].cpu().numpy()]
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i])
            class_name = model.names.get(class_id, 'unknown')
            box_area = get_box_area(box)

            print(f"    [{i}] 클래스: '{class_name}' (ID: {class_id}), 신뢰도: {confidence:.3f}, 면적: {box_area}")

            if box_area < min_area:
                print(f"         ❌ 면적이 너무 작아서 제외됨")
                continue

            all_detections.append({
                'box': box,
                'confidence': confidence,
                'original_class': class_name,
                'class_id': class_id
            })
            print(f"         ✅ 탐지 목록에 추가됨")

    print(f"\n📊 면적 필터링 후 탐지 수: {len(all_detections)}")

    class_count = {}
    for det in all_detections:
        cls = det['original_class']
        class_count[cls] = class_count.get(cls, 0) + 1

    print("\n📈 클래스별 탐지 분포 (NMS 전):")
    for cls, count in class_count.items():
        print(f"  - {cls}: {count}개")

    filtered_detections = non_max_suppression_custom(all_detections, iou_threshold=0.4)
    print(f"\n🔧 NMS 후 탐지 수: {len(filtered_detections)}")

    final_detections = {'pointers': [], 'arrows': [], 'target_objects': []}

    print("\n🏷️ 클래스 매핑 과정:")
    for det in filtered_detections:
        name_lower = det['original_class'].lower()
        print(f"  - 원본 클래스: '{det['original_class']}' -> 소문자: '{name_lower}'")

        mapped = False
        if 'pointer' in name_lower or 'point' in name_lower:
            final_detections['pointers'].append(det)
            print(f"    ✅ pointers에 매핑됨")
            mapped = True
        elif 'arrow' in name_lower:
            final_detections['arrows'].append(det)
            print(f"    ✅ arrows에 매핑됨")
            mapped = True
        elif 'target' in name_lower or 'object' in name_lower:
            final_detections['target_objects'].append(det)
            print(f"    ✅ target_objects에 매핑됨")
            mapped = True

        if not mapped:
            print(f"    ❌ 매핑되지 않음 - 알 수 없는 클래스")

    for category in final_detections:
        if final_detections[category]:
            original_count = len(final_detections[category])
            final_detections[category] = non_max_suppression_custom(
                final_detections[category], iou_threshold=0.3)
            if original_count != len(final_detections[category]):
                print(f"🎯 {category} NMS: {original_count} -> {len(final_detections[category])}")

    return final_detections

def recognize_pointer_text(image, pointer_detections):
    """포인터 텍스트 OCR"""
    print("\n--- 포인터 텍스트 OCR 시작 ---")
    if not pointer_detections:
        print("  - 포인터가 없어서 OCR을 건너뜁니다.")
        return

    for i, p_det in enumerate(pointer_detections):
        x1, y1, x2, y2 = p_det['box']
        padding = 10
        crop_img = image[max(0, y1-padding):min(image.shape[0], y2+padding),
                         max(0, x1-padding):min(image.shape[1], x2+padding)]

        if crop_img.size == 0:
            p_det.update({'text': 'CROP_ERROR', 'text_confidence': 0.0})
            continue

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        results = reader.readtext(enhanced, detail=True, paragraph=False)
        if results:
            best_res = max(results, key=lambda x: x[2])
            text, confidence = best_res[1].strip(), best_res[2]
            p_det['text'] = text if confidence >= TEXT_CONFIDENCE_THRESHOLD else 'N/A'
            p_det['text_confidence'] = confidence
        else:
            p_det.update({'text': 'NO_TEXT', 'text_confidence': 0.0})

        print(f"  - Pointer {i}: '{p_det['text']}' (신뢰도: {p_det.get('text_confidence', 0):.3f})")

def visualize_detections(image, detections):
    """탐지된 객체들을 시각화"""
    vis_image = image.copy()
    colors = {
        'pointers': (255, 100, 100),
        'arrows': (100, 255, 100),
        'target_objects': (100, 100, 255)
    }

    print("\n--- 시각화 시작 ---")

    for category, items in detections.items():
        color = colors.get(category, (128, 128, 128))
        print(f"\n📍 {category.replace('_', ' ').capitalize()} 시각화:")

        for i, item in enumerate(items):
            x1, y1, x2, y2 = item['box']

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 1)

            if category == 'pointers' and 'text' in item:
                label = f"{category[:-1]}_{i}: {item['text']} ({item['confidence']:.2f})"
            else:
                label = f"{category[:-1]}_{i} ({item['confidence']:.2f})"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image, (x1, y1 - 20), (x1 + label_size[0] + 4, y1), color, -1)

            cv2.putText(vis_image, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            center = get_box_center(item['box'])
            cv2.circle(vis_image, center, 3, color, -1)

            print(f"  - {category[:-1]}_{i}: {item['original_class']} (신뢰도: {item['confidence']:.3f})")

    return vis_image


def save_extracted_objects(original_image, detections, json_data, output_dir):
    """탐지된 객체들을 개별 이미지 파일로 잘라내어 저장합니다."""
    print("\n--- 탐지된 객체 이미지 추출 및 저장 시작 ---")
    print(f"  - 저장 경로: {output_dir}")

    for category, items in json_data.items():
        for item in items:
            obj_id = item['id']
            x1, y1, x2, y2 = item['box']

            filename = f"{obj_id}.png"
            save_path = os.path.join(output_dir, filename)

            cropped_img = original_image[y1:y2, x1:x2]

            if cropped_img.size == 0:
                print(f"    - ⚠️ {filename} 크롭 실패 (영역 크기가 0).")
                continue

            try:
                cv2.imwrite(save_path, cropped_img)
                print(f"    - ✅ {filename} 저장 완료.")
            except Exception as e:
                print(f"    - ❌ {filename} 저장 실패: {e}")

def main():
    print("=== 단순화된 Pointer-Arrow-Target 객체 탐지 시스템 ===")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"오류: 이미지 로드 실패. '{IMAGE_PATH}' 경로를 확인해주세요.")
        return

    print(f"이미지 크기: {image.shape}")
    print(f"설정 - 신뢰도: {CONFIDENCE_THRESHOLD}, 최소면적: {MIN_BOX_AREA}")

    detections = detect_objects_yolo(image, model)

    if not detections['target_objects']:
        print("\n⚠️  Target 객체가 탐지되지 않았습니다. 더 관대한 설정으로 재시도합니다...")
        retry_detections = detect_objects_yolo(image, model, conf_threshold=0.001, min_area=10)

        if retry_detections['target_objects']:
            print(f"✅ 재시도에서 target 객체 {len(retry_detections['target_objects'])}개 발견!")
            if not detections['pointers'] and retry_detections['pointers']:
                detections['pointers'] = retry_detections['pointers']
            if not detections['arrows'] and retry_detections['arrows']:
                detections['arrows'] = retry_detections['arrows']
            detections['target_objects'] = retry_detections['target_objects']

    recognize_pointer_text(image, detections['pointers'])

    print("\n" + "="*60)
    print("최종 탐지 결과 요약")
    print("="*60)

    total_objects = sum(len(items) for items in detections.values())
    print(f"총 탐지된 객체 수: {total_objects}개")

    for category, items in detections.items():
        print(f"\n📋 {category.replace('_', ' ').capitalize()}: {len(items)}개")
        for i, item in enumerate(items):
            if category == 'pointers' and 'text' in item:
                print(f"  - {category[:-1]}_{i}: '{item['text']}' (신뢰도: {item['confidence']:.3f})")
            else:
                print(f"  - {category[:-1]}_{i}: {item['original_class']} (신뢰도: {item['confidence']:.3f})")

    print(f"\n{'-'*40}\nJSON 결과 생성 및 저장\n{'-'*40}")
    
    json_data = {}
    for category, items in detections.items():
        json_data[category] = []
        for i, item in enumerate(items):
            obj_data = {
                'id': f"{category[:-1]}_{i}",
                'class': item['original_class'],
                'confidence': round(item['confidence'], 3),
                'box': item['box']
            }
            if category == 'pointers' and 'text' in item:
                obj_data['text'] = item['text']
                obj_data['text_confidence'] = round(item.get('text_confidence', 0), 3)
            json_data[category].append(obj_data)
    
    json_output = json.dumps(json_data, indent=2, ensure_ascii=False)
    print("--- JSON 내용 ---")
    print(json_output)

    try:
        with open(JSON_SAVE_PATH, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"\n✅ JSON 결과가 파일로 저장되었습니다: {JSON_SAVE_PATH}")
    except Exception as e:
        print(f"\n❌ JSON 파일 저장 실패: {e}")

    save_extracted_objects(image, detections, json_data, CROPPED_OBJECTS_DIR)

    result_image = visualize_detections(image, detections)
    
    cv2.imshow("Object Detection Result", result_image)
    print(f"\n결과 창이 나타났습니다. 아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0)
    
    try:
        cv2.imwrite(VISUALIZED_IMAGE_SAVE_PATH, result_image)
        print(f"시각화된 결과 이미지가 저장되었습니다: {VISUALIZED_IMAGE_SAVE_PATH}")
    except Exception as e:
        print(f"결과 이미지 저장 실패: {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()