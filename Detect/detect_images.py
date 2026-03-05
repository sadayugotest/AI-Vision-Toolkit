import os
# สั่งให้ห้ามออกเน็ตเด็ดขาด
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ใช้ backend สำหรับบันทึกไฟล์
import matplotlib.pyplot as plt
from pathlib import Path
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from torch.utils.data import DataLoader

def detect_single_image(image_path, checkpoint_path, output_dir="detection_results"):
    """
    ตรวจสอบรูปภาพเดียวและแสดง Predicted Heat Map
    
    Args:
        image_path: path ของรูปที่ต้องการตรวจสอบ
        checkpoint_path: path ของ model checkpoint (.ckpt)
        output_dir: folder สำหรับบันทึกผลลัพธ์
    """
    
    # สร้าง output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Anomaly Detection")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Model: {checkpoint_path}")
    
    # 1. สร้าง Model (ต้องตรงกับตอน train)
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=False
    )
    
    # 2. สร้าง Engine
    engine = Engine(task="classification")
    
    # 3. สร้าง PredictDataset สำหรับรูปเดียว
    dataset = PredictDataset(path=image_path)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # 4. ทำการ Predict
    print("\nProcessing image...")
    predictions = engine.predict(
        model=model,
        ckpt_path=checkpoint_path,
        dataloaders=dataloader
    )
    
    # 5. ดึงผลลัพธ์
    result = predictions[0]
    pred_score = result["pred_scores"][0].item()
    pred_label = result["pred_labels"][0].item()
    anomaly_map = result["anomaly_maps"][0].cpu().numpy()
    
    # จัดการมิติของ anomaly map
    while anomaly_map.ndim > 2:
        anomaly_map = anomaly_map.squeeze(0)
    
    # แสดงผลลัพธ์
    status = "ABNORMAL ❌" if pred_label else "NORMAL ✓"
    print(f"\n{'='*60}")
    print(f"DETECTION RESULT")
    print(f"{'='*60}")
    print(f"Anomaly Score: {pred_score:.4f}")
    print(f"Status: {status}")
    print(f"{'='*60}\n")
    
    # 6. อ่านรูปต้นฉบับ
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"Error: Cannot read image from {image_path}")
        return
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 7. Resize heatmap ให้ตรงกับรูปต้นฉบับ
    h, w = original_img.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
    
    # 8. สร้าง visualization (3 แบบ)
    fig = plt.figure(figsize=(18, 6))
    
    # 8.1 รูปต้นฉบับ
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original_img_rgb)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 8.2 Predicted Heat Map
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(anomaly_map_resized, cmap='jet', vmin=0, vmax=1)
    ax2.set_title(f"Predicted Heat Map\nScore: {pred_score:.4f}", 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 8.3 Overlay (รูปต้นฉบับ + Heatmap ทับ)
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(original_img_rgb)
    # สร้าง heatmap overlay แบบโปร่งใส
    heatmap_colored = plt.cm.jet(anomaly_map_resized)[:, :, :3]  # ลบ alpha channel
    ax3.imshow(heatmap_colored, alpha=0.5)  # ความโปร่งใส 50%
    ax3.set_title(f"Overlay\nStatus: {status}", 
                  fontsize=14, fontweight='bold',
                  color='red' if pred_label else 'green')
    ax3.axis('off')
    
    plt.tight_layout()
    
    # 9. บันทึกผลลัพธ์
    image_name = Path(image_path).stem
    result_filename = f"{image_name}_result.png"
    result_path = output_path / result_filename
    
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"✓ Result saved to: {result_path}")
    
    # บันทึก heatmap แยก
    heatmap_filename = f"{image_name}_heatmap.png"
    heatmap_path = output_path / heatmap_filename
    plt.figure(figsize=(8, 6))
    plt.imshow(anomaly_map_resized, cmap='jet', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Anomaly Heat Map\nScore: {pred_score:.4f}", fontsize=12)
    plt.axis('off')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {heatmap_path}")
    
    plt.close('all')
    
    # 10. แสดงผลด้วย OpenCV
    result_display = cv2.imread(str(result_path))
    if result_display is not None:
        # Resize ถ้ารูปใหญ่เกินไป
        max_height = 800
        h, w = result_display.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            result_display = cv2.resize(result_display, (new_w, max_height))
        
        cv2.imshow("Detection Result - Press any key to close", result_display)
        print("\n👁️ Displaying result... Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        'score': pred_score,
        'label': pred_label,
        'status': 'abnormal' if pred_label else 'normal',
        'result_path': str(result_path),
        'heatmap_path': str(heatmap_path)
    }


def detect_multiple_images(image_folder, checkpoint_path, output_dir="detection_results"):
    """
    ตรวจสอบรูปหลายๆ รูปในโฟลเดอร์
    
    Args:
        image_folder: path ของโฟลเดอร์ที่มีรูปภาพ
        checkpoint_path: path ของ model checkpoint
        output_dir: folder สำหรับบันทึกผลลัพธ์
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_folder = Path(image_folder)
    
    # หารูปภาพทั้งหมดในโฟลเดอร์
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))
    
    print(f"\nFound {len(image_files)} images in {image_folder}")
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n--- Processing image {i}/{len(image_files)} ---")
        result = detect_single_image(str(img_path), checkpoint_path, output_dir)
        results.append(result)
    
    # สรุปผลลัพธ์
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    normal_count = sum(1 for r in results if r['status'] == 'normal')
    abnormal_count = len(results) - normal_count
    print(f"Total images: {len(results)}")
    print(f"Normal: {normal_count}")
    print(f"Abnormal: {abnormal_count}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # ===== Configuration =====
    # path ของ model checkpoint ที่ train เสร็จแล้ว
    CHECKPOINT_PATH = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/Model/model.ckpt"
    
    # กรณีที่ 1: ตรวจสอบรูปเดียว
    IMAGE_PATH = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/OK.jpg"
    
    # กรณีที่ 2: ตรวจสอบทั้งโฟลเดอร์ (uncomment บรรทัดด้านล่าง)
    # IMAGE_FOLDER = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/dataset_folder2/abnormal"
    
    OUTPUT_DIR = r"C:/Users/c028479/PycharmProjects/pythonProject/Anomalib_new/detection_results"
    
    # ===== Run Detection =====
    
    # ตรวจสอบว่า checkpoint มีอยู่หรือไม่
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first using anomalib_train 18.py")
        exit(1)
    
    # ตรวจสอบรูปเดียว
    if Path(IMAGE_PATH).exists():
        result = detect_single_image(IMAGE_PATH, CHECKPOINT_PATH, OUTPUT_DIR)
        print(f"\n✓ Detection completed!")
    else:
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("\n💡 Tip: Update IMAGE_PATH in the script to point to your test image")
    
    # หรือตรวจสอบทั้งโฟลเดอร์ (uncomment เพื่อใช้งาน)
    # results = detect_multiple_images(IMAGE_FOLDER, CHECKPOINT_PATH, OUTPUT_DIR)
