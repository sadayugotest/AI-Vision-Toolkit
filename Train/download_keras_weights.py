"""
download_keras_weights.py
=========================
สคริปต์สำหรับดาวน์โหลด Keras ImageNet pretrained weights ล่วงหน้า
เพื่อใช้งานแบบ offline บนเครื่องที่ไม่มีอินเทอร์เน็ต

วิธีใช้:
  1. รันบนเครื่องที่มีอินเทอร์เน็ต:
     python download_keras_weights.py

  2. คัดลอกโฟลเดอร์ ~/.keras/models/ ไปยังเครื่อง offline
     ที่เดียวกัน: C:\\Users\\<username>\\.keras\\models\\

  หรือจะเลือกดาวน์โหลดเฉพาะ model ที่ต้องการ:
     python download_keras_weights.py MobileNetV2 ResNet50
"""

import os
import sys
import urllib.request
import hashlib

KERAS_WEIGHT_INFO = {
    'MobileNetV2': {
        'file': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
        'url': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
    },
    'ResNet50': {
        'file': 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'url': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    },
    'EfficientNetB0': {
        'file': 'efficientnetb0_notop.h5',
        'url': 'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
    },
    'InceptionV3': {
        'file': 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'url': 'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    },
    'DenseNet121': {
        'file': 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'url': 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    },
}


def keras_cache_dir():
    """ดาวน์โหลดไว้ที่เดียวกับ PyTorch hub checkpoints"""
    return os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints')


def download_weight(model_name: str, info: dict):
    cache = keras_cache_dir()
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, info['file'])

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  ✅ {model_name}: already exists ({size_mb:.1f} MB) → {dest}")
        return True

    print(f"  ⏳ {model_name}: downloading...")
    print(f"     URL: {info['url']}")
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
                print(f"\r     [{bar}] {pct:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end='')

        urllib.request.urlretrieve(info['url'], dest, reporthook=progress)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"\n  ✅ {model_name}: downloaded ({size_mb:.1f} MB) → {dest}")
        return True
    except Exception as e:
        print(f"\n  ❌ {model_name}: FAILED → {e}")
        if os.path.isfile(dest):
            os.remove(dest)
        return False


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else list(KERAS_WEIGHT_INFO.keys())

    print("=" * 60)
    print("Keras ImageNet Weights Downloader (offline preparation)")
    print(f"Cache directory: {keras_cache_dir()}")
    print(f"Models to download: {', '.join(models)}")
    print("=" * 60)

    success = 0
    failed = 0
    for name in models:
        info = KERAS_WEIGHT_INFO.get(name)
        if not info:
            print(f"  ⚠️  Unknown model: {name} (available: {', '.join(KERAS_WEIGHT_INFO.keys())})")
            failed += 1
            continue
        if download_weight(name, info):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Done! ✅ {success} succeeded, ❌ {failed} failed")
    if success > 0:
        print(f"\nCopy this folder to offline machine:")
        print(f"  {keras_cache_dir()}")
        print(f"  → same path on target: C:\\Users\\<user>\\.keras\\models\\")
    print("=" * 60)


if __name__ == '__main__':
    main()
