import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np 
import tqdm
import os 
import shutil

def resize_image(args):
    img_path, source_dir, dest_dir, target_height, target_aspect_ratio = args
    try:
        relative_path = img_path.relative_to(source_dir)
        output_path = dest_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(img_path) as img:
            w, h = img.size
            if target_aspect_ratio is None:
                new_width = int(w * (target_height / h))
            else:
                new_width = int(target_height * target_aspect_ratio)
            if img.mode in ("RGB", "L"):
                img = img.resize((new_width, target_height), Image.LANCZOS)
            else:
                img = Image.fromarray(np.array(img).astype(np.float32) / 65535, mode="F")
                img = img.resize((new_width, target_height), Image.LANCZOS)
                img = Image.fromarray((np.array(np.clip(img, 0, 1)) * 65535).astype(np.uint16), mode="I;16")
            img.save(output_path)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

def main(source_dir, target_height, target_aspect_ratio):
    target_height = int(target_height)
    source = Path(source_dir)
    dest = Path(f"{source}_{target_height}")

    os.makedirs(dest, exist_ok=True)
    if not (dest / "sparse").exists():
        shutil.copytree(source / "sparse", dest / "sparse")
    if os.path.exists(source / "transforms_train.json"):
        shutil.copyfile(source / "transforms_train.json", dest / "transforms_train.json")
        shutil.copyfile(source / "transforms_test.json", dest / "transforms_test.json")

    images = list(source.rglob("*.png"))

    with ThreadPoolExecutor() as executor:
        args = ((img, source, dest, target_height, target_aspect_ratio) for img in images)
        for _ in tqdm.tqdm(executor.map(resize_image, args), total=len(images)):
            pass

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python resize_priors.py SOURCE_DIR TARGET_IMG_HEIGHT ASPECT_RATIO ('None' to preserve aspect ratio)")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], None if sys.argv[3] == "None" else float(sys.argv[3]))