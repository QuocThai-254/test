import os
import hashlib
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def get_image_info(img_path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            mode = img.mode
            format = img.format
            return width, height, mode, format
    except Exception as e:
        return None, None, None, str(e)


def calculate_hash(img_path):
    hash_md5 = hashlib.md5()
    with open(img_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def perform_eda(data_dir, output_dir="eda_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    classes = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    print(f"Scanning {data_dir}...")
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = os.listdir(cls_path)
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                img_path = os.path.join(cls_path, f)
                w, h, m, fmt = get_image_info(img_path)
                md5 = calculate_hash(img_path)
                data.append(
                    {
                        "class": cls,
                        "filename": f,
                        "path": img_path,
                        "width": w,
                        "height": h,
                        "mode": m,
                        "format": fmt,
                        "md5": md5,
                        "size_kb": os.path.getsize(img_path) / 1024,
                    }
                )

    df = pd.DataFrame(data)

    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="class")
    plt.title(f"Class Distribution in {data_dir}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"class_dist_{os.path.basename(data_dir)}.png")
    )
    plt.close()

    # 2. Image Dimensions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="width", y="height", hue="class")
    plt.title("Image Dimensions by Class")
    plt.savefig(os.path.join(output_dir, f"img_dims_{os.path.basename(data_dir)}.png"))
    plt.close()

    # 3. Duplicate Detection
    dupes = df[df.duplicated("md5", keep=False)]
    if not dupes.empty:
        print(f"Found {len(dupes)} duplicate images (by MD5 hash) in {data_dir}!")
        dupes.to_csv(
            os.path.join(output_dir, f"duplicates_{os.path.basename(data_dir)}.csv"),
            index=False,
        )
    else:
        print(f"No exact duplicates found in {data_dir}.")

    return df


if __name__ == "__main__":
    train_df = perform_eda("data_train")
    test_df = perform_eda("data_test")

    print("\nTraining set summary:")
    print(train_df["class"].value_counts())

    print("\nTest set summary:")
    print(test_df["class"].value_counts())

    # Check for leakage (train in test)
    leakage = train_df[train_df["md5"].isin(test_df["md5"])]
    if not leakage.empty:
        print(
            f"\nWARNING: Found {len(leakage)} images from training set in test set (Data Leakage)!"
        )
        leakage.to_csv("eda_results/data_leakage.csv", index=False)
    else:
        print("\nNo data leakage detected between train and test sets.")
