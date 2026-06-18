"""
Satu file untuk:
1) Training CNN daun tomat dari plant96.npz
2) Evaluasi + confusion matrix + grafik akurasi/loss
3) Demo scan 1 gambar random per kelas
4) Web Streamlit untuk upload gambar daun dan prediksi penyakit

Cara pakai:
- Training + evaluasi + demo scan:
    python tomato_leaf_cnn_train_streamlit.py --train

- Membuka website Streamlit:
    streamlit run tomato_leaf_cnn_train_streamlit.py

Syarat training:
- File plant96.npz harus satu folder dengan script ini.

Output training:
- hasil_tomato_cnn_plant96/model_tomato_cnn_plant96.keras
- hasil_tomato_cnn_plant96/classification_report.txt
- hasil_tomato_cnn_plant96/confusion_matrix.png
- hasil_tomato_cnn_plant96/grafik_akurasi.png
- hasil_tomato_cnn_plant96/grafik_loss.png
- hasil_tomato_cnn_plant96/demo_pipeline_<kelas>.png
- hasil_tomato_cnn_plant96/demo_scan_per_kelas.csv
"""

import os
import sys
import time
from pathlib import Path

# Mengurangi log TensorFlow yang terlalu ramai.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")

import numpy as np
import pandas as pd


# =========================================================
# KONFIGURASI UMUM
# =========================================================

SEED = 42
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 30
SAMPLES_PER_CLASS = 300
NPZ_PATH = Path("plant96.npz")

OUTPUT_DIR = Path("hasil_tomato_cnn_plant96")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_OUTPUT_PATH = OUTPUT_DIR / "model_tomato_cnn_plant96.keras"

ORIGINAL_CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple healthy",
    "Blueberry healthy",
    "Cherry healthy",
    "Cherry Powdery Mildew",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn healthy",
    "Corn Northern Leaf Blight",
    "Grape Black Rot",
    "Grape Black Measles",
    "Grape Leaf Blight",
    "Grape healthy",
    "Orange Huanglongbing",
    "Peach Bacterial Spot",
    "Peach healthy",
    "Bell Pepper Bacterial Spot",
    "Bell Pepper healthy",
    "Potato Early Blight",
    "Potato healthy",
    "Potato Late Blight",
    "Raspberry healthy",
    "Soybean healthy",
    "Squash Powdery Mildew",
    "Strawberry Healthy",
    "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Two Spotted Spider Mite",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato healthy",
]

TARGET_CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Healthy",
]

TARGET_ORIGINAL_NAMES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato healthy",
]

TARGET_ORIGINAL_LABELS = [ORIGINAL_CLASS_NAMES.index(name) for name in TARGET_ORIGINAL_NAMES]
LABEL_REMAP = {old_label: new_label for new_label, old_label in enumerate(TARGET_ORIGINAL_LABELS)}


# =========================================================
# FUNGSI TRAINING DAN EVALUASI
# =========================================================

def import_training_dependencies():
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    return plt, tf, train_test_split, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_npz_dataset(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(
            f"File {npz_path} tidak ditemukan.\n"
            "Download plant96.npz dari release GitHub, lalu letakkan satu folder dengan script ini."
        )

    try:
        data = np.load(npz_path)
    except Exception as e:
        raise RuntimeError(
            f"Gagal membuka {npz_path}. Kemungkinan file rusak atau bukan file .npz asli.\n"
            "Solusi: hapus plant96.npz, download ulang dari GitHub release, lalu coba lagi.\n"
            f"Detail error: {e}"
        )

    required_keys = ["train_images", "train_labels", "test_images", "test_labels"]
    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Key '{key}' tidak ditemukan di file .npz. Key tersedia: {data.files}")

    train_images = data["train_images"]
    train_labels = data["train_labels"].astype(int)
    test_images = data["test_images"]
    test_labels = data["test_labels"].astype(int)

    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    images = images.astype("float32")
    if images.max() > 1.0:
        images = images / 255.0

    return images, labels


def make_tomato_subset(images, labels):
    selected_images = []
    selected_labels = []
    rng = np.random.default_rng(SEED)

    print("Label asli yang digunakan:")
    for old_label, target_name in zip(TARGET_ORIGINAL_LABELS, TARGET_CLASS_NAMES):
        idx = np.where(labels == old_label)[0]
        print(f"- {target_name:16s} | label asli {old_label:2d} | jumlah tersedia: {len(idx)}")

        if len(idx) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Kelas {target_name} hanya memiliki {len(idx)} data, "
                f"kurang dari target {SAMPLES_PER_CLASS}."
            )

        chosen_idx = rng.choice(idx, size=SAMPLES_PER_CLASS, replace=False)
        selected_images.append(images[chosen_idx])
        selected_labels.append(np.full(SAMPLES_PER_CLASS, LABEL_REMAP[old_label], dtype=np.int32))

    x = np.concatenate(selected_images, axis=0)
    y = np.concatenate(selected_labels, axis=0)

    order = rng.permutation(len(y))
    x = x[order]
    y = y[order]

    return x, y


def make_dataset(tf, images, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_cnn_model(tf):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(8 / 360),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(96, 96, 3)),
        data_augmentation,
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(TARGET_CLASS_NAMES), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def choose_one_random_test_image_per_class(labels, class_names, seed=SEED):
    rng = np.random.default_rng(seed + 123)
    selected_indices = []

    for class_index, class_name in enumerate(class_names):
        indices = np.where(labels == class_index)[0]
        if len(indices) == 0:
            print(f"[WARNING] Tidak ada data uji untuk kelas: {class_name}")
            continue
        selected_indices.append(int(rng.choice(indices)))

    return selected_indices


def prepare_pipeline_image(tf, image_array):
    original = image_array.astype("float32")

    if original.max() > 1.0:
        original_display = original / 255.0
    else:
        original_display = original.copy()

    resized = tf.image.resize(original_display, IMG_SIZE).numpy()

    normalized = resized.astype("float32")
    if normalized.max() > 1.0:
        normalized = normalized / 255.0

    return original_display, resized, normalized


def predict_pipeline_image(model, normalized_image):
    input_image = np.expand_dims(normalized_image, axis=0)
    probs = model.predict(input_image, verbose=0)[0]

    pred_index = int(np.argmax(probs))
    pred_class = TARGET_CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index] * 100)

    return probs, pred_index, pred_class, confidence


def save_pipeline_visual(plt, tf, model, x_test, y_test, index, output_dir=OUTPUT_DIR):
    image_array = x_test[index]
    true_index = int(y_test[index])
    true_class = TARGET_CLASS_NAMES[true_index]

    original_display, resized, normalized = prepare_pipeline_image(tf, image_array)
    probs, pred_index, pred_class, confidence = predict_pipeline_image(model, normalized)

    status = "BENAR" if pred_index == true_index else "SALAH"

    print("\n" + "=" * 70)
    print("DEMO SCAN DAUN TOMAT")
    print("=" * 70)
    print(f"Index data uji       : {index}")
    print(f"Label asli           : {true_class}")
    print(f"Hasil prediksi       : {pred_class}")
    print(f"Confidence           : {confidence:.2f}%")
    print(f"Status               : {status}")
    print(f"Ukuran input         : {image_array.shape}")
    print(f"Setelah resize       : {resized.shape}")
    print(f"Rentang piksel awal  : {image_array.min():.4f} sampai {image_array.max():.4f}")
    print(f"Rentang normalisasi  : {normalized.min():.4f} sampai {normalized.max():.4f}")

    print("\nProbabilitas tiap kelas:")
    for class_name, prob in zip(TARGET_CLASS_NAMES, probs):
        print(f"- {class_name:16s}: {prob * 100:.2f}%")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    axes[0].imshow(np.clip(original_display, 0, 1))
    axes[0].set_title("1. Gambar Daun")
    axes[0].axis("off")

    axes[1].imshow(np.clip(resized, 0, 1))
    axes[1].set_title("2. Resize 96×96")
    axes[1].axis("off")

    axes[2].imshow(np.clip(normalized, 0, 1))
    axes[2].set_title("3. Normalisasi 0–1")
    axes[2].axis("off")

    axes[3].barh(TARGET_CLASS_NAMES, probs * 100)
    axes[3].set_xlim(0, 100)
    axes[3].set_xlabel("Probabilitas (%)")
    axes[3].set_title("4. Output Model")

    fig.suptitle(
        f"Label Asli: {true_class} | Prediksi: {pred_class} | Confidence: {confidence:.2f}% | {status}",
        fontsize=12,
    )

    plt.tight_layout()

    safe_class_name = true_class.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"demo_pipeline_{safe_class_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "index": index,
        "label_asli": true_class,
        "prediksi": pred_class,
        "confidence": confidence,
        "status": status,
        "output_gambar": str(output_path),
    }


def run_demo_scan_per_class(plt, tf, model, x_test, y_test):
    print("\n\n============================================================")
    print("DEMO PIPELINE SCAN: 1 GAMBAR RANDOM PER KELAS")
    print("Alur: Gambar -> Resize -> Normalisasi -> Model -> Probabilitas -> Output")
    print("============================================================")

    selected_indices = choose_one_random_test_image_per_class(y_test, TARGET_CLASS_NAMES, SEED)

    demo_results = []
    for index in selected_indices:
        result = save_pipeline_visual(plt, tf, model, x_test, y_test, index)
        demo_results.append(result)

    demo_df = pd.DataFrame(demo_results)
    demo_csv_path = OUTPUT_DIR / "demo_scan_per_kelas.csv"
    demo_df.to_csv(demo_csv_path, index=False)

    print("\nRingkasan demo scan per kelas:")
    print(demo_df[["index", "label_asli", "prediksi", "confidence", "status"]])
    print("\nFile ringkasan demo disimpan di:")
    print(demo_csv_path.resolve())
    print("\nGambar pipeline tiap kelas disimpan di folder:")
    print(OUTPUT_DIR.resolve())


def run_training_pipeline():
    plt, tf, train_test_split, classification_report, confusion_matrix, ConfusionMatrixDisplay = import_training_dependencies()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    all_images, all_labels = load_npz_dataset(NPZ_PATH)
    x, y = make_tomato_subset(all_images, all_labels)

    print("\nJumlah total subset:", len(y))
    print("Distribusi kelas subset:")
    for i, name in enumerate(TARGET_CLASS_NAMES):
        print(f"- {name:16s}: {np.sum(y == i)} citra")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    print("\nPembagian data:")
    print("Data latih    :", len(y_train))
    print("Data validasi :", len(y_val))
    print("Data uji      :", len(y_test))

    train_ds = make_dataset(tf, x_train, y_train, shuffle=True)
    val_ds = make_dataset(tf, x_val, y_val)
    test_ds = make_dataset(tf, x_test, y_test)

    model = build_cnn_model(tf)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    training_time = time.time() - start_time
    print(f"\nWaktu pelatihan: {training_time:.2f} detik")

    test_loss, test_accuracy = model.evaluate(test_ds)

    print("\nHasil Evaluasi Data Uji")
    print(f"Loss    : {test_loss:.4f}")
    print(f"Akurasi : {test_accuracy * 100:.2f}%")

    y_pred_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test

    report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_CLASS_NAMES,
        digits=4,
    )

    print("\nClassification Report:")
    print(report)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write("Classification Report\n")
        file.write("=====================\n\n")
        file.write(report)
        file.write(f"\nTest Loss: {test_loss:.4f}")
        file.write(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
        file.write(f"\nTraining Time: {training_time:.2f} detik")

    summary_df = pd.DataFrame({
        "Metrik": ["Akurasi", "Loss", "Waktu Training (detik)"],
        "Nilai": [f"{test_accuracy * 100:.2f}%", f"{test_loss:.4f}", f"{training_time:.2f}"],
    })
    summary_df.to_csv(OUTPUT_DIR / "ringkasan_hasil.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_CLASS_NAMES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix CNN Daun Tomat")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Grafik Akurasi Training dan Validasi")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grafik_akurasi.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Grafik Loss Training dan Validasi")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grafik_loss.png", dpi=300)
    plt.close()

    model.save(MODEL_OUTPUT_PATH)

    print("\nModel dan hasil evaluasi berhasil disimpan di folder:")
    print(OUTPUT_DIR.resolve())
    print("\nModel tersimpan di:")
    print(MODEL_OUTPUT_PATH.resolve())

    run_demo_scan_per_class(plt, tf, model, x_test, y_test)


# =========================================================
# STREAMLIT WEB APP
# =========================================================

def is_running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit_app():
    import streamlit as st
    from PIL import Image
    import tensorflow as tf

    st.set_page_config(
        page_title="Klasifikasi Penyakit Daun Tomat",
        page_icon="🍅",
        layout="wide",
    )

    model_candidates = [
        OUTPUT_DIR / "model_tomato_cnn_plant96.keras",
        OUTPUT_DIR / "model_tomato_cnn.keras",
        Path("model_tomato_cnn_plant96.keras"),
        Path("model_tomato_cnn.keras"),
    ]

    @st.cache_resource
    def load_trained_model():
        for model_path in model_candidates:
            if model_path.exists():
                return tf.keras.models.load_model(model_path), model_path
        return None, None

    def preprocess_uploaded_image(uploaded_image):
        original_image = Image.open(uploaded_image).convert("RGB")
        resized_image = original_image.resize(IMG_SIZE)
        image_array = np.array(resized_image).astype(np.float32)
        normalized_image = image_array / 255.0
        input_batch = np.expand_dims(normalized_image, axis=0)
        return original_image, resized_image, image_array, normalized_image, input_batch

    def predict_uploaded_image(model, input_batch):
        prediction = model.predict(input_batch, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = TARGET_CLASS_NAMES[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        probability_df = pd.DataFrame({
            "Kelas": TARGET_CLASS_NAMES,
            "Probabilitas (%)": prediction * 100,
        }).sort_values("Probabilitas (%)", ascending=False)

        return prediction, predicted_class, confidence, probability_df

    st.title("🍅 Klasifikasi Penyakit Daun Tomat Menggunakan CNN")
    st.write(
        "Website ini memakai model hasil training dari program yang sama. "
        "Upload gambar daun tomat, lalu sistem akan melakukan resize, normalisasi, "
        "prediksi penyakit, dan menampilkan probabilitas setiap kelas."
    )

    model, model_path = load_trained_model()

    with st.sidebar:
        st.header("Menu")
        st.write("**Mode:** Prediksi gambar daun")
        st.write("**Input model:** `.keras`")
        st.write("**Ukuran input:** 96×96 piksel")
        st.divider()
        st.write("**Kelas:**")
        for class_name in TARGET_CLASS_NAMES:
            st.write(f"- {class_name}")

    if model is None:
        st.error("Model .keras belum ditemukan.")
        st.write("Jalankan training dulu dengan perintah:")
        st.code("python tomato_leaf_cnn_train_streamlit.py --train", language="bash")
        st.write("Atau pastikan model ada di salah satu lokasi berikut:")
        st.code("\n".join(str(p) for p in model_candidates), language="text")
        st.stop()

    st.success(f"Model berhasil dimuat: {model_path}")

    with st.expander("Lihat alur kerja program"):
        st.markdown(
            """
            **Alur pembacaan gambar:**

            1. Gambar daun di-upload oleh pengguna.  
            2. Gambar diubah ke format RGB.  
            3. Gambar di-resize menjadi **96×96 piksel**.  
            4. Nilai piksel dinormalisasi dari rentang **0–255** menjadi **0–1**.  
            5. Gambar masuk ke model CNN.  
            6. Model menghasilkan probabilitas untuk setiap kelas.  
            7. Kelas dengan probabilitas tertinggi menjadi hasil prediksi.  
            """
        )

    uploaded_file = st.file_uploader(
        "Upload gambar daun tomat",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        original_image, resized_image, image_array, normalized_image, input_batch = preprocess_uploaded_image(uploaded_file)
        prediction, predicted_class, confidence, probability_df = predict_uploaded_image(model, input_batch)

        st.divider()
        st.subheader("Hasil Prediksi")

        col_result_1, col_result_2, col_result_3 = st.columns(3)
        with col_result_1:
            st.metric("Prediksi Penyakit", predicted_class)
        with col_result_2:
            st.metric("Confidence", f"{confidence:.2f}%")
        with col_result_3:
            if confidence >= 70:
                st.success("Keyakinan tinggi")
            elif confidence >= 50:
                st.warning("Keyakinan sedang")
            else:
                st.error("Keyakinan rendah")

        if confidence < 50:
            st.info(
                "Confidence masih rendah. Ini wajar pada model baseline CNN sederhana. "
                "Gunakan hasil sebagai demonstrasi sistem, bukan diagnosis pertanian final."
            )

        st.divider()
        st.subheader("Proses Pembacaan Gambar")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_image, caption="1. Gambar daun asli", use_container_width=True)
            st.write(f"Ukuran asli: `{original_image.size}`")
        with col2:
            st.image(resized_image, caption="2. Resize menjadi 96×96", use_container_width=True)
            st.write(f"Ukuran resize: `{resized_image.size}`")
        with col3:
            st.image(normalized_image, caption="3. Normalisasi 0–1", use_container_width=True)
            st.write(
                f"Rentang piksel normalisasi: "
                f"`{normalized_image.min():.4f}` sampai `{normalized_image.max():.4f}`"
            )

        st.divider()
        st.subheader("Probabilitas Tiap Kelas")
        st.dataframe(probability_df, use_container_width=True, hide_index=True)
        st.bar_chart(probability_df.set_index("Kelas"))

        st.divider()
        st.subheader("Kesimpulan Output")
        st.write(
            f"Berdasarkan probabilitas tertinggi, gambar daun diprediksi sebagai "
            f"**{predicted_class}** dengan confidence **{confidence:.2f}%**."
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    if is_running_under_streamlit():
        run_streamlit_app()
    elif "--train" in sys.argv:
        run_training_pipeline()
    else:
        print("Program gabungan CNN + Streamlit daun tomat")
        print("\nCara pakai:")
        print("1. Training + evaluasi + demo scan:")
        print("   python tomato_leaf_cnn_train_streamlit.py --train")
        print("\n2. Membuka website upload gambar:")
        print("   streamlit run tomato_leaf_cnn_train_streamlit.py")
        print("\nPastikan plant96.npz tersedia untuk training.")
