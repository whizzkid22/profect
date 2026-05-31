import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_CUDA"] = "False"
import base64
import io
import json
import re
import sys
import asyncio
import httpx
import math
import warnings
import tempfile
import numpy as np
from typing import Any, Optional
from fastapi.responses import JSONResponse, FileResponse, Response

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Hair Vision Analyzer Main")
_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_HERE, "static")

# ── API KEY ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── OpenAI / OpenRouter ───────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

# ── ML-стек (ленивая загрузка) ────────────────────────────────────────────────
_ML_LOADED = False
_ML_BUNDLE = {}
_ML_ERROR  = ""

FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "deepseek/deepseek-v4-flash:free",
]

# ══════════════════════════════════════════════════════════════════════════════
def _load_ml_models_internal():
    global _ML_LOADED, _ML_ERROR, _ML_BUNDLE
    if _ML_LOADED:
        return
    try:
        import cv2
        import mediapipe as mp
        import torch
        import tensorflow as tf
        from deepface import DeepFace
        from PIL import Image

        # ── Вспомогательная функция пути ──────────────────────────────────────
        def _model_path(env_var, default_name):
            from_env = os.environ.get(env_var)
            if from_env:
                return from_env
            candidate = os.path.join(_HERE, default_name)
            return candidate if os.path.exists(candidate) else default_name

        # ── MediaPipe Hair Segmenter ──────────────────────────────────────────
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        _HAIR_SEG_PATH = os.path.join(_HERE, "hair_segmenter.tflite")
        if not os.path.exists(_HAIR_SEG_PATH):
            import urllib.request
            print("⬇️  Скачиваю hair_segmenter.tflite…")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
                _HAIR_SEG_PATH,
            )
            print("✅ hair_segmenter.tflite скачан")

        _seg_opts = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_HAIR_SEG_PATH),
            output_category_mask=True,
        )
        hair_segmenter = mp_vision.ImageSegmenter.create_from_options(_seg_opts)

        def _get_hair_mask(rgb_img: np.ndarray) -> np.ndarray:
            """Возвращает uint8 маску 0/1 того же размера что входное изображение."""
            # rgb_img должен быть contiguous uint8
            img_c = np.ascontiguousarray(rgb_img.astype(np.uint8))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_c)
            result   = hair_segmenter.segment(mp_image)
            # category_mask — это mp.Image, извлекаем numpy через numpy_view()
            cat = result.category_mask.numpy_view()  # HxW, uint8
            # Копируем чтобы освободить буфер MediaPipe
            return (cat == 1).astype(np.uint8).copy()

        print("✅ MediaPipe Hair Segmenter загружен")

        # ── MediaPipe Face Landmarker ─────────────────────────────────────────
        _LANDMARKER_PATH = _model_path("MEDIAPIPE_MODEL", "face_landmarker.task")
        if not os.path.exists(_LANDMARKER_PATH):
            import urllib.request
            print("⬇️  Скачиваю face_landmarker.task…")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                _LANDMARKER_PATH,
            )
            print("✅ face_landmarker.task скачан")

        _lm_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_LANDMARKER_PATH),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(_lm_opts)
        print("✅ Face Landmarker загружен")

        # ── Keras / TF патч ───────────────────────────────────────────────────
        import keras

        @classmethod
        def _bn_from_config(cls, config):
            config = dict(config)
            ax = config.get("axis")
            if isinstance(ax, (list, tuple)):
                config["axis"] = ax[0] if ax else -1
            for key in ("virtual_batch_size", "adjustment"):
                config.pop(key, None)
            return super(keras.layers.BatchNormalization, cls).from_config(config)

        keras.layers.BatchNormalization.from_config = _bn_from_config

        # ── Табличные модели ──────────────────────────────────────────────────
        import json as _json
        from xgboost import XGBClassifier
        import lightgbm as _lgb
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        _F = lambda name: os.path.join(_HERE, name)

        def _need(fname):
            p = _F(fname)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Файл не найден: {p}")
            return p

        xgb_model = XGBClassifier()
        xgb_model.load_model(_need("xgb_model.json"))

        lgb_booster = _lgb.Booster(model_file=_need("lgb_model.txt"))

        class _LGBWrapper:
            def predict_proba(self, X):
                raw = lgb_booster.predict(X)
                if raw.ndim == 1:
                    return np.column_stack([1 - raw, raw])
                return raw

        lgb_model_final = _LGBWrapper()

        import pickle as _pkl
        with open(_need("gb_model.bin"), "rb") as _f:
            gb_model = _pkl.load(_f)

        _sd = _json.load(open(_need("scaler.json")))
        scaler = StandardScaler()
        scaler.mean_           = np.array(_sd["mean_"])
        scaler.scale_          = np.array(_sd["scale_"])
        scaler.var_            = np.array(_sd["var_"])
        scaler.n_features_in_  = _sd["n_features_in_"]
        scaler.n_samples_seen_ = _sd["n_samples_seen_"]

        _te = _json.load(open(_need("target_enc.json")))
        target_enc = LabelEncoder()
        target_enc.classes_ = np.array(_te["classes_"])
        nc = len(target_enc.classes_)

        feature_cols = _json.load(open(_need("feature_cols.json")))
        weights      = tuple(_json.load(open(_need("weights.json"))))

        _ce = _json.load(open(_need("cat_encoders.json")))
        cat_encoders = {}
        for col, classes in _ce.items():
            enc = LabelEncoder()
            enc.classes_ = np.array(classes)
            cat_encoders[col] = enc

        # ── CNN ───────────────────────────────────────────────────────────────
        _cnn_npy   = os.path.join(_HERE, "cnn_weights.npy")
        _cnn_h5    = os.path.join(_HERE, "cnn_model.h5")
        _cnn_keras = os.path.join(_HERE, "cnn_bisenet.keras")

        def _build_cnn_arch(n_classes):
            from tensorflow.keras import layers as KL, models as KM
            m = KM.Sequential([
                KL.Input(shape=(96, 96, 3)),
                KL.RandomFlip("horizontal"), KL.RandomRotation(0.05),
                KL.RandomZoom(0.1), KL.RandomBrightness(0.1),
                KL.Conv2D(32,  3, activation="relu", padding="same"),
                KL.BatchNormalization(), KL.MaxPooling2D(), KL.Dropout(0.2),
                KL.Conv2D(64,  3, activation="relu", padding="same"),
                KL.BatchNormalization(), KL.MaxPooling2D(), KL.Dropout(0.2),
                KL.Conv2D(128, 3, activation="relu", padding="same"),
                KL.BatchNormalization(), KL.GlobalAveragePooling2D(),
                KL.Dense(256, activation="relu"), KL.Dropout(0.4),
                KL.Dense(128, activation="relu"), KL.Dropout(0.3),
                KL.Dense(n_classes, activation="softmax"),
            ])
            return m

        cnn_model = None

        if os.path.exists(_cnn_npy):
            try:
                cnn_model = _build_cnn_arch(nc)
                cnn_model(np.zeros((1, 96, 96, 3), dtype="float32"), training=False)
                saved_weights = np.load(_cnn_npy, allow_pickle=True)
                for w, val in zip(cnn_model.weights, saved_weights):
                    w.assign(val)
                print("✅ CNN: загружена из cnn_weights.npy")
            except Exception as _e_npy:
                cnn_model = None

        if cnn_model is None:
            for _cnn_path in [_cnn_h5, _cnn_keras]:
                if not os.path.exists(_cnn_path):
                    continue
                try:
                    try:
                        cnn_model = tf.keras.models.load_model(
                            _cnn_path, compile=False, safe_mode=False)
                    except TypeError:
                        cnn_model = tf.keras.models.load_model(
                            _cnn_path, compile=False)
                    break
                except Exception:
                    pass

        if cnn_model is None:
            raise FileNotFoundError("CNN не загружена")

        cnn_model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

        # ── Bundle — только то что реально существует ─────────────────────────
        _ML_BUNDLE = {
            "cv2": cv2,
            "np": np,
            "Image": Image,
            "DeepFace": DeepFace,
            "mp": mp,
            "get_hair_mask": _get_hair_mask,
            "face_landmarker": face_landmarker,
            "cnn": cnn_model,
            "xgb": xgb_model,
            "lgb": lgb_model_final,
            "gb":  gb_model,
            "scaler": scaler,
            "tenc": target_enc,
            "fcols": feature_cols,
            "cencs": cat_encoders,
            "weights": weights,
        }
        _ML_LOADED = True
        print("✅ ML-модели загружены")

    except Exception as e:
        import traceback
        _ML_ERROR = f"{e}\n\nTraceback:\n{traceback.format_exc()}"
        _ML_LOADED = True
        print(f"❌ Ошибка загрузки ML: {e}")

# ══════════════════════════════════════════════════════════════════════════════

_LM_IDX = {
    "jaw_left": 234, "jaw_right": 454, "chin": 152,
    "forehead_left": 10, "forehead_right": 338, "forehead_center": 8,
    "cheek_left": 50, "cheek_right": 280,
    "temple_left": 127, "temple_right": 356,
    "jaw_tip_1": 172, "jaw_tip_2": 397,
    "left_eye_left": 33, "left_eye_right": 133,
    "right_eye_left": 362, "right_eye_right": 263,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "nose_tip": 1, "nose_bridge": 6,
    "nose_left": 49, "nose_right": 279,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top": 0, "mouth_bottom": 17,
    "left_eyebrow_center": 70, "right_eyebrow_center": 300,
}

_FACE_TIPS = {
    "oval":   "Универсальная форма — подходит почти любая стрижка. Можно подчеркнуть скулы объёмом на висках.",
    "round":  "Удлиняющие стрижки: длинные слои, асимметрия, высокий топ. Избегай каре на уровне щёк.",
    "square": "Смягчай углы: волны, слои, чёлка-занавеска. Избегай прямых чётких линий.",
    "heart":  "Добавляй объём снизу: боб до подбородка, волны от скул. Избегай пышных причёсок на макушке.",
    "oblong": "Широкие стрижки: боб, каре, чёлка визуально укорачивают лицо. Избегай длинных прямых волос.",
}

_HAIR_TYPE_TIPS = {
    "Straight":   "Прямые волосы хорошо держат объём у корней. Подходят укладки с феном и брашингом.",
    "Wavy":       "Волнистые волосы выглядят лучше с лёгкими слоями и диффузором.",
    "Curly":      "Кудрявым волосам нужно увлажнение. Метод CG (Curly Girl) даёт отличные результаты.",
    "Kinky":      "Очень кудрявые волосы требуют максимального увлажнения и минимальной термообработки.",
    "Dreadlocks": "Дреды требуют специального ухода: разделение у корней, увлажнение маслами.",
}

_HAIR_TYPE_LABELS_RU = {
    "Straight":   "Прямые",
    "Wavy":       "Волнистые",
    "Curly":      "Кудрявые",
    "Kinky":      "Афро (очень кудрявые)",
    "Dreadlocks": "Дреды",
}


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _face_shape(fr, fw, jw):
    if fr > 1.55:            return "oblong"
    if fr < 1.05:            return "round"
    if abs(fw - jw) < 0.05: return "square"
    if fw > jw + 0.08:      return "heart"
    return "oval"


def _make_bald(img_rgb: np.ndarray, get_hair_mask, face_landmarker=None) -> np.ndarray:
    """
    Симуляция лысины через MediaPipe Hair Segmenter + HSV заливка цветом кожи.
    """
    import cv2

    h, w = img_rgb.shape[:2]

    # 1. Точная маска волос (MediaPipe Hair Segmenter работает на любом размере)
    hair_mask = get_hair_mask(img_rgb)  # uint8 0/1, HxW

    # 2. Морфология — заполнить дыры внутри волос
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, k)

    hair_px = int(np.sum(hair_mask))
    print(f"DEBUG _make_bald: hair_pixels={hair_px}, img={h}x{w}")

    if hair_px < 300:
        print("⚠️ Волосы не найдены, возвращаем оригинал")
        return img_rgb

    # 3. Цвет кожи через HSV — ищем телесные пиксели вне зоны волос
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    skin_mask_bool = (
        (hsv[:, :, 0] >= 0)   & (hsv[:, :, 0] <= 30)  &
        (hsv[:, :, 1] >= 15)  & (hsv[:, :, 1] <= 170) &
        (hsv[:, :, 2] >= 70)
    ) & (hair_mask == 0)
    skin_px = img_rgb[skin_mask_bool]

    if len(skin_px) > 50:
        skin_color = np.median(skin_px, axis=0).astype(np.float32)
    else:
        skin_color = np.array([200, 170, 140], dtype=np.float32)
    print(f"DEBUG skin_color: RGB={skin_color.astype(int).tolist()}, px={len(skin_px)}")

    # 4. Заливка с адаптацией к освещению
    img_f    = img_rgb.astype(np.float32)
    lum_orig = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    lum_skin = (0.299*skin_color[0] + 0.587*skin_color[1] +
                0.114*skin_color[2] + 1e-3)
    lum_sm   = cv2.GaussianBlur(lum_orig, (31, 31), 0)
    ratio    = np.clip(lum_sm / lum_skin, 0.5, 1.5)[:, :, None]
    fill_map = np.clip(skin_color[None, None, :] * ratio, 0, 255)

    # 5. Мягкие края через размытие маски
    alpha  = cv2.GaussianBlur(hair_mask.astype(np.float32), (21, 21), 0)
    alpha  = np.clip(alpha, 0, 1)[:, :, None]

    result = fill_map * alpha + img_f * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _run_ml_inference(image_bytes: bytes) -> dict:
    b  = _ML_BUNDLE
    cv2          = b["cv2"]
    np_          = b["np"]        # не перекрываем глобальный np
    Image        = b["Image"]
    DeepFace     = b["DeepFace"]
    mp           = b["mp"]
    get_hair_mask= b["get_hair_mask"]

    arr     = np_.frombuffer(image_bytes, dtype=np_.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Не удалось декодировать изображение")

    img_rgb           = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig    = img_bgr.shape[:2]

    # ── Маска волос для CNN-thumb (256x256) ───────────────────────────────────
    small_bgr = cv2.resize(img_bgr, (256, 256))
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    hair_mask_raw = get_hair_mask(small_rgb)
    kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hair_uint     = cv2.morphologyEx(hair_mask_raw, cv2.MORPH_CLOSE,  kernel)
    hair_uint     = cv2.morphologyEx(hair_uint,     cv2.MORPH_DILATE, kernel)
    hair_mask_bool= hair_uint.astype(bool)

    print(f"DEBUG hair_mask (256x256): sum={np_.sum(hair_mask_raw)}")

    # Цвет кожи для CNN thumb
    fz       = np_.zeros(small_rgb.shape[:2], bool)
    fz[int(256 * 0.15):, :] = True
    skin_px  = small_rgb[fz & ~hair_mask_bool]
    skin_col = (np_.median(skin_px, axis=0).astype(np_.uint8)
                if len(skin_px) > 50 else np_.array([210, 180, 150], np_.uint8))

    result_img = small_rgb.copy()
    result_img[hair_mask_bool] = skin_col
    blur       = cv2.GaussianBlur(hair_uint.astype(np_.float32), (15, 15), 0)[:, :, None]
    result_img = (result_img * blur + small_rgb * (1 - blur)).astype(np_.uint8)
    result_img[hair_mask_bool] = skin_col

    thumb = cv2.resize(result_img, (96, 96))

    # ── Face landmarks ────────────────────────────────────────────────────────
    face_landmarker = b["face_landmarker"]
    mp_image        = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    det             = face_landmarker.detect(mp_image)
    if not det.face_landmarks:
        raise ValueError("Лицо не обнаружено на фото")

    lms = det.face_landmarks[0]
    pts = {name: (lms[idx].x * w_orig, lms[idx].y * h_orig)
           for name, idx in _LM_IDX.items()}

    face_w   = _dist(pts["jaw_left"],       pts["jaw_right"])
    face_h   = _dist(pts["chin"],           pts["forehead_center"])
    fore_w   = _dist(pts["forehead_left"],  pts["forehead_right"])
    cheek_w  = _dist(pts["cheek_left"],     pts["cheek_right"])
    jaw_w    = _dist(pts["jaw_tip_1"],      pts["jaw_tip_2"])
    temple_w = _dist(pts["temple_left"],    pts["temple_right"])
    le_w     = _dist(pts["left_eye_left"],  pts["left_eye_right"])
    re_w     = _dist(pts["right_eye_left"], pts["right_eye_right"])
    le_h     = _dist(pts["left_eye_top"],   pts["left_eye_bottom"])
    re_h     = _dist(pts["right_eye_top"],  pts["right_eye_bottom"])
    eye_d    = _dist(pts["left_eye_right"], pts["right_eye_left"])
    nose_len = _dist(pts["nose_bridge"],    pts["nose_tip"])
    nose_w   = _dist(pts["nose_left"],      pts["nose_right"])
    mouth_w  = _dist(pts["mouth_left"],     pts["mouth_right"])
    mouth_h  = _dist(pts["mouth_top"],      pts["mouth_bottom"])
    eps = 1e-8

    feats = {
        "face_width":               face_w,
        "face_height":              face_h,
        "face_ratio":               face_h  / (face_w  + eps),
        "forehead_width":           fore_w,
        "forehead_to_face_width":   fore_w  / (face_w  + eps),
        "cheek_width":              cheek_w,
        "cheek_to_face_width":      cheek_w / (face_w  + eps),
        "jaw_width":                jaw_w,
        "jaw_to_face_width":        jaw_w   / (face_w  + eps),
        "temple_width":             temple_w,
        "temple_to_face_width":     temple_w / (face_w + eps),
        "jaw_angle":                math.degrees(math.atan2(
            abs(pts["jaw_left"][1]-pts["chin"][1]),
            abs(pts["jaw_left"][0]-pts["chin"][0]))),
        "cheek_to_forehead":        cheek_w / (fore_w + eps),
        "cheek_to_jaw":             cheek_w / (jaw_w  + eps),
        "forehead_to_jaw":          fore_w  / (jaw_w  + eps),
        "left_eye_width":           le_w,
        "right_eye_width":          re_w,
        "left_eye_height":          le_h,
        "right_eye_height":         re_h,
        "eye_distance":             eye_d,
        "eye_distance_ratio":       eye_d  / (face_w + eps),
        "eye_width_ratio":          le_w   / (re_w   + eps),
        "eye_width_to_height":      (le_w+re_w) / (le_h+re_h + eps),
        "eye_asymmetry":            abs(le_w/(le_h+eps) - re_w/(re_h+eps)),
        "eye_size_symmetry":        abs(le_w*le_h - re_w*re_h) / (le_w*le_h+re_w*re_h+eps),
        "nose_length":              nose_len,
        "nose_width":               nose_w,
        "nose_ratio":               nose_len / (nose_w  + eps),
        "nose_width_to_face_width": nose_w   / (face_w + eps),
        "nose_to_face_height":      nose_len / (face_h  + eps),
        "mouth_width":              mouth_w,
        "mouth_height":             mouth_h,
        "mouth_ratio":              mouth_w / (mouth_h + eps),
        "mouth_to_face_width":      mouth_w / (face_w  + eps),
        "mouth_to_nose_width":      mouth_w / (nose_w  + eps),
    }

    face_ratio = feats["face_ratio"]
    fore_ratio = feats["forehead_to_face_width"]
    jaw_ratio  = feats["jaw_to_face_width"]
    face_shape = _face_shape(face_ratio, fore_ratio, jaw_ratio)
    feats["face_shape_algo"] = face_shape

    # ── DeepFace: возраст и пол ───────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf_:
        tmp_path = tf_.name
        cv2.imwrite(tmp_path, img_bgr)
    try:
        da = DeepFace.analyze(img_path=tmp_path, actions=["age", "gender"],
                              detector_backend="opencv",
                              enforce_detection=False, silent=True)
        if isinstance(da, list):
            da = da[0]
        age         = int(da["age"])
        gs          = da["gender"]
        gender_text = max(gs, key=gs.get)
        raw_conf    = gs[gender_text]
        gender_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
    except Exception:
        age = -1; gender_text = "Unknown"; gender_conf = 0.0
    finally:
        os.unlink(tmp_path)

    feats["gender"]            = 1 if gender_text == "Man" else 0
    feats["age"]               = age
    feats["gender_text"]       = gender_text
    feats["age_group"]         = ("young"  if age < 30 else
                                  "middle" if age < 50 else "senior")
    feats["gender_confidence"] = gender_conf

    cencs = b["cencs"]
    for col, enc in cencs.items():
        if col in feats:
            try:    feats[col] = enc.transform([str(feats[col])])[0]
            except: feats[col] = 0

    fcols  = b["fcols"]
    scaler = b["scaler"]
    X_row  = scaler.transform([[feats.get(c, 0) for c in fcols]])
    X_img  = np_.expand_dims(thumb.astype(np_.float32) / 255.0, 0)

    p_cnn = b["cnn"].predict(X_img, verbose=0)
    p_xgb = b["xgb"].predict_proba(X_row)
    p_lgb = b["lgb"].predict_proba(X_row)
    p_gb  = b["gb"].predict_proba(X_row)

    w        = b["weights"]
    combined = (w[0]*p_cnn + w[1]*p_xgb + w[2]*p_lgb + w[3]*p_gb)[0]
    tenc     = b["tenc"]
    top3     = combined.argsort()[::-1][:3]

    # ── Симуляция лысины ──────────────────────────────────────────────────────
    bald_b64 = None
    try:
        bald_rgb = _make_bald(img_rgb, get_hair_mask,
                              face_landmarker=b["face_landmarker"])

        bald_res = np_.clip(bald_rgb, 0, 255).astype(np_.uint8)
        bald_bgr = cv2.cvtColor(bald_res, cv2.COLOR_RGB2BGR)
        _, buf   = cv2.imencode(".jpg", bald_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        bald_b64 = base64.b64encode(buf.tobytes()).decode()
        print("✅ Симуляция лысины готова")

    except Exception as _bald_err:
        import traceback
        print(f"❌ Ошибка симуляции лысины: {_bald_err}\n{traceback.format_exc()}")

    # ── Результат ─────────────────────────────────────────────────────────────
    geo = {
        "face_ratio":             round(float(feats["face_ratio"]), 3),
        "forehead_to_face_width": round(float(feats["forehead_to_face_width"]), 3),
        "jaw_to_face_width":      round(float(feats["jaw_to_face_width"]), 3),
        "cheek_to_face_width":    round(float(feats["cheek_to_face_width"]), 3),
        "eye_distance_ratio":     round(float(feats["eye_distance_ratio"]), 3),
        "nose_width_to_face":     round(float(feats["nose_width_to_face_width"]), 3),
        "mouth_to_face_width":    round(float(feats["mouth_to_face_width"]), 3),
    }

    hair_type_en  = str(tenc.classes_[top3[0]])
    hair_type_ru  = _HAIR_TYPE_LABELS_RU.get(hair_type_en, hair_type_en)
    gender_conf_pct = min(round(float(gender_conf) * 100, 1), 100.0)

    return {
        "hair_type":      hair_type_ru,
        "confidence":     round(float(combined[top3[0]]) * 100, 1),
        "top3": [
            {"label": _HAIR_TYPE_LABELS_RU.get(str(tenc.classes_[i]), str(tenc.classes_[i])),
             "prob":  round(float(combined[i]) * 100, 1)}
            for i in top3
        ],
        "face_shape":     str(face_shape),
        "face_shape_tip": _FACE_TIPS.get(face_shape, ""),
        "hair_type_tip":  _HAIR_TYPE_TIPS.get(hair_type_en, ""),
        "gender":         str(gender_text),
        "gender_conf":    gender_conf_pct,
        "age":            int(age),
        "age_group":      str(feats.get("age_group", "unknown")),
        "geo":            geo,
        "model_weights":  {"cnn": float(w[0]), "xgb": float(w[1]),
                           "lgb": float(w[2]), "gb":  float(w[3])},
        "bald_image":     bald_b64,
    }


# ── API ЭНДПОИНТЫ ─────────────────────────────────────────────────────────────

@app.get("/")
async def read_index():
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Hair Vision Analyzer API is running. Check /static/index.html"}


@app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join(_STATIC_DIR, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)


@app.post("/api/ml-analyze")
async def ml_analyze(file: UploadFile = File(...)):
    """Локальный анализ лица (форма, черты, возраст) через ML стек."""
    _load_ml_models_internal()

    if _ML_ERROR and not _ML_BUNDLE:
        raise HTTPException(status_code=500,
                            detail=f"ML не загружен: {_ML_ERROR[:300]}")
    try:
        content = await file.read()
        data    = _run_ml_inference(content)
        return JSONResponse(data)
    except Exception as e:
        import traceback
        print(f"❌ ml-analyze error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


COLAB_WORKER_URL = "https://ungeological-unfelicitating-nanci.ngrok-free.dev"


@app.post("/api/hair-transfer")
async def hair_transfer(
    face_file:  UploadFile = File(...),
    shape_file: UploadFile = File(...),
    color_file: UploadFile = File(...),
):
    """Перенос причёски через Colab GPU Worker."""
    target = f"{COLAB_WORKER_URL.rstrip('/')}/generate"
    try:
        face_bytes  = await face_file.read()
        shape_bytes = await shape_file.read()
        color_bytes = await color_file.read()

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                target,
                files={
                    "face_file":  ("face.jpg",  face_bytes,  "image/jpeg"),
                    "shape_file": ("shape.jpg", shape_bytes, "image/jpeg"),
                    "color_file": ("color.jpg", color_bytes, "image/jpeg"),
                },
            )
            if resp.status_code != 200:
                print(f"❌ Ошибка Colab: {resp.status_code} - {resp.text}")
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": "Colab error", "details": resp.text},
                )
            return JSONResponse(resp.json())

    except Exception as e:
        import traceback
        print(f"🚨 КРИТИЧЕСКАЯ ОШИБКА hair-transfer:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_json_object(raw: str) -> dict:
    raw     = (raw or "").strip()
    cleaned = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    cleaned = cleaned.replace("\ufeff", "").strip()

    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    def _close_unbalanced_braces(s: str) -> str:
        depth = 0
        in_str, esc = False, False
        for ch in s:
            if in_str:
                if esc:          esc = False
                elif ch == "\\": esc = True
                elif ch == '"':  in_str = False
                continue
            if ch == '"':   in_str = True
            elif ch == "{": depth += 1
            elif ch == "}": depth = max(0, depth - 1)
        return s + ("}" * depth) if depth > 0 else s

    def _first_balanced_object(s: str):
        start = s.find("{")
        if start == -1: return None
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:          esc = False
                elif ch == "\\": esc = True
                elif ch == '"':  in_str = False
            else:
                if ch == '"':   in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: return s[start:i + 1]
        return None

    candidates = [cleaned]
    balanced   = _first_balanced_object(cleaned)
    if balanced: candidates.append(balanced)

    last_err = None
    for cand in candidates:
        for attempt in (cand,
                        _remove_trailing_commas(cand),
                        _close_unbalanced_braces(cand)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict): return parsed
            except Exception as e:
                last_err = e
    raise ValueError(f"Не удалось распарсить JSON: {last_err}")


def _normalize_llm_ru(parsed: dict) -> dict:
    value_map = {
        "oval": "овальная", "round": "круглая", "square": "квадратная",
        "heart": "сердцевидная", "oblong": "продолговатая",
        "straight": "прямые", "wavy": "волнистые", "curly": "кудрявые",
        "kinky": "афро", "dreadlocks": "дреды",
    }
    def _map_v(v):
        if isinstance(v, str):  return value_map.get(v.lower().strip(), v)
        if isinstance(v, list): return [_map_v(x) for x in v]
        return v
    return {k: _map_v(v) for k, v in parsed.items()}


def _is_low_quality(parsed: dict) -> bool:
    required = {"face_shape", "recommended_cuts", "description"}
    if not required.issubset(parsed.keys()): return True
    return len(str(parsed.get("description", ""))) < 30


@app.post("/api/analyze")
async def llm_analyze(
    file:  UploadFile = File(...),
    prefs: str        = Form(""),
):
    """Анализ лица и волос через OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY не задан.",
        )

    img_bytes = await file.read()
    img_b64   = base64.b64encode(img_bytes).decode("utf-8")
    data_url  = f"data:{file.content_type or 'image/jpeg'};base64,{img_b64}"

    instructions = (
        "Ты — профессиональный стилист. Дай рекомендации на РУССКОМ языке.\n"
        "Ответ ТОЛЬКО в формате JSON с ключами: "
        "face_shape, hair_type, current_style, recommended_cuts (массив 3 шт), "
        "recommended_colors (массив 3 шт), avoid (массив), care_tips (массив), "
        "description (2 предложения)."
    )
    user_msg = f"Пожелания клиента: {prefs}" if prefs.strip() else "Проанализируй фото."

    async with httpx.AsyncClient(timeout=60.0) as client:
        last_resp_text = ""
        for model in FREE_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": f"{instructions}\n\n{user_msg}"},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    "response_format": {"type": "json_object"},
                }
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "http://localhost:8000",
                        "X-Title":       "Hair Vision Analyzer",
                    },
                    json=payload,
                )

                if resp.status_code == 200:
                    resp_json = resp.json()
                    if "error" in resp_json:
                        last_resp_text = str(resp_json["error"])
                        print(f"⚠️ Модель {model} вернула ошибку: {last_resp_text}")
                        continue

                    raw_content = resp_json["choices"][0]["message"]["content"]
                    parsed      = _normalize_llm_ru(_extract_json_object(raw_content))

                    if not _is_low_quality(parsed):
                        return JSONResponse(parsed)
                    last_resp_text = raw_content
                    print(f"⚠️ Модель {model} вернула низкокачественный ответ")
                else:
                    last_resp_text = f"HTTP {resp.status_code}: {resp.text}"
                    print(f"⚠️ Модель {model}: {resp.status_code}")

            except Exception as e:
                import traceback
                last_resp_text = str(e)
                print(f"⚠️ Модель {model} упала: {traceback.format_exc()}")
                continue

    return JSONResponse(
        status_code=200,
        content={
            "description":      "Не удалось получить качественный анализ от LLM.",
            "details":          last_resp_text[:300],
            "face_shape":       "не определено",
            "recommended_cuts": [],
            "care_tips":        [],
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
