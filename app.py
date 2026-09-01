import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy import signal
from vmdpy import VMD
import antropy as ant


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MODMA MDD Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("MODMA EEG Depression Classification")

st.write(
    """
    Upload a three-channel resting-state EEG TXT file.
    The application extracts VMD-based Sample Entropy,
    RMS and PSD features and applies the trained
    LightGBM classifier.
    """
)

st.warning(
    "Research-use prototype only. "
    "This application is not a medical diagnostic device."
)


# ============================================================
# LOAD EXPORTED MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(
        "lightgbm_model.pkl"
    )

    medians = joblib.load(
        "feature_medians.pkl"
    )

    with open(
        "model_config.json",
        "r"
    ) as f:

        config = json.load(f)

    return model, medians, config


try:

    model, train_medians, config = (
        load_model_files()
    )

except Exception as error:

    st.error(
        f"Could not load model files: {error}"
    )

    st.stop()


st.success(
    "Trained LightGBM model loaded successfully."
)


# ============================================================
# EXPORTED MODEL CONFIGURATION
# ============================================================

BEST_FEATURES = config["features"]

CHANNEL_NAMES = config.get(
    "channels",
    ["Fp1", "Fpz", "Fp2"]
)

N_SELECTED_IMFS = int(
    config.get(
        "selected_imfs",
        3
    )
)

FS = int(
    config.get(
        "sampling_frequency",
        250
    )
)


# ============================================================
# SIGNAL PROCESSING SETTINGS
# ============================================================

LOWCUT = 1.0
HIGHCUT = 45.0

FIR_TAPS = 251

VMD_ALPHA = 2000
VMD_TAU = 0.0

# Three IMFs are required by the trained 27-feature model
VMD_K = N_SELECTED_IMFS

VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7

SAMPEN_ORDER = 2


# ============================================================
# WINDOW SETTINGS
# ============================================================

WINDOW_SECONDS = 1.0

WINDOW_SAMPLES = int(
    FS * WINDOW_SECONDS
)

# feature_df contained:
# 660 windows / 55 subjects = 12 windows per subject
N_WINDOWS = 12


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

with st.expander(
    "Model configuration"
):

    st.write(
        "Feature set:",
        config.get(
            "feature_set",
            "VMD_Fusion"
        )
    )

    st.write(
        "Number of model features:",
        len(BEST_FEATURES)
    )

    st.write(
        "EEG channels:",
        CHANNEL_NAMES
    )

    st.write(
        "Selected IMFs:",
        N_SELECTED_IMFS
    )

    st.write(
        "Sampling frequency:",
        f"{FS} Hz"
    )

    st.write(
        "Analysis window:",
        f"{WINDOW_SECONDS:.1f} second "
        f"({WINDOW_SAMPLES} samples)"
    )

    st.write(
        "Windows per recording:",
        N_WINDOWS
    )


# ============================================================
# EEG FILE LOADER
# ============================================================

def load_eeg(uploaded_file):

    """
    Load a MODMA EEG TXT file.

    Supported orientations:

    samples x 3
    3 x samples

    samples x 8
    8 x samples

    For 8-channel MODMA files,
    the first three channels are used:
    Fp1, Fpz, Fp2.
    """

    try:

        eeg = np.loadtxt(
            uploaded_file
        )

    except Exception:

        uploaded_file.seek(0)

        eeg = np.loadtxt(
            uploaded_file,
            delimiter=","
        )

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    if eeg.ndim != 2:

        raise ValueError(
            "EEG data must be a two-dimensional matrix."
        )


    # --------------------------------------------------------
    # samples x 3
    # --------------------------------------------------------

    if eeg.shape[1] == 3:

        return eeg


    # --------------------------------------------------------
    # 3 x samples
    # --------------------------------------------------------

    if eeg.shape[0] == 3:

        return eeg.T


    # --------------------------------------------------------
    # samples x 8
    # --------------------------------------------------------

    if eeg.shape[1] >= 8:

        return eeg[:, :3]


    # --------------------------------------------------------
    # 8 x samples
    # --------------------------------------------------------

    if eeg.shape[0] >= 8:

        return eeg[:3, :].T


    raise ValueError(
        "Could not identify the three EEG channels. "
        f"Detected shape: {eeg.shape}"
    )


# ============================================================
# REPAIR INVALID VALUES
# ============================================================

def repair_nonfinite(eeg):

    eeg = eeg.copy()

    indices = np.arange(
        len(eeg)
    )

    for channel in range(
        eeg.shape[1]
    ):

        x = eeg[:, channel]

        valid = np.isfinite(
            x
        )

        if valid.all():
            continue

        if valid.sum() < 2:

            raise ValueError(
                f"Channel {channel + 1} "
                "contains insufficient valid samples."
            )

        eeg[:, channel] = np.interp(
            indices,
            indices[valid],
            x[valid]
        )

    return eeg


# ============================================================
# EEG PREPROCESSING
# ============================================================

def preprocess_eeg(eeg):

    # --------------------------------------------------------
    # Repair NaN / infinite values
    # --------------------------------------------------------

    eeg = repair_nonfinite(
        eeg
    )


    # --------------------------------------------------------
    # Linear detrending
    # --------------------------------------------------------

    eeg = signal.detrend(
        eeg,
        axis=0,
        type="linear"
    )


    # --------------------------------------------------------
    # FIR 1-45 Hz band-pass
    # --------------------------------------------------------

    coefficients = signal.firwin(
        FIR_TAPS,
        [LOWCUT, HIGHCUT],
        pass_zero=False,
        fs=FS
    )


    # --------------------------------------------------------
    # Zero-phase filtering
    # --------------------------------------------------------

    eeg = signal.filtfilt(
        coefficients,
        [1.0],
        eeg,
        axis=0
    )

    return eeg


# ============================================================
# MEMORY-SAFE WINDOW SEGMENTATION
# ============================================================

def create_windows(eeg):

    """
    Create 12 memory-safe 1-second EEG windows.

    The 12 windows are distributed approximately evenly
    across the complete recording.

    Each window contains only 250 samples at 250 Hz,
    preventing VMD from allocating extremely large arrays.
    """

    total_samples = eeg.shape[0]

    if total_samples < WINDOW_SAMPLES:

        raise ValueError(
            f"Recording is too short. "
            f"At least {WINDOW_SAMPLES} samples are required."
        )


    # --------------------------------------------------------
    # Latest possible start position
    # --------------------------------------------------------

    max_start = (
        total_samples
        -
        WINDOW_SAMPLES
    )


    # --------------------------------------------------------
    # Select 12 positions distributed across the recording
    # --------------------------------------------------------

    starts = np.linspace(
        0,
        max_start,
        N_WINDOWS,
        dtype=int
    )


    windows = []


    for start in starts:

        end = (
            start
            +
            WINDOW_SAMPLES
        )

        window = eeg[
            start:end,
            :
        ].copy()

        windows.append(
            window
        )


    return windows


# ============================================================
# VMD NORMALIZATION
# ============================================================

def normalize_signal(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    x = (
        x
        -
        np.mean(x)
    )

    std = np.std(
        x
    )

    if std > 1e-12:

        x = (
            x
            /
            std
        )

    return x


# ============================================================
# VMD
# ============================================================

def apply_vmd(x):

    x = normalize_signal(
        x
    )

    modes, _, _ = VMD(
        x,
        VMD_ALPHA,
        VMD_TAU,
        VMD_K,
        VMD_DC,
        VMD_INIT,
        VMD_TOL
    )

    return modes[
        :N_SELECTED_IMFS
    ]


# ============================================================
# SAMPLE ENTROPY
# ============================================================

def sample_entropy_value(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    if len(x) < 20:

        return np.nan

    std = np.std(
        x
    )

    if std < 1e-12:

        return 0.0

    x = (
        x
        -
        np.mean(x)
    ) / std

    try:

        value = ant.sample_entropy(
            x,
            order=SAMPEN_ORDER
        )

        return float(
            value
        )

    except Exception:

        return np.nan


# ============================================================
# RMS
# ============================================================

def rms_value(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(x)
            )
        )
    )


# ============================================================
# PSD ENERGY
# ============================================================

def psd_energy(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    frequencies, psd = signal.welch(
        x,
        fs=FS,
        nperseg=min(
            len(x),
            256
        )
    )

    mask = (
        (frequencies >= LOWCUT)
        &
        (frequencies <= HIGHCUT)
    )

    if np.sum(mask) < 2:

        return np.nan

    return float(
        np.trapezoid(
            psd[mask],
            frequencies[mask]
        )
    )


# ============================================================
# FEATURE EXTRACTION FROM ONE WINDOW
# ============================================================

def extract_window_features(
    window
):

    features = {}

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):

        channel_signal = window[
            :,
            channel_index
        ]

        modes = apply_vmd(
            channel_signal
        )

        for mode_index in range(
            N_SELECTED_IMFS
        ):

            mode_signal = modes[
                mode_index
            ]

            prefix = (
                f"{channel_name}_"
                f"IMF{mode_index + 1}"
            )


            # ------------------------------------------------
            # Sample Entropy
            # ------------------------------------------------

            features[
                f"{prefix}_SampEn"
            ] = sample_entropy_value(
                mode_signal
            )


            # ------------------------------------------------
            # RMS
            # ------------------------------------------------

            features[
                f"{prefix}_RMS"
            ] = rms_value(
                mode_signal
            )


            # ------------------------------------------------
            # PSD
            # ------------------------------------------------

            features[
                f"{prefix}_PSD"
            ] = psd_energy(
                mode_signal
            )

    return features


# ============================================================
# EXTRACT FEATURES FROM COMPLETE RECORDING
# ============================================================

def extract_recording_features(
    eeg
):

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessed = preprocess_eeg(
        eeg
    )


    # --------------------------------------------------------
    # Create 12 memory-safe 1-second windows
    # --------------------------------------------------------

    windows = create_windows(
        preprocessed
    )


    rows = []


    progress = st.progress(
        0
    )

    status = st.empty()


    for i, window in enumerate(
        windows
    ):

        status.write(
            f"Processing EEG window "
            f"{i + 1}/{len(windows)}..."
        )


        features = extract_window_features(
            window
        )


        features[
            "window_idx"
        ] = i


        rows.append(
            features
        )


        progress.progress(
            (i + 1)
            /
            len(windows)
        )


    progress.empty()
    status.empty()


    return pd.DataFrame(
        rows
    )


# ============================================================
# PREPARE LIGHTGBM INPUT MATRIX
# ============================================================

def prepare_model_input(
    feature_df
):

    missing_features = [
        feature
        for feature in BEST_FEATURES
        if feature not in feature_df.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing trained-model features: "
            +
            ", ".join(
                missing_features
            )
        )


    # --------------------------------------------------------
    # Select exact trained feature order
    # --------------------------------------------------------

    X = feature_df[
        BEST_FEATURES
    ].copy()


    # --------------------------------------------------------
    # Replace infinity with NaN
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Convert saved medians to pandas Series
    # --------------------------------------------------------

    if isinstance(
        train_medians,
        pd.Series
    ):

        medians = train_medians


    elif isinstance(
        train_medians,
        dict
    ):

        medians = pd.Series(
            train_medians
        )


    else:

        medians = pd.Series(
            np.asarray(
                train_medians
            ).reshape(-1),
            index=BEST_FEATURES
        )


    # --------------------------------------------------------
    # Fill missing feature values
    # using TRAINING-set medians
    # --------------------------------------------------------

    for feature in BEST_FEATURES:

        if feature in medians.index:

            X[feature] = (
                X[feature]
                .fillna(
                    medians[
                        feature
                    ]
                )
            )


    remaining_nan = (
        X.isna()
        .sum()
        .sum()
    )


    if remaining_nan > 0:

        raise ValueError(
            f"{remaining_nan} missing feature "
            "values remain after median imputation."
        )


    return X.astype(
        np.float32
    )


# ============================================================
# SUBJECT-LEVEL PREDICTION
# ============================================================

def predict_recording(
    feature_df
):

    X = prepare_model_input(
        feature_df
    )


    # --------------------------------------------------------
    # Window-level probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )


    # --------------------------------------------------------
    # Subject-level probability:
    # mean of all 12 window probabilities
    # --------------------------------------------------------

    subject_probability = float(
        np.mean(
            probabilities
        )
    )


    # --------------------------------------------------------
    # Decision threshold
    #
    # Uses exported threshold if available.
    # Falls back to 0.50 otherwise.
    # --------------------------------------------------------

    threshold = float(
        config.get(
            "threshold",
            0.50
        )
    )


    prediction = int(
        subject_probability
        >=
        threshold
    )


    return (
        probabilities,
        subject_probability,
        prediction,
        threshold
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.divider()

st.subheader(
    "Upload EEG Recording"
)

uploaded_file = st.file_uploader(
    "Choose a MODMA three-channel EEG TXT file",
    type=["txt"]
)


# ============================================================
# PROCESS UPLOAD
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Load EEG
        # ----------------------------------------------------

        eeg = load_eeg(
            uploaded_file
        )


        # ----------------------------------------------------
        # Recording information
        # ----------------------------------------------------

        duration = (
            len(eeg)
            /
            FS
        )


        st.success(
            "EEG file loaded successfully."
        )


        col1, col2, col3 = st.columns(
            3
        )


        col1.metric(
            "Samples",
            f"{len(eeg):,}"
        )


        col2.metric(
            "Channels",
            eeg.shape[1]
        )


        col3.metric(
            "Duration",
            f"{duration:.2f} s"
        )


        # ====================================================
        # RAW EEG PREVIEW
        # ====================================================

        st.subheader(
            "Raw EEG Preview"
        )


        preview_samples = min(
            len(eeg),
            FS * 5
        )


        preview = pd.DataFrame(
            eeg[
                :preview_samples
            ],
            columns=CHANNEL_NAMES
        )


        preview.index = (
            np.arange(
                preview_samples
            )
            /
            FS
        )


        preview.index.name = (
            "Time (seconds)"
        )


        st.line_chart(
            preview
        )


        # ====================================================
        # ANALYSIS BUTTON
        # ====================================================

        if st.button(
            "Analyze EEG and Predict",
            type="primary"
        ):

            with st.spinner(
                "Running preprocessing, VMD, "
                "feature extraction and "
                "LightGBM prediction..."
            ):


                # --------------------------------------------
                # Extract 27 VMD-derived features
                # from each of 12 windows
                # --------------------------------------------

                extracted_df = (
                    extract_recording_features(
                        eeg
                    )
                )


                # --------------------------------------------
                # Predict
                # --------------------------------------------

                (
                    probabilities,
                    subject_probability,
                    prediction,
                    threshold
                ) = predict_recording(
                    extracted_df
                )


            # =================================================
            # CLASSIFICATION RESULT
            # =================================================

            st.divider()

            st.subheader(
                "Classification Result"
            )


            if prediction == 1:

                st.error(
                    "Model classification: "
                    "Major Depressive Disorder (MDD)"
                )

            else:

                st.success(
                    "Model classification: "
                    "Healthy Control (HC)"
                )


            result_col1, result_col2 = (
                st.columns(2)
            )


            result_col1.metric(
                "Mean MDD probability",
                f"{subject_probability:.4f}"
            )


            result_col2.metric(
                "Decision threshold",
                f"{threshold:.4f}"
            )


            # =================================================
            # WINDOW-LEVEL RESULTS
            # =================================================

            results_df = pd.DataFrame({

                "Window":
                    np.arange(
                        1,
                        len(probabilities) + 1
                    ),

                "MDD Probability":
                    probabilities
            })


            st.subheader(
                "Window-Level MDD Probabilities"
            )


            st.line_chart(
                results_df.set_index(
                    "Window"
                )
            )


            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # EXTRACTED FEATURE TABLE
            # =================================================

            with st.expander(
                "View extracted EEG features"
            ):

                st.dataframe(
                    extracted_df,
                    use_container_width=True
                )


            # =================================================
            # FEATURE INFORMATION
            # =================================================

            with st.expander(
                "View model feature names"
            ):

                for i, feature in enumerate(
                    BEST_FEATURES,
                    start=1
                ):

                    st.write(
                        f"{i}. {feature}"
                    )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.warning(
                """
                This prediction is generated by an
                experimental machine-learning model
                trained on the MODMA EEG dataset.

                It must not be interpreted as a clinical
                diagnosis of depression and should not
                replace assessment by a qualified
                healthcare professional.
                """
            )


    except MemoryError:

        st.error(
            "EEG processing ran out of memory."
        )

        st.write(
            "The application uses memory-safe "
            "1-second windows, but the system "
            "still does not have enough available RAM."
        )


    except Exception as error:

        st.error(
            "EEG processing failed."
        )

        st.exception(
            error
        )