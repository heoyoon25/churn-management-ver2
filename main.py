import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Matplotlib setting: English-friendly font only
# ─────────────────────────────────────────────────────────────
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root{
        --bg: #f6f8fc;
        --card: #ffffff;
        --text: #1f2937;
        --muted: #6b7280;
        --line: #e5e7eb;
        --primary: #5b6cff;
        --primary2: #7b61ff;
        --accent: #14b8a6;
        --danger: #ef4444;
        --success: #16a34a;
        --shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
        --radius: 16px;
    }

    .stApp {
        background: linear-gradient(180deg, #f8faff 0%, #f5f7fb 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #182132 0%, #1d2840 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .sidebar-brand {
        background: linear-gradient(135deg, rgba(91,108,255,0.22), rgba(123,97,255,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        margin-bottom: 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sidebar-brand-sub {
        font-size: 0.82rem;
        opacity: 0.82;
        line-height: 1.45;
    }

    .sidebar-section-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.72;
        margin: 1rem 0 0.55rem 0;
        font-weight: 700;
    }

    .sidebar-data-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.9rem;
        margin-top: 0.6rem;
    }
    .sidebar-mini-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        margin-top: 0.7rem;
    }
    .sidebar-mini-box {
        background: rgba(255,255,255,0.96);
        border-radius: 12px;
        padding: 0.75rem 0.5rem;
        text-align: center;
    }
    .sidebar-mini-label {
        color: #6b7280 !important;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .sidebar-mini-value {
        color: #111827 !important;
        font-size: 1.05rem;
        font-weight: 900;
    }

    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.58rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 800;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .status-done {
        background: rgba(34,197,94,0.18);
        color: #bbf7d0 !important;
    }
    .status-wait {
        background: rgba(255,255,255,0.08);
        color: #e5e7eb !important;
    }

    .page-wrap {
        padding-top: 0.3rem;
    }

    .hero {
        background: linear-gradient(135deg, #5b6cff 0%, #7b61ff 55%, #14b8a6 100%);
        border-radius: 24px;
        padding: 2rem 2rem 1.8rem 2rem;
        color: white;
        box-shadow: 0 16px 36px rgba(91,108,255,0.18);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 220px;
        height: 220px;
        background: rgba(255,255,255,0.10);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1.15;
        margin-bottom: 0.35rem;
        position: relative;
        z-index: 2;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.94;
        line-height: 1.6;
        max-width: 760px;
        position: relative;
        z-index: 2;
    }
    .hero-badges {
        margin-top: 1rem;
        position: relative;
        z-index: 2;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.16);
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0 0.4rem 0.35rem 0;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 900;
        color: #111827;
        margin-top: 0.35rem;
        margin-bottom: 0.15rem;
    }
    .section-subtitle {
        font-size: 0.92rem;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid #eef2f7;
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: var(--shadow);
        min-height: 165px;
    }
    .feature-icon {
        width: 48px;
        height: 48px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius: 14px;
        font-size: 1.45rem;
        background: linear-gradient(135deg, rgba(91,108,255,0.12), rgba(123,97,255,0.16));
        margin-bottom: 0.9rem;
    }
    .feature-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.35rem;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #6b7280;
        line-height: 1.55;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
        border: 1px solid #edf2f7;
        border-radius: 18px;
        padding: 1rem 1rem;
        box-shadow: var(--shadow);
        text-align: center;
    }
    .metric-value {
        font-size: 1.72rem;
        font-weight: 900;
        color: #111827;
        line-height: 1.1;
    }
    .metric-label {
        margin-top: 0.25rem;
        font-size: 0.82rem;
        color: #6b7280;
        font-weight: 700;
    }

    .info-chip {
        display: inline-block;
        padding: 0.28rem 0.58rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4f46e5;
        font-size: 0.76rem;
        font-weight: 800;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    .upload-panel {
        background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        border: 1px solid #edf2f7;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: var(--shadow);
    }

    [data-testid="stFileUploader"] {
        border: 2px dashed #b9c2ff !important;
        background: #f8faff !important;
        border-radius: 16px !important;
        padding: 0.6rem !important;
    }

    .divider-space {
        height: 0.4rem;
    }

    .simple-note {
        font-size: 0.86rem;
        color: #6b7280;
        line-height: 1.55;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #5b6cff 0%, #7b61ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.58rem 1.05rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(91,108,255,0.18);
    }
    .stButton > button:hover {
        opacity: 0.95;
    }

    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 800;
    }

    .tiny-muted {
        font-size: 0.8rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
defaults = {
    "df_raw": None,
    "df": None,
    "df_processed": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "lr_model": None,
    "dt_model": None,
    "dnn_model": None,
    "lr_result": None,
    "dt_result": None,
    "dnn_result": None,
    "selected_X": [],
    "selected_y": None,
    "split_ratio": "7:3",
    "missing_handled": False,
    "outlier_handled": False,
    "encoded": False,
    "convert_handled": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"""
    <div class="section-title">{title}</div>
    <div class="section-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)

def metric_card(value, label):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def feature_card(icon, title, desc):
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

def check_data():
    if st.session_state.df is None:
        st.warning("⚠️ 먼저 메인 페이지에서 데이터를 업로드해 주세요.")
        st.stop()

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_prob = proba[:, 1]
        except Exception:
            y_prob = None

    fpr, tpr, roc_auc = None, None, None
    try:
        if y_prob is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
    except Exception:
        pass

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "y_pred": y_pred,
        "cm": confusion_matrix(y_test, y_pred),
    }

def plot_single_roc(result, title, color="#5b6cff"):
    if result is None or result["fpr"] is None:
        st.info("ROC curve is available for binary classification with probability output.")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax.plot(result["fpr"], result["tpr"], color=color, lw=2.4,
            label=f"AUC = {result['auc']:.4f}")
    ax.fill_between(result["fpr"], result["tpr"], alpha=0.10, color=color)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">📊 고객 이탈 예측</div>
        <div class="sidebar-brand-sub">
            업로드 → 탐색 → 전처리 → 모델 학습 → 결과 비교까지
            한 번에 진행할 수 있는 Streamlit 분석 대시보드
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
    pages = {
        "🏠 메인 / 업로드": "main",
        "🔍 데이터 탐색": "eda",
        "⚙️ 데이터 전처리": "preprocess",
        "🤖 모델 학습": "model",
        "📈 결과 분석": "result",
    }
    page = st.radio("페이지 선택", list(pages.keys()), label_visibility="collapsed")
    current = pages[page]

    st.markdown('<div class="sidebar-section-title">Workspace</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"""
        <div class="sidebar-data-card">
            <div style="font-size:0.88rem;font-weight:800;">현재 데이터 상태</div>
            <div class="sidebar-mini-grid">
                <div class="sidebar-mini-box">
                    <div class="sidebar-mini-label">Rows</div>
                    <div class="sidebar-mini-value">{df_info.shape[0]:,}</div>
                </div>
                <div class="sidebar-mini-box">
                    <div class="sidebar-mini-label">Columns</div>
                    <div class="sidebar-mini-value">{df_info.shape[1]:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        def pill(label, done):
            cls = "status-done" if done else "status-wait"
            txt = "Done" if done else "Pending"
            return f'<span class="status-pill {cls}">{label}: {txt}</span>'

        st.markdown(
            pill("Missing", st.session_state.missing_handled) +
            pill("Outlier", st.session_state.outlier_handled) +
            pill("Encoding", st.session_state.encoded),
            unsafe_allow_html=True
        )
    else:
        st.info("데이터를 업로드하면 상태 정보가 여기에 표시됩니다.")

# ─────────────────────────────────────────────────────────────
# PAGE 1: Main
# ─────────────────────────────────────────────────────────────
if current == "main":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-title">고객 이탈 예측</div>
        <div class="hero-sub">
            고객 데이터를 업로드하고, 탐색·전처리·모델 학습·성능 비교까지
            한 화면 흐름으로 수행할 수 있는 머신러닝 분석 앱입니다.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">EDA</span>
            <span class="hero-badge">Preprocessing</span>
            <span class="hero-badge">Logistic Regression</span>
            <span class="hero-badge">Decision Tree</span>
            <span class="hero-badge">DNN(MLP)</span>
            <span class="hero-badge">ROC / Confusion Matrix</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("주요 기능", "이상치 퍼센테이지 설정과 DNN 모형을 포함한 리팩토링 버전입니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("📂", "데이터 업로드", "CSV/Excel 파일을 업로드하고 행·열·결측치 현황을 즉시 확인합니다.")
    with c2:
        feature_card("🧪", "탐색 & 전처리", "분포 확인, 결측치/이상치 처리, 문자열 숫자 변환, 인코딩을 지원합니다.")
    with c3:
        feature_card("📈", "모델 비교", "Logistic Regression, Decision Tree, DNN(MLP)을 학습하고 비교합니다.")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("데이터 업로드", "분석할 CSV 또는 Excel 파일을 업로드하세요.")

    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "파일 선택 (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        help="UTF-8 CSV 또는 Excel 파일을 지원합니다."
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded, encoding="utf-8-sig")
            else:
                df = pd.read_excel(uploaded)

            st.session_state.df_raw = df.copy()
            st.session_state.df = df.copy()
            st.session_state.df_processed = None
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.lr_model = None
            st.session_state.dt_model = None
            st.session_state.dnn_model = None
            st.session_state.lr_result = None
            st.session_state.dt_result = None
            st.session_state.dnn_result = None
            st.session_state.missing_handled = False
            st.session_state.outlier_handled = False
            st.session_state.encoded = False
            st.session_state.convert_handled = False

            st.success(f"✅ **{uploaded.name}** 업로드 완료")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card(f"{df.shape[0]:,}", "총 행 수")
            with m2:
                metric_card(f"{df.shape[1]:,}", "총 열 수")
            with m3:
                metric_card(f"{int(df.isnull().sum().sum()):,}", "결측치 수")
            with m4:
                metric_card(f"{df.select_dtypes(include=np.number).shape[1]:,}", "수치형 변수")

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📋 데이터 미리보기", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            with st.expander("📊 기술 통계량"):
                st.dataframe(df.describe(include="all").T, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {e}")

    else:
        st.markdown("""
        <div class="simple-note">
            샘플 데이터로 먼저 화면을 테스트해보고 싶다면 아래 버튼을 눌러 예시 데이터를 생성할 수 있습니다.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎲 샘플 데이터 생성"):
            np.random.seed(42)
            n = 500
            sample_df = pd.DataFrame({
                "age": np.random.randint(20, 70, n),
                "tenure": np.random.randint(1, 60, n),
                "balance": np.random.uniform(0, 200000, n).round(2),
                "num_products": np.random.randint(1, 5, n),
                "credit_score": np.random.randint(300, 850, n),
                "is_active": np.random.randint(0, 2, n),
                "gender": np.random.choice(["Male", "Female"], n),
                "geography": np.random.choice(["France", "Germany", "Spain"], n),
                "salary": np.random.uniform(20000, 150000, n).round(2),
                "churn": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            })

            for col in ["balance", "credit_score", "salary"]:
                idx = np.random.choice(n, int(n * 0.05), replace=False)
                sample_df.loc[idx, col] = np.nan

            st.session_state.df_raw = sample_df.copy()
            st.session_state.df = sample_df.copy()
            st.session_state.missing_handled = False
            st.session_state.outlier_handled = False
            st.session_state.encoded = False
            st.session_state.convert_handled = False
            st.session_state.lr_model = None
            st.session_state.dt_model = None
            st.session_state.dnn_model = None
            st.session_state.lr_result = None
            st.session_state.dt_result = None
            st.session_state.dnn_result = None

            st.success("✅ 샘플 데이터가 생성되었습니다.")
            st.dataframe(sample_df.head(), use_container_width=True)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2: EDA
# ─────────────────────────────────────────────────────────────
elif current == "eda":
    check_data()
    df = st.session_state.df

    section_header("데이터 탐색", "변수 유형, 기초 정보, 분포 및 상관관계를 확인합니다.")

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        metric_card(f"{df.shape[0]:,}", "행 수")
    with i2:
        metric_card(f"{df.shape[1]:,}", "열 수")
    with i3:
        metric_card(f"{int(df.isnull().sum().sum()):,}", "결측치")
    with i4:
        metric_card(f"{int(df.duplicated().sum()):,}", "중복 행")

    st.markdown("<br>", unsafe_allow_html=True)

    section_header("변수 정보", "수치형/범주형 변수 현황을 확인합니다.")

    col_left, col_right = st.columns([1.3, 1])
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    with col_left:
        dtype_df = pd.DataFrame({
            "변수명": df.columns.tolist(),
            "데이터 타입": df.dtypes.astype(str).values,
            "변수 구분": ["Numeric" if c in num_cols else "Categorical" for c in df.columns],
            "결측치 수": df.isnull().sum().values,
            "결측치 비율(%)": (df.isnull().mean() * 100).round(2).values,
            "고유값 수": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=320)

    with col_right:
        type_counts = pd.Series([
            "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical"
            for col in df.columns
        ]).value_counts()

        fig_pie, ax_pie = plt.subplots(figsize=(4, 3.4))
        colors = ["#5b6cff", "#14b8a6"]
        ax_pie.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 11}
        )
        ax_pie.set_title("Variable Type Distribution", fontsize=12, fontweight="bold")
        st.pyplot(fig_pie, use_container_width=True)
        plt.close()

    section_header("변수 시각화", "선택한 변수 조합으로 기본 차트를 생성합니다.")

    all_cols = df.columns.tolist()
    v1, v2, v3 = st.columns([1, 1, 1])
    with v1:
        x_var = st.selectbox("X variable", all_cols, key="eda_x")
    with v2:
        y_var = st.selectbox("Y variable", ["(None)"] + all_cols, key="eda_y")
    with v3:
        chart_type = st.selectbox(
            "Chart type",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"],
            key="eda_chart"
        )

    if st.button("📊 그래프 생성", key="btn_chart"):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        palette = "#5b6cff"

        try:
            x_data = df[x_var].copy()
            y_data = df[y_var].copy() if y_var != "(None)" else None

            x_is_num = pd.api.types.is_numeric_dtype(x_data)
            y_is_num = pd.api.types.is_numeric_dtype(y_data) if y_data is not None else False

            if chart_type == "Histogram":
                if x_is_num:
                    ax.hist(
                        x_data.dropna().astype(float),
                        bins=30, color=palette, edgecolor="white", alpha=0.88
                    )
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("Frequency")
                else:
                    counts = x_data.value_counts()
                    ax.bar(range(len(counts)), counts.values, color=palette, edgecolor="white", alpha=0.88)
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha="right")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("Frequency")

            elif chart_type == "Box Plot":
                if not x_is_num:
                    st.warning("Box Plot의 X 변수는 수치형이어야 합니다.")
                    plt.close()
                    st.stop()

                if y_data is not None and not y_is_num:
                    groups = []
                    labels = []
                    for g in df[y_var].dropna().unique():
                        grp = df[df[y_var] == g][x_var].dropna().astype(float)
                        if len(grp) > 0:
                            groups.append(grp.values)
                            labels.append(str(g))
                    ax.boxplot(
                        groups, patch_artist=True,
                        boxprops=dict(facecolor=palette, alpha=0.65),
                        medianprops=dict(color="red", linewidth=2)
                    )
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_xlabel(y_var)
                    ax.set_ylabel(x_var)
                else:
                    ax.boxplot(
                        x_data.dropna().astype(float).values,
                        patch_artist=True,
                        boxprops=dict(facecolor=palette, alpha=0.65),
                        medianprops=dict(color="red", linewidth=2)
                    )
                    ax.set_ylabel(x_var)
                    ax.set_xticks([1])
                    ax.set_xticklabels([x_var])

            elif chart_type == "Scatter Plot":
                if y_var == "(None)":
                    st.warning("Scatter Plot은 Y 변수가 필요합니다.")
                    plt.close()
                    st.stop()
                if not x_is_num or not y_is_num:
                    st.warning("Scatter Plot은 X/Y 모두 수치형이어야 합니다.")
                    plt.close()
                    st.stop()

                valid = df[[x_var, y_var]].dropna()
                ax.scatter(
                    valid[x_var].astype(float),
                    valid[y_var].astype(float),
                    alpha=0.45, color=palette, edgecolors="white", linewidths=0.25
                )
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)

            elif chart_type == "Bar Chart":
                if not x_is_num:
                    counts = x_data.value_counts()
                    ax.bar(range(len(counts)), counts.values, color=palette, edgecolor="white", alpha=0.88)
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha="right")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("Frequency")
                else:
                    if y_data is not None and y_is_num:
                        valid = df[[x_var, y_var]].dropna().copy()
                        valid[x_var] = valid[x_var].astype(float)
                        valid[y_var] = valid[y_var].astype(float)
                        bins = pd.cut(valid[x_var], bins=10)
                        grouped = valid.groupby(bins, observed=True)[y_var].mean()
                        labels = [str(i) for i in grouped.index]
                        ax.bar(range(len(grouped)), grouped.values, color=palette, edgecolor="white", alpha=0.88)
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                        ax.set_xlabel(x_var)
                        ax.set_ylabel(f"Mean of {y_var}")
                    else:
                        counts = x_data.value_counts().sort_index()
                        ax.bar(range(len(counts)), counts.values, color=palette, edgecolor="white", alpha=0.88)
                        ax.set_xticks(range(len(counts)))
                        ax.set_xticklabels([str(i) for i in counts.index], rotation=45, ha="right", fontsize=8)
                        ax.set_xlabel(x_var)
                        ax.set_ylabel("Frequency")

            elif chart_type == "Line Chart":
                if not x_is_num:
                    st.warning("Line Chart의 X 변수는 수치형이어야 합니다.")
                    plt.close()
                    st.stop()

                if y_data is not None and y_is_num:
                    valid = df[[x_var, y_var]].dropna().sort_values(x_var)
                    ax.plot(valid[x_var].astype(float), valid[y_var].astype(float), color=palette, linewidth=1.8)
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                else:
                    sorted_data = x_data.dropna().astype(float).reset_index(drop=True)
                    ax.plot(sorted_data.index, sorted_data.values, color=palette, linewidth=1.8)
                    ax.set_xlabel("Index")
                    ax.set_ylabel(x_var)

            title = f"{chart_type} | {x_var}"
            if y_var != "(None)":
                title += f" vs {y_var}"
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.25, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"그래프 생성 오류: {e}")
        finally:
            plt.close()

    if len(num_cols) >= 2:
        section_header("상관관계 히트맵", "수치형 변수 간 상관관계를 확인합니다.")
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, ax=ax_hm,
            linewidths=0.5, cbar_kws={"shrink": 0.8}
        )
        ax_hm.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_hm, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────
# PAGE 3: Preprocess
# ─────────────────────────────────────────────────────────────
elif current == "preprocess":
    check_data()
    df = st.session_state.df.copy()

    section_header("데이터 전처리", "결측치, 이상치, 문자열 숫자 변환, 인코딩을 수행합니다.")

    tab1, tab2, tab3, tab4 = st.tabs(["결측치 처리", "이상치 처리", "수치형 변환", "인코딩"])

    with tab1:
        missing = st.session_state.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("✅ 결측치가 없습니다.")
        else:
            miss_df = pd.DataFrame({
                "변수명": missing.index,
                "결측치 수": missing.values,
                "결측치 비율(%)": (missing / len(st.session_state.df) * 100).round(2).values
            })
            st.dataframe(miss_df, use_container_width=True)

        method = st.radio(
            "처리 방법",
            ["평균값 대체 (수치형)", "중앙값 대체 (수치형)", "최빈값 대체 (범주형)", "행 삭제"],
            horizontal=True,
            key="missing_method"
        )

        if st.button("✅ 결측치 처리 실행", key="btn_missing"):
            df_work = st.session_state.df.copy()
            num_cols = [c for c in df_work.columns if pd.api.types.is_numeric_dtype(df_work[c])]
            cat_cols = [c for c in df_work.columns if not pd.api.types.is_numeric_dtype(df_work[c])]

            if method == "행 삭제":
                df_work = df_work.dropna().reset_index(drop=True)
            else:
                for c in num_cols:
                    if df_work[c].isnull().any():
                        if method == "평균값 대체 (수치형)":
                            df_work[c] = df_work[c].fillna(df_work[c].mean())
                        else:
                            df_work[c] = df_work[c].fillna(df_work[c].median())

                for c in cat_cols:
                    if df_work[c].isnull().any():
                        mode_val = df_work[c].mode()
                        if len(mode_val) > 0:
                            df_work[c] = df_work[c].fillna(mode_val[0])

            remaining = df_work.isnull().sum().sum()
            if remaining > 0:
                for c in df_work.columns:
                    if df_work[c].isnull().any():
                        if pd.api.types.is_numeric_dtype(df_work[c]):
                            df_work[c] = df_work[c].fillna(df_work[c].median())
                        else:
                            df_work[c] = df_work[c].fillna("Unknown")

            st.session_state.df = df_work
            st.session_state.missing_handled = True

            if df_work.isnull().sum().sum() == 0:
                st.success("✅ 결측치 처리 완료")
            else:
                st.warning(f"⚠️ 남은 결측치: {df_work.isnull().sum().sum()}개")
            st.rerun()

    with tab2:
        df_cur = st.session_state.df

        num_cols = []
        for c in df_cur.columns:
            if pd.api.types.is_numeric_dtype(df_cur[c]):
                try:
                    arr = df_cur[c].dropna().astype(float).values
                    _ = np.quantile(arr, 0.25)
                    num_cols.append(c)
                except Exception:
                    pass

        if not num_cols:
            st.info("수치형 변수가 없습니다.")
        else:
            st.markdown("#### 이상치 기준 설정")
            detect_basis = st.radio(
                "이상치 판정 기준",
                ["IQR 기준", "Percentile 기준"],
                horizontal=True,
                key="outlier_basis"
            )

            info_records = []
            if detect_basis == "IQR 기준":
                iqr_multiplier = st.slider(
                    "IQR multiplier (이상치 강도)",
                    min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                    help="작을수록 더 많은 값을 이상치로 판정합니다."
                )
                for c in num_cols:
                    try:
                        arr = df_cur[c].dropna().astype(float).values
                        Q1 = float(np.quantile(arr, 0.25))
                        Q3 = float(np.quantile(arr, 0.75))
                        IQR = Q3 - Q1
                        lo = Q1 - iqr_multiplier * IQR
                        hi = Q3 + iqr_multiplier * IQR
                        mask = (arr < lo) | (arr > hi)
                        out_pct = mask.mean() * 100
                        info_records.append({
                            "Variable": c,
                            "Outliers": int(mask.sum()),
                            "Outlier %": round(out_pct, 2),
                            "Lower Bound": round(lo, 3),
                            "Upper Bound": round(hi, 3),
                        })
                    except Exception:
                        info_records.append({
                            "Variable": c,
                            "Outliers": "N/A",
                            "Outlier %": "N/A",
                            "Lower Bound": "-",
                            "Upper Bound": "-",
                        })
            else:
                total_pct = st.slider(
                    "Total outlier percentage (%)",
                    min_value=1.0, max_value=20.0, value=5.0, step=0.5,
                    help="예: 5%면 하위 2.5% + 상위 2.5%를 이상치로 간주합니다."
                )
                tail_pct = total_pct / 2.0
                st.caption(f"Lower tail: {tail_pct:.2f}% / Upper tail: {tail_pct:.2f}%")
                for c in num_cols:
                    try:
                        arr = df_cur[c].dropna().astype(float).values
                        lo = float(np.percentile(arr, tail_pct))
                        hi = float(np.percentile(arr, 100 - tail_pct))
                        mask = (arr < lo) | (arr > hi)
                        out_pct = mask.mean() * 100
                        info_records.append({
                            "Variable": c,
                            "Outliers": int(mask.sum()),
                            "Outlier %": round(out_pct, 2),
                            "Lower Bound": round(lo, 3),
                            "Upper Bound": round(hi, 3),
                        })
                    except Exception:
                        info_records.append({
                            "Variable": c,
                            "Outliers": "N/A",
                            "Outlier %": "N/A",
                            "Lower Bound": "-",
                            "Upper Bound": "-",
                        })

            st.dataframe(pd.DataFrame(info_records), use_container_width=True)

            out_method = st.radio(
                "처리 방법",
                ["Clipping", "Row Removal"],
                horizontal=True,
                key="outlier_method"
            )
            out_cols = st.multiselect(
                "처리할 변수 선택",
                num_cols,
                default=num_cols,
                key="outlier_cols"
            )

            if st.button("✅ 이상치 처리 실행", key="btn_outlier"):
                df_work = st.session_state.df.copy()
                processed = []
                skipped = []

                for c in out_cols:
                    try:
                        arr = df_work[c].dropna().astype(float).values

                        if detect_basis == "IQR 기준":
                            Q1 = float(np.quantile(arr, 0.25))
                            Q3 = float(np.quantile(arr, 0.75))
                            IQR = Q3 - Q1
                            lo = Q1 - iqr_multiplier * IQR
                            hi = Q3 + iqr_multiplier * IQR
                        else:
                            lo = float(np.percentile(arr, tail_pct))
                            hi = float(np.percentile(arr, 100 - tail_pct))

                        df_work[c] = df_work[c].astype(float)

                        if out_method == "Clipping":
                            df_work[c] = df_work[c].clip(lo, hi)
                        else:
                            df_work = df_work[(df_work[c] >= lo) & (df_work[c] <= hi)]

                        processed.append(c)
                    except Exception as e:
                        skipped.append(f"{c} ({e})")

                df_work = df_work.reset_index(drop=True)
                st.session_state.df = df_work
                st.session_state.outlier_handled = True

                if processed:
                    st.success(f"✅ 처리 완료: {processed}")
                if skipped:
                    st.warning(f"⚠️ 처리 실패: {skipped}")
                st.rerun()

    with tab3:
        df_cur = st.session_state.df
        cat_cols_all = [c for c in df_cur.columns if not pd.api.types.is_numeric_dtype(df_cur[c])]

        st.markdown("#### 🔢 문자열 → 수치형 변환")
        st.markdown("`%`, `₩`, `,` 같은 특수문자가 포함된 숫자 문자열을 수치형으로 변환합니다.")

        if not cat_cols_all:
            st.success("✅ 변환할 범주형 변수가 없습니다.")
        else:
            auto_detect = []
            for c in cat_cols_all:
                sample = df_cur[c].dropna().astype(str).head(50)
                cleaned = sample.str.replace(r"[%,₩$\s]", "", regex=True)
                try:
                    pd.to_numeric(cleaned)
                    auto_detect.append(c)
                except Exception:
                    numeric_ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()
                    if numeric_ratio > 0.8:
                        auto_detect.append(c)

            if auto_detect:
                st.info(f"💡 변환 가능한 변수 감지: {auto_detect}")

            convert_cols = st.multiselect(
                "수치형으로 변환할 변수 선택",
                cat_cols_all,
                default=auto_detect,
                key="convert_cols"
            )
            remove_chars = st.text_input(
                "제거할 특수문자",
                value="%,₩$ ",
                key="remove_chars"
            )

            cp1, cp2 = st.columns(2)
            with cp1:
                st.markdown("**변환 전 미리보기**")
                if convert_cols:
                    st.dataframe(df_cur[convert_cols].head(5).astype(str), use_container_width=True)
            with cp2:
                st.markdown("**변환 후 미리보기**")
                if convert_cols:
                    preview = df_cur[convert_cols].head(5).copy()
                    for c in convert_cols:
                        pattern = f"[{re.escape(remove_chars)}]"
                        preview[c] = pd.to_numeric(
                            preview[c].astype(str).str.replace(pattern, "", regex=True),
                            errors="coerce"
                        )
                    st.dataframe(preview, use_container_width=True)

            if st.button("✅ 수치형 변환 실행", key="btn_convert"):
                if not convert_cols:
                    st.warning("변환할 변수를 선택해 주세요.")
                else:
                    df_work = st.session_state.df.copy()
                    success_cols = []
                    fail_cols = []

                    for c in convert_cols:
                        try:
                            pattern = f"[{re.escape(remove_chars)}]"
                            converted = pd.to_numeric(
                                df_work[c].astype(str).str.replace(pattern, "", regex=True),
                                errors="coerce"
                            )
                            success_rate = converted.notna().mean()

                            if success_rate >= 0.8:
                                if converted.isnull().any():
                                    converted = converted.fillna(converted.median())
                                df_work[c] = converted
                                success_cols.append(f"{c} ({success_rate:.0%})")
                            else:
                                fail_cols.append(f"{c} ({success_rate:.0%})")
                        except Exception as e:
                            fail_cols.append(f"{c} ({e})")

                    st.session_state.df = df_work
                    st.session_state.convert_handled = True

                    if success_cols:
                        st.success(f"✅ 변환 완료: {success_cols}")
                    if fail_cols:
                        st.warning(f"⚠️ 변환 실패: {fail_cols}")
                    st.rerun()

    with tab4:
        df_cur = st.session_state.df
        cat_cols = [c for c in df_cur.columns if not pd.api.types.is_numeric_dtype(df_cur[c])]

        if not cat_cols:
            st.success("✅ 인코딩할 범주형 변수가 없습니다.")
        else:
            cat_info = pd.DataFrame({
                "변수명": cat_cols,
                "고유값 수": [df_cur[c].nunique() for c in cat_cols],
                "샘플값": [str(df_cur[c].dropna().unique()[:3].tolist()) for c in cat_cols],
            })
            st.dataframe(cat_info, use_container_width=True)

            high_cardinality = [c for c in cat_cols if df_cur[c].nunique() > 10]
            if high_cardinality:
                st.warning(f"⚠️ 고유값이 많은 변수: {high_cardinality}")

            safe_cols = [c for c in cat_cols if df_cur[c].nunique() <= 10]

            enc_method = st.radio(
                "인코딩 방법",
                ["One-Hot Encoding", "Label Encoding"],
                horizontal=True,
                key="enc_method"
            )

            if enc_method == "One-Hot Encoding":
                enc_cols = st.multiselect(
                    "인코딩할 변수 선택",
                    cat_cols,
                    default=safe_cols,
                    key="enc_cols_ohe"
                )
            else:
                enc_cols = st.multiselect(
                    "인코딩할 변수 선택",
                    cat_cols,
                    default=cat_cols,
                    key="enc_cols_le"
                )

            if st.button("✅ 인코딩 실행", key="btn_encode"):
                if not enc_cols:
                    st.warning("인코딩할 변수를 선택해 주세요.")
                else:
                    df_work = st.session_state.df.copy()
                    try:
                        if enc_method == "One-Hot Encoding":
                            df_work = pd.get_dummies(df_work, columns=enc_cols, drop_first=True)
                            bool_cols = [c for c in df_work.columns if df_work[c].dtype == bool]
                            if bool_cols:
                                df_work[bool_cols] = df_work[bool_cols].astype(int)
                        else:
                            le = LabelEncoder()
                            for c in enc_cols:
                                if df_work[c].isnull().any():
                                    mode_val = df_work[c].mode()
                                    df_work[c] = df_work[c].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")
                                df_work[c] = le.fit_transform(df_work[c].astype(str))

                        if df_work.isnull().sum().sum() > 0:
                            for c in df_work.columns:
                                if df_work[c].isnull().any():
                                    if pd.api.types.is_numeric_dtype(df_work[c]):
                                        df_work[c] = df_work[c].fillna(df_work[c].median())
                                    else:
                                        df_work[c] = df_work[c].fillna("Unknown")

                        st.session_state.df = df_work
                        st.session_state.encoded = True
                        st.success(f"✅ 인코딩 완료 (현재 열 수: {df_work.shape[1]})")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 인코딩 오류: {e}")

    section_header("Feature Selection", "독립변수(X)와 종속변수(Y)를 선택합니다.")
    df_cur = st.session_state.df
    all_cols = df_cur.columns.tolist()

    f1, f2 = st.columns(2)
    with f1:
        selected_y = st.selectbox(
            "종속변수 Y",
            all_cols,
            index=len(all_cols) - 1,
            key="sel_y"
        )
    with f2:
        x_options = [c for c in all_cols if c != selected_y]
        selected_X = st.multiselect(
            "독립변수 X",
            x_options,
            default=x_options,
            key="sel_x"
        )

    if st.button("✅ Feature Selection 저장", key="btn_fs"):
        if not selected_X:
            st.error("❌ 독립변수를 1개 이상 선택해 주세요.")
        else:
            st.session_state.selected_X = selected_X
            st.session_state.selected_y = selected_y
            st.success(f"✅ 저장 완료 | X: {len(selected_X)}개 / Y: {selected_y}")

    if st.session_state.selected_X:
        st.markdown("**현재 선택된 변수**")
        st.markdown(f'<span class="info-chip">Y: {st.session_state.selected_y}</span>', unsafe_allow_html=True)
        for c in st.session_state.selected_X:
            st.markdown(f'<span class="info-chip">{c}</span>', unsafe_allow_html=True)

    section_header("Data Partitioning", "Train/Test 분할을 수행합니다.")

    d1, d2 = st.columns([1, 2])
    with d1:
        split_ratio = st.radio("Train : Test 비율", ["7:3", "8:2"], index=0, key="split_ratio_radio")
        random_seed = st.number_input("Random Seed", value=42, min_value=0, key="random_seed")

    with d2:
        ratio_val = 0.7 if split_ratio == "7:3" else 0.8
        n_total = len(df_cur)
        n_train = int(n_total * ratio_val)
        n_test = n_total - n_train

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.96rem;font-weight:800;margin-bottom:0.6rem;">분할 미리보기</div>
            <div style="display:flex;gap:0.8rem;">
                <div style="flex:1;background:linear-gradient(135deg,#5b6cff,#7b61ff);border-radius:14px;padding:1rem;color:white;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:900;">{n_train:,}</div>
                    <div style="font-size:0.82rem;opacity:0.92;">Train ({int(ratio_val*100)}%)</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#14b8a6,#0ea5e9);border-radius:14px;padding:1rem;color:white;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:900;">{n_test:,}</div>
                    <div style="font-size:0.82rem;opacity:0.92;">Test ({int((1-ratio_val)*100)}%)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("✅ 데이터 분할 실행", key="btn_split"):
        if not st.session_state.selected_X or not st.session_state.selected_y:
            st.error("❌ Feature Selection을 먼저 완료해 주세요.")
        else:
            try:
                X = df_cur[st.session_state.selected_X]
                y = df_cur[st.session_state.selected_y]

                non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
                if non_num:
                    st.warning(f"⚠️ 비수치형 변수는 제외됩니다: {non_num}")
                    X = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]]

                test_size = 1 - ratio_val
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=int(random_seed),
                    stratify=y
                )
                st.session_state.X_train = X_tr
                st.session_state.X_test = X_te
                st.session_state.y_train = y_tr
                st.session_state.y_test = y_te
                st.session_state.split_ratio = split_ratio

                st.success(f"✅ 분할 완료 | Train: {len(X_tr):,} / Test: {len(X_te):,}")
            except Exception as e:
                st.error(f"❌ 분할 오류: {e}")

# ─────────────────────────────────────────────────────────────
# PAGE 4: Model
# ─────────────────────────────────────────────────────────────
elif current == "model":
    check_data()
    section_header("모델 학습", "Logistic Regression, Decision Tree, DNN(MLP)을 학습합니다.")

    if st.session_state.X_train is None:
        st.warning("⚠️ 전처리 페이지에서 데이터 분할을 먼저 완료해 주세요.")
        st.stop()

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:1rem;font-weight:800;margin-bottom:0.4rem;">학습 데이터 현황</div>
        <div class="tiny-muted">
            Train: <b>{len(X_train):,}</b> |
            Test: <b>{len(X_test):,}</b> |
            Feature 수: <b>{X_train.shape[1]}</b> |
            Target: <b>{st.session_state.selected_y}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_tab1, model_tab2, model_tab3 = st.tabs([
        "📉 Logistic Regression",
        "🌳 Decision Tree",
        "🧠 DNN (MLP)"
    ])

    with model_tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            lr_C = st.select_slider("C", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        with c2:
            lr_max_iter = st.slider("Max Iter", 100, 2000, 1000, 100)
        with c3:
            lr_solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga", "newton-cg"])

        if st.button("🚀 Logistic Regression 학습", key="btn_lr"):
            with st.spinner("학습 중..."):
                try:
                    lr = LogisticRegression(
                        C=lr_C,
                        max_iter=lr_max_iter,
                        solver=lr_solver,
                        random_state=42
                    )
                    lr.fit(X_train, y_train)
                    st.session_state.lr_model = lr
                    st.session_state.lr_result = compute_metrics(lr, X_test, y_test)
                    st.success("✅ Logistic Regression 학습 완료")
                except Exception as e:
                    st.error(f"❌ 학습 오류: {e}")

        if st.session_state.lr_result:
            r = st.session_state.lr_result
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card(f"{r['accuracy']:.4f}", "Accuracy")
            with m2:
                metric_card(f"{r['precision']:.4f}", "Precision")
            with m3:
                metric_card(f"{r['recall']:.4f}", "Recall")
            with m4:
                metric_card(f"{r['f1']:.4f}", "F1-Score")

            c_left, c_right = st.columns([1, 1])
            with c_left:
                fig_cm, ax_cm = plt.subplots(figsize=(4.2, 3.5))
                sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax_cm, linewidths=0.5)
                ax_cm.set_title("Confusion Matrix", fontweight="bold")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()
            with c_right:
                plot_single_roc(r, "ROC Curve", "#5b6cff")

    with model_tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            dt_max_depth = st.slider("Max Depth", 1, 20, 5)
        with c2:
            dt_min_samples = st.slider("Min Samples Split", 2, 50, 2)
        with c3:
            dt_criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])

        if st.button("🚀 Decision Tree 학습", key="btn_dt"):
            with st.spinner("학습 중..."):
                try:
                    dt = DecisionTreeClassifier(
                        max_depth=dt_max_depth,
                        min_samples_split=dt_min_samples,
                        criterion=dt_criterion,
                        random_state=42
                    )
                    dt.fit(X_train, y_train)
                    st.session_state.dt_model = dt
                    st.session_state.dt_result = compute_metrics(dt, X_test, y_test)
                    st.success("✅ Decision Tree 학습 완료")
                except Exception as e:
                    st.error(f"❌ 학습 오류: {e}")

        if st.session_state.dt_result:
            r = st.session_state.dt_result
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card(f"{r['accuracy']:.4f}", "Accuracy")
            with m2:
                metric_card(f"{r['precision']:.4f}", "Precision")
            with m3:
                metric_card(f"{r['recall']:.4f}", "Recall")
            with m4:
                metric_card(f"{r['f1']:.4f}", "F1-Score")

            row1, row2 = st.columns([1, 1])
            with row1:
                plot_single_roc(r, "Decision Tree ROC Curve", "#14b8a6")
            with row2:
                fi = pd.Series(
                    st.session_state.dt_model.feature_importances_,
                    index=X_train.columns
                ).sort_values(ascending=True).tail(15)

                fig_fi, ax_fi = plt.subplots(figsize=(6.5, 4.2))
                fi.plot(kind="barh", ax=ax_fi, color="#5b6cff", edgecolor="white")
                ax_fi.set_title("Feature Importance (Top 15)", fontweight="bold")
                ax_fi.set_xlabel("Importance")
                ax_fi.set_ylabel("Feature")
                ax_fi.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_fi, use_container_width=True)
                plt.close()

            fig_cm, ax_cm = plt.subplots(figsize=(4.2, 3.5))
            sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax_cm, linewidths=0.5)
            ax_cm.set_title("Confusion Matrix", fontweight="bold")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=False)
            plt.close()

    with model_tab3:
        st.markdown("#### DNN (MLPClassifier) Hyperparameters")
        d1, d2, d3 = st.columns(3)
        with d1:
            hidden_option = st.selectbox(
                "Hidden Layers",
                ["64", "128", "64,32", "128,64", "128,64,32"],
                index=2
            )
        with d2:
            dnn_alpha = st.select_slider(
                "Alpha",
                options=[0.0001, 0.001, 0.01, 0.1],
                value=0.0001
            )
        with d3:
            dnn_lr = st.select_slider(
                "Learning Rate Init",
                options=[0.0001, 0.001, 0.01],
                value=0.001
            )

        d4, d5, d6 = st.columns(3)
        with d4:
            dnn_max_iter = st.slider("Max Iter", 100, 1000, 300, 50)
        with d5:
            dnn_activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])
        with d6:
            dnn_batch_size = st.selectbox("Batch Size", ["auto", 16, 32, 64, 128], index=2)

        if st.button("🚀 DNN 학습", key="btn_dnn"):
            with st.spinner("DNN 학습 중..."):
                try:
                    hidden_layers = tuple(int(x.strip()) for x in hidden_option.split(","))

                    dnn_pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("mlp", MLPClassifier(
                            hidden_layer_sizes=hidden_layers,
                            activation=dnn_activation,
                            alpha=dnn_alpha,
                            learning_rate_init=dnn_lr,
                            max_iter=dnn_max_iter,
                            batch_size=dnn_batch_size,
                            random_state=42
                        ))
                    ])

                    dnn_pipe.fit(X_train, y_train)
                    st.session_state.dnn_model = dnn_pipe
                    st.session_state.dnn_result = compute_metrics(dnn_pipe, X_test, y_test)
                    st.success("✅ DNN 학습 완료")
                except Exception as e:
                    st.error(f"❌ DNN 학습 오류: {e}")

        if st.session_state.dnn_result:
            r = st.session_state.dnn_result
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card(f"{r['accuracy']:.4f}", "Accuracy")
            with m2:
                metric_card(f"{r['precision']:.4f}", "Precision")
            with m3:
                metric_card(f"{r['recall']:.4f}", "Recall")
            with m4:
                metric_card(f"{r['f1']:.4f}", "F1-Score")

            c_left, c_right = st.columns([1, 1])
            with c_left:
                fig_cm, ax_cm = plt.subplots(figsize=(4.2, 3.5))
                sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Purples", ax=ax_cm, linewidths=0.5)
                ax_cm.set_title("Confusion Matrix", fontweight="bold")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()
            with c_right:
                plot_single_roc(r, "DNN ROC Curve", "#7b61ff")

# ─────────────────────────────────────────────────────────────
# PAGE 5: Result
# ─────────────────────────────────────────────────────────────
elif current == "result":
    check_data()
    section_header("결과 분석", "세 모델의 성능을 비교하고 결과를 저장합니다.")

    lr_r = st.session_state.lr_result
    dt_r = st.session_state.dt_result
    dnn_r = st.session_state.dnn_result

    if lr_r is None and dt_r is None and dnn_r is None:
        st.warning("⚠️ 모델 학습 페이지에서 모델을 먼저 학습해 주세요.")
        st.stop()

    metrics_rows = []
    for name, r in [
        ("Logistic Regression", lr_r),
        ("Decision Tree", dt_r),
        ("DNN (MLP)", dnn_r)
    ]:
        if r:
            metrics_rows.append({
                "Model": name,
                "Accuracy": float(r["accuracy"]),
                "Precision": float(r["precision"]),
                "Recall": float(r["recall"]),
                "F1-Score": float(r["f1"]),
                "AUC": float(r["auc"]) if r["auc"] is not None else np.nan,
            })

    if metrics_rows:
        result_df = pd.DataFrame(metrics_rows).set_index("Model")
        display_df = result_df.copy().round(4)
        st.dataframe(
            display_df.style.highlight_max(
                axis=0,
                color="#dcfce7",
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
            ),
            use_container_width=True
        )

    section_header("성능 비교 차트", "Accuracy, Precision, Recall, F1-Score를 시각적으로 비교합니다.")

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_results = {
        "Logistic Regression": lr_r,
        "Decision Tree": dt_r,
        "DNN (MLP)": dnn_r
    }
    model_colors = {
        "Logistic Regression": "#5b6cff",
        "Decision Tree": "#14b8a6",
        "DNN (MLP)": "#7b61ff"
    }

    available_models = [name for name, r in model_results.items() if r is not None]
    values = {
        name: [model_results[name]["accuracy"], model_results[name]["precision"],
               model_results[name]["recall"], model_results[name]["f1"]]
        for name in available_models
    }

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metric_names))
    total_models = max(len(available_models), 1)
    width = 0.22 if total_models >= 3 else 0.35

    offsets = np.linspace(-width, width, total_models) if total_models > 1 else [0]

    for idx, name in enumerate(available_models):
        vals = values[name]
        bars = ax_bar.bar(
            x + offsets[idx], vals, width,
            label=name,
            color=model_colors[name], edgecolor="white", alpha=0.92
        )
        for bar in bars:
            ax_bar.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=11)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, alpha=0.28)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    section_header("ROC Curve", "각 모델의 ROC Curve와 AUC를 비교합니다.")

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random (AUC = 0.50)")

    for name, r in model_results.items():
        if r and r["fpr"] is not None:
            ax_roc.plot(
                r["fpr"], r["tpr"],
                color=model_colors[name], lw=2.5,
                label=f"{name} (AUC = {r['auc']:.4f})"
            )
            ax_roc.fill_between(r["fpr"], r["tpr"], alpha=0.08, color=model_colors[name])

    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax_roc.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_roc, use_container_width=True)
    plt.close()

    section_header("Confusion Matrix", "각 모델의 confusion matrix를 비교합니다.")

    available_cm_models = [(name, r) for name, r in model_results.items() if r is not None]
    cols = st.columns(len(available_cm_models)) if available_cm_models else []

    for col, (name, r) in zip(cols, available_cm_models):
        with col:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.3))
            sns.heatmap(
                r["cm"], annot=True, fmt="d",
                cmap="Blues", ax=ax_cm, linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            ax_cm.set_title(name, fontsize=11, fontweight="bold")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close()

    if st.session_state.dt_model is not None:
        section_header("Decision Tree Feature Importance", "Decision Tree가 어떤 변수를 중요하게 보는지 확인합니다.")
        fi = pd.Series(
            st.session_state.dt_model.feature_importances_,
            index=st.session_state.X_train.columns
        ).sort_values(ascending=True).tail(15)

        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        fi.plot(kind="barh", ax=ax_fi, color="#5b6cff", edgecolor="white")
        ax_fi.set_title("Decision Tree Feature Importance (Top 15)", fontweight="bold")
        ax_fi.set_xlabel("Importance")
        ax_fi.set_ylabel("Feature")
        ax_fi.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

    section_header("결과 저장", "모델 성능 비교표를 CSV로 다운로드할 수 있습니다.")
    if metrics_rows:
        csv_result = result_df.round(4).to_csv(encoding="utf-8-sig")
        st.download_button(
            label="📥 성능 지표 CSV 다운로드",
            data=csv_result,
            file_name="model_performance.csv",
            mime="text/csv"
        )
