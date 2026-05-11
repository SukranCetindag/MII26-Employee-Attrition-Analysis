
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Çalışan İstifa Tahmin Dashboard",
    layout="wide"
)

st.title("🤖 Çalışan İstifa Tahmin Dashboard")
st.write("Bu sistem, çalışan bilgilerine göre istifa riskini tahmin eder.")

df = pd.read_csv("MII26-Employee-Attrition-Analysis-main/catboost_info/WA_Fn-UseC_-HR-Employee-Attrition.csv")
model = joblib.load("MII26-Employee-Attrition-Analysis-main/catboost_info/employee_attrition_model.pkl")

st.subheader("📌 Veri Seti Önizleme")
st.dataframe(df.head(), use_container_width=True)

st.subheader("🧑‍💼 Çalışan Bilgileriyle Tahmin Yap")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Yaş", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input("Aylık Gelir", min_value=1000, max_value=25000, value=5000)
    overtime = st.selectbox("Fazla Mesai", ["Yes", "No"])

with col2:
    distance = st.number_input("Eve Uzaklık", min_value=1, max_value=30, value=5)
    years_at_company = st.number_input("Şirkette Geçirilen Yıl", min_value=0, max_value=40, value=3)
    job_satisfaction = st.selectbox("İş Tatmini", [1, 2, 3, 4])

with col3:
    department = st.selectbox("Departman", df["Department"].unique())
    gender = st.selectbox("Cinsiyet", df["Gender"].unique())
    work_life_balance = st.selectbox("İş-Yaşam Dengesi", [1, 2, 3, 4])

input_data = df.drop("Attrition", axis=1).iloc[[0]].copy()

input_data["Age"] = age
input_data["MonthlyIncome"] = monthly_income
input_data["OverTime"] = overtime
input_data["DistanceFromHome"] = distance
input_data["YearsAtCompany"] = years_at_company
input_data["JobSatisfaction"] = job_satisfaction
input_data["Department"] = department
input_data["Gender"] = gender
input_data["WorkLifeBalance"] = work_life_balance

if st.button("Tahmin Et"):
    prediction = model.predict(input_data)[0]

    if prediction == 1 or prediction == "Yes":
        st.error("⚠️ Bu çalışanın istifa riski yüksek görünüyor.")
    else:
        st.success("✅ Bu çalışanın istifa riski düşük görünüyor.")

        st.markdown("---")
st.header("📊 Veri Analizi ve Görselleştirmeler")

def grafik_karti(baslik, fig, aciklama):
    st.markdown(f"#### {baslik}")
    st.pyplot(fig, use_container_width=True)
    st.markdown(f"""
    <div style="
        background-color:#f7f7f7;
        padding:12px;
        border-radius:10px;
        font-size:14px;
        min-height:95px;
    ">
    <b>Yorum:</b><br>{aciklama}
    </div>
    """, unsafe_allow_html=True)

# =========================
# 1. SATIR
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.countplot(
        data=df,
        x="Attrition",
        hue="Attrition",
        palette=["#C3B1E1", "#F7D154"],
        legend=False,
        ax=ax
    )
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )
    ax.set_title("İstifa Dağılımı")
    ax.set_xlabel("İstifa Durumu")
    ax.set_ylabel("Çalışan Sayısı")
    ax.set_xticklabels(["İstifa Eden", "İstifa Etmeyen"])

    grafik_karti(
        "İstifa Dağılımı",
        fig,
        "Bu grafik, istifa eden ve etmeyen çalışanların sayısal dağılımını göstermektedir. Veri setinde istifa etmeyen çalışan sayısı daha yüksektir."
    )

with col2:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.histplot(
        data=df,
        x="Age",
        hue="Attrition",
        multiple="stack",
        kde=True,
        palette=["#F7D154", "#C3B1E1"],
        ax=ax
    )
    ax.set_title("Yaş ve İstifa İlişkisi")
    ax.set_xlabel("Yaş")
    ax.set_ylabel("Çalışan Sayısı")

    grafik_karti(
        "Yaş ve İstifa",
        fig,
        "Yaş dağılımı çalışanların hangi yaş aralıklarında yoğunlaştığını gösterir. Genç ve orta yaş grubundaki çalışanların ayrılma eğilimi daha dikkatli incelenmelidir."
    )

with col3:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.boxplot(
        data=df,
        x="Attrition",
        y="MonthlyIncome",
        hue="Attrition",
        palette=["#F7D154", "#C3B1E1"],
        legend=False,
        ax=ax
    )
    ax.set_title("Aylık Gelir ve İstifa")
    ax.set_xlabel("İstifa Durumu")
    ax.set_ylabel("Aylık Gelir")
    ax.set_xticklabels(["İstifa Eden", "İstifa Etmeyen"])

    grafik_karti(
        "Maaş ve İstifa",
        fig,
        "Bu grafik, aylık gelir düzeyi ile istifa durumu arasındaki ilişkiyi gösterir. Düşük gelir seviyeleri çalışan ayrılma riskini artırabilir."
    )

# =========================
# 2. SATIR
# =========================
col4, col5, col6 = st.columns(3)

with col4:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.countplot(
        data=df,
        x="Department",
        hue="Attrition",
        palette=["#F7D154", "#C3B1E1"],
        ax=ax
    )
    ax.set_title("Departman Bazlı İstifa")
    ax.set_xlabel("Departman")
    ax.set_ylabel("Çalışan Sayısı")
    ax.tick_params(axis="x", rotation=15)

    grafik_karti(
        "Departman Bazlı İstifa",
        fig,
        "Departmanlara göre istifa dağılımı incelenmiştir. Özellikle çalışan sayısının fazla olduğu departmanlarda ayrılma oranı daha dikkatli değerlendirilmelidir."
    )

with col5:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.countplot(
        data=df,
        x="JobSatisfaction",
        hue="Attrition",
        palette=["#F7D154", "#C3B1E1"],
        ax=ax
    )
    ax.set_title("İş Tatmini ve İstifa")
    ax.set_xlabel("İş Tatmini Seviyesi")
    ax.set_ylabel("Çalışan Sayısı")

    grafik_karti(
        "İş Tatmini",
        fig,
        "İş tatmini seviyesi çalışan bağlılığı üzerinde önemli bir etkendir. Düşük iş tatmini, çalışanların istifa etme eğilimini artırabilir."
    )

with col6:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.countplot(
        data=df,
        x="OverTime",
        hue="Attrition",
        palette=["#800020", "#C8A2FF"],
        ax=ax
    )
    ax.set_title("Fazla Mesai ve İstifa")
    ax.set_xlabel("Fazla Mesai")
    ax.set_ylabel("Çalışan Sayısı")

    grafik_karti(
        "Fazla Mesai",
        fig,
        "Fazla mesai yapan çalışanlarda istifa riski daha yüksek olabilir. Bu durum iş-yaşam dengesi açısından kritik bir göstergedir."
    )

# =========================
# 3. SATIR
# =========================
col7, col8, col9 = st.columns(3)

with col7:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.boxplot(
        data=df,
        x="Attrition",
        y="DistanceFromHome",
        hue="Attrition",
        palette=["#4CAF50", "#81C784"],
        legend=False,
        ax=ax
    )
    ax.set_title("Eve Uzaklık ve İstifa")
    ax.set_xlabel("İstifa Durumu")
    ax.set_ylabel("Eve Uzaklık")
    ax.set_xticklabels(["İstifa Eden", "İstifa Etmeyen"])

    grafik_karti(
        "Eve Uzaklık",
        fig,
        "Eve uzaklık çalışan memnuniyetini etkileyebilir. Ulaşım mesafesi arttıkça çalışanların ayrılma riski değişkenlik gösterebilir."
    )

with col8:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.histplot(
        data=df,
        x="YearsAtCompany",
        hue="Attrition",
        multiple="stack",
        palette=["#C8A2FF", "#9C6BFF"],
        ax=ax
    )
    ax.set_title("Şirkette Geçirilen Yıl")
    ax.set_xlabel("Şirkette Geçirilen Yıl")
    ax.set_ylabel("Çalışan Sayısı")

    grafik_karti(
        "Şirkette Geçirilen Yıl",
        fig,
        "Şirkette geçirilen yıl çalışan bağlılığını gösteren önemli bir değişkendir. Kısa süredir çalışan kişilerde ayrılma riski daha yüksek olabilir."
    )

with col9:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.countplot(
        data=df,
        x="WorkLifeBalance",
        hue="Attrition",
        palette=["#4CAF50", "#C8A2FF"],
        ax=ax
    )
    ax.set_title("İş-Yaşam Dengesi")
    ax.set_xlabel("İş-Yaşam Dengesi")
    ax.set_ylabel("Çalışan Sayısı")

    grafik_karti(
        "İş-Yaşam Dengesi",
        fig,
        "İş-yaşam dengesi çalışanların şirkette kalma kararını etkileyebilir. Düşük denge puanı, istifa riskini artıran bir unsur olabilir."
    )

# =========================
# 4. SATIR - MODEL ANALİZLERİ
# =========================
st.markdown("---")
st.header("🤖 Model Analiz Grafikleri")

col10, col11, col12 = st.columns(3)

with col10:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    cm = [[226, 21], [20, 27]]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title("Final Model Başarı Matrisi")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")

    grafik_karti(
        "Final Model Başarı Matrisi",
        fig,
        "Bu matris modelin doğru ve yanlış tahminlerini gösterir. Model, istifa eden ve etmeyen çalışanları ayırt etmek için kullanılmıştır."
    )

with col11:
    model_names = [
        "LogisticRegression", "LinearSVC", "CalibratedClassifierCV",
        "LinearDiscriminant", "GaussianNB", "PassiveAggressive",
        "BernoulliNB", "Perceptron", "ExtraTreeClassifier", "NearestCentroid"
    ]
    accuracies = [89.1, 88.8, 88.4, 88.1, 84.0, 84.0, 83.0, 82.0, 76.2, 69.4]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.barplot(x=accuracies, y=model_names, color="#800020", ax=ax)
    ax.set_title("Model Kıyaslaması")
    ax.set_xlabel("Doğruluk Oranı (%)")
    ax.set_ylabel("Modeller")
    ax.tick_params(axis="y", labelsize=7)

    grafik_karti(
        "LazyPredict Model Kıyaslaması",
        fig,
        "Farklı makine öğrenmesi modelleri karşılaştırılmıştır. En yüksek doğruluk oranına sahip modeller final model seçimi için referans alınmıştır."
    )

with col12:
    features = [
        "MonthlyIncome", "OverTime", "Age", "DailyRate", "HourlyRate",
        "TotalWorkingYears", "DistanceFromHome", "YearsAtCompany",
        "YearsInCurrentRole", "NumCompaniesWorked"
    ]
    importance = [0.132, 0.124, 0.116, 0.108, 0.098, 0.090, 0.082, 0.074, 0.066, 0.058]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.barplot(
        x=importance,
        y=features,
        palette=["#F7D154", "#C8A2FF"] * 5,
        ax=ax
    )
    ax.set_title("Tahmin Kriterleri")
    ax.set_xlabel("Önem Derecesi")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)

    grafik_karti(
        "Yapay Zekanın Tahmin Kriterleri",
        fig,
        "Modelin tahmin yaparken en çok dikkate aldığı değişkenler gösterilmiştir. Aylık gelir, fazla mesai ve yaş öne çıkan kriterlerdir."
    )

# =========================
# 5. SATIR
# =========================
col13, col14, col15 = st.columns(3)

with col13:
    trigger_features = [
        "OverTime", "YearsAtCompany", "YearsInCurrentRole",
        "NumCompaniesWorked", "YearsSinceLastPromotion", "JobLevel",
        "YearsWithCurrManager", "MaritalStatus", "JobSatisfaction",
        "TotalWorkingYears"
    ]
    effects = [0.86, 0.75, -0.70, 0.47, 0.00, 0.00, -0.45, 0.45, -0.41, -0.40]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.barplot(x=effects, y=trigger_features, palette="mako", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("İstifayı Tetikleyen Faktörler")
    ax.set_xlabel("Etki Gücü")
    ax.set_ylabel("Özellik")
    ax.tick_params(axis="y", labelsize=7)

    grafik_karti(
        "İstifayı En Çok Tetikleyen Faktörler",
        fig,
        "Pozitif değerler istifa riskini artıran, negatif değerler ise azaltan faktörleri gösterir. Fazla mesai en güçlü tetikleyicilerden biridir."
    )
