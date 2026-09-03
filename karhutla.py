import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# =========================
# MEMBACA DATA
# =========================

file = "luas-karhutla-provinsi.xlsx"
df = pd.read_excel(file)

# Hanya mengambil data tahun 2026
# Baris "Total" DIHAPUS karena bukan provinsi
df_provinsi = df[df["Provinsi"].str.strip().str.lower() != "total"].copy()

data = (
    df_provinsi[2026]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)

data = pd.to_numeric(data, errors="coerce")
data = data.dropna()

# Data positif untuk distribusi kontinu
data_positif = data[data > 0]


# =========================
# INFORMASI DATA
# =========================

print("JUMLAH PROVINSI :", len(data))
print("JUMLAH NILAI NOL:", (data == 0).sum())
print("JUMLAH NILAI POSITIF:", len(data_positif))

print("MINIMUM :", data.min())
print("MAKSIMUM :", data.max())
print("MEAN :", data.mean())
print("MEDIAN :", data.median())


# =========================
# HISTOGRAM
# =========================

bins = np.logspace(
    np.log10(data_positif.min()),
    np.log10(data_positif.max()),
    10
)

plt.figure(figsize=(10, 6))

plt.hist(
    data_positif,
    bins=bins,
    edgecolor="black",
    alpha=0.7
)

plt.xscale("log")

plt.title(
    "Distribusi Luas Karhutla Provinsi Tahun 2026",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Luas Karhutla (Hektare)")
plt.ylabel("Frekuensi")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "histogram_karhutla_2026.png",
    dpi=300
)

plt.show()


# =========================
# UJI DISTRIBUSI PROBABILITAS
# =========================

distribusi = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min
}

hasil = []

for nama, dist in distribusi.items():

    parameter = dist.fit(data_positif)

    ks, p_value = stats.kstest(
        data_positif,
        dist.cdf,
        args=parameter
    )

    hasil.append([
        nama,
        ks,
        p_value
    ])


hasil = pd.DataFrame(
    hasil,
    columns=[
        "Distribusi",
        "KS",
        "p-value"
    ]
)

print("\nHASIL UJI DISTRIBUSI")
print(hasil)


# =========================
# DISTRIBUSI TERBAIK
# =========================

terbaik = hasil.loc[
    hasil["p-value"].idxmax()
]

print("\nDistribusi yang paling sesuai:")
print(terbaik["Distribusi"])

print("KS:", terbaik["KS"])
print("p-value:", terbaik["p-value"])