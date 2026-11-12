# 🏆 More than a Hack 2027 — Project Tracker & Team Alignment

> **Dokumen ini dibuat sebagai panduan kerja tim & AI Agent** agar setiap anggota dan agen memiliki pemahaman yang selaras (*single source of truth*) mengenai status proyek, temuan riset (EDA), keputusan teknis, dan langkah kerja berikutnya.

---

## 📌 Ringkasan Singkat Kompetisi
* **Penyelenggara:** Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital Barcelona.
* **Tantangan:** Menentukan penguasaan bola (*ball possession*) frame-by-frame untuk 1 pertandingan LaLiga: **FC Barcelona (`H`) vs Athletic Club (`A`)**, 24 Agustus 2024.
* **Target Output:** File CSV (`frame,label`) tepat **199.319 baris** (frame `0 .. 199318`).
* **Label Resmi:**
  * `H` : Home (FC Barcelona) menguasai bola (bola hidup)
  * `A` : Away (Athletic Club) menguasai bola (bola hidup)
  * `D` : Dead ball (bola mati / keluar lapangan / jeda wasit)
  * `X` : Abstain (digunakan di luar babak aktif: pre-match, halftime, post-match)
* **Metrik Utama Peringkat:** **`spellF1`** (Temporal IoU matching per possession spell, bukan per-frame accuracy!).
* **Floor Baseline Resmi yang Harus Dikalahkan:** `baseline_smooth50` dengan `spellF1` = **0.2433**.
* **Deadline Submisi:** **17 November 2026** (Wajib 3–5 anggota tim terdaftar).

---

## 🚦 Aturan Emas & Batasan (*DOs and DON'Ts*)
* ✅ **DO:** Semua prediksi harus dihasilkan murni dari algoritma/kode.
* ✅ **DO:** Selalu validasi file output dengan `python evaluation/validate_submission.py <file>.csv` sebelum submit.
* ❌ **DON'T:** Dilarang keras melakukan hand-labeling / koreksi manual pada pertandingan Barcelona vs Athletic ini.
* ❌ **DON'T:** Dilarang memasukkan event data resmi beranotasi manusia (seperti Opta/StatsBomb) khusus laga ini ke dalam pipeline.
* ❌ **DON'T:** Dilarang push data pertandingan (`data/`) ke GitHub publik (sudah diamankan via `.gitignore`).
* ❌ **DON'T:** Jangan mengunggah video atau data mentah ke LLM cloud publik (pelanggaran ToS data).

---

## 🔍 Temuan Kunci dari Exploratory Data Analysis (EDA)

Skrip EDA tersimpan di: [`scratch/eda.py`](scratch/eda.py)

### 1. Struktur Match & Graded Windows (Total 199.319 Frame)
* **Frame `0 .. 11184` (Pre-match):** Tidak dinilai $\to$ Label `X`.
* **Frame `11185 .. 90651` (Babak 1):** Dinilai (79.467 frame) $\to$ Prediksi `H`/`A`/`D`.
* **Frame `90652 .. 110807` (Halftime):** Tidak dinilai $\to$ Label `X`.
* **Frame `110808 .. 113741` (Pre-Kickoff Babak 2):** **DINILAI (2.934 frame) $\to$ 100% PASTI `D`!** *(Poin gratis yang sering dilewatkan)*.
* **Frame `113742 .. 190197` (Babak 2 Play):** Dinilai (76.456 frame) $\to$ Prediksi `H`/`A`/`D`.
* **Frame `190198 .. 199318` (Post-match):** Tidak dinilai $\to$ Label `X`.

### 2. Isu Bola Hilang (*Missing Ball* 40.7%)
* Pada 158.857 frame aktif yang dinilai, kamera kehilangan bola pada **64.643 frame (40.69%)**.
* **Kelemahan Baseline Resmi:** Saat bola hilang sesaat (umpan lambung / kemelut), baseline langsung menebak `D`. Ini memotong satu penguasaan bola utuh menjadi puluhan potongan kecil sehingga `spellF1` anjlok.
* **Solusi Tim Kita:** *State persistence / Forward-fill*. Jika bola hilang hanya sesaat ($< 1.5 - 2.0$ detik), penguasaan tetap dipegang oleh tim terakhir.

### 3. Fase Duel & Perebutan Bola (*Contested Phases* 17.2%)
* Sebanyak 17.2% frame aktif memiliki pemain Barcelona dan Athletic dalam selisih jarak $< 1.0$ meter dari bola.
* **Fisika Lapangan:** Jangkauan kaki/tekel pemain adalah $0.8 - 1.2$ m, dan tracking kamera memiliki jitter $\pm 20-50$ cm.
* **Aturan Penguasaan Bola:** Sesuai `CHALLENGE.md`, pada fase duel/perebutan, *possession stays with the last team that had it*. Kita menerapkan threshold selisih ($\Delta d > 1.0$ m) agar tidak terjadi kedipan (*flickering*) label.

### 4. Smart Tagging (59 Set Pieces, 638 Fase Taktis)
* Smart Tagging adalah output Computer Vision otomatis (bukan human truth), sehingga ada variasi (contoh: 12 corner di Smart Tagging vs 11 corner resmi LaLiga).
* **Fungsi Utama:** Digunakan sebagai **sinyal/prior kuat**, bukan aturan kaku:
  * Jendela event `SET PIECES` (throw-in, corner, free-kick, goal-kick) adalah indikator kuat untuk bola mati `D`.
  * Jendela fase aktif (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`) adalah indikator kuat bahwa bola masih hidup meskipun tracker kehilangan bola sesaat.

---

## 📈 Roadmap & Status Pengerjaan

| Step | Deskripsi | Status | Output / File |
| :---: | :--- | :---: | :--- |
| **0** | **Aturan & Pipeline Setup** | ✅ Selesai | Gitignore diamankan, scorer teruji |
| **1** | **Penyediaan Data Taktikal** | ✅ Selesai | `data/tactical/` terisi 4 file kunci (ATD, XML, Tagging, Homography) |
| **2** | **Deep EDA & Validasi Hipotesis** | ✅ Selesai | [`scratch/eda.py`](scratch/eda.py) (analisis missing ball, duel, boundaries) |
| **3** | **Enhanced Heuristic Baseline v1** | ✅ Selesai | [`models/v1_enhanced_baseline.py`](models/v1_enhanced_baseline.py) (`spellF1` naik ke **0.4000**) |
| **4** | **State Machine / HMM Model v2** | ✅ Selesai | [`models/v2_smart_tagging_hmm.py`](models/v2_smart_tagging_hmm.py) (membuktikan batasan Markovian) |
| **5** | **Hybrid Smart Tagging Mask v3** | ✅ Selesai | [`models/v3_hybrid_smart_tagging.py`](models/v3_hybrid_smart_tagging.py) (Stoppage mask set piece) |
| **6** | **Kinematic Physics (Spearman) v4** | ✅ Selesai | [`models/v4_kinematic_physics.py`](models/v4_kinematic_physics.py) (Time-to-Intercept) |
| **7** | **Two-Branch 5NN (Borghesi 2023) v5**| ✅ Selesai | [`models/v5_two_branch_5nn.py`](models/v5_two_branch_5nn.py) (`macroF1` All-time #1 **0.6661**) |
| **8** | **Voronoi & Pitch Control Surface v7**| ✅ Selesai | [`models/v7_voronoi_pitch_control.py`](models/v7_voronoi_pitch_control.py) (**`transF1 = 0.3043`**) |
| **9** | **Turnover Bridge Champion v8** | ✅ Selesai | [`models/v8_turnover_bridge_champion.py`](models/v8_turnover_bridge_champion.py) (`spellF1 = 0.4824`) |
| **10**| **Ball Flight Dynamics Champion v9** | ✅ Selesai | [`models/v9_ball_flight_kinematics.py`](models/v9_ball_flight_kinematics.py) (`spellF1 = 0.5500`, `macroF1 = 0.6681`) |
| **11**| **Multimodal Set-Piece Kinematics v10** | ✅ Selesai | [`models/v10_multimodal_kinematics.py`](models/v10_multimodal_kinematics.py) (`spellF1 = 0.5697`, `macroF1 = 0.6803`) |
| **12**| **Collective Flocking Champion v11** | ✅ Selesai | [`models/v11_collective_flocking.py`](models/v11_collective_flocking.py) (**`spellF1 = 0.5697`**, **`macroF1 = 0.7551` / `0.7728`** 🏆 DOMINASI TOTAL KEDUA METRIK!) |
| **13**| **Validasi & Technical Document** | ✅ Selesai | [`technical_document.md`](technical_document.md) (Dokumen teknis 7 bab lengkap siap submit) |

---

## 📊 Hasil Evaluasi Papan Skor Resmi (Public Ground Truth)

Hasil perbandingan model terhadap baseline panitia di `evaluation/scoreboard.csv`:

| Peringkat | Model Submission | macroF1 | spellF1 (Primary) | Pred Spells | Durasi Rata-rata | transF1 | Latency | shareErr | Frag |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| - | **Ground Truth (Kenyataan)** | *1.0000* | *1.0000* | **17** | **14.2 s** | *-* | *0.0* | *0.0000* | **1.00** |
| 🥇 **1** | **`v12_dead_ball_w82` (Ours)** | **0.7820** | **0.5765** | **17 (EXACT)** | **15.1 s** | **0.3256** | **+1.0 f (0.04s)** | **0.003** | **1.14** |
| 🥈 **2** | **`v12_dead_ball_w80` (Ours)** | **0.7809** | **0.5765** | **17 (EXACT)** | **15.1 s** | **0.2791** | -2.5 f | **0.003** | **1.14** |
| 🥉 **3** | `v11_player_speed_w80` | 0.7551 | 0.5697 | 16 | 16.4 s | 0.2500 | -7.8 f | 0.000 | 1.00 |
| 4 | `v10_set_pieces_w80` | 0.6564 | 0.5697 | 16 | 17.5 s | 0.2051 | -11.0 f | 0.007 | 0.95 |
| 5 | `v11_player_speed_w75` | 0.7728 | 0.5529 | 17 | 15.3 s | 0.2857 | -8.2 f | 0.005 | 1.10 |
| 6 | `v10_set_pieces_w75` | 0.6803 | 0.5529 | 17 | 16.3 s | 0.2439 | -10.8 f | 0.003 | 1.05 |
| 7 | `v9_flight_filter_w80` | 0.6433 | 0.5500 | 15 | 18.8 s | 0.2162 | -11.2 f | 0.010 | 0.86 |
| 10 | `v8_turnover_bridge_w90` | 0.6279 | 0.4824 | 17 | 16.6 s | 0.2000 | -11.2 f | 0.037 | 1.00 |
| 13 | `baseline_smooth125` *(Panitia)* | 0.5812 | 0.4242 | 16 | 18.5 s | 0.1667 | -7.7 f | 0.052 | 0.81 |
| 26 | `baseline_smooth50` *(Floor Resmi)* | 0.6207 | 0.3256 | 26 | 10.8 s | 0.0816 | +4.0 f | 0.048 | 1.43 |

> 📌 **Perbandingan Head-to-Head vs Model v6 Teman:**
> * **Model Teman (v6):** `spellF1 = 0.5250`, `macroF1 = 0.7454`, `Possession = 15`, `Team Acc = 89.1%`.
> * **Model Kita (`v12_dead_ball_w82`):** **`spellF1 = 0.5765` (+0.0515 lebih tinggi)**, **`macroF1 = 0.7820` (+0.0366 lebih tinggi)**, **`Possession = 17` (Exact match)**, **`Team Acc = 91.3%` (+2.2% lebih tinggi)**, **`transF1 = 0.3256` (Latency +1.0 frame = 0.04s)**.


---

## 🛠️ Perintah Penting untuk Rekan Tim / Agent

```bash
# 1. Menjalankan evaluasi papan skor pada Public Ground Truth (5 menit)
python evaluation/run_evaluation.py

# 2. Memvalidasi format file submission sebelum diserahkan
python evaluation/validate_submission.py evaluation/results/<nama_file>.csv

# 3. Menjalankan skrip EDA
python scratch/eda.py
```

