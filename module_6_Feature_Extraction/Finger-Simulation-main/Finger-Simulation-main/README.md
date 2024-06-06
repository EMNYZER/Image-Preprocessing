# Finger Simulation Experiment

## Project Overview

Pada praktikum modul ini, Anda akan melakukan experiment klasifikasi citra jari berdasarkan jumlah jarinya dengan menggunakan dataset jari yang telah Anda siapkan sebelumnya. Hal ini bertujuan untuk menguji kemampuan Anda dalam mengimplemetasikan teknik pengolahan citra untuk mengklasifikasikan citra jari. Anda diharuskan untuk memilih tahapan preprocessing yang tepat sesuai dengan karakteristik data yang ada. Setelah itu Anda akan melakukan feature extraction dan melakukan pembuatan model klasifikasi.

Anda akan diminta untuk melakukan eksperimen 3-5 kali percobaan dengan notebook yang berbeda (format notebook terdapat pada folder "starting pack"). Pada setiap percobaannya, Anda diharuskan melakukan improvement pada model klasifikasi yang telah Anda buat sebelumnya. Anda dapat melakukan improvement dengan cara menyesuaikan teknik preprocessing, mengubah model klasifikasi, melakukan hyperparameter tuning, atau melakukan teknik lain yang Anda anggap perlu. Pada "starting pack" terdapat file `validation.csv` yang berisi data hasil ekstraksi fitur yang digunakan untuk menguji model klasifikasi dengan data yang belum pernah dilihat oleh model sebelumnya. Pada modul ini, Anda diharuskan membuat resume hasil akhir eksperimen yang telah Anda lakukan (Format resume terdapat pada foler "starting pack"). Pada resume tersebut, Anda diharuskan untuk menjelaskan hasil eksperimen yang telah Anda lakukan pada tiap percobaannya.

## Data Understanding

Pada tahapan ini, pertama-tama lakukanlah data loading beserta labelnya, gunakan nama folder sebagai labelnya. Lakukan penyeragaman ukuran dari dataset dengan resize, bisa menggunakan 150x150 atau 300x300. Selanjutnya, Anda diminta untuk melakukan eksplorasi data untuk memahami karakteristik data yang digunakan. Anda dapat menampilkan jumlah data, karakteristik data (kondisi background, noise, pencahyaan, dll), distribusi data, sampel data, dan lainnya. Hal ini bertujuan untuk memahami data yang akan digunakan dalam proses klasifikasi, sehingga dapat memilih teknik preprocessing yang tepat ataupun penanganan jika terdapat data yang tidak seimbang. Berikut merupakan contohnya:

![alt text](https://github.com/mausneg/Finger-Simulation/blob/main/module%20source/image.png?raw=true)

Gambar 1. Contoh Visualisasi Distribusi Data

![alt text](https://github.com/mausneg/Finger-Simulation/blob/main/module%20source/image-1.png?raw=true)

Gambar 2. Contoh Sampel Data

## Data Preparation

### Image Augmentation

Pada bagian awal dari tahap ini, Anda **diwajibkan** untuk menerapkan teknik **image augmentation** untuk menambah jumlah data.

![alt text](https://github.com/mausneg/Finger-Simulation/blob/main/module%20source/image-2.png?raw=true)

Gambar 3. Contoh Image Augmentation

Gambar di atas merupakan contoh image augmentation yang dapat Anda lakukan dengan mengombinasikan teknik-teknik pada modul 1, sehingga menghasilkan augmented image yang bersifat random. Anda dapat menggunakan teknik image augmentation yang lain yang Anda anggap perlu. Ingat gunakan jumlah hasil augmentasi sewajarnya saja.

### Preprocessing

Selanjutnya, Anda dapat melakukan teknik preprocessing lainnya yang Anda anggap perlu. Jelaskan alasan Anda menggunakan teknik tersebut, gunakan modul-modul yang telah Anda pelajari sebelumnya ataupun teknik preprocessing lain yang Anda ketahui.

### Feature Extraction

Pada tahapan ini, Anda diminta untuk melakukan ekstraksi fitur dengan metode Gray Level Co-occurrence Matrix (GLCM). Dengan GLCM sudut 0, 45, 90, dan 135 derajat, simetris, dan lakukan uji coba dengan distance 1-5. Anda dapat menghitung nilai dari beberapa fitur berikut:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- Entropy
- ASM

Tabel 1. Contoh Hasil Ekstraksi Fitur
| | contrast_0 | contrast_45 | contrast_90 | contrast_135 | dissimilarity_0 | dissimilarity_45 | dissimilarity_90 | dissimilarity_135 | homogeneity_0 | homogeneity_45 | ... | correlation_135 | asm_0 | entropy_0 | asm_45 | entropy_45 | asm_90 | entropy_90 | asm_135 | entropy_135 | label |
|-------|------------|-------------|-------------|--------------|-----------------|------------------|------------------|-------------------|---------------|----------------|-----|-----------------|-------|-----------|--------|------------|--------|------------|---------|-------------|-------|
| 0 | 1768.320952| 1930.804999 | 1137.489905 | 1365.584606 | 23.408952 | 26.309066 | 17.052571 | 19.286253 | 0.358650 | 0.291555 | ... | 0.604250 | 0.063717 | 7.732198 | 0.050296 | 8.081334 | 0.088581 | 7.687207 | 0.070051 | 7.888678 | 3.0 |
| 1 | 1812.106476| 962.718707 | 1484.389905 | 2024.235469 | 21.899238 | 13.042055 | 17.873333 | 24.230708 | 0.553610 | 0.616566 | ... | 0.373513 | 0.273481 | 5.519998 | 0.323768 | 5.276858 | 0.315946 | 5.237368 | 0.259917 | 5.575754 | 3.0 |

### Feature Selection

Pada tahap ini, Anda diminta untuk melakukan seleksi fitur. Anda dapat menggunakan teknik seleksi fitur seperti correlation, PCA, atau teknik seleksi fitur lain yang Anda ketahui. Jelaskan alasan Anda menggunakan teknik tersebut dan berikan alasan mengapa seleksi fitur diperlukan.

### Splitting Data

Pada tahap ini, Anda diminta untuk membagi data menjadi data training dan data testing. Anda dapat menggunakan perbandingan 80:20 atau 70:30, atau perbandingan lain yang Anda anggap perlu. Jelaskan alasan Anda menggunakan perbandingan tersebut.

### Normalization

Pada tahap ini, Anda diminta untuk melakukan normalisasi data. Anda dapat menggunakan teknik normalisasi standarization atau min-max normalization. Jelaskan alasan Anda menggunakan teknik tersebut dan berikan alasan mengapa normalisasi diperlukan.

## Modeling

Pada tahap ini, Anda diminta untuk membuat model klasifikasi. Berikut merupakan model yang Anda dapat gunakan:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Gunakan akurasi sebagai metrik yang digunakan untuk menampilkan hasil klasifikasi.

## Evaluation

Pada bagian ini Anda perlu mengevaluasi model klasifikasi yang telah Anda buat. Anda dapat menggunakan confusion matrix, accuracy, precision, atau metrik evaluasi lain yang Anda anggap perlu. Jelaskan hasil evaluasi yang Anda dapatkan dan berikan analisis mengenai hasil evaluasi tersebut.

Tabel 2. Contoh Hasil Evaluasi dalam Format Tabel

|               | Accuracy | Precision | Recall   | F1-Score |
| ------------- | -------- | --------- | -------- | -------- |
| KNN           | 0.948667 | 0.948664  | 0.948667 | 0.948504 |
| SVM           | 0.976333 | 0.976319  | 0.976333 | 0.976333 |
| Random Forest | 0.959667 | 0.959822  | 0.959667 | 0.959615 |

![alt text](https://github.com/mausneg/Finger-Simulation/blob/main/module%20source/image-3.png?raw=true)

Gambar 4. Contoh Hasil Evaluasi dengan Confusion Matrix

Ingat, Anda perlu melakukan eksperimen sebanyak 3-5 kali percobaan dengan notebook yang berbeda. Jika hasilnya belum sesuai dengan yang diharapkan, jelaskan alasan mengapa hal tersebut terjadi dan apa yang akan Anda lakukan selanjutnya. Lakukan perbandingan hasil eksperimen dengan eksperimen yang Anda lakukan sebelumnya. Jelaskan perbedaan hasil eksperimen tersebut dan apa yang menyebabkan perbedaan tersebut.

_**Jika Anda memiliki pertanyaan, jangan ragu untuk bertanya kepada asisten praktikum. Selamat mengerjakan!**_

_**Penanggungjawab Modul:**_

_**1. Maulana Surya Negara (F1D021110)**_

_**2. Amdila Rahmadi (F1D021078)**_

_**3. Muh Restu Aliza Akbar (F1D021113)**_
