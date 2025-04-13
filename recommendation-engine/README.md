# Sistem Rekomendasi Proyek Web3

Sistem rekomendasi proyek Web3 berbasis _feature-enhanced collaborative filtering_ yang menganalisis popularitas dan tren investasi untuk memberikan rekomendasi cryptocurrency yang dipersonalisasi.


## 📋 Deskripsi

Sistem ini menggunakan data dari CoinGecko API untuk menyediakan rekomendasi proyek Web3 (cryptocurrency, token, NFT, DeFi, dll) berdasarkan:

-   **Popularitas** (market cap, volume, metrik sosial)
-   **Tren Investasi** (perubahan harga, sentimen pasar)
-   **Interaksi Pengguna** (view, favorite, portfolio)
-   **Fitur Kategori** (DeFi, GameFi, Layer-1, dll)
-   **Blockchain** (Ethereum, Solana, Binance, dll)

Pendekatan _feature-enhanced collaborative filtering_ memungkinkan rekomendasi yang menggabungkan pola penggunaan kolaboratif dengan fitur spesifik proyek, menghasilkan rekomendasi yang lebih akurat dan kontekstual.

## 🚀 Fitur Utama

 1. Collaborative Filtering Berbasis:
	-   **User-based CF**: Rekomendasi berdasarkan kesamaan pengguna
	-   **Item-based CF**: Rekomendasi berdasarkan kesamaan proyek
	-   **Feature-enhanced CF**: CF diperkaya dengan fitur proyek
	-   **Neural Collaborative Filtering**: Deep learning untuk rekomendasi

2. Analisis Tren dan Popularitas:
	-   Integrasi skor tren dan popularitas untuk rekomendasi yang lebih relevan
	-   Analisis sentimen dan aktivitas sosial
	-   Deteksi peristiwa pasar seperti pump, dump, dan volatilitas tinggi

3. Penanganan Masalah Cold-Start:
	-  Strategi rekomendasi untuk pengguna baru
	-   Pendekatan berbasis fitur untuk proyek baru

4. REST API dan Aplikasi Web:
	-   Endpoint API untuk integrasi dengan aplikasi lain
	-   Sistem caching untuk meningkatkan performa
	-   Penanganan rate-limiting cerdas

5. Update Data Otomatis:
	-   Pengumpulan data berkala dari CoinGecko
	-   Pelacakan perubahan tren dan popularitas

## 🏗️ Arsitektur Sistem

Sistem terdiri dari tiga komponen utama:

 1. **Recommendation Engine** - Core sistem yang melakukan algoritma collaborative filtering
 2. **API Service** - Layanan REST API untuk akses ke rekomendasi
 3. **Data Pipeline** - Proses pengumpulan dan pemrosesan data

## 📁 Struktur Proyek

```
recommender-system-web3
└──recommendation-engine/
    │
    ├── config/
    │   └── config.py                # Konfigurasi API, database, dll
    │
    ├── data/
    │   ├── raw/                     # Data mentah dari API
    │   │   ├── web3_projects_*.csv  # Data proyek Web3 dari CoinGecko
    │   │   ├── coin_details_*.json  # Detail koin individual 
    │   │   ├── coins_markets_*.json # Data market dari CoinGecko
    │   │   └── trending_coins.json  # Data trending coins
    │   │
    │   └── processed/               # Data yang sudah diproses
    │       ├── processed_projects.csv            # Proyek yang sudah diproses
    │       ├── user_interactions.csv             # Interaksi pengguna sintetis
    │       ├── feature_matrix.csv                # Matriks fitur
    │       ├── user_item_matrix_*.csv            # Matriks user-item
    │       ├── item_similarity_*.csv             # Matriks similaritas item
    │       ├── user_similarity_*.csv             # Matriks similaritas user
    │       ├── feature_similarity_*.csv          # Matriks similaritas fitur
    │       ├── combined_similarity_*.csv         # Matriks similaritas gabungan
    │       ├── model_performance_*.json          # Metrik evaluasi model
    │       ├── recommendation_model_*.pkl        # Model rekomendasi tersimpan
    │       └── recommendations_*_*.json          # Hasil rekomendasi
    │
    ├── src/
    │   ├── collectors/              # Pengumpulan data
    │   ├── processors/              # Pemrosesan data
    │   ├── models/                  # Model rekomendasi
    │   ├── utils/                   # Utilitas
    │   └── evaluation/              # Metrik evaluasi
    │
    ├── scripts/                     # Script untuk tugas spesifik
    ├── logs/                        # File log
    │
    ├── central_logging.py           # Konfigurasi logging terpusat
    ├── app.py                       # Aplikasi API untuk recommendation engine
    ├── requirements.txt             # Dependensi package Python
    ├── README.md                    # Dokumentasi proyek
    └── main.py                      # Entry point utama
```

## 🛠️ Instalasi

### Prasyarat

-   Python 3.8+
-   pip (Package manager Python)
-   API key CoinGecko (untuk pengumpulan data)
-   PostgreSQL (opsional, untuk penyimpanan data)

### Langkah-langkah Instalasi

 1. Clone repository:
	```
	    git clone https://github.com/username/web3-recommendation-system.git
	    cd web3-recommendation-system
	```
2. Buat dan aktifkan virtual environment:
	```
		python -m venv venv
		source venv/bin/activate  		# Linux/Mac
		venv\Scripts\activate     		# Windows
		source venv/Scripts/activate 	# Git Bash
	```
3. Install dependensi:
	```
	    pip install -r requirements.txt
	```

4. Konfigurasi API key:
	-   Buat file `.env` di direktori root (atau edit `config/config.py`)
	-   Tambahkan API key CoinGecko:
		> COINGECKO_API_KEY="your-api-key"

5. Setup database (opsional):
	-   Buat database PostgreSQL
	-   Update konfigurasi database di `config/config.py`

## 🚀 Penggunaan

### Menggunakan Command Line Interface
Proyek ini menyediakan CLI yang komprehensif untuk semua fungsi utama:
```
# Menjalankan dalam mode interaktif
python main.py interactive

# Menjalankan seluruh pipeline dari awal hingga akhir
python main.py run

# Mengumpulkan data dari CoinGecko
python main.py collect --limit 500 --detail-limit 100

# Memproses data yang sudah dikumpulkan
python main.py process --users 500

# Membangun matriks untuk collaborative filtering
python main.py build

# Melatih model rekomendasi
python main.py train --include-all --save-model --eval-cold-start

# Menghasilkan rekomendasi
python main.py recommend --user_id user_1 --type hybrid --num 10 --save
atau
python main.py recommend --user_id user_2 --type user-based --num 10

# Menganalisis hasil rekomendasi
python main.py analyze --detailed --output analysis_report.md
```

### Menggunakan API
API server dapat dijalankan dengan:
```
# Development
python app.py

# Production (dengan Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
Contoh permintaan API:
```
# Mendapatkan rekomendasi untuk pengguna
curl -X GET "http://localhost:5000/api/recommendations?user_id=user_1&type=hybrid&limit=10"

# Mencatat interaksi pengguna
curl -X POST "http://localhost:5000/api/interactions" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "project_id": "bitcoin", "interaction_type": "view", "weight": 3}'

# Mendapatkan daftar proyek dengan filter
curl -X GET "http://localhost:5000/api/projects?category=defi&chain=ethereum&limit=20"
```

## 📈 Algoritma Rekomendasi

 1. Collaborative Filtering Berbasis Pengguna
 Merekomendasikan proyek berdasarkan kesamaan preferensi pengguna:
	 ```
	 # Pseudocode implementasi
	def user_based_cf(user_id, n=10):
	    user_ratings = user_item_matrix.loc[user_id]
	    unrated_projects = get_unrated_projects(user_ratings)
	    similar_users = find_similar_users(user_id, threshold=SIMILARITY_THRESHOLD)
	    
	    recommendations = {}
	    for project_id in unrated_projects:
	        weighted_sum = sum(similarity[sim_user] * rating[sim_user][project_id] 
	                         for sim_user in similar_users 
	                         if project_id in rated_projects[sim_user])
	        similarity_sum = sum(similarity[sim_user] for sim_user in similar_users)
	        recommendations[project_id] = weighted_sum / similarity_sum
	    
	    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
	```
	
2. Feature-Enhanced Collaborative Filtering
Menggabungkan CF tradisional dengan fitur proyek untuk rekomendasi yang lebih akurat:
	```
	# Pseudocode implementasi
	def feature_enhanced_cf(user_id, n=10):
	    # Langkah 1: Dapatkan matriks kesamaan berbasis fitur
	    feature_sim_matrix = calculate_feature_similarity(projects_df)
	    
	    # Langkah 2: Gabungkan dengan matriks kesamaan CF
	    enhanced_similarity = combine_similarities(
	        cf_similarity_matrix, 
	        feature_sim_matrix,
	        alpha=0.7  # Bobot untuk CF vs. feature similarity
	    )
	    
	    # Langkah 3: Buat rekomendasi menggunakan matriks gabungan
	    recommendations = predict_with_enhanced_similarity(
	        user_id, user_item_matrix, enhanced_similarity
	    )
	    
	    return recommendations[:n]
	```

3. Neural Collaborative Filtering (NCF)
Menggunakan deep learning untuk menangkap hubungan non-linear:
	```
	# Implementasi dengan PyTorch
	class NCF(nn.Module):
	    def __init__(self, num_users, num_items, embedding_size=64, layers=[128, 64, 32, 16]):
	        super(NCF, self).__init__()
	        
	        # GMF path
	        self.user_embedding_gmf = nn.Embedding(num_users, embedding_size)
	        self.item_embedding_gmf = nn.Embedding(num_items, embedding_size)
	        
	        # MLP path
	        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
	        self.item_embedding_mlp = nn.Embedding(num_items, embedding_size)
	        
	        # Bangun MLP layers
	        self.mlp_layers = nn.ModuleList()
	        input_size = 2 * embedding_size
	        
	        for layer_size in layers:
	            self.mlp_layers.append(nn.Linear(input_size, layer_size))
	            self.mlp_layers.append(nn.ReLU())
	            self.mlp_layers.append(nn.BatchNorm1d(layer_size))
	            self.mlp_layers.append(nn.Dropout(p=0.1))
	            input_size = layer_size
	        
	        # Output layer combines GMF and MLP
	        self.output_layer = nn.Linear(layers[-1] + embedding_size, 1)
	        self.sigmoid = nn.Sigmoid()
	```

## 📊 Metrik Evaluasi

Model dievaluasi menggunakan metrik standar untuk sistem rekomendasi:
-   **Precision@K**: Rasio proyek relevan dari K rekomendasi
-   **Recall@K**: Rasio proyek relevan yang berhasil direkomendasikan
-   **F1@K**: Rata-rata harmonis dari precision dan recall
-   **NDCG@K**: Normalized Discounted Cumulative Gain
-   **MRR (Mean Reciprocal Rank)**: Peringkat rata-rata dari rekomendasi yang relevan pertama
-   **Hit Ratio**: Rasio pengguna yang menerima setidaknya satu rekomendasi yang relevan

Metrik evaluasi disimpan di:
```
data/processed/model_performance_{timestamp}.json
```

## 🔍 Pemecahan Masalah Umum

1. Rate Limiting CoinGecko API
	```
	# Gunakan delay yang lebih panjang antara requests
	python scripts/collect_data.py --rate-limit 3
	```
2. Kesalahan JSON Parsing
Jika menemui kesalahan parsing data JSON:
	```
	# Coba parsing dengan penanganan kesalahan
	try:
	    data = json.loads(json_string)
	except json.JSONDecodeError:
	    # Coba clean up dan parse ulang
	    cleaned = json_string.replace('\"\"', '\"').replace('\\"', '"')
	    data = json.loads(cleaned)
	```
3. Masalah Memory untuk Dataset Besar
	```
	# Split processing ke dalam batch
	python scripts/process_data.py --batch-size 1000

	# Mengurangi jumlah koin yang dikumpulkan
	python scripts/collect_data.py --limit 500
	```
4. Error PostgreSQL Connection
Periksa konfigurasi database di `config/config.py`:
	```
	DB_HOST = "localhost"
	DB_PORT = 5432
	DB_NAME = "web3_recommender"
	DB_USER = "postgres"
	DB_PASSWORD = "password"
	```

## 📅 Jadwal Update

Sistem ini optimal ketika data diperbarui secara berkala:
-   **Harian**: Update data pasar dan trending
-   **Mingguan**: Pembaruan project details lengkap
-   **Bulanan**: Retraining model dan evaluasi

Skrip cron untuk update otomatis:
```
# Update harian (data pasar dan trending)
0 0 * * * cd /path/to/project && python scripts/update_coingecko_data.py --type market,trending

# Update mingguan (detail proyek)
0 0 * * 0 cd /path/to/project && python scripts/update_coingecko_data.py --type full

# Update bulanan (retraining model)
0 0 1 * * cd /path/to/project && python scripts/train_model.py --include-all --save-model
```

## 🔮 Pengembangan Mendatang

1. Ekspansi Data Source:
	-   Integrasi dengan CoinMarketCap API
	-   Pengambilan data sentimen dari Twitter/X
	-   Analisis berita kripto dari API berita
2. Peningkatan Algoritma:
	-   Implementasi Factorization Machines
	-   Hibridisasi dengan Knowledge Graph
	-   Analisis deret waktu advanced untuk prediksi tren
3. Fitur Platform:
	-   Dashboard admin untuk monitoring
	-   Visualisasi interaktif rekomendasi
	-   Fitur notifikasi untuk tren dan peluang
4. Integrasi:
	-   API webhooks untuk integrasi dengan platform lain
	-   Eksport data ke format yang umum digunakan
	-   Autentikasi OAuth untuk integrasi with wallet


## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan ikuti langkah-langkah berikut:
1.  Fork repository
2.  Buat branch fitur baru (`git checkout -b feature/amazing-feature`)
3.  Commit perubahan Anda (`git commit -m 'Add some amazing feature'`)
4.  Push ke branch (`git push origin feature/amazing-feature`)
5.  Buka Pull Request

Pastikan untuk menjalankan pengujian dan update dokumentasi jika diperlukan.

## 📜 Lisensi

Distributed under the MIT License.

## 📱 Kontak

Nama Proyek: Web3 Recommendation System
Email: [feifeifry@gmail.com](mailto:feifeifry@gmail.com)
Project Link: [https://github.com/feyfry/web3-recommendation-system](https://github.com/feyfry/web3-recommendation-system)

## 🙏 Pengakuan

-   [CoinGecko API](https://www.coingecko.com/en/api) untuk data cryptocurrency
-   [Scikit-learn](https://scikit-learn.org/) untuk algoritma machine learning
-   [PyTorch](https://pytorch.org/) untuk Neural Collaborative Filtering
-   [Flask](https://flask.palletsprojects.com/) untuk REST API
-   [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) untuk manipulasi data
