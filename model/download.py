import os
import subprocess

# ID folder Google Drive yang bisa diakses publik
folder_id = '1rzlWb0y0Cw9Syy-wOHkjHjCe7WEUO--Y'

# Folder tujuan penyimpanan
download_folder = './model/'
os.makedirs(download_folder, exist_ok=True)

# Perintah gdown untuk download folder
cmd = [
    'gdown',
    f'--folder',
    f'https://drive.google.com/drive/folders/{folder_id}',
    '-O', download_folder
]

# Jalankan perintah
try:
    print("⏬ Mulai mengunduh model dari Google Drive...")
    subprocess.run(cmd, check=True)
    print("✅ Semua file berhasil diunduh ke folder:", download_folder)
except subprocess.CalledProcessError as e:
    print("❌ Terjadi kesalahan saat mengunduh:", e)
