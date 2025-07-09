# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 'datas' listesi, kod dışındaki varlıkları uygulamaya dahil eder.
# PyInstaller'a 'ffmpeg' ve 'music' klasörlerini kopyalamasını söylüyoruz.
# ('Kaynak klasör/dosya', 'Paket içindeki hedef klasör')
datas_to_include = [
    ('music', 'music'),                   # Müzik dosyalarını ekle
    ('ffmpeg/windows', 'ffmpeg/windows'), # Windows için FFmpeg
    ('ffmpeg/macos', 'ffmpeg/macos'),     # macOS için FFmpeg (platformlar arası uyumluluk için)
]

a = Analysis(
    ['main.py'],  # ANA PYTHON DOSYANIZIN ADI
    pathex=['.'],          # Proje kök dizini
    binaries=[],
    datas=datas_to_include,
    hiddenimports=[
        # PyInstaller'ın gözden kaçırabileceği önemli modülleri ekliyoruz.
        'pydub.utils',
        'mutagen',
        'platform'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- Windows için EXE OLUŞTURMA BÖLÜMÜ ---
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ProEqualizer', # Oluşturulacak .exe dosyasının adı
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # GUI uygulaması olduğu için konsol penceresi olmasın.
    # icon='icon.ico' # İSTEĞE BAĞLI: .ico uzantılı ikon dosyanız varsa bu satırı aktif edin.
)