# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- EN ÖNEMLİ BÖLÜM BURASI ---
# 'datas' listesi, Python kodunuz dışındaki dosyaları ve klasörleri
# uygulamanıza eklemek için kullanılır.
# Format: ('kaynak_dosya_veya_klasör', 'uygulamanın_içinde_gideceği_yer')
datas_to_include = [
    ('music', 'music'),   # 'music' klasörünü al ve uygulamanın içine 'music' olarak koy.
    ('ffmpeg', 'ffmpeg')    # 'ffmpeg' klasörünü al ve uygulamanın içine 'ffmpeg' olarak koy.
]
# -----------------------------

a = Analysis(
    ['main.py'],
    pathex=['.'], # Proje ana dizini
    binaries=[],
    datas=datas_to_include, # Yukarıda tanımladığımız listeyi burada kullanıyoruz.
    hiddenimports=[
        # PyInstaller'ın bazen gözden kaçırdığı gizli bağımlılıklar.
        # Bunları eklemek olası hataları önler.
        'pydub.utils',
        'scipy.signal._sosfilt',
        'scipy._lib.messagestream'
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ProEqualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # GUI uygulaması için 'False' olmalı.
)

# --- macOS için APP BUNDLE OLUŞTURMA ---
# Bu bölüm, .exe yerine .app uzantılı bir uygulama paketi oluşturur.
app = BUNDLE(
    exe,
    name='ProEqualizer.app',
    icon=None, # İsterseniz bir .icns dosya yolu ekleyebilirsiniz.
    bundle_identifier=None,
    info_plist={
        'NSHighResolutionCapable': 'True'
    }
)