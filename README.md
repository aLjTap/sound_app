# Pro Equalizer - Python Müzik Çalar ve Equalizer

Modern ve kullanıcı dostu bir müzik çalar ve real-time equalizer uygulaması.

## Özellikler

- 🎵 MP3 dosya oynatma
- 🎛️ 10 bantlı parametrik equalizer
- 📊 Real-time spektrum analiz görselleştirmesi
- 🎨 Modern dark tema arayüz
- 📁 Müzik kütüphanesi yönetimi
- ⏯️ Tam oynatma kontrolleri (play/pause/seek)

## Gereksinimler

- Python 3.12
- macOS (FFmpeg dahil)

## Kurulum

### 1. Projeyi klonlayın

```bash
git clone <repository-url>
cd sound_App_last
```

### 2. Virtual environment oluşturun

```bash
python3.12 -m venv venv
```

### 3. Virtual environment'ı aktifleştirin

```bash
source venv/bin/activate
# veya
./activate_env.sh
```

### 4. Gereksinimleri yükleyin

```bash
pip install -r requirements.txt
```

## Kullanım

### Hızlı Başlangıç

```bash
./run_app.sh
```

### Manuel Çalıştırma

```bash
source venv/bin/activate
python main.py
```

## Kütüphaneler

- **librosa**: Gelişmiş ses işleme (scipy yerine)
- **customtkinter**: Modern GUI
- **pyaudio**: Real-time ses oynatma
- **pydub**: Ses formatı dönüştürme
- **numpy**: Numerik hesaplamalar
- **mutagen**: Metadata okuma

## Equalizer Bantları

1. **Sub-Bass** (60 Hz) - Derin bas tonları
2. **Betonung** (170 Hz) - Vurgu ve güç
3. **Wärme** (310 Hz) - Sıcaklık ve dolgunluk
4. **Klarheit** (600 Hz) - Netlik ve anlaşılırlık
5. **Vokal** (1 kHz) - Vokal frekansları
6. **Lebendigkeit** (3 kHz) - Canlılık
7. **Helligkeit** (6 kHz) - Parlaklık
8. **Glanz** (12 kHz) - Işıltı
9. **Luft** (14 kHz) - Hava
10. **Ultra-Höhen** (16 kHz) - Ultra yüksek tonlar

## Geliştirme

### Kod Yapısı

- `main.py`: Ana uygulama dosyası
- `requirements.txt`: Python bağımlılıkları
- `ffmpeg/`: Platform-specific FFmpeg binaries
- `music/`: Örnek müzik dosyaları

### Virtual Environment Yönetimi

```bash
# Aktifleştirme
source venv/bin/activate

# Deaktifleştirme
deactivate

# Paket listesi
pip list

# Yeni paket ekleme
pip install <package-name>
pip freeze > requirements.txt
```

## Sorun Giderme

### Ses problemi

- PyAudio kurulumunu kontrol edin
- Ses kartı ayarlarını kontrol edin

### FFmpeg hatası

- FFmpeg binary'lerin doğru konumda olduğunu kontrol edin
- Platform-specific binary'lerin executable olduğunu kontrol edin

### Import hatası

- Virtual environment'ın aktif olduğunu kontrol edin
- requirements.txt'deki tüm paketlerin yüklü olduğunu kontrol edin

## Lisans

Bu proje kişisel kullanım için geliştirilmiştir.
