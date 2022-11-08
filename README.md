# DerinSesChat-
1.İlk adımda github kaynaklı dosyayı indirerek veya --git clone https://github.com/onurcelal/DerinSesChat yöntemiyle indirilmelidir. 

2.İkinci adımda eğitim dosyasını çalıştırıp .pkl ve .h5 uzantılı dosyalar üretilmesi gerekmektedir.

3.Dosyalar elde edildikten sonrası gerekli kütüphaneler --pip ile yüklenmesi gerekmektedir. Kütüphaneler gereklilikler.txt dosyasında sunulmuştur.

4.Gereklilikler yüklendikten sonrası python sürümünüz ile uyumlu pyaudio dosyasını yüklenmelidir. Bu dosya "https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio" resmi sitesinde mevcuttur. Bu dosya .whl formatında indirilip --pip install "dosya_adı.whl" yöntemiyle yüklenmelidir. Ayrıca --pipwin yöntemi ile de bu dosya indirilebilinir.

5.Tüm kurulum tamamlandıktan sonra dosyanın çalıştırıldığı klasöre en az 3 saniyelik bir ses dosyasını .wav formatında main.py dosyasında yer alan "dosya_adı" isimli kısma tanımlanmalıdır. 

6.Gerekli ses dosyası .wav formatında tanımladıktan sonra main.py dosyasını çalıştırınız.

7.Yazılım çalışmaya başladığında "Siz:" ile başlayan bir komut yazım alanı görülecek. Buraya dilediğiniz bir kelime veya yazı yazarak yazılımın buna yanıt vermesi için gerekli süreci başlatmış olacaksınız. Aksi takdirde algoritma çalışmayacaktır.
