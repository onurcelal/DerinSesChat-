import argparse
import os
from pathlib import Path
import random
import getpass
import librosa
from nltk import text
import numpy as np
import soundfile as sf
import torch
import sohpet 
import json
import pickle
import threading
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
from IPython.display import Audio
from itertools import count

veri = json.loads(open('metin.json').read())
sınıflar = pickle.load(open('sınıflar.pkl','rb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)
    
   
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Yapılandırmanızın bir testi çalıştırılıyor......\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("CPU çıkarıma hazırlanıyor:\n")
    
   
    print("Kodlayıcı, sentezleyici ve ses kodlayıcı hazırlanıyor...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

        
    

    ## Teste başlıyor
    print("Yapılandırmanız birincil girdi ile test ediliyor.")
   
   
    print("\Kodlayıcı test ediyor...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    print("\Sentezci test ediyor... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    
   
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None
    print("\Ses kodlayıcı test ediyor...")
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    num_generated = ""    
    girdi_yolu = Path("/content/drive/MyDrive/DerinSesChat/steve.wav")
    ## Hafızaya İşleniyor
    preprocessed_wav = encoder.preprocess_wav(girdi_yolu)
    original_wav, sampling_rate = librosa.load(str(girdi_yolu))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("HAFIZAYA BAŞARIYLA ALINDI")

  
    embed = encoder.embed_utterance(preprocessed_wav)
    print("KLONLANIYOR")
    for bşlk in veri["intents"]:         
      yanıt = bşlk["response"]  
  
    text = random.choice(yanıt)
    
    
    ## Spectogram oluşturuluyor
    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)
  
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Spectrogram melodisi oluşturuluyor")


    ## Dalga formu oluşturuluyor
    print("Dalga formu sentezleniyor:")

    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)
        
    generated_wav = vocoder.infer_waveform(spec)
    ## Sonraki jenerasyon
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    dosya_adı = "demo_çıktı.wav" 
    sf.write(dosya_adı, generated_wav.astype(np.float32), synthesizer.sample_rate)

  
    
        
      

    

