import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

def wav2melSpec(audio):
    sr = 16000
    return librosa.feature.melspectrogram(y=audio, sr=sr)

def imgSpec(recorded_audio,output):
    spec = wav2melSpec(recorded_audio)
    plt.figure(figsize=(24, 12))
    plt.subplot(3,2,1)
    plt.title('Бичигдсэн аудионы спектрограм')
    ms_dB = librosa.power_to_db(spec, ref=np.max)
    img = librosa.display.specshow(ms_dB, x_axis='time', y_axis='mel')
    plt.colorbar(img, format='%+2.0f dB')

    plt.subplot(3,2,2)
    plt.title('Бичигдсэн дохионы хэлбэр')
    plt.plot(recorded_audio)
    #librosa.display.waveshow(ms_feature, sr=16000)

    spec = wav2melSpec(output)
    plt.subplot(3,2,3)
    plt.title('Таамагласан аудионы спектрограм')
    ms_dB = librosa.power_to_db(spec, ref=np.max)
    img = librosa.display.specshow(ms_dB, x_axis='time', y_axis='mel')
    plt.colorbar(img, format='%+2.0f dB')

    plt.subplot(3,2,4)
    plt.title('Таамагласан дохионы хэлбэр')
    plt.plot(output)
    #librosa.display.waveshow(ms_feature, sr=16000)


def hear_audio(recorded_audio,output):
    sr = 16000
    print("Бичигдсэн аудио")
    ipd.display(ipd.Audio(data=recorded_audio, rate=sr))
    print("Таамагласан аудио")
    ipd.display(ipd.Audio(data=output, rate=sr))
    
def get_audio_info(recorded_audio,output, show_melspec=True, label=None):
    if label is not None:
        print("Label:", label)
    if show_melspec is not False:
        imgSpec(recorded_audio,output)
    hear_audio(recorded_audio,output)