from multiprocessing import Process, Queue, Event
import datetime
import time
import numpy as np
import soundcard as sc
import soundfile as sf
import pydub
import audio_processing as ap
from play_zero_sound import play_zero_sound
from functools import partial

def rec_process(device, rec_queue, start_event, rec_filename_header, sampling_rate, filters):
    start_event.wait()

    raw_audio_data, start_time_ns = record_audio(device, sampling_rate)
    # save_audio(raw_audio_data, "pre-"+str(rec_filename_header), sampling_rate)
    processed_data = raw_audio_data
    for filter_func in filters:
        processed_data = filter_func(audio_data=processed_data)
    rec_wavfile = save_audio(processed_data, rec_filename_header, sampling_rate)
    rec_queue.put((rec_wavfile, start_time_ns))

def record_audio(device, sampling_rate):
    with device.recorder(samplerate=sampling_rate) as sound_input:
        device_name = device.name
        data = []
        print(f"Rec Started on device: {device_name}")
        print("Press Ctrl+c to Stop...")

        data.append(sound_input.record(numframes=sampling_rate))
        start_time_ns = time.perf_counter_ns()

        try:
            while True:
                data.append(sound_input.record(numframes=sampling_rate))
        except KeyboardInterrupt:
            pass

    return np.concatenate(data, axis=0)[:, 0], start_time_ns

def save_audio(audio_data, output_filename_header, sampling_rate):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{output_filename_header}_{timestamp}.wav"

    sf.write(file=output_file_name, data=audio_data, samplerate=sampling_rate)
    return output_file_name

def mix_audio_files(audio_files):
    mixed_audio = pydub.AudioSegment.from_file(audio_files[0], format="wav")
    for audio_file in audio_files[1:]:
        mixed_audio = mixed_audio.overlay(pydub.AudioSegment.from_file(audio_file, format="wav"))
    return mixed_audio

if __name__ == '__main__':
    sampling_rate = 24000

    mic_device = sc.get_microphone(id=str(sc.default_microphone().name))
    out_device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

    mic_rec_filename_header = "mic_rec"
    out_rec_filename_header = "out_rec"

    mic_filters = [
        partial(ap.reduce_gain_of_loud_sections, sampling_rate=sampling_rate, threshold_dB=-19, attack_release_ms=50, overcomp_ratio=0.35),
        partial(ap.apply_auto_gain, target_peak_db=-1),
        partial(ap.mask_voiceless_sections, sampling_rate=sampling_rate, low_freq=180, high_freq=550, order=20, threshold_rms=0.030, window_size=0.05, pre_duration=0.2, post_duration=0.6),
        partial(ap.apply_auto_gain, target_peak_db=-7),
    ]

    out_filters = [
        partial(ap.apply_auto_gain, target_peak_db=-8),
    ]

    ready_event = Event()
    play_zero_sound_process = Process(target=play_zero_sound, args=(sampling_rate, ready_event))
    play_zero_sound_process.start()

    ready_event.wait()

    start_event = Event()

    out_rec_queue = Queue()
    out_rec_process = Process(target=rec_process, args=(out_device, out_rec_queue, start_event, out_rec_filename_header, sampling_rate, out_filters))
    out_rec_process.start()

    mic_rec_queue = Queue()
    mic_rec_process = Process(target=rec_process, args=(mic_device, mic_rec_queue, start_event, mic_rec_filename_header, sampling_rate, mic_filters))
    mic_rec_process.start()

    start_event.set()
    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopped.")
        print("Post Processing...")
        pass
    finally:
        play_zero_sound_process.join()
        mic_rec_process.join()
        out_rec_process.join()
        print("Done!")
        print("Mix and Save mp3 file...")
        mic_rec_wavfile, mic_rec_start_time_ns = mic_rec_queue.get()
        out_rec_wavfile, out_rec_start_time_ns = out_rec_queue.get()
        time_diff_ms = (out_rec_start_time_ns - mic_rec_start_time_ns) / 1e6

        if time_diff_ms > 0:
            mic_rec_audio = pydub.AudioSegment.from_wav(mic_rec_wavfile)
            mic_rec_audio = mic_rec_audio[time_diff_ms:]
            mic_rec_audio.export(mic_rec_wavfile, format="wav")
        elif time_diff_ms < 0:
            out_rec_audio = pydub.AudioSegment.from_wav(out_rec_wavfile)
            out_rec_audio = out_rec_audio[-time_diff_ms:]
            out_rec_audio.export(out_rec_wavfile, format="wav")

    mixed_audio = mix_audio_files([mic_rec_wavfile, out_rec_wavfile])
    trimmed_audio = ap.trim_silence(mixed_audio,
                                  silence_thresh=-40,
                                  min_silence_len=2000,
                                  padding_before=250,
                                  padding_after=250)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    result_filename  = f"rec_result_{timestamp}.mp3"
    trimmed_audio.export(result_filename, format="mp3")
    print(f"Saved to {result_filename}.")
