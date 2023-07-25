import numpy as np
from scipy.signal import butter, lfilter, sosfilt
import noisereduce as nr
from pydub import silence

def reduce_gain_of_loud_sections(audio_data, sampling_rate, threshold_dB, attack_release_ms, overcomp_ratio):
    threshold = (10 ** (threshold_dB / 20))
    window_size = int(attack_release_ms * sampling_rate / 1000)

    loud_sections = np.abs(audio_data) > threshold

    def apply_window(start, end, gain_reduction):
        length = end - start
        if length > 0:
            if length <= window_size:
                audio_data[start:end] *= np.linspace(1, gain_reduction, length)
            else:
                audio_data[start:start + window_size] = audio_data[start:start + window_size] * np.linspace(1, gain_reduction, window_size)
                audio_data[start + window_size:end - window_size] = audio_data[start + window_size:end - window_size] * gain_reduction
                audio_data[end - window_size:end] = audio_data[end - window_size:end] * np.linspace(gain_reduction, 1, window_size)

    start = None
    for i in range(len(audio_data) - window_size * 2):
        if loud_sections[i:i + window_size].any() or (start is not None and loud_sections[i - window_size:i].any()):
            if start is None:
                start = i
        else:
            if start is not None:
                max_amplitude = np.max(np.abs(audio_data[start:start + window_size * 2]))
                gain_reduction = threshold / max_amplitude
                gain_reduction *= overcomp_ratio
                apply_window(start, i + window_size, gain_reduction)
                start = None

    if start is not None:
        max_amplitude = np.max(np.abs(audio_data[start:]))
        gain_reduction = threshold / max_amplitude
        gain_reduction *= overcomp_ratio
        apply_window(start, len(audio_data), gain_reduction)

    return audio_data

def apply_limiter(audio_data, limit_threshold_db, lookahead_samples=8):
    limit_threshold_amplitude = db_to_amplitude(limit_threshold_db)
    limited_data = np.copy(audio_data)
    num_samples = len(audio_data)

    for i in range(num_samples):
        lookahead_start = i
        lookahead_end = min(i + lookahead_samples, num_samples)

        max_amplitude_in_window = np.max(np.abs(audio_data[lookahead_start:lookahead_end]))
        if max_amplitude_in_window > limit_threshold_amplitude:
            gain_reduction = limit_threshold_amplitude / max_amplitude_in_window
            limited_data[i] = audio_data[i] * gain_reduction
        else:
            limited_data[i] = audio_data[i]

    return limited_data

def mask_voiceless_sections(audio_data, sampling_rate, low_freq, high_freq, order, threshold_rms, window_size, pre_duration, post_duration):
    sos = butter(order, [low_freq, high_freq], btype='band', fs=sampling_rate, output='sos')
    filtered_audio_data = sosfilt(sos, audio_data)

    window_samples = int(window_size * sampling_rate)
    step_size = window_samples // 2
    mask = []

    for i in range(0, len(filtered_audio_data), step_size):
        window_start = max(0, i - window_samples // 2)
        window_end = min(len(filtered_audio_data), i + window_samples // 2)
        window_rms = np.sqrt(np.mean(filtered_audio_data[window_start:window_end] ** 2))

        if window_rms >= threshold_rms:
            mask.extend([1] * step_size)
        else:
            mask.extend([0] * step_size)

    mask = np.array(mask[:len(filtered_audio_data)])

    pre_samples = int(pre_duration * sampling_rate)
    post_samples = int(post_duration * sampling_rate)
    extended_mask = np.copy(mask)

    for i in range(len(mask)):
        if mask[i] == 1:
            extended_mask[max(0, i - pre_samples):i] = 1
            extended_mask[i:min(len(mask), i + post_samples)] = 1

    masked_audio_data = audio_data * extended_mask

    return masked_audio_data

def apply_normalize(audio, target_level_db):
    current_level_db = amplitude_to_db(np.max(np.abs(audio)))
    gain = db_to_amplitude(target_level_db - current_level_db)
    normalized_audio = audio * gain
    return normalized_audio

def apply_auto_gain(audio_data, target_peak_db):
    peak_value = np.max(np.abs(audio_data))
    if peak_value == 0:
        return audio_data
    peak_value_db = amplitude_to_db(peak_value)
    gain_adjustment_db = target_peak_db - peak_value_db
    gain_adjustment = db_to_amplitude(gain_adjustment_db)
    adjusted_audio_data = audio_data * gain_adjustment
    return adjusted_audio_data

def apply_noise_reduction(audio_data, sampling_rate, time_constant_s, stationary):
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sampling_rate, time_constant_s=time_constant_s, stationary=stationary)
    return reduced_noise

def apply_eq(audio_data, sampling_rate, low_freq, high_freq, order):
    b, a = butter(order, [low_freq / (sampling_rate / 2), high_freq / (sampling_rate / 2)], btype="band")
    filtered_audio = lfilter(b, a, audio_data)
    return filtered_audio

def db_to_amplitude(db):
    return 10 ** (db / 20)

def amplitude_to_db(amplitude):
    if amplitude == 0:
        return 0.0
    else:
        return 20 * np.log10(np.abs(amplitude))

def trim_silence(audio, silence_thresh, min_silence_len, padding_before, padding_after):
    trimmed_audio = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    non_silent_audio = [audio[max(0, segment[0] - padding_before):min(len(audio), segment[1] + padding_after)] for segment in trimmed_audio]
    result_audio = sum(non_silent_audio)
    return result_audio