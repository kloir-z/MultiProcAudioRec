# Web Conference Audio Recorder

This script is primarily designed for recording web conferences. It uses multiprocessing to record system sounds and microphone inputs simultaneously. During recording system sounds, a sound at volume level 0 is played to prevent timestamp irregularities. After recording, the audio files are automatically adjusted for volume, and voiceless sections in the microphone input are silenced.

## Key Features
- Multiprocessing to record system and microphone sounds simultaneously
- Avoids timestamp irregularities by playing a sound at volume 0
- Automatic volume adjustments post-recording
- Voiceless sections in the microphone input are silenced

## How it Works
1. The main script sets up two recording processes, one for the microphone and one for the system sound output.
2. The system sound recording is started with a zero volume sound to avoid time irregularities.
3. The microphone and system sound recordings are processed independently, each one applying a series of filters on the audio data.
4. The processed audio data is saved as a .wav file for each recording process.
5. The two .wav files are mixed together and saved as an mp3 file.

## Code Overview
The main body of the script:
- Retrieves the default microphone and speaker from the system using the `soundcard` library.
- Defines a series of filters to be applied to the recorded audio data.
- Initiates the multiprocessing environment and starts the recording processes.
- Records the system sound and microphone audio simultaneously, using multiprocessing to prevent blocking.
- Saves the recorded data from both the microphone and system sound as .wav files.
- Mixes the audio files from both sources together and exports it as a .mp3 file.

## Dependencies
This script requires Python 3.6 or above, and the following Python packages:
- numpy
- soundcard
- soundfile
- pydub
- scipy
- pyaudio
- noisereduce

## Usage
To run this script, simply navigate to the directory containing the script in your terminal and execute it using the command `python main.py`. Make sure your system's default microphone and speakers are set to the devices from which you wish to record. 

This script will run until manually stopped. To stop recording, press Ctrl+C in the terminal.
