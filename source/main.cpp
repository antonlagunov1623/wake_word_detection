#include <iostream>
#include <cmath>
#include "AudioFile.h"
#include <vector>
#include "utils.h"

using namespace std;

const int MAX_MEM_BLOCK = pow(2, 8) * pow(2, 10);

int min(int a, int b) {
    if (a < b) {
        return a;
    }
    else {
        return b;
    }
}

vector<vector<double>> frame(const vector<double>& x, int frame_length, int hop_length) {
    int ax1 = frame_length;
    int ax2 = (int)((x.size() - frame_length) / hop_length) + 1;
    vector<vector<double>> frames(ax1, vector<double>(ax2));
    for (size_t i = 0; i < ax1; ++i) {
        for (size_t j = 0; j < ax2; ++j) {
            frames.at(i).at(j) = x.at(i + j*hop_length);
        }
    }
    return frames;
}

double hzToMel(double frequencies) {
    double f_min = 0.0;
    double f_sp = 200.0 / 3;
    double mels = (frequencies - f_min) / f_sp;
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = log(6.4) / 27.0;
    if (frequencies >= min_log_hz) {
        mels = min_log_mel + (log(frequencies / min_log_hz) / logstep);
    }
    return mels;
}

vector<double> melToHz(const vector<double>& mels) {
    double f_min = 0.0;
    double f_sp = 200.0 / 3;
    vector<double> freqs = oneDimScalarMultiplication(mels, f_sp);
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = log(6.4) / 27.0;
    for (size_t i = 0; i < mels.size(); ++i) {
        if (mels.at(i) >= min_log_mel) {
            freqs.at(i) = min_log_hz * exp(logstep * (mels.at(i) - min_log_mel));
        }
    }
    return freqs;
}

vector<double> fftFrequencies(int sr, int n_fft) {
    return linspace(0, sr / 2, (int)(n_fft / 2) + 1);
}

vector<double> melFrequencies(int n_mels, double fmin, double fmax) {
    double min_mel = hzToMel(fmin);
    double max_mel = hzToMel(fmax);
    vector<double> mels = linspace(min_mel, max_mel, n_mels);
    return melToHz(mels);
}

vector<double> getWindow(int nx) {
    vector<double> fac = linspace(-M_PI, M_PI, nx + 1);
    vector<double> window = oneDimZeros(nx + 1);
    for (size_t i = 0; i < 2; ++i) {
        vector<double> res = cosArray(fac, (double)i);
        res = oneDimScalarMultiplication(res, 0.5);
        window = oneDimAddition(window, res);
    }
    window.pop_back();
    return window;
}

vector<vector<double>> stft(const vector<double> y_samples, int n_fft, int hop_length, int win_length) {
    if (n_fft == -1) {
        n_fft = 2048;
    }
    if (win_length == -1) {
        win_length = n_fft;
    }
    if (hop_length == -1) {
        hop_length = win_length / 4;
    }
    vector<double> window = getWindow(win_length);
    vector<double> padded_window = constantOneDimPad(window, n_fft);
    vector<vector<double>> fft_window = expandDim(padded_window, -1, 1);
    vector<double> y = reflectOneDimPad(y_samples, int(n_fft / 2));
    vector<vector<double>> y_frames = frame(y, n_fft, hop_length);
    vector<vector<double>> stft_matrix = twoDimZeros((int)(n_fft / 2) + 1, y_frames[0].size());
    int n_columns = (int)(MAX_MEM_BLOCK / (stft_matrix.size() * __SIZEOF_DOUBLE__));
    for (size_t bl_s = 0; bl_s < stft_matrix.at(0).size(); bl_s+=n_columns) {
        int bl_t = min(bl_s + n_columns, stft_matrix.at(0).size());
        for (size_t i = bl_s; i < bl_t; ++i) {
            vector<double> multi;
            for (size_t j = 0; j < y_frames.size(); ++j) {
                multi.push_back(y_frames.at(j).at(i) * fft_window.at(j).at(0));
            }
            vector<double> rfft_multi = rfft(multi, multi.size());
            for (size_t j = 0; j < stft_matrix.size(); ++j) {
                stft_matrix.at(j).at(i) = pow(abs(rfft_multi.at(j)), 2);
            }
        }
    }
    return stft_matrix;
}

vector<vector<double>> mel(int sr, int n_fft) {
    double fmin = 0.0;
    double fmax = sr / 2;
    int n_mels = 128;
    vector<vector<double>> weights = twoDimZeros(n_mels, (int)(n_fft / 2) + 1);
    vector<double> fftfreqs = fftFrequencies(sr, n_fft);
    vector<double> mel_f = melFrequencies(n_mels + 2, fmin, fmax);
    vector<double> fdiff = diff(mel_f);
    vector<vector<double>> ramps = subtractOuter(mel_f, fftfreqs);
    for (size_t i = 0; i < n_mels; ++i) {
        vector<double> lower;
        for (size_t j = 0; j < ramps.at(0).size(); ++j) {
            lower.push_back(-ramps.at(i).at(j) / fdiff.at(i));
        }
        vector<double> upper;
        for (size_t j = 0; j < ramps.at(0).size(); ++j) {
            upper.push_back(ramps.at(i + 2).at(j) / fdiff.at(i + 1));
        }
        vector<double> zero_v = oneDimZeros(lower.size());
        vector<double> min_v = compareOneDimMinimum(lower, upper);
        vector<double> max_v = compareOneDimMaximum(min_v, zero_v);
        for (size_t j = 0; j < max_v.size(); ++j) {
            weights.at(i).at(j) = max_v.at(j);
        }
    }
    vector<double> denorm;
    for (size_t i = 0; i < n_mels; ++i) {
        denorm.push_back(2.0 / (mel_f.at(i + 2) - mel_f.at(i)));
    }
    vector<vector<double>> enorm = expandDim(denorm, -1, 1);
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights.at(0).size(); ++j) {
            weights.at(i).at(j) *= enorm.at(i).at(0);
        }
    }
    return weights;
}

vector<vector<double>> melspectrogram(vector<double>& y, int sr, int n_fft, int hop_length, int win_length) {
    vector<vector<double>> spectrogram = stft(y, n_fft, hop_length, win_length);
    vector<vector<double>> mels = mel(sr, n_fft);
    return multiplication(mels, spectrogram);
}

int main() {
    AudioFile<double> audio_file;
    audio_file.load("/home/anton/Documents/test/1/1904.wav");

    int sample_rate = audio_file.getSampleRate();
    cout << "---------------------------------------------------------" << endl;
    cout << "Sample rate = " << sample_rate << endl;
    cout << "---------------------------------------------------------" << endl;

    int num_samples = audio_file.getNumSamplesPerChannel();
    vector<double> signal;
    for (int i = 0; i < num_samples; i++) {
	    signal.push_back(audio_file.samples[0][i]);
    }
    format1Cout(signal, "signal");

    int n_fft = 1024;
    int hop_length = 128;
    int win_length = n_fft;

    vector<double> window = getWindow(win_length);
    format1Cout(window, "window");

    vector<double> padded_window = constantOneDimPad(window, n_fft);
    format1Cout(padded_window, "padded window");

    vector<vector<double>> fft_window = expandDim(padded_window, -1, 1);
    format2Cout(fft_window, "fft window");

    vector<double> y = reflectOneDimPad(signal, int(n_fft / 2));
    format1Cout(y, "reflected signal");

    vector<vector<double>> y_frames = frame(y, n_fft, hop_length);
    format2Cout(y_frames, "y frames");

    vector<vector<double>> spectrogram = stft(signal, n_fft, hop_length, win_length);
    format2Cout(spectrogram, "spectrogram");

    int n_mels = 128;
    double fmin = 0.0;
    double fmax = sample_rate / 2;

    vector<double> fftfreqs = fftFrequencies(sample_rate, n_fft);
    format1Cout(fftfreqs, "fft frequencies");

    vector<double> mel_f = melFrequencies(n_mels + 2, fmin, fmax);
    format1Cout(mel_f, "mel frequencies");

    vector<double> fdiff = diff(mel_f);
    format1Cout(fdiff, "fdiff");

    vector<vector<double>> ramps = subtractOuter(mel_f, fftfreqs);
    format2Cout(ramps, "ramps");

    vector<vector<double>> mels = mel(sample_rate, n_fft);
    format2Cout(mels, "mels");

    vector<vector<double>> melspectrum = melspectrogram(signal, sample_rate, n_fft, hop_length, win_length);
    format2Cout(melspectrum, "melspectrogram");

    return 0;
}