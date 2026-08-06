#pragma once

// Completion beep for example mains. Blocks so the process stays alive
// long enough for the tone to play (important when the console exits quickly).

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>

#include <cmath>
#include <cstdint>
#include <vector>
// MinGW / MSVC: link winmm (see CMakeLists.txt wtf_add_exe).
#else
#include <chrono>
#include <cstdio>
#include <thread>
#endif

namespace wtf_ex {

#ifdef _WIN32
/// Play a mono PCM sine tone via PlaySound(SND_MEMORY). freq_hz e.g. 800–4000.
/// Returns false if playback fails.
inline bool PlayToneHz(int freq_hz, int duration_ms, int volume = 12000)
{
    if (freq_hz < 50 || freq_hz > 12000 || duration_ms < 1)
        return false;

    constexpr int sample_rate = 44100;
    const int n_samples =
        static_cast<int>((static_cast<long long>(sample_rate) * duration_ms) / 1000);
    if (n_samples < 1)
        return false;

    // Minimal WAV: RIFF header + 16-bit mono PCM.
#pragma pack(push, 1)
    struct WavHdr
    {
        char riff[4] = {'R', 'I', 'F', 'F'};
        std::uint32_t file_size = 0; // filled below
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt_[4] = {'f', 'm', 't', ' '};
        std::uint32_t fmt_size = 16;
        std::uint16_t audio_format = 1; // PCM
        std::uint16_t num_channels = 1;
        std::uint32_t sample_rate_ = sample_rate;
        std::uint32_t byte_rate = sample_rate * 2;
        std::uint16_t block_align = 2;
        std::uint16_t bits_per_sample = 16;
        char data[4] = {'d', 'a', 't', 'a'};
        std::uint32_t data_size = 0;
    };
#pragma pack(pop)

    const std::uint32_t data_bytes =
        static_cast<std::uint32_t>(n_samples) * 2u;
    std::vector<std::uint8_t> buf(sizeof(WavHdr) + data_bytes);
    auto* hdr = reinterpret_cast<WavHdr*>(buf.data());
    *hdr = WavHdr{};
    hdr->data_size = data_bytes;
    hdr->file_size = 36u + data_bytes;

    auto* pcm = reinterpret_cast<std::int16_t*>(buf.data() + sizeof(WavHdr));
    constexpr double pi = 3.14159265358979323846;
    const double step =
        2.0 * pi * static_cast<double>(freq_hz) / static_cast<double>(sample_rate);
    // Short attack/release to avoid clicks.
    const int fade = std::min(n_samples / 8, sample_rate / 100); // ~10 ms cap
    for (int i = 0; i < n_samples; ++i)
    {
        double env = 1.0;
        if (fade > 0)
        {
            if (i < fade)
                env = static_cast<double>(i) / static_cast<double>(fade);
            else if (i > n_samples - fade)
                env = static_cast<double>(n_samples - 1 - i)
                      / static_cast<double>(fade);
        }
        const double s = std::sin(step * static_cast<double>(i)) * env;
        int v = static_cast<int>(s * static_cast<double>(volume));
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        pcm[i] = static_cast<std::int16_t>(v);
    }

    // PlaySound needs a pointer that remains valid for the duration of SND_SYNC.
    return ::PlaySoundW(reinterpret_cast<LPCWSTR>(buf.data()), nullptr,
                        SND_MEMORY | SND_SYNC | SND_NODEFAULT)
           != FALSE;
}
#endif

/// High-frequency completion chirp (three rising tones). Blocks until done.
inline void DoneBeep()
{
#ifdef _WIN32
    // Programmable PCM — not system aliases (those are fixed, often low pitch).
    // Typical "done" series around 1.5–2.5 kHz (clear, not piercing).
    static const int kFreqs[] = {1800, 2200, 2600};
    bool any = false;
    for (int f : kFreqs)
    {
        if (PlayToneHz(f, /*duration_ms=*/390, /*volume=*/32767))   //Practical max: 32767
        {
            any = true;
            ::Sleep(50);
        }
    }
    if (!any)
    {
        // Last resort if SND_MEMORY fails (rare).
        for (int i = 0; i < 3; ++i)
        {
            ::MessageBeep(MB_ICONASTERISK);
            ::Sleep(300);
        }
    }
    ::Sleep(1500);
#else
    for (int i = 0; i < 3; ++i)
    {
        std::fputc('\a', stderr);
        std::fflush(stderr);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
#endif
}

} // namespace wtf_ex
