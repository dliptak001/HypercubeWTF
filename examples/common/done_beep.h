#pragma once

// Completion beep for example mains. Blocks so the process stays alive
// long enough for the tone to play (important when the console exits quickly).

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <chrono>
#include <cstdio>
#include <thread>
#endif

namespace wtf_ex {

/// Play a short completion beep; returns only after the tone has finished.
inline void DoneBeep()
{
#ifdef _WIN32
    // Synchronous: blocks for duration_ms (freq Hz).
    ::Beep(1500, 400);
    // Extra hold so the console / runner does not tear down mid-tail.
    ::Sleep(1500);
#else
    std::fputc('\a', stderr);
    std::fflush(stderr);
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
#endif
}

} // namespace wtf_ex
