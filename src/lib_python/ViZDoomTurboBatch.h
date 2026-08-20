/*
 Copyright (C) 2026 by the env-ViZDoom-turbo contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZDOOM_TURBO_BATCH_H__
#define __VIZDOOM_TURBO_BATCH_H__

#include "ViZDoomGamePython.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vizdoom {

    class TurboBatchStepper {
    public:
        TurboBatchStepper(
            pyb::list games,
            unsigned int frameSkip,
            bool treatTimeoutAsTruncation,
            pyb::array_t<double, pyb::array::c_style> actions,
            pyb::array_t<uint8_t, pyb::array::c_style> frames,
            pyb::array_t<uint8_t, pyb::array::c_style> palettes,
            pyb::array_t<float, pyb::array::c_style> rewards,
            pyb::array_t<bool, pyb::array::c_style> terminated,
            pyb::array_t<bool, pyb::array::c_style> truncated,
            pyb::array_t<double, pyb::array::c_style> gameVariables);

        void stepLaneInto(size_t lane);
        void resetLaneInto(size_t lane, unsigned int seed);
        void readLaneInto(size_t lane);
        pyb::array_t<uint8_t> indexedFrameView(size_t lane);
        pyb::array_t<uint8_t> paletteView(size_t lane);
        pyb::tuple nativeApi();

    private:
        void stepLaneNative(size_t lane);
        void startLaneNative(size_t lane);
        void finishLaneNative(size_t lane);
        void startResetLaneNative(size_t lane, unsigned int seed);
        void resetLaneNative(size_t lane, unsigned int seed);
        void clearNativeError() noexcept;
        void recordNativeError(const char *phase, size_t lane, const char *message) noexcept;
        static uint64_t nativeStepLane(void *context, size_t lane) noexcept;
        static uint64_t nativeStartLane(void *context, size_t lane) noexcept;
        static uint64_t nativeStartAll(void *context) noexcept;
        static uint64_t nativeFinishLane(void *context, size_t lane) noexcept;
        static unsigned int nativeStartResetLane(
            void *context,
            size_t lane,
            unsigned int seed) noexcept;
        static unsigned int nativeResetLane(
            void *context,
            size_t lane,
            unsigned int seed) noexcept;
        static const uint8_t *nativeFrame(void *context, size_t lane) noexcept;
        static const uint8_t *nativePalette(void *context, size_t lane) noexcept;
        static const uint64_t *nativeBackgroundData(void *context, size_t lane) noexcept;
        static void nativeClearError(void *context) noexcept;
        static size_t nativeCopyError(void *context, char *destination, size_t capacity) noexcept;

        pyb::list gameOwners;
        std::vector<DoomGamePython *> games;
        unsigned int frameSkip;
        bool treatTimeoutAsTruncation;
        size_t actionWidth;
        size_t frameSize;
        size_t gameVariablesWidth;
        std::vector<uint32_t> screenUpdateSequences;
        std::mutex nativeErrorMutex;
        std::string nativeError;

        pyb::array_t<double, pyb::array::c_style> actions;
        pyb::array_t<uint8_t, pyb::array::c_style> frames;
        pyb::array_t<uint8_t, pyb::array::c_style> palettes;
        pyb::array_t<float, pyb::array::c_style> rewards;
        pyb::array_t<bool, pyb::array::c_style> terminated;
        pyb::array_t<bool, pyb::array::c_style> truncated;
        pyb::array_t<double, pyb::array::c_style> gameVariables;

        double *actionsData;
        uint8_t *framesData;
        uint8_t *palettesData;
        float *rewardsData;
        bool *terminatedData;
        bool *truncatedData;
        double *gameVariablesData;
    };
}

#endif
