/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors
 Copyright (C) 2023 - 2026 by Marek Wydmuch, Farama Foundation, and the respective contributors

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

#include "ViZDoomGamePython.h"
#include "ViZDoomController.h"
#include "ViZDoomExceptions.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace vizdoom {
    DoomGamePython::DoomGamePython() {
        this->grayShape.resize(2);
        this->audioShape.resize(2);
        this->variablesShape.resize(1);
        this->turboTics = 0;
        this->turboResetInFlight = false;
        this->turboResetSeed = 0;
    }

    void DoomGamePython::setAction(pyb::object const &pyAction) {
        auto action = DoomGamePython::pyObjectToVector<double>(pyAction);
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::setAction(action);
    }

    double DoomGamePython::makeAction(pyb::object const &pyAction, unsigned int tics) {
        auto action = DoomGamePython::pyObjectToVector<double>(pyAction);
        ReleaseGIL gil = ReleaseGIL();
        return DoomGame::makeAction(action, tics);
    }

    GameStatePython* DoomGamePython::getState() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        if (this->state == nullptr) return nullptr;

        // TODO: the following line causes:
        // Fatal Python error: PyEval_SaveThread: NULL tstate
        //ReleaseGIL gil = ReleaseGIL();
        this->pyState = new GameStatePython();

        this->pyState->number = this->state->number;
        this->pyState->tic = this->state->tic;

        /* Update buffers */
        this->updateBuffersShapes();

        if (this->state->screenBuffer != nullptr)
            this->pyState->screenBuffer = this->dataToNumpyArray(this->colorShape, this->state->screenBuffer->data());
        else this->pyState->screenBuffer = pyb::none();

        if (this->state->audioBuffer != nullptr)
            this->pyState->audioBuffer = this->dataToNumpyArray(this->audioShape, this->state->audioBuffer->data());
        else this->pyState->audioBuffer = pyb::none();

        if (this->state->depthBuffer != nullptr)
            this->pyState->depthBuffer = this->dataToNumpyArray(this->grayShape, this->state->depthBuffer->data());
        else this->pyState->depthBuffer = pyb::none();

        if (this->state->labelsBuffer != nullptr) {
            this->pyState->labelsBuffer = this->dataToNumpyArray(this->grayShape, this->state->labelsBuffer->data());

            /* Update labels */
            this->pyState->labels = DoomGamePython::vectorToPyList<Label>(this->state->labels);
        }  else {
            this->pyState->labelsBuffer = pyb::none();
            this->pyState->labels = pyb::none();
        }

        if (this->state->automapBuffer != nullptr)
            this->pyState->automapBuffer = this->dataToNumpyArray(this->colorShape, this->state->automapBuffer->data());
        else this->pyState->automapBuffer = pyb::none();

        /* Update notifications buffer */
        if (this->doomController->isNotificationsEnabled())
            this->pyState->notificationsBuffer = pyb::str(this->state->notificationsBuffer);
        else this->pyState->notificationsBuffer = pyb::none();

        /* Updates vars */
        if (!this->state->gameVariables.empty()) {
            // Numpy array version
            this->variablesShape[0] = this->state->gameVariables.size();
            this->pyState->gameVariables = dataToNumpyArray(this->variablesShape, this->state->gameVariables.data());

            // Python list version
            //this->pyState->gameVariables = DoomGamePython::vectorToPyList<double>(this->state->gameVariables);
        }
        else this->pyState->gameVariables = pyb::none();

        /* Update objects */
        if (this->isObjectsInfoEnabled()) {
            this->pyState->objects = DoomGamePython::vectorToPyList<Object>(this->state->objects);
        } else this->pyState->objects = pyb::none();

        /* Update sectors */
        if (this->isSectorsInfoEnabled()) {
            pyb::list pySectors;
            for (auto& sector : this->state->sectors){
                SectorPython pySector;
                pySector.id = sector.id;
                pySector.floorHeight = sector.floorHeight;
                pySector.ceilingHeight = sector.ceilingHeight;
                pySector.lines = DoomGamePython::vectorToPyList<Line>(sector.lines);
                pySectors.append(pySector);
            }
            this->pyState->sectors = pySectors;
            //this->pyState->sectors = DoomGamePython::vectorToPyList<Sectors>(this->state->objects);
        } else this->pyState->sectors = pyb::none();

        return this->pyState;
    }

    ServerStatePython* DoomGamePython::getServerState() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        ServerStatePython* pyServerState = new ServerStatePython();

        pyServerState->tic = this->serverState->tic;
        pyServerState->playerCount = this->serverState->playerCount;

        pyb::list pyPlayersInGame, pyPlayersNames, pyPlayersFrags,
                pyPlayersAfk, pyPlayersLastActionTic, pyPlayersLastKillTic;
        for(int i = 0; i < MAX_PLAYERS; ++i) {
            pyPlayersInGame.append(this->serverState->playersInGame[i]);
            pyPlayersNames.append(pyb::str(this->serverState->playersNames[i].c_str()));
            pyPlayersFrags.append(this->serverState->playersFrags[i]);
            pyPlayersAfk.append(this->serverState->playersAfk[i]);
            pyPlayersLastActionTic.append(this->serverState->playersLastActionTic[i]);
            pyPlayersLastKillTic.append(this->serverState->playersLastKillTic[i]);
        }

        pyServerState->playersInGame = pyPlayersInGame;
        pyServerState->playersNames = pyPlayersNames;
        pyServerState->playersFrags = pyPlayersFrags;
        pyServerState->playersAfk = pyPlayersAfk;
        pyServerState->playersLastActionTic = pyPlayersLastActionTic;
        pyServerState->playersLastKillTic = pyPlayersLastKillTic;

        return pyServerState;
    }

    pyb::list DoomGamePython::getLastAction() {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        return DoomGamePython::vectorToPyList(this->lastAction);
    }

    pyb::list DoomGamePython::getAvailableButtons(){
        return DoomGamePython::vectorToPyList(this->availableButtons);
    }

    void DoomGamePython::setAvailableButtons(pyb::list const &pyButtons){
        DoomGame::setAvailableButtons(DoomGamePython::pyListToVector<Button>(pyButtons));
    }

    pyb::list DoomGamePython::getAvailableGameVariables(){
        return DoomGamePython::vectorToPyList(this->availableGameVariables);
    }

    void DoomGamePython::setAvailableGameVariables(pyb::list const &pyGameVariables){
        DoomGame::setAvailableGameVariables(DoomGamePython::pyListToVector<GameVariable>(pyGameVariables));
    }

    bool DoomGamePython::setConfig(pyb::object const &config) {
        // If config is a string, pass it directly to the C++ setConfig
        if (pyb::isinstance<pyb::str>(config)) {
            std::string configStr = config.cast<std::string>();
            return DoomGame::setConfig(configStr);
        }

        // If config is a dict, convert it to a config string
        if (pyb::isinstance<pyb::dict>(config)) {
            pyb::dict configDict = config.cast<pyb::dict>();
            std::ostringstream configStream;

            for (auto item : configDict) {
                std::string key = pyb::str(item.first).cast<std::string>();
                pyb::object value = pyb::reinterpret_borrow<pyb::object>(item.second);

                // Convert key to lowercase and replace spaces with underscores
                std::transform(key.begin(), key.end(), key.begin(), ::tolower);
                std::replace(key.begin(), key.end(), ' ', '_');

                configStream << key << " = ";

                // Handle different value types
                if (pyb::isinstance<pyb::list>(value)) {
                    // List values (e.g., available_buttons, available_game_variables)
                    pyb::list listValue = value.cast<pyb::list>();
                    configStream << "{\n";
                    for (auto listItem : listValue) {
                        // Check if it's an enum value
                        if (pyb::hasattr(listItem, "name")) {
                            // It's an enum, get its name
                            std::string enumName = pyb::str(pyb::getattr(listItem, "name")).cast<std::string>();
                            configStream << "    " << enumName << "\n";
                        } else {
                            // It's a regular value, convert to string
                            configStream << "    " << pyb::str(listItem).cast<std::string>() << "\n";
                        }
                    }
                    configStream << "}\n";
                } else if (pyb::isinstance<pyb::bool_>(value)) {
                    // Boolean values
                    bool boolValue = value.cast<bool>();
                    configStream << (boolValue ? "true" : "false") << "\n";
                } else if (pyb::isinstance<pyb::int_>(value)) {
                    // Integer values
                    configStream << value.cast<int>() << "\n";
                } else if (pyb::isinstance<pyb::float_>(value)) {
                    // Float values
                    configStream << value.cast<double>() << "\n";
                } else if (pyb::isinstance<pyb::str>(value)) {
                    // String values
                    configStream << value.cast<std::string>() << "\n";
                } else if (pyb::hasattr(value, "name")) {
                    // Enum value, get its name
                    std::string enumName = pyb::str(pyb::getattr(value, "name")).cast<std::string>();
                    configStream << enumName << "\n";
                } else {
                    // Fallback: convert to string
                    configStream << pyb::str(value).cast<std::string>() << "\n";
                }
            }

            return DoomGame::setConfig(configStream.str());
        }

        // If it's neither a string nor a dict, throw an error
        throw std::invalid_argument("config must be either a string or a dict");
    }

    // These functions are wrapped for manual GIL management
    void DoomGamePython::init(){
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::init();
    }

    void DoomGamePython::newEpisode(std::string filePath) {
        ReleaseGIL gil = ReleaseGIL();  // this prevents the deadlock during the start of multiplayer game, if different Doom instances are started from different Python threads
        DoomGame::newEpisode(filePath);
    }

    void DoomGamePython::advanceAction(unsigned int tics, bool updateState){
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::advanceAction(tics, updateState);
    }

    void DoomGamePython::respawnPlayer(){
        ReleaseGIL gil = ReleaseGIL();
        DoomGame::respawnPlayer();
    }

    void DoomGamePython::turboStepStart(
        const double *action,
        size_t actionSize,
        unsigned int tics) {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        if (actionSize != this->availableButtons.size())
            throw std::invalid_argument("action width does not match available buttons");

        for (size_t index = 0; index < actionSize; ++index) {
			this->nextAction[index] = action[index];
			this->doomController->setButtonState(
				this->availableButtons[index],
				this->nextAction[index]);
        }
        this->turboTotalBefore = this->summaryReward;
        this->turboStepAdvanced = this->doomController->isTicPossible();
        if (this->turboStepAdvanced) {
            if (this->turboTics != tics) {
                this->turboTics = tics;
                this->turboTicsCount = std::to_string(tics);
            }
			this->doomController->startTicsBatched(
				tics, true, &this->turboTicsCount, true);
        }
    }

    void DoomGamePython::turboStepFinish(
        uint8_t *frame,
        size_t frameSize,
        uint8_t *palette,
        size_t paletteSize,
        float &reward,
        bool &terminated,
        bool &truncated,
        double *gameVariables,
        size_t gameVariablesSize,
        bool treatTimeoutAsTruncation) {
        if (frame != nullptr && frameSize != this->doomController->getScreenSize())
            throw std::invalid_argument("frame size does not match the Doom screen");
        if (palette != nullptr && paletteSize != 256 * 3)
            throw std::invalid_argument("palette must contain 256 RGB entries");
        if ((frame == nullptr) != (palette == nullptr))
            throw std::invalid_argument("frame and palette outputs must both be present or absent");
        if (gameVariablesSize != this->availableGameVariables.size())
            throw std::invalid_argument("game variable width does not match available variables");

        if (this->turboStepAdvanced) {
            this->doomController->finishTicsBatched();
            if (this->doomController->isAllowDoomInput() ||
                this->doomController->isReplaying()) {
                for (size_t index = 0; index < this->availableButtons.size(); ++index) {
                    this->lastAction[index] =
                        this->doomController->getButtonState(this->availableButtons[index]);
                }
            }
            else {
				this->lastAction = this->nextAction;
            }
            this->updateReward();
            if (this->doomController->isRunDoomAsync())
                this->lastMapTic = this->doomController->getMapTic();
            else
                this->lastMapTic = this->doomController->getMapLastTic();
        }

        const bool finished = this->isEpisodeFinished();
        const bool timeout = finished && this->isEpisodeTimeoutReached();
        truncated = finished && timeout && treatTimeoutAsTruncation;
        terminated = finished;
        if (!finished && frame != nullptr) {
            uint8_t *screen = this->doomController->getScreenBuffer();
            if (screen == nullptr) throw std::runtime_error("Doom screen buffer is unavailable");
            std::memcpy(frame, screen, frameSize);
            std::memcpy(palette, this->doomController->getScreenPalette(), paletteSize);
        }
        reward = static_cast<float>(this->summaryReward - this->turboTotalBefore);

        for (size_t index = 0; index < gameVariablesSize; ++index) {
            gameVariables[index] =
                this->doomController->getGameVariable(this->availableGameVariables[index]);
        }
    }

    void DoomGamePython::turboStepInto(
        const double *action,
        size_t actionSize,
        unsigned int tics,
        uint8_t *frame,
        size_t frameSize,
        uint8_t *palette,
        size_t paletteSize,
        float &reward,
        bool &terminated,
        bool &truncated,
        double *gameVariables,
        size_t gameVariablesSize,
        bool treatTimeoutAsTruncation) {
        this->turboStepStart(action, actionSize, tics);
        this->turboStepFinish(
            frame,
            frameSize,
            palette,
            paletteSize,
            reward,
            terminated,
            truncated,
            gameVariables,
            gameVariablesSize,
            treatTimeoutAsTruncation);
    }

    void DoomGamePython::turboReset(
        unsigned int seed,
        double *gameVariables,
        size_t gameVariablesSize) {
        if (gameVariablesSize != this->availableGameVariables.size())
            throw std::invalid_argument("game variable width does not match available variables");
        if (this->turboResetInFlight) {
            this->doomController->finishRestartMapBatched();
            const bool requestedResetCompleted = this->turboResetSeed == seed;
            this->turboResetInFlight = false;
            if (!requestedResetCompleted) {
                DoomGame::setSeed(seed);
                this->doomController->restartMapBatched();
            }
        }
        else {
            DoomGame::setSeed(seed);
            this->doomController->restartMapBatched();
        }
        this->resetState();
        for (size_t index = 0; index < gameVariablesSize; ++index) {
            gameVariables[index] =
                this->doomController->getGameVariable(this->availableGameVariables[index]);
        }
    }

    void DoomGamePython::turboResetStart(unsigned int seed) {
        if (this->turboResetInFlight) {
            if (this->turboResetSeed != seed)
                throw std::logic_error("a different Turbo reset is already in flight");
            return;
        }
        DoomGame::setSeed(seed);
        this->doomController->startRestartMapBatched();
        this->turboResetSeed = seed;
        this->turboResetInFlight = true;
    }

    void DoomGamePython::turboReadIndexedInto(
        uint8_t *frame,
        size_t frameSize,
        uint8_t *palette,
        size_t paletteSize) {
        if (!this->isRunning()) throw ViZDoomIsNotRunningException();
        if (frameSize != this->doomController->getScreenSize())
            throw std::invalid_argument("frame size does not match the Doom screen");
        if (paletteSize != 256 * 3)
            throw std::invalid_argument("palette must contain 256 RGB entries");
        std::memcpy(frame, this->doomController->getScreenBuffer(), frameSize);
        std::memcpy(palette, this->doomController->getScreenPalette(), paletteSize);
    }

    void DoomGamePython::updateBuffersShapes(){
        int channels = this->getScreenChannels();
        int width = this->getScreenWidth();
        int height = this->getScreenHeight();

        switch(this->getScreenFormat()){
            case CRCGCB:
            case CBCGCR:
                this->colorShape.resize(3);
                this->colorShape[0] = channels;
                this->colorShape[1] = height;
                this->colorShape[2] = width;
                break;

            case GRAY8:
            case DOOM_256_COLORS8:
                this->colorShape.resize(2);
                this->colorShape[0] = height;
                this->colorShape[1] = width;
                break;

            default:
                this->colorShape.resize(3);
                this->colorShape[0] = height;
                this->colorShape[1] = width;
                this->colorShape[2] = channels;
        }

        this->grayShape[0] = height;
        this->grayShape[1] = width;

        this->audioShape[0] = this->getAudioSamplesPerTic() * this->getAudioBufferSize();
        this->audioShape[1] = 2;
    }


    template<class T> pyb::list DoomGamePython::vectorToPyList(const std::vector<T>& vector){
        pyb::list pyList;
        for (auto& i : vector) pyList.append(i);
        return pyList;
    }

    template<class T> std::vector<T> DoomGamePython::pyListToVector(pyb::list const &pyList){
        size_t pyLen = pyb::len(pyList);
        std::vector<T> vector = std::vector<T>(pyLen);
        for (size_t i = 0; i < pyLen; ++i) vector[i] = pyb::cast<T>(pyList[i]);
        return vector;
    }

    template<class T> std::vector<T> DoomGamePython::pyArrayToVector(pyb::array_t<T> const &pyArray){
        if (pyArray.ndim() != 1)
            throw std::runtime_error("The number of dimensions larger than 1, the array should be 1D ndarray");

        size_t pyLen = pyArray.shape(0);
        std::vector<T> vector = std::vector<T>(pyLen);
        for (size_t i = 0; i < pyLen; ++i) vector[i] = pyArray.at(i);
        return vector;
    }

    template<typename T> std::vector<T> DoomGamePython::pyObjectToVector(pyb::object const &pyObject) {
        if(pyb::isinstance<pyb::list>(pyObject) || pyb::isinstance<pyb::tuple>(pyObject))
            return pyListToVector<T>(pyObject);
        else if(pyb::isinstance<pyb::array>(pyObject))
            return pyArrayToVector<T>(pyObject);
        else throw std::runtime_error("Unsupported type, should be list, tuple, or 1D ndarray of numeric or boolean values");
    }

    template<class T> pyb::array_t<T> DoomGamePython::dataToNumpyArray(std::vector<pyb::ssize_t> dims, T *data){
        return pyb::array(dims, data);
    }
}
