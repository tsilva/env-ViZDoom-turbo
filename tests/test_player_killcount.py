#!/usr/bin/env python3

from pathlib import Path

import vizdoom as vzd


def test_player_killcount_tracks_player_kills_but_not_world_kills() -> None:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "basic.cfg"))
    game.set_window_visible(False)
    game.add_game_args("+sv_cheats 1")
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
    game.add_available_game_variable(vzd.GameVariable.PLAYER_KILLCOUNT)
    game.init()

    try:
        game.new_episode()
        game.send_game_command("summon zombieman")
        game.advance_action(1)
        game.send_game_command("mdk")
        game.advance_action(1)

        assert game.get_game_variable(vzd.GameVariable.KILLCOUNT) == 1
        assert game.get_game_variable(vzd.GameVariable.PLAYER_KILLCOUNT) == 1

        game.new_episode()
        assert game.get_game_variable(vzd.GameVariable.KILLCOUNT) == 0
        assert game.get_game_variable(vzd.GameVariable.PLAYER_KILLCOUNT) == 0

        game.send_game_command("kill monsters")
        game.advance_action(1)

        assert game.get_game_variable(vzd.GameVariable.KILLCOUNT) == 1
        assert game.get_game_variable(vzd.GameVariable.PLAYER_KILLCOUNT) == 0
    finally:
        game.close()
