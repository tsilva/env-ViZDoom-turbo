from __future__ import annotations

import numpy as np
import pytest
from env_vizdoom_turbo._env_vizdoom_turbo import ActionHistory, ImageProcessor, preprocess_into


def test_area_resize_grayscale_and_maxpool_are_batched() -> None:
    current = np.asarray(
        [
            [
                [[0, 0, 0], [40, 80, 120]],
                [[80, 40, 0], [255, 255, 255]],
            ],
            [
                [[10, 20, 30], [10, 20, 30]],
                [[10, 20, 30], [10, 20, 30]],
            ],
        ],
        dtype=np.uint8,
    )
    previous = np.zeros_like(current)
    previous[1] = 200
    output = np.empty((2, 1, 1, 1), dtype=np.uint8)

    preprocess_into(current, output, [0, 0, 0, 0], False, 0, "area", previous)

    pooled_lane_zero = current[0].mean(axis=(0, 1))
    expected_zero = int(
        round(
            (
                pooled_lane_zero[0] * 77
                + pooled_lane_zero[1] * 150
                + pooled_lane_zero[2] * 29
            )
            / 256
        )
    )
    assert output[:, 0, 0, 0].tolist() == [expected_zero, 200]


def test_crop_mask_preserves_geometry_and_uses_fill() -> None:
    current = np.full((1, 4, 4, 3), 100, dtype=np.uint8)
    output = np.empty_like(current)

    preprocess_into(current, output, [1, 1, 1, 1], True, 7, "nearest")

    assert np.all(output[:, 0] == 7)
    assert np.all(output[:, -1] == 7)
    assert np.all(output[:, :, 0] == 7)
    assert np.all(output[:, :, -1] == 7)
    assert np.all(output[:, 1:3, 1:3] == 100)


@pytest.mark.parametrize("mask_crop", [False, True])
def test_indexed_crop_matches_rgb_reference_on_all_four_edges(mask_crop: bool) -> None:
    rng = np.random.default_rng(508)
    crop = [7, 32, 11, 13]
    fill = 19
    indexed = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    palette = rng.integers(0, 256, size=(256, 3), dtype=np.uint8)
    rgb = palette[indexed][None]
    expected = np.empty((1, 84, 84, 1), dtype=np.uint8)
    preprocess_into(rgb, expected, crop, mask_crop, fill, "area")

    processor = ImageProcessor(
        1,
        240,
        320,
        84,
        84,
        1,
        crop,
        mask_crop,
        fill,
        "area",
        4,
        "chw",
        1,
    )
    stack = np.zeros((4, 84, 84, 1), dtype=np.uint8)
    head = np.zeros(1, dtype=np.int64)
    output = np.empty((4, 84, 84), dtype=np.uint8)
    processor.reset_indexed_lane_into(indexed, palette, stack, head, output)

    np.testing.assert_array_equal(output, np.repeat(expected[0].transpose(2, 0, 1), 4, axis=0))


def _write_reference_observation(
    stack: np.ndarray,
    heads: np.ndarray,
    *,
    layout: str,
) -> np.ndarray:
    num_envs, frame_stack, height, width, channels = stack.shape
    shape = (
        (num_envs, channels * frame_stack, height, width)
        if layout == "chw"
        else (num_envs, height, width, channels * frame_stack)
    )
    output = np.empty(shape, dtype=np.uint8)
    for lane in range(num_envs):
        head = int(heads[lane])
        for output_slot in range(frame_stack):
            source_slot = (head + 1 + output_slot) % frame_stack
            frame = stack[lane, source_slot]
            channel_start = output_slot * channels
            channel_end = channel_start + channels
            if layout == "chw":
                output[lane, channel_start:channel_end] = frame.transpose(2, 0, 1)
            else:
                output[lane, :, :, channel_start:channel_end] = frame
    return output


@pytest.mark.parametrize("algorithm", ["nearest", "bilinear", "area"])
@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("layout", ["hwc", "chw"])
@pytest.mark.parametrize("mask_crop", [False, True])
def test_persistent_processor_matches_legacy_pipeline(
    algorithm: str,
    grayscale: bool,
    layout: str,
    mask_crop: bool,
) -> None:
    rng = np.random.default_rng(8917)
    num_envs, raw_height, raw_width = 3, 7, 9
    out_height, out_width = 4, 5
    channels, frame_stack = (1 if grayscale else 3), 3
    crop = [1, 1, 2, 1]
    processor = ImageProcessor(
        num_envs,
        raw_height,
        raw_width,
        out_height,
        out_width,
        channels,
        crop,
        mask_crop,
        17,
        algorithm,
        frame_stack,
        layout,
        2,
    )
    current = rng.integers(
        0,
        256,
        size=(num_envs, raw_height, raw_width, 3),
        dtype=np.uint8,
    )
    stack = np.zeros(
        (num_envs, frame_stack, out_height, out_width, channels),
        dtype=np.uint8,
    )
    heads = np.zeros(num_envs, dtype=np.int64)
    output_shape = (
        (num_envs, channels * frame_stack, out_height, out_width)
        if layout == "chw"
        else (num_envs, out_height, out_width, channels * frame_stack)
    )
    output = np.empty(output_shape, dtype=np.uint8)

    expected_frame = np.empty(
        (num_envs, out_height, out_width, channels),
        dtype=np.uint8,
    )
    preprocess_into(
        current,
        expected_frame,
        crop,
        mask_crop,
        17,
        algorithm,
    )
    expected_stack = np.repeat(expected_frame[:, None], frame_stack, axis=1)
    processor.reset_into(
        current,
        stack,
        heads,
        output,
        np.ones(num_envs, dtype=np.bool_),
    )
    np.testing.assert_array_equal(stack, expected_stack)
    np.testing.assert_array_equal(
        output,
        _write_reference_observation(expected_stack, heads, layout=layout),
    )

    for _ in range(frame_stack + 1):
        previous = current
        current = rng.integers(
            0,
            256,
            size=(num_envs, raw_height, raw_width, 3),
            dtype=np.uint8,
        )
        preprocess_into(
            current,
            expected_frame,
            crop,
            mask_crop,
            17,
            algorithm,
            previous,
        )
        expected_heads = (heads + 1) % frame_stack
        for lane in range(num_envs):
            expected_stack[lane, expected_heads[lane]] = expected_frame[lane]
        processor.step_into(current, stack, heads, output, previous)
        np.testing.assert_array_equal(heads, expected_heads)
        np.testing.assert_array_equal(stack, expected_stack)
        np.testing.assert_array_equal(
            output,
            _write_reference_observation(expected_stack, heads, layout=layout),
        )


def test_native_action_history_supports_independent_lane_resets() -> None:
    history = ActionHistory(3, 2)
    first = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    second = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    history.append(first)
    history.append(second)
    history.clear(np.asarray([False, True, False], dtype=np.bool_))
    replacement = np.asarray([[0.25, 0.75]], dtype=np.float64)
    history.replace_lane(2, replacement)

    np.testing.assert_array_equal(history.lane(0), [first[0], second[0]])
    assert history.lane(1) == []
    np.testing.assert_array_equal(history.lane(2), replacement)
