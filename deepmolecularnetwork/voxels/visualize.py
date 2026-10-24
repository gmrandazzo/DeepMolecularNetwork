#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""Visualization of 3D Convolutional Neural Network layer activations in OpenDX format."""

import argparse
from pathlib import Path
import sys

import numpy as np

from deepmolecularnetwork.framework import Model
from deepmolecularnetwork.utility.modelhelpers import GetLoadModelFnc
from deepmolecularnetwork.voxels.opendx import readOpenDX, writeOpenDX


def write_grid_layer(
    activations: list[np.ndarray],
    origin: tuple[float, float, float] | list[float],
    delta: tuple[float, float, float] | list[float],
    outname: str,
    outdir: str | Path | None = None,
) -> list[str]:
    """Write each 3D activation channel across layers to an OpenDX file.

    Args:
        activations: List of layer activation numpy arrays.
        origin: 3D origin tuple (xmin, ymin, zmin).
        delta: 3D step size tuple (dx, dy, dz).
        outname: Base name for output files.
        outdir: Optional directory to store output .dx files.

    Returns:
        List of generated file paths.
    """
    written_files: list[str] = []
    target_dir = Path(outdir) if outdir is not None else Path(".")
    target_dir.mkdir(parents=True, exist_ok=True)

    for ly_idx, act in enumerate(activations):
        if act.ndim == 5:
            # Check for channels_first: (1, C, D0, D1, D2) vs channels_last: (1, D0, D1, D2, C)
            if act.shape[2] == act.shape[3] == act.shape[4] and act.shape[1] != act.shape[2]:
                num_channels = act.shape[1]
                for ch in range(num_channels):
                    t = act[0, ch, :, :, :]
                    gout = target_dir / f"ly{ly_idx}_ch{ch}_{outname}.dx"
                    writeOpenDX(t, origin, delta, str(gout))
                    written_files.append(str(gout))
            else:
                num_channels = act.shape[-1]
                for ch in range(num_channels):
                    t = act[0, :, :, :, ch]
                    gout = target_dir / f"ly{ly_idx}_ch{ch}_{outname}.dx"
                    writeOpenDX(t, origin, delta, str(gout))
                    written_files.append(str(gout))
        elif act.ndim == 4:
            if act.shape[1] == act.shape[2] == act.shape[3]:
                num_channels = act.shape[0]
                for ch in range(num_channels):
                    t = act[ch, :, :, :]
                    gout = target_dir / f"ly{ly_idx}_ch{ch}_{outname}.dx"
                    writeOpenDX(t, origin, delta, str(gout))
                    written_files.append(str(gout))
            else:
                num_channels = act.shape[-1]
                for ch in range(num_channels):
                    t = act[:, :, :, ch]
                    gout = target_dir / f"ly{ly_idx}_ch{ch}_{outname}.dx"
                    writeOpenDX(t, origin, delta, str(gout))
                    written_files.append(str(gout))

    return written_files


def visualize_3d_layers(
    model_path: str,
    dx_path: str,
    maxlayer: int | None = None,
    outdir: str | None = None,
) -> list[str]:
    """Extract and export internal layer activations of a 3D CNN model to OpenDX files.

    Args:
        model_path: Filepath of the trained Keras model.
        dx_path: Filepath to the input OpenDX (.dx) file.
        maxlayer: Upper layer index bound to extract activations from.
        outdir: Output directory to write activation .dx files.

    Returns:
        List of exported file paths.
    """
    load_fnc = GetLoadModelFnc()
    m = load_fnc(model_path)

    x_pred, origin, delta = readOpenDX(dx_path)
    s0, s1, s2 = x_pred.shape

    is_channels_first = False
    in_shape = getattr(m, "input_shape", None)
    if in_shape is not None and len(in_shape) == 5:
        if in_shape[1] == 1 or (in_shape[-1] != 1 and in_shape[1] <= 4):
            is_channels_first = True
    else:
        for ly in getattr(m, "layers", []):
            if getattr(ly, "data_format", None) == "channels_first":
                is_channels_first = True
                break

    if is_channels_first:
        x_input = x_pred.reshape(1, 1, s0, s1, s2)
    else:
        x_input = x_pred.reshape(1, s0, s1, s2, 1)

    layer_limit = maxlayer if maxlayer is not None and maxlayer > 0 else len(m.layers)
    target_layers = []
    for ly in m.layers[:layer_limit]:
        try:
            if len(ly.output.shape) in (4, 5):
                target_layers.append(ly)
        except Exception:
            target_layers.append(ly)

    if not target_layers:
        target_layers = m.layers[:layer_limit]

    layer_outputs = [layer.output for layer in target_layers]

    inputs = getattr(m, "inputs", None)
    if not inputs:
        if hasattr(m, "layers") and m.layers:
            try:
                inputs = getattr(m.layers[0], "input", None)
                if inputs is not None and not isinstance(inputs, list):
                    inputs = [inputs]
            except Exception:
                inputs = None
    if not inputs and hasattr(m, "input"):
        try:
            inputs = [m.input]
        except Exception:
            inputs = None

    activation_model = Model(inputs=inputs, outputs=layer_outputs)
    activations = activation_model.predict(x_input, verbose=0)

    if not isinstance(activations, list):
        activations = [activations]

    basename = Path(dx_path).resolve().stem
    return write_grid_layer(activations, origin, delta, basename, outdir=outdir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for 3D CNN layer activation visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize internal 3D CNN feature map activations " "exported to OpenDX (.dx) files."
        )
    )
    parser.add_argument("--model", default=None, type=str, help="Path to trained Keras model (.h5)")
    parser.add_argument(
        "--dx", default=None, type=str, help="Path to input OpenDX (.dx) file to analyze"
    )
    parser.add_argument(
        "--maxlayer",
        default=None,
        type=int,
        help="Maximum layer index to inspect (default: all layers)",
    )
    parser.add_argument(
        "--outdir", default=None, type=str, help="Output directory to save generated DX files"
    )

    args = parser.parse_args(argv)

    if args.model is None or args.dx is None:
        parser.print_help()
        print("\nERROR! Both --model and --dx arguments are required.")
        return 1

    visualize_3d_layers(
        model_path=args.model,
        dx_path=args.dx,
        maxlayer=args.maxlayer,
        outdir=args.outdir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
