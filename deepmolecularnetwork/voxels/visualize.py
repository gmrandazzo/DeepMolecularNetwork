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
from typing import Any

import numpy as np

from deepmolecularnetwork.framework import Model, keras
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
        num_channels = act.shape[-1]
        for ch in range(num_channels):
            if act.ndim == 4:
                t = act[:, :, :, ch]
            elif act.ndim == 5:
                t = act[0, :, :, :, ch]
            else:
                continue

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
    x_input = x_pred.reshape(1, x_pred.shape[0], x_pred.shape[1], x_pred.shape[2], 1)

    layer_limit = maxlayer if maxlayer is not None and maxlayer > 0 else len(m.layers)
    target_layers = m.layers[:layer_limit]
    layer_outputs = [layer.output for layer in target_layers]

    inputs = getattr(m, "inputs", [m.input])
    activation_model = Model(inputs=inputs, outputs=layer_outputs)
    activations = activation_model.predict(x_input)

    if not isinstance(activations, list):
        activations = [activations]

    basename = Path(dx_path).resolve().stem
    return write_grid_layer(activations, origin, delta, basename, outdir=outdir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for 3D CNN layer activation visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize internal 3D CNN feature map activations exported to OpenDX (.dx) files."
    )
    parser.add_argument("--model", default=None, type=str, help="Path to trained Keras model (.h5)")
    parser.add_argument("--dx", default=None, type=str, help="Path to input OpenDX (.dx) file to analyze")
    parser.add_argument(
        "--maxlayer",
        default=None,
        type=int,
        help="Maximum layer index to inspect (default: all layers)",
    )
    parser.add_argument("--outdir", default=None, type=str, help="Output directory to save generated DX files")

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
