#!/usr/bin/env python3
# (c) 2018-2026 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use, modify, and distribute it under
# the terms of the GNU General Public License, version 3.
# See the file LICENSE for details
"""SMILES representation to 2D one-hot encoding matrix converter."""

import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SMILE_CHARS = [
    "C",  # aliphatic carbon
    "c",  # aromatic carbon
    "O",  # aliphatic oxygen
    "o",  # aromatic oxygen
    "N",  # aliphatic nitrogen
    "n",  # aromatic nitrogen
    "S",  # aliphatic sulfur
    "s",  # aromatic sulfur
    "P",  # aliphatic phosphor
    "p",  # aromatic phosphor
    "F",
    "Cl",
    "Br",
    "I",
    "H",
    "Si",
    "Se",
    "se",
    "As",
    "B",
    "b",
    "Na",
    "Fe",
    "Mg",
    "=",
    "#",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "(",
    ")",
    "[",
    "]",
    "@",
    "@@",
    "/",
    "\\",
    "-",
    "+",
    "%",
]


class SMILES2MX:
    """Converts a SMILES string into a 2D one-hot encoded matrix."""

    def __init__(self, row_padding: int = 256, smile_chars: list[str] | None = None) -> None:
        """Initialize converter.

        Args:
            row_padding: Maximum length of SMILES tokens to pad or truncate.
            smile_chars: List of recognizable SMILES tokens/atoms. Defaults to standard alphabet.
        """
        self.row_padding = row_padding
        self.smile_chars = (
            list(smile_chars) if smile_chars is not None else list(DEFAULT_SMILE_CHARS)
        )
        self._char_to_index: dict[str, int] = {char: i for i, char in enumerate(self.smile_chars)}

    def smilechars2dict(self) -> dict[str, int]:
        """Return the dictionary mapping SMILES token to row index."""
        return self._char_to_index.copy()

    def getshape(self) -> tuple[int, int]:
        """Return the shape (n_tokens, row_padding) of the generated 2D matrix."""
        return (len(self.smile_chars), self.row_padding)

    def parse(self, smi: str) -> list[str]:
        """Parse a SMILES string into a list of constituent atom and bond tokens.

        Args:
            smi: Input SMILES string.

        Returns:
            List of parsed SMILES token strings.
        """
        tokens: list[str] = []
        size_smi = len(smi)
        j = 0
        while j < size_smi:
            if j + 1 < size_smi:
                ch2 = smi[j : j + 2]
                if ch2 in self._char_to_index:
                    tokens.append(ch2)
                    j += 2
                    continue
            ch1 = smi[j]
            if ch1 in self._char_to_index:
                tokens.append(ch1)
            else:
                logger.warning("SMILES character '%s' not recognized in vocabulary", ch1)
            j += 1
        return tokens

    def smi2mx(self, smi: str) -> list[list[int]]:
        """Convert a SMILES string into a 2D integer list of shape (n_tokens, row_padding).

        Args:
            smi: Input SMILES string.

        Returns:
            2D nested list with binary 1s marking token presence at corresponding positions.
        """
        tokens = self.parse(smi)
        num_rows = len(self.smile_chars)
        matrix: list[list[int]] = [[0] * self.row_padding for _ in range(num_rows)]

        size_smi = len(tokens)
        if size_smi > self.row_padding:
            logger.debug(
                "SMILES size %d exceeds padding limit %d; truncating", size_smi, self.row_padding
            )
            size_smi = self.row_padding

        for col_idx in range(size_smi):
            token = tokens[col_idx]
            if token in self._char_to_index:
                row_idx = self._char_to_index[token]
                matrix[row_idx][col_idx] = 1
            else:
                logger.warning("Token '%s' not found during matrix population", token)
        return matrix

    def smi2array(self, smi: str, dtype: Any = np.float32) -> np.ndarray:
        """Convert a SMILES string into a numpy ndarray of shape (n_tokens, row_padding).

        Args:
            smi: Input SMILES string.
            dtype: Numpy data type of the returned array.

        Returns:
            2D numpy array.
        """
        return np.asarray(self.smi2mx(smi), dtype=dtype)
