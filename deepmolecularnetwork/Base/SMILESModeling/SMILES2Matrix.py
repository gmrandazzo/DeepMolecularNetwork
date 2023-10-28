#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

class SMILES2MX(object):
    def __init__(self, row_padding=256):
        self.row_padding = row_padding
        self.smile_chars = ['C', # aliphatic carbon
                            'c', # aromatic carbon
                            'O', # aliphatic oxigen
                            'o', # aromatic oxigen
                            'N', # aliphatic nitrogen
                            'n', # aromatic nitrogen
                            'S', # aliphatic sulfur
                            's', # aromatic sulfur
                            'P', # aliphatic phosphor
                            'p', # aromatic phosphor
                            'F', # ...
                            'Cl',
                            'Br',
                            'I',
                            'H',
                            'Si',
                            'Se',
                            'se',
                            'As',
                            'B',
                            'b',
                            'Na',
                            'Fe',
                            'Mg',
                            '=',
                            '#',
                            '0',
                            '1',
                            '2',
                            '3',
                            '4',
                            '5',
                            '6',
                            '7',
                            '8',
                            '9',
                            '(',
                            ')',
                            '[',
                            ']',
                            '@',
                            '@@',
                            '/',
                            '\\',
                            '-',
                            '+',
                            '%']

    def smilechars2dict(self):
        d = {}
        i = 0
        for item in self.smile_chars:
            d[item] = i
            i += 1
        return d

    def getshape(self):
        return (len(self.smile_chars), self.row_padding)

    def parse(self, smi):
        s = []
        size_smi = len(smi)
        j = 0
        while j < size_smi:
            if j+1 < size_smi:
                ch = smi[j]+smi[j+1]
                if ch in self.smile_chars:
                    s.append(ch)
                    j += 2
                else:
                    ch = smi[j]
                    if ch in self.smile_chars:
                        s.append(ch)
                        j += 1
                    else:
                        print("Error: %s not found" % (ch))
                        j += 1
            else:
                ch = smi[j]
                if ch in self.smile_chars:
                    s.append(ch)
                    j += 1
                else:
                    print("Error: %s not found" % (ch))
                    j += 1
        return s

    def smi2mx(self, smi):
        psmi = self.parse(smi)
        d = self.smilechars2dict()
        m = []
        for i in range(len(d.keys())):
            m.append([])
            for j in range(self.row_padding):
                m[-1].append(0)

        size_smi = len(psmi)
        if size_smi > self.row_padding:
            print("SMILES %s size %d > %d" % (smi,
                                              size_smi,
                                              self.row_padding))
            size_smi = self.row_padding

        for j in range(size_smi):
            if psmi[j] in d.keys():
                i = d[psmi[j]]
                m[i][j] = 1
            else:
                print("Error: char %s not found!" % (psmi[j]))
        return m


if __name__ == '__main__':
    s = SMILES2MX(30)
    #smi = "c1cc(ccc1/C=C/C(=O)O)OCl"
    smi = "C[C@@H]1CC[C@@]2(CC[C@@]3(C(=CC[C@H]4[C@]3(CC[C@@H]5[C@@]4(CC[C@@H](C5(C)C)O)C)C)[C@@H]2[C@H]1C)C)C(=O)O"
    psmi = s.parse(smi)
    print(psmi)
    m = s.smi2mx(smi)
    for row in m:
        print(row)
