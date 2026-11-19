#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter short, corrupted, repetitive and boilerplate text using shared quality criteria."""
from __future__ import annotations
import re

_STOP = set(
    "the a an and or but if of to in on at by for with as is are was were be been "
    "being it its this that these those i you he she we they them his her their our "
    "your my me him us do does did have has had not no so than then there here what "
    "which who whom from about into over after before between out up down".split()
)
_WORD = re.compile(r"[A-Za-z']+")
_BOILER = ("slideshow controls", "see the looks", "photos:", "advertisement")


def is_clean(text, min_words=40, min_stop_ratio=0.15, min_unique_ratio=0.25):
    t = (text or "").strip()
    low = t.lower()
    if any(b in low for b in _BOILER):
        return False
    words = _WORD.findall(low)
    if len(words) < min_words:                    # sub-sentence quip
        return False
    if sum(1 for w in words if w in _STOP) / len(words) < min_stop_ratio:  # corrupted
        return False
    if len(set(words)) / len(words) < min_unique_ratio:                    # boilerplate/repeat
        return False
    return True


def filter_pair(ai, hum, n, **kw):
    """Filter both lists, truncate each to n, and report how many were dropped."""
    ca = [t for t in ai if is_clean(t, **kw)]
    ch = [t for t in hum if is_clean(t, **kw)]
    dropped_ai = len(ai) - len(ca); dropped_hu = len(hum) - len(ch)
    return ca[:n], ch[:n], dropped_ai, dropped_hu
