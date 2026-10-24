"""The per-clip steps the flows run (flows.py): clean, transform, split.

    audio.py      clean: decode, mono, 32 kHz, trim, loudness; reject or flag bad clips
    features.py   transform: log-mel features and summary vectors, as the QVIM baseline computes them
    splits.py     train/val/test by a seeded hash of each reference sound
"""
