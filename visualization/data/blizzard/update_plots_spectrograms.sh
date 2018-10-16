#!/bin/sh
# =============================================================================
# Regenerate the spectrograms, crop and store them in the target folder
# =============================================================================

BASE="/tmp"
TARGET="/thesis/images/implementation/features"

pdfcrop $BASE/linear_spectrogram_raw_mag_db.pdf  $TARGET/linear_spectrogram_raw_mag_db.pdf
pdfcrop $BASE/mel_spectrogram_raw_mag_db.pdf     $TARGET/mel_spectrogram_raw_mag_db.pdf
