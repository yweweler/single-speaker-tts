#!/bin/sh
# =============================================================================
# Regenerate the attention alignments, crop and store them in the target folder
# =============================================================================

BASE="/tmp"
TARGET="/thesis/tex/images/results/alignments"

pdfcrop $BASE/alignments_step_11501.pdf $TARGET/alignments_step_11501.pdf
pdfcrop $BASE/alignments_step_13001.pdf $TARGET/alignments_step_13001.pdf
pdfcrop $BASE/alignments_step_14001.pdf $TARGET/alignments_step_14001.pdf
pdfcrop $BASE/alignments_step_15001.pdf $TARGET/alignments_step_15001.pdf
