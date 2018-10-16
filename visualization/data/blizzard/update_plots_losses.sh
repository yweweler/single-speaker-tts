#!/bin/sh
# =============================================================================
# Regenerate the training curves, crop and store them in the target folder
# =============================================================================

BASE="/thesis/project/visualization/data/blizzard/nancy/survey"
TARGET="/thesis/tex/images/loss/blizzard"

pdfcrop $BASE/loss_loss_decoder.pdf          $TARGET/loss_loss_decoder.pdf
pdfcrop $BASE/loss_loss_eval.pdf             $TARGET/loss_loss_eval.pdf
pdfcrop $BASE/loss_loss_post_processing.pdf  $TARGET/loss_loss_post_processing.pdf
pdfcrop $BASE/loss_loss_train.pdf            $TARGET/loss_loss_train.pdf
