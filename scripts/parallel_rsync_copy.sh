#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SRC_MODEL_PATH=$1
DEST_PATH=$2
MODEL_NAME=$(basename $SRC_MODEL_PATH)
# DEST_PATH="/scratch/slurm_tmpdir/$SLURM_JOB_ID/step$SLURM_ARRAY_TASK_ID"

# mkdir dir beforehand to avoid parallel rsync to create dir in parallel
mkdir -p $DEST_PATH/$MODEL_NAME/

# run parallel rsyncs
# ls --indicator-style=none $SRC_MODEL_PATH/* | xargs -n1 -P5 -I% rsync -Pah % $DEST_PATH/$MODEL_NAME/
find "$SRC_MODEL_PATH" -mindepth 1 -maxdepth 1 \
  | xargs -n1 -P5 -I% rsync -Pah % "$DEST_PATH/$MODEL_NAME/"

# run one last rsync to double check that everything was transferred
rsync -Pah $SRC_MODEL_PATH/ $DEST_PATH/$MODEL_NAME/
