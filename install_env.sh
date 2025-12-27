#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e  # Exit immediately if a command exits with a non-zero status

install_libsndfile() {
    # configurations
    target_lib_dir=$1

    echo "[INFO] Installing libsndfile to $target_lib_dir..."

    # clean up the target directory
    rm -rf "$target_lib_dir/libsndfile*"
    rm -rf "$target_lib_dir/libvorbis*"

    # copy all the .so files into the target directory
    # -rwxrwxr-x 1 xxx 576K xxx libsndfile.so.1.0.31
    # -rwxrwxr-x 1 xxx 245K xxx libvorbis.so.0
    # -rwxrwxr-x 1 xxx 789K xxx libvorbisenc.so.2
    # -rwxrwxr-x 1 xxx  53K xxx libvorbisfile.so.3
    wget -q https://dl.fbaipublicfiles.com/seamless_next/lib/libsndf.tar.gz
    tar -xzf libsndf.tar.gz -C $target_lib_dir
    rm libsndf.tar.gz

    # create symlinks
    # lrwxrwxrwx 1 xxx   20 xxx libsndfile.so -> libsndfile.so.1.0.31*
    # lrwxrwxrwx 1 xxx   20 xxx libsndfile.so.1 -> libsndfile.so.1.0.31*
    # lrwxrwxrwx 1 xxx   17 xxx libvorbisenc.so -> libvorbisenc.so.2*
    # lrwxrwxrwx 1 xxx   17 xxx libvorbisenc.so.2.0.12 -> libvorbisenc.so.2*
    # lrwxrwxrwx 1 xxx   18 xxx libvorbisfile.so -> libvorbisfile.so.3*
    # lrwxrwxrwx 1 xxx   18 xxx libvorbisfile.so.3.3.8 -> libvorbisfile.so.3*
    # lrwxrwxrwx 1 xxx   14 xxx libvorbis.so -> libvorbis.so.0*
    # lrwxrwxrwx 1 xxx   14 xxx libvorbis.so.0.4.9 -> libvorbis.so.0*
    cd $target_lib_dir
    ln -sf libsndfile.so.1.0.31 libsndfile.so
    ln -sf libsndfile.so.1.0.31 libsndfile.so.1
    ln -sf libvorbis.so.0 libvorbis.so.0.4.9
    ln -sf libvorbis.so.0 libvorbis.so
    ln -sf libvorbisenc.so.2 libvorbisenc.so
    ln -sf libvorbisenc.so.2 libvorbisenc.so.2.0.12
    ln -sf libvorbisfile.so.3 libvorbisfile.so
    ln -sf libvorbisfile.so.3 libvorbisfile.so.3.3.8
    cd - > /dev/null

    echo "[INFO] libsndfile installation completed."
}

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell config to make uv available in current session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment..."
uv venv ./env

echo "Activating virtual environment..."
source ./env/bin/activate

echo "Syncing fairseq2 dependencies..."
cd fairseq2
uv sync --extra fs2v05-pt271-cu128 --group llm --active

echo "Installing additional packages..."
uv pip install polars retrying pandas xxhash importlib_resources ruamel.yaml importlib_metadata pyarrow torcheval sacrebleu editdistance

install_libsndfile $VIRTUAL_ENV/lib64

echo "Environment setup complete!"
