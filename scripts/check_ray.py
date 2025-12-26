# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from fire import Fire
import ray
import sys
import logging

# Configure logging to suppress excessive Ray connection messages
logging.basicConfig(level=logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)


def check_ray(ip_address: str, port: int = 10001):
    """
    Checks if a Ray head node is accessible at the given IP and port.

    Prints 'RUNNING' to stdout and exits with 0 if successful.
    Exits with 1 otherwise.
    """
    print(f"Checking Ray connection to {ip_address}:{port}...", file=sys.stderr)
    address = f"ray://{ip_address}:{port}"
    try:
        # Short timeout, we just want to see if connection is possible
        # ignore_version_mismatch=True might be useful if workers/head differ slightly
        ray.init(
            address=address,
            namespace="check_ray_connection",  # Use a temporary namespace
            logging_level=logging.FATAL,  # Suppress info messages
            ignore_reinit_error=True,
            _system_config={
                "lineage_pinning_enabled": False
            },  # Avoid hangs on shutdown with lineage
        )
        # If init succeeds, Ray is running
        print("RUNNING")
        ray.shutdown()  # Disconnect cleanly
        sys.exit(0)  # Exit with success code
    except (ConnectionError, TimeoutError, ray.exceptions.RaySystemError) as e:
        # Could not connect or other Ray client init issue
        # print(f"Debug: Ray connection failed: {e}", file=sys.stderr) # Optional debug line
        sys.exit(1)  # Exit with failure code
    except Exception as e:
        # Catch any other unexpected errors during init
        # print(f"Debug: Unexpected error during Ray check: {e}", file=sys.stderr) # Optional debug line
        sys.exit(1)  # Exit with failure code


if __name__ == "__main__":
    Fire(check_ray)
