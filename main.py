#!/usr/bin/env python3
"""Run the whole pipeline: fetch from EDGAR, parse, write the dataset.

    python main.py --user-agent "Jane Doe jane@example.com"

The pipeline lives in curator/pipeline.py. This file imports it only when it
runs as a script. The parse step starts worker processes, and on Windows each
worker imports this file again. The guard keeps that import empty.
"""

import sys

if __name__ == "__main__":
    from curator.pipeline import main

    sys.exit(main())
