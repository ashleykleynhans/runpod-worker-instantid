# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.3] - 2026-08-31

### Changed
- Fixed pip dependency conflict by bumping transformers to 5.x and huggingface-hub to <2.0 (diffusers 0.40.0 requires huggingface-hub>=1.23.0)
- Bumped diffusers to 0.40.0
- Bumped release to 2.4.3

## [2.4.2] - 2026-05-05

### Changed
- Bumped diffusers requirement from >=0.37.1,<1.0 to >=0.38.0,<1.0
- Removed Docker Pulls badge from README

## [2.4.1] - 2026-04-25

### Changed
- Pinned torch==2.6.0 and added constraints file to prevent xformers compatibility breakage
- Configured Dependabot to ignore torch, xformers, torchvision, and torchaudio updates
- Bumped huggingface-hub, transformers, peft, diffusers, and accelerate dependencies
- Added upgrading section for network volume to docs

## [2.4.0] - 2026-04-08

### Added
- Added Dependabot config to block major version bumps and RC versions
- Pinned dependency version ranges with upper bounds

## [2.2.3] - 2026-02-24

### Changed
- Upgraded Python to version 3.11

## [2.2.2] - 2026-02-24

### Changed
- Fixed issues with using venv on the Network Volume
- Included runpod module in requirements.txt

## [2.2.0] - 2026-02-24

### Added
- Added tests
- Added GitHub workflow to automatically run the tests
- Added parameter names

### Changed
- Renamed `rp_handler.py` to `handler.py`
- Updated the GitHub workflow that automatically builds the Docker images to only build the images if the tests are successful
- Updated the GitHub workflow that automatically builds the Docker images to build both Docker image variants

## [1.1.5] - 2026-02-16

### Added
- Added GitHub workflow to automatically build the Docker image

### Changed
- Bumped CUDA to version 12.4
- Bumped torch to version 2.6.0
- Bumped xformers to version 0.0.29.post3
- Bumped diffusers to version 0.36.0
- Bumped transformers to version 4.57.6

## [1.0.12] - 2024-04-11

### Added
- Added support for specifying width and height
- Added badges

### Changed
- Switched out Hugging Face repo with my own
- Updated instructions for testing
- Fixed typo in the docs

## [1.0.10] - 2024-01-30

### Changed
- Load libtcmalloc.so to improve memory management

## [1.0.9] - 2024-01-30

Initial release.
