## Building the Worker with a Network Volume

This is the simpler option.  No network volume is required.
The entire application will be stored within the Docker image
but will obviously create a more bulky Docker image as a result.

You can either use my pre-built Docker image:
```
ghcr.io/ashleykleynhans/runpod-worker-instantid:2.2.0
```

Or alternatively, you can build it yourself by following the
instructions below.

### Clone the repo

```bash
git clone https://github.com/ashleykleynhans/runpod-worker-instantid.git
```

### Build the Docker image

```bash
docker build -t dockerhub-username/runpod-worker-instantid:1.0.0 -f Dockerfile.Standalone .
docker login
docker push dockerhub-username/runpod-worker-instantid:1.0.0
```
