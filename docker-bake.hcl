variable "REGISTRY" {
    default = "docker.io"
}

variable "REGISTRY_USER" {
    default = "ashleykza"
}

variable "APP" {
    default = "runpod-worker-instantid"
}

variable "RELEASE" {
    default = "1.1.5"
}

variable "CU_VERSION" {
    default = "124"
}

variable "CUDA_VERSION" {
    default = "12.4.1"
}

variable "TORCH_VERSION" {
    default = "2.6.0"
}

variable "XFORMERS_VERSION" {
    default = "0.0.29.post3"
}

target "default" {
    dockerfile = "Dockerfile.Standalone"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}"]
    args = {
        RELEASE = "${RELEASE}"
        CUDA_VERSION = "${CUDA_VERSION}"
        INDEX_URL = "https://download.pytorch.org/whl/cu${CU_VERSION}"
        TORCH_VERSION = "${TORCH_VERSION}+cu${CU_VERSION}"
        XFORMERS_VERSION = "${XFORMERS_VERSION}"
    }
}

target "network-volume" {
    dockerfile = "Dockerfile.Network_Volume"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}-network-volume"]
    args = {
        CUDA_VERSION = "${CUDA_VERSION}"
    }
}
