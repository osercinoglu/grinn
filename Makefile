# Makefile for gRINN Docker builds

# Image names
IMAGE_NAME = grinn
DEV_IMAGE_NAME = grinn-dev

# Build production image (with tests)
.PHONY: build
build:
	docker build --build-arg CACHEBUST=$$(date +%s) -t $(IMAGE_NAME) .

# Build development image (without tests, faster)
.PHONY: build-dev
build-dev:
	docker build -f Dockerfile.dev -t $(DEV_IMAGE_NAME) .

# Build without cache (forces complete rebuild)
.PHONY: build-no-cache
build-no-cache:
	docker build --no-cache --build-arg CACHEBUST=$$(date +%s) -t $(IMAGE_NAME) .

# Build development image without cache
.PHONY: build-dev-no-cache
build-dev-no-cache:
	docker build --no-cache -f Dockerfile.dev -t $(DEV_IMAGE_NAME) .

# Quick rebuild for code changes (uses cache-busting)
.PHONY: rebuild
rebuild:
	docker build --build-arg CACHEBUST=$$(date +%s) -t $(IMAGE_NAME) .

# Quick rebuild for development
.PHONY: rebuild-dev
rebuild-dev:
	docker build -f Dockerfile.dev -t $(DEV_IMAGE_NAME) .

# Test the built image
.PHONY: test
test:
	docker run --rm $(IMAGE_NAME) help

# Test the development image
.PHONY: test-dev
test-dev:
	docker run --rm $(DEV_IMAGE_NAME) help

# Interactive shell in production image
.PHONY: shell
shell:
	docker run --rm -it $(IMAGE_NAME) bash

# Interactive shell in development image
.PHONY: shell-dev
shell-dev:
	docker run --rm -it $(DEV_IMAGE_NAME) bash

# Clean up Docker images
.PHONY: clean
clean:
	docker rmi $(IMAGE_NAME) $(DEV_IMAGE_NAME) 2>/dev/null || true
	docker system prune -f

# Show help
.PHONY: help
help:
	@echo "gRINN Docker Build Commands:"
	@echo ""
	@echo "Production builds (with tests):"
	@echo "  make build           - Build production image"
	@echo "  make rebuild         - Quick rebuild with cache-busting"
	@echo "  make build-no-cache  - Force complete rebuild"
	@echo ""
	@echo "Development builds (faster, no tests):"
	@echo "  make build-dev           - Build development image"
	@echo "  make rebuild-dev         - Quick rebuild development image"
	@echo "  make build-dev-no-cache  - Force complete rebuild (dev)"
	@echo ""
	@echo "Testing:"
	@echo "  make test      - Test production image"
	@echo "  make test-dev  - Test development image"
	@echo "  make shell     - Interactive shell (production)"
	@echo "  make shell-dev - Interactive shell (development)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean     - Remove images and cleanup"
	@echo ""
	@echo "Examples for code development:"
	@echo "  make build-dev    # Initial build"
	@echo "  make rebuild-dev  # After code changes"
