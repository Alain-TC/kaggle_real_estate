MODULE := kaggle_blueprint
BLUE='\033[0;34m'
NC='\033[0m' # No Color
MESSAGE="Message"


# Where to push the docker image.
#REGISTRY ?= docker.pkg.github.com/martinheinz/python-project-blueprint
REGISTRY ?= python-project-blueprint

IMAGE := $(REGISTRY)/$(MODULE)

# This version-strategy uses git tags to set the version string
TAG := $(shell git describe --tags --always --dirty)

run:
	@python -m $(MODULE)

submit:
	@kaggle competitions submit -c house-prices-advanced-regression-techniques -f $(MODULE)/data/submission.csv -m $(MESSAGE)
	@echo "\n"
test:
	@pytest

lint:
	@echo "\n${BLUE}Running Pylint against source and test files...${NC}\n"
	@pylint --rcfile=setup.cfg **/*py **/**/*py || exit 0
	@echo "\n${BLUE}Running Flake8 against source and test files...${NC}\n"
	@flake8 || exit 0
	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
	@bandit -r --ini setup.cfg || exit 0

# Example: make build-prod VERSION=1.0.0
build-prod:
	@echo "\n${BLUE}Building Production image with labels:\n"
	@echo "name: $(MODULE)"
	@echo "version: $(VERSION)${NC}\n"
	@sed                                     \
	    -e 's|{NAME}|$(MODULE)|g'            \
	    -e 's|{VERSION}|$(VERSION)|g'        \
	    prod.Dockerfile | docker build -t $(IMAGE):$(VERSION) -f- .


build-dev:
	@echo "\n${BLUE}Building Development image with labels:\n"
	@echo "name: $(MODULE)"
	@echo "version: $(TAG)${NC}\n"
	@sed                                 \
	    -e 's|{NAME}|$(MODULE)|g'        \
	    -e 's|{VERSION}|$(TAG)|g'        \
	    dev.Dockerfile | docker build -t $(IMAGE):$(TAG) -f- .

# Example: make shell CMD="-c 'date > datefile'"
shell: build-dev
	@echo "\n${BLUE}Launching a shell in the containerized build environment...${NC}\n"
		@docker run                                                 \
			-ti                                                     \
			--rm                                                    \
			--entrypoint /bin/bash                                  \
			-u $$(id -u):$$(id -g)                                  \
			$(IMAGE):$(TAG)										    \
			$(CMD)


shell2:
	@echo "\n${BLUE}Launching a shell in the containerized build environment...${NC}\n"
		@docker run                                                 \
			-ti                                                     \
			--rm                                                    \
			--entrypoint /bin/bash                                  \
			-u $$(id -u):$$(id -g)                                  \
			$(IMAGE):$(TAG)                                                                             \
			$(CMD)




# Example: make push VERSION=0.0.2
push: build-prod
	@echo "\n${BLUE}Pushing image to GitHub Docker Registry...${NC}\n"
	@docker push $(IMAGE):$(VERSION)

version:
	@echo $(TAG)

.PHONY: clean image-clean build-prod push test

clean:
	rm -rf .pytest_cache .coverage .pytest_cache coverage.xml

.PHONY: clean test

docker-clean:
	@docker system prune -f --filter "label=name=$(MODULE)"
