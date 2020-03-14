MODULE := kaggle
BLUE='\033[0;34m'
NC='\033[0m' # No Color
FILES="*py|**/*py"
MESSAGE="Message"

run:
	@python  main.py

submit:
	@kaggle competitions submit -c house-prices-advanced-regression-techniques -f data/submission.csv -m $(MESSAGE)
test:
	@pytest

lint:
	@echo "\n${BLUE}Running Pylint against main files...${NC}\n"	
	@pylint --rcfile=setup.cfg *py || exit 0
	@echo "\n${BLUE}Running Pylint against package files...${NC}\n"
	@pylint --rcfile=setup.cfg **/*py || exit 0
	@echo "\n${BLUE}Running Flake8 against source and test files...${NC}\n"
	@flake8 || exit 0
	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
	@bandit -r --ini setup.cfg
clean:
	rm -rf .pytest_cache .coverage .pytest_cache coverage.xml

.PHONY: clean test
