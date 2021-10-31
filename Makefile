flake:
	flake8 ./src ./examples ./tests

isort:
	isort ./src ./examples ./tests

black:
	black ./src ./examples ./tests

mypy:
	mypy ./src ./examples ./tests

lint:
	make isort && make black && make flake  && make mypy

test:
	pytest ./tests
