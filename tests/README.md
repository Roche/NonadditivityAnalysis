# Nonadditivity Tests


This is the unit and integration tests part of the nonadditivity package.
It is based on the [pytest](https://docs.pytest.org/) framework.

## Organisation


The tests themselves are written in [this](.) directory. Files needed or temporarily
created by the teses are in the [_test_files](_test_files) folder. The
[Fixtures](https://docs.pytest.org/en/6.2.x/fixture.html#fixture) used for the tests are
stored in the [fixtures](fixtures) folder and imported into the [conftest.py](conftest.py)
file. Pytest recognizes the fixtures automatically, so one does not need to import them in the test
files again.

```
    $ NonadditivityAnalysis/tests
    ├── __init__.py
    ├── conftest.py
    ├── test_1.py
    ├── ...
    ├── classificatoin/
        ├── test_classification_1.py
        ├── ...
    ├── fixtures/
        ├── __init__.py
        ├── 1_fixtures.py
        ├── ...
    ├── test_files/
        ├── __init__.py
        ├── test_file_1.py
        ├── ...
```

## Running the tests

Make sure that pytest is installed and run

```bash
    pytest
```

If you want to get an in depth coverage report run

```bash
    pytest --cov-report=term-missing --cov=nonadditivity tests/
```

## Add new tests

create a test case in an existing or a new file (filename: test_*.py). If you need fixtures, that are not
already part, add them in the [fixtures](fixtures) folder.
