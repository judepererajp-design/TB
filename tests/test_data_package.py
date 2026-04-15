import importlib


def test_data_package_exposes_database_module():
    data_pkg = importlib.import_module("data")
    database_module = data_pkg.database

    assert database_module is importlib.import_module("data.database")
