def test_package_imports():
    import wmagent
    import wmagent.data
    import wmagent.models
    import wmagent.train
    import wmagent.eval
    assert wmagent.__version__ == "0.1.0"
