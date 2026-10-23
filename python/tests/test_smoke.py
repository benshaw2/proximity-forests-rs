import proximity_forests


def test_native_module_imports():
    version, architecture = proximity_forests.runtime_info()
    assert version == proximity_forests.__version__
    assert architecture
