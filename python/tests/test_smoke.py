import platform
import proximity_forests


def _pep440_from_cargo(version):
    return version.replace("-alpha.", "a").replace("-beta.", "b").replace("-rc.", "rc")


def test_native_module_imports():
    native_version, architecture = proximity_forests.runtime_info()
    assert _pep440_from_cargo(native_version) == proximity_forests.__version__
    assert architecture == platform.machine()
